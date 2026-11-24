"""`extract` writes the trajectory generator's training features, stored per scene (numpy only).

Each scene once (`scene_record`: `waymo_dataset.encode_scene`, scene frame) with the road users
learned from (`scene_centers`): each one's route in its frame (`routes`) and whether it starts from
rest. Training and generation assemble a road user's inputs in its frame with the same function
(`waymo_dataset.assemble`). Shards are processed one at a time (`extract_scene_shard`) and written
as one compressed file each (`write_scenes`), split by scene hash into train and validation like
the scene export (`data.waymo.split.partition`).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ....core.io import file_sha256, same_name_groups
from ....core.types import Scene, Track
from ..routes import logged_route, route_chunks, scene_lanes
from .waymo_dataset import FRAME_KEYS, LAYOUT, TYPES, FeatureConfig, encode_scene, pad_frames

if TYPE_CHECKING:  # for annotations only: no new import when running
    from concurrent.futures import Executor

# package data (under the repository root, an installed package would have none)
EXCLUDE_DEFAULT = Path(__file__).with_name("safeshift_scored.txt")
SCENE_KEYS = tuple(LAYOUT)
SAMPLE_KEYS = ("center", "route", "route_ok", "rest")
EXTRACT_WORKERS = 4  # shards extracted at once (one process each)
# the vehicles' order in a scene: a seed from the first hex digits of the scene id's md5
SEED_HEX_DIGITS, HEX_BASE = 8, 16
# a sample starts from rest (up-weighted in training): standing over its history (speed below
# REST_SPEED) and moving more than REST_LEAVE_M over its future
REST_SPEED, REST_LEAVE_M = 0.3, 3.0


# A stored record may carry fields of the removed per-sample layout (rs_w352's): accepted only
# with the value the scene layout always had (absent: that value).
_SCENE_LAYOUT = {"layout": "scene", "chunk_points": 20, "radius_m": None, "map_radius_m": None,
                 "map_direction": True, "route_signals": True}  # fmt: skip


def feature_config(data: dict) -> FeatureConfig:
    """The FeatureConfig of a features.json or checkpoint record, the one entry for reading one;
    features of another layout or version are refused (ValueError)."""
    if not data or data.get("version") != FeatureConfig.version:
        raise ValueError(f"features of version {(data or {}).get('version')}: only version 5")
    data = dict(data)
    for key, only in _SCENE_LAYOUT.items():
        value = data.pop(key, only)
        if value != only:
            raise ValueError(f"features {key} {value!r}: only {only!r} (the scene layout)")
    return FeatureConfig(**data)


def extract(
    shards: list[str | Path],
    out_dir: str | Path,
    cfg: FeatureConfig | None = None,
    *,
    workers: int = EXTRACT_WORKERS,
    limit: int | None = None,
    version: str | None = None,
    split: str = "hash",
    exclude: str | Path | None = None,
) -> dict:
    """Every shard not yet extracted (resumable); records the feature settings, the scenes kept
    out (`exclude`: an exclusion file, recorded in `excluded.json`) and, per shard, its counts in
    `extract.jsonl`. SystemExit before anything is written for shards that share a file name
    (each would write OUT/<name>.<split>.npz and its mark)."""
    import json
    from concurrent.futures import ProcessPoolExecutor, as_completed

    clashes = same_name_groups(shards)
    if clashes:
        raise SystemExit(f"raw shards with one file name would write one feature file: {clashes}")
    cfg = cfg or FeatureConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, record = read_exclusions(exclude) if exclude else (frozenset(), None)
    # one directory, one extraction: a resumed run must match the recorded settings, so a
    # directory never mixes features of two settings
    settings = {  # (the split and limit are per run: recorded in each shard's mark)
        "features": asdict(cfg),
        "dataset_version": version,
        "exclusions": record,
    }
    run = out_dir / "extract.json"
    if run.exists():
        old = json.loads(run.read_text())
        differing = sorted(
            key for key in set(old) | set(settings) if old.get(key) != settings.get(key)
        )
        if differing:
            raise SystemExit(
                f"{out_dir}: extracted with other settings ({differing}); use a new dir"
            )
    elif list(out_dir.glob("*.npz")):
        raise SystemExit(f"{out_dir}: features without a record of their settings; use a new dir")
    else:
        run.write_text(json.dumps(settings, indent=1) + "\n")
    (out_dir / "excluded.json").write_text(json.dumps(record) + "\n")
    (out_dir / "features.json").write_text(json.dumps(asdict(cfg)))
    done = []
    with ProcessPoolExecutor(workers) as pool:
        todo = shards_to_extract(out_dir, shards, split, limit, pool)
        futures = [
            pool.submit(extract_scene_shard, shard, out_dir, cfg, limit, version, split, ids)
            for shard in todo
        ]
        for future in as_completed(futures):
            done.append({**future.result(), "split": split, "limit": limit})
            (out_dir / f"{done[-1]['shard']}.done").write_text(json.dumps(done[-1]) + "\n")
            with open(out_dir / "extract.jsonl", "a") as log_file:  # per shard: errors, counts
                log_file.write(json.dumps(done[-1]) + "\n")
    return {"shards": len(done), "skipped": len(list(shards)) - len(todo), "results": done}


def read_exclusions(path: str | Path) -> tuple[frozenset, dict]:
    """Scene ids of an exclusion file (one per line, '#' comments) and its record (file, md5,
    count) for `excluded.json`."""
    data = Path(path).read_bytes()
    lines = (line.split("#")[0].strip() for line in data.decode().splitlines())
    ids = frozenset(line for line in lines if line)
    record = {"file": Path(path).name, "md5": hashlib.md5(data).hexdigest(), "scenes": len(ids)}
    return ids, record


def shards_to_extract(
    out_dir: Path, shards: list[str | Path], split: str, limit: int | None, pool: Executor
) -> list[Path]:
    """The shards of `shards` not yet extracted to `out_dir`. A shard is done once all its outputs
    are written (its `.done` mark, with its counts, also of an empty split, and the raw shard's
    sha256): an interrupted one is done again whole. SystemExit for a shard done with another
    split or from other raw content (the done shards hashed in `pool`), or a directory holding
    shards of another record limit (else a shard done from a few records would be skipped as if
    extracted in full)."""
    import json

    marks = {
        mark.name[: -len(".done")]: json.loads(mark.read_text()) for mark in out_dir.glob("*.done")
    }
    other = sorted({mark.get("limit") for mark in marks.values()} - {limit}, key=str)
    if other:
        raise SystemExit(f"{out_dir}: shards extracted with record limit {other[0]}, this run "
                         f"{limit}; use a new dir")  # fmt: skip
    for shard in shards:
        mark = marks.get(Path(shard).name)
        if mark is not None and mark.get("split") != split:
            raise SystemExit(f"{out_dir}: {Path(shard).name} extracted with split "
                             f"{mark.get('split')}, this run {split}; use a new dir")  # fmt: skip
    extracted = [Path(shard) for shard in shards if Path(shard).name in marks]
    for shard, digest in zip(extracted, pool.map(file_sha256, extracted), strict=True):
        if marks[shard.name].get("sha256") != digest:
            raise SystemExit(f"{shard}: its content changed since it was extracted to {out_dir}; "
                             "use a new dir")  # fmt: skip
    return [Path(shard) for shard in shards if Path(shard).name not in marks]


def extract_scene_shard(
    shard: Path,
    out_dir: Path,
    cfg: FeatureConfig,
    limit: int | None = None,
    version: str | None = None,
    split: str = "hash",
    exclude: frozenset = frozenset(),
) -> dict:
    """The features of one raw shard, written as `<shard>.<split>.npz` (`write_scenes`). split
    "hash": train / val by scene hash (`data.waymo.split.partition`, 20% held out); "train" or
    "val": the whole shard (the official training split for training, the official validation
    split for validation). The result records the raw shard's sha256 (its mark keeps it)."""
    from ....data.waymo.scenes import DATASET_VERSION, load_record
    from ....data.waymo.split import partition
    from ....data.waymo.tfrecord import index_records

    shard_sha256 = file_sha256(shard)
    buckets: dict[str, list[dict]] = {"train": [], "val": []}
    errors = empty = excluded = 0
    for ref in index_records(shard, limit):
        try:  # the raw 10 Hz frames, as MTR (resampling to an exact 0.1 s grid drops the last
            # frame of the ~9% of scenes whose 91 timestamps span a little under 9 s)
            scene = load_record(ref, dt=None, dataset_version=version or DATASET_VERSION)
        except Exception:  # one bad record must not stop a shard
            errors += 1
            continue
        if scene.scene_id in exclude:
            excluded += 1
            continue
        bucket = (
            split if split != "hash" else "train" if partition(scene.scene_id) == "train" else "val"
        )
        record = scene_record(scene, cfg)
        if record is not None:
            buckets[bucket].append(record)
        else:
            empty += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, records in buckets.items():
        counts[name] = sum(len(record["center"]) for record in records)
        if records:
            write_scenes(records, out_dir / f"{shard.name}.{name}.npz")
    scenes = sum(len(records) for records in buckets.values())
    return {
        "shard": shard.name,
        "sha256": shard_sha256,
        "errors": errors,
        "excluded": excluded,
        "scenes": scenes,
        "no_samples": empty,
        **counts,
    }


def scene_record(scene: Scene, cfg: FeatureConfig) -> dict | None:
    """One scene in the scene layout: `waymo_dataset.encode_scene` plus the road users learned
    from, each with its route (in its frame) and whether it starts from rest (training weights)."""
    tracks = [track for track in scene.tracks.values() if track.agent_type in TYPES]
    centers = scene_centers(scene, cfg, tracks)
    if not centers:
        return None
    record = encode_scene(scene)
    anchor, history, future = scene.anchor, cfg.history, cfg.future
    lanes = scene_lanes(scene)
    routes, route_ok, rest = [], [], []
    for index in centers:
        track = tracks[index]
        origin, h0 = track.states[anchor, :2], track.states[anchor, 2]
        future_states = track.states[anchor + 1 : anchor + future + 1]
        world_future = np.vstack((track.states[anchor : anchor + 1, :2], future_states[:, :2]))
        headings = np.r_[track.states[anchor, 2], future_states[:, 2]]  # recorded headings
        route, usable = route_chunks(
            logged_route(track.agent_type, world_future, lanes, headings), origin, h0
        )
        routes.append(route)
        route_ok.append(usable)
        past = track.states[anchor - history + 1 : anchor + 1]
        still = (
            past[:, 3].max() < REST_SPEED and track.valid[anchor - history + 1 : anchor + 1].all()
        )
        moved = np.linalg.norm(future_states[-1, :2] - origin) > REST_LEAVE_M
        rest.append(bool(still and moved))
    record.update(
        center=np.array(centers, np.int32),
        route=np.array(routes, np.float32),
        route_ok=np.array(route_ok, np.float32),
        rest=np.array(rest, bool),
        a0=np.int32(anchor),
    )
    return record


def scene_centers(scene: Scene, cfg: FeatureConfig, tracks: list[Track]) -> list[int]:
    """Indices in `tracks` of the road users learned from: those with a complete future, every
    pedestrian and cyclist first, then the vehicles in a random order fixed by the scene id."""
    anchor, future = scene.anchor, cfg.future
    if anchor < cfg.history - 1 or len(scene.timestamps) < anchor + future + 1:
        return []
    complete = [
        index
        for index, track in enumerate(tracks)
        if track.valid[anchor] and track.valid[anchor + 1 : anchor + future + 1].all()
    ]
    seed = int(hashlib.md5(scene.scene_id.encode()).hexdigest()[:SEED_HEX_DIGITS], HEX_BASE)
    rng = np.random.default_rng(seed)
    vulnerable = [index for index in complete if tracks[index].agent_type != "vehicle"]
    vehicles = [index for index in complete if tracks[index].agent_type == "vehicle"]
    rng.shuffle(vehicles)
    return (vulnerable + vehicles)[: cfg.per_scene]


def write_scenes(records: list[dict], path: Path) -> None:
    """Scene records concatenated into one file: per-scene arrays with offsets (agent_off,
    map_off), the sample table (scene index, center, route, route_ok, rest), each scene's anchor
    frame (key a0)."""
    frames = max(record["agents"].shape[1] for record in records)  # a few scenes have 92 frames

    def padded(record: dict, key: str) -> np.ndarray:
        array = record[key]
        if key in FRAME_KEYS and array.shape[1] < frames:
            array = pad_frames(array, frames)
        return array

    out = {key: np.concatenate([padded(record, key) for record in records]) for key in SCENE_KEYS}
    out["agent_off"] = np.cumsum([0] + [len(record["agents"]) for record in records]).astype(
        np.int64
    )
    out["map_off"] = np.cumsum([0] + [len(record["map_pts"]) for record in records]).astype(
        np.int64
    )
    out["a0"] = np.array([record["a0"] for record in records], np.int32)
    out["sample_scene"] = np.concatenate(
        [np.full(len(record["center"]), index, np.int32) for index, record in enumerate(records)]
    )
    for key in SAMPLE_KEYS:
        out[key] = np.concatenate([record[key] for record in records])
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **out)
    temporary.replace(path)
