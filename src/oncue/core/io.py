"""Canonical JSON, content digests, atomic writes, and scene (de)serialization."""

from __future__ import annotations

import dataclasses
import enum
import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import Config
from .replay import fill_gaps
from .types import (
    ANCHOR_FILE_KEY,
    FILE_KEY,
    BoundarySegment,
    LaneNeighbor,
    MapFeature,
    Scene,
    SceneSource,
    SignalTrack,
    Track,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

    from .config import ReplayConfig

FILE_DIGEST_CHUNK_BYTES = 1 << 20  # `file_sha256` reads this much at a time


def jsonable(value: Any) -> Any:
    """Convert ``value`` to JSON types; a dataclass's fields are keyed by ``FILE_KEY`` or name."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.metadata.get(FILE_KEY, f.name): jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON string for ``value``: sorted keys, no whitespace, no NaN."""
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """Return the sha256 hex digest of ``canonical_json(value)``."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the sha256 of a file's content, read in chunks."""
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(FILE_DIGEST_CHUNK_BYTES), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: str | Path, value: Any, *, indent: int | None = None) -> Path:
    """Atomically write ``value`` to ``path`` as JSON, keys sorted when ``indent`` is None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(jsonable(value), indent=indent, allow_nan=False, sort_keys=indent is None)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON file and return the decoded value."""
    return json.loads(Path(path).read_text())


def read_lines(path: str | Path) -> Iterator:
    """Yield the JSON value on each non-empty line of a JSONL file."""
    with open(path) as handle:
        for line in handle:
            text = line.rstrip("\n")
            if text:
                yield json.loads(text)


def scene_from_dict(data: dict) -> Scene:
    def feature(f: dict) -> MapFeature:
        boundaries = {
            side: tuple(BoundarySegment(**b) for b in f.get(side, ()))
            for side in ("left_boundaries", "right_boundaries")
        }
        neighbors = {
            side: tuple(LaneNeighbor(**n) for n in f.get(side, ()))
            for side in ("left_neighbors", "right_neighbors")
        }
        return MapFeature(
            f["feature_id"],
            f["kind"],
            f["points"],
            lane_type=f.get("lane_type"),
            edge_type=f.get("edge_type"),
            line_type=f.get("line_type"),
            entry_lanes=tuple(f.get("entry_lanes", ())),
            exit_lanes=tuple(f.get("exit_lanes", ())),
            controlled_lanes=tuple(f.get("controlled_lanes", ())),
            speed_limit_mps=f.get("speed_limit_mps"),
            z=f.get("z"),
            **boundaries,
            **neighbors,
        )

    return Scene(
        scene_id=data["scene_id"],
        timestamps=data["timestamps"],
        anchor=data[ANCHOR_FILE_KEY],
        ego_id=data["ego_id"],
        tracks={k: Track(**t) for k, t in data["tracks"].items()},
        map_features=tuple(feature(f) for f in data.get("map_features", ())),
        drivable_area=tuple(data.get("drivable_area", ())),
        signals=tuple(SignalTrack(**g) for g in data.get("signals", ())),
        source=SceneSource(**data["source"]) if data.get("source") else None,
        notes=data.get("notes", {}),
    )


# Run records a scene directory may hold besides scenes; these are not scenes.
METADATA_FILES = ("export.json", "extract.json", "features.json", "run.json")


SCENE_SHARD_SUFFIX = ".scenes.jsonl.gz"  # an ``oncue export`` shard; other shards are raw WOMD


def scene_inputs(paths: Iterable[str | Path]) -> list[Path]:
    """Expand ``paths`` into scene files, each once; a directory gives its shards and scene JSON."""
    out = []
    for p in map(Path, paths):
        if not p.is_dir():
            out.append(p)
            continue
        out += sorted(p.glob(f"*{SCENE_SHARD_SUFFIX}"))
        out += sorted(
            f
            for f in p.glob("*.json")
            if f.name not in METADATA_FILES and not f.name.endswith(".settings.json")
        )
    seen, unique = set(), []
    for f in out:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def same_name_groups(paths: Iterable[str | Path]) -> list[list[str]]:
    """Return groups of inputs sharing a file name, in input order; only groups of size >= 2."""
    by_name: dict[str, list[str]] = {}
    for path in map(Path, paths):
        by_name.setdefault(path.name, []).append(str(path))
    return [group for group in by_name.values() if len(group) > 1]


def save_scene(scene: Scene, path: str | Path) -> Path:
    """Write ``scene`` to ``path`` atomically."""
    return write_json(path, scene)


def load_scene(path: str | Path) -> Scene:
    """Load a scene from ``path``."""
    return scene_from_dict(read_json(path))


def load_config(run_json: str | Path | None = None) -> Config:
    """Return the generation config: from `run_json`'s "config" if given, otherwise the defaults."""
    if run_json is None:
        return Config()
    data = read_json(run_json)
    return Config.from_dict(data.get("config", data))


def iter_scenes(
    path: str | Path,
    part: int = 0,
    parts: int = 1,
    replay: ReplayConfig | None = None,
    wanted: Callable[[str], bool] | None = None,
) -> Iterator[Scene]:
    """Yield scenes of a shard (*.scenes.jsonl.gz, every `parts`-th from `part`) or a scene file.

    Scenes are gap-filled with `replay` (None: as logged); `wanted(scene_id)` filters them before
    they are built.
    """
    path = Path(path)
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt") as f:
            records = (json.loads(line) for i, line in enumerate(f) if i % parts == part)
            for record in records:
                if wanted is not None and not wanted(record["scene_id"]):
                    continue
                scene = scene_from_dict(record)
                yield scene if replay is None else fill_gaps(scene, replay)
    elif part == 0:
        scene = load_scene(path)
        if wanted is None or wanted(scene.scene_id):
            yield scene if replay is None else fill_gaps(scene, replay)
