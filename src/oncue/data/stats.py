"""``oncue stats``: log statistics behind the target parameters, per road-user type.

Each road user valid over the 8 s future window gives one row of ``COLUMNS``; ``summarize``
turns the rows into percentiles with scene-bootstrap 95 % intervals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.io import SCENE_SHARD_SUFFIX, iter_scenes, scene_inputs
from ..core.kinematics import SMOOTH_FRAMES, measures, speeds
from ..core.road import VehicleLanes
from ..utils.geometry import heading_unit_vector
from ..utils.parallel import run_jobs
from ..utils.tracks import frame_of

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..core.types import Scene, Track

TYPES = ("pedestrian", "cyclist", "vehicle")
QUANTITIES = (
    "length",
    "width",
    "height",
    "peak_speed",
    "accel",
    "jerk",
    "lateral",
    "yaw_rate",
    "moving_speed",
)
# measures that apply only to some road users (NaN for the rest)
EXTRA = ("crossing_speed", "crossing_deviation_deg", "limit_ratio", "over_limit", "start_jump")
COLUMNS = (*QUANTITIES, "start_accel", *EXTRA)  # a road user's row of numbers
# p1 for the too-slow label, p5 and p95 for the common speeds, p99 for the rare-motion label.
PERCENTILES = (1, 5, 25, 50, 75, 95, 99, 99.5)
BOOTSTRAP_RESAMPLES = 1000  # scene resamples behind each percentile's 95 % interval
BOOTSTRAP_SEED = 0
ROWS_SUFFIX = ".rows.npz"  # a batch's rows (``--rows-out``)
RAW_MARK = ".tfrecord-"  # substring in a raw WOMD shard's name (``training.tfrecord-00000-...``)
MOVING_MPS = 1.0  # a road user with a peak speed at least this is "moving"
FIRST_KINEMATIC = QUANTITIES.index("peak_speed")  # the first motion column
START_COLUMN = COLUMNS.index(
    "start_accel"
)  # column of ``start_accel`` (NaN for road users that never start)
STANDSTILL_MPS = 0.5  # standstill: a step slower than this
START_WINDOW_S = 1.0  # window for the start acceleration after a road user leaves standstill
WORKERS = 8  # processes, one shard each at a time
MIN_FUTURE_FRAMES = 60  # minimum future frames a road user needs to be measured
CROSSING_LEAD_S = 1.0  # a lane entry requires this long outside first; the speed covers that span
MIN_WAY_M = 0.5  # the crossing angle needs a way at least this long over the lead time


@dataclass
class Rows:
    """Measured road users, flattened: type, scene, ``COLUMNS`` values, and metadata."""

    kinds: np.ndarray  # [N] road-user type
    scenes: np.ndarray  # [N] scene id
    values: np.ndarray  # [N, len(COLUMNS)]
    files: list[str]
    min_frames: int


def run(
    paths: Sequence[str | Path],
    out: str | Path | None = None,
    workers: int = WORKERS,
    min_frames: int = MIN_FUTURE_FRAMES,
    rows_out: str | Path | None = None,
) -> dict | None:
    """Compute statistics for ``paths``, write them to ``out`` and return them.

    Returns ``None`` without ``out``. ``rows_out`` saves the rows; passing row files as
    ``paths`` later merges them instead of re-reading the scenes.
    """
    rows = gather(paths, workers, min_frames)
    if rows_out is not None:
        save_rows(rows, Path(rows_out))
    if out is None:
        return None
    result = summarize(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")
    return result


def compute(
    paths: Sequence[str | Path], workers: int = WORKERS, min_frames: int = MIN_FUTURE_FRAMES
) -> dict:
    """Return the statistics dict for ``paths`` without writing anything."""
    return summarize(gather(paths, workers, min_frames))


def gather(paths: Sequence[str | Path], workers: int, min_frames: int) -> Rows:
    """Collect rows from ``paths``: merge row files, or measure scenes."""
    files = scene_files(paths)
    if not files:  # never an empty result: row files in a directory are not picked up (give them)
        raise ValueError(f"no scene shards, raw shards or row files in {list(map(str, paths))}")
    saved = [f for f in files if f.name.endswith(ROWS_SUFFIX)]
    if saved and len(saved) != len(files):
        raise ValueError("inputs mix row files and scene shards: give one kind")
    return load_rows(saved) if saved else collect(files, workers, min_frames)


def collect(files: Sequence[Path], workers: int, min_frames: int) -> Rows:
    """Measure every eligible road user in ``files``, one shard per process."""
    kinds, scenes, values = [], [], []
    jobs = [(str(f), min_frames) for f in files]
    for shard_rows in run_jobs(_shard, jobs, workers, ordered=True):
        for kind, scene_id, *numbers in shard_rows:
            kinds.append(kind)
            scenes.append(scene_id)
            values.append(numbers)
    return Rows(
        np.array(kinds, str),
        np.array(scenes, str),
        np.array(values, float).reshape(-1, len(COLUMNS)),
        [f.name for f in files],
        min_frames,
    )


def save_rows(rows: Rows, path: Path) -> None:
    """Write ``rows`` to ``path`` atomically (temp file beside, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial.npz")
    np.savez_compressed(
        partial,
        kinds=rows.kinds,
        scenes=rows.scenes,
        values=rows.values,
        columns=np.array(COLUMNS),
        files=np.array(rows.files),
        min_frames=rows.min_frames,
    )
    partial.replace(path)


def load_rows(paths: Sequence[Path]) -> Rows:
    """Merge row files, refusing other columns, mixed future windows and repeated shards."""
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            if tuple(saved["columns"]) != COLUMNS:
                raise ValueError(f"{path}: columns {tuple(saved['columns'])}, not {COLUMNS}")
            parts.append(
                Rows(
                    saved["kinds"],
                    saved["scenes"],
                    saved["values"],
                    [str(f) for f in saved["files"]],
                    int(saved["min_frames"]),
                )
            )
    windows = {part.min_frames for part in parts}
    if len(windows) != 1:
        raise ValueError(f"row files with different future windows: {sorted(windows)}")
    files = [f for part in parts for f in part.files]
    if len(set(files)) != len(files):
        raise ValueError("a shard is in two row files")
    return Rows(
        np.concatenate([part.kinds for part in parts]),
        np.concatenate([part.scenes for part in parts]),
        np.concatenate([part.values for part in parts]),
        files,
        windows.pop(),
    )


def summarize(rows: Rows) -> dict:
    """Return per-type percentiles with scene-bootstrap intervals and counts.

    Kinematic measures count only moving road users (peak speed at least ``MOVING_MPS``).
    """
    out = {
        "files": rows.files,
        "min_frames": rows.min_frames,
        "moving_mps": MOVING_MPS,
        "percentiles": list(PERCENTILES),
        "types": {},
    }
    for kind in TYPES:
        mine = rows.kinds == kind
        if not mine.any():
            continue
        x, scenes = rows.values[mine], rows.scenes[mine]
        moving = x[:, FIRST_KINEMATIC] >= MOVING_MPS

        started = np.isfinite(x[:, START_COLUMN])
        starts = x[:, START_COLUMN][started]
        counts = {
            "n": len(x),
            "n_moving": int(moving.sum()),
            "n_starts": len(starts),
            "n_scenes": len(set(scenes)),
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        row_masks = {
            q: moving if i >= FIRST_KINEMATIC else np.ones(len(x), bool)
            for i, q in enumerate(QUANTITIES)
        }
        extra = _extra_values(x, moving)
        out["types"][kind] = (
            counts
            | _extra_counts(x, moving)
            | {q: _percentiles(x[rows_of, i]) for i, (q, rows_of) in enumerate(row_masks.items())}
            | {"start_accel": _percentiles(starts)}
            | {name: _percentiles(values) for name, (values, _) in extra.items()}
            | {
                "ci95": {
                    q: scene_bootstrap(x[rows_of, i], scenes[rows_of], rng)
                    for i, (q, rows_of) in enumerate(row_masks.items())
                    if i >= FIRST_KINEMATIC
                }
                | {"start_accel": scene_bootstrap(starts, scenes[started], rng)}
                | {
                    name: scene_bootstrap(values, scenes[which], rng)
                    for name, (values, which) in extra.items()
                }
            }
        )
    return out


def _extra_values(x: np.ndarray, moving: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{measure: (values, row mask)}`` for ``EXTRA`` and ``peak_over_moving``."""
    out = {}
    for name in EXTRA:
        column = x[:, COLUMNS.index(name)]
        which = np.isfinite(column)
        if name == "crossing_speed":
            which &= column >= STANDSTILL_MPS
        if name == "over_limit":
            which &= moving
        out[name] = (column[which], which)
    pace = x[:, COLUMNS.index("moving_speed")]
    which = moving & (pace > 0)
    out["peak_over_moving"] = (x[which, COLUMNS.index("peak_speed")] / pace[which], which)
    return out


def _extra_counts(x: np.ndarray, moving: np.ndarray) -> dict:
    """Return the count per ``EXTRA`` measure and the share of lane entries from standing."""
    counts = {f"n_{name}": int(np.isfinite(x[:, COLUMNS.index(name)]).sum()) for name in EXTRA}
    counts["n_over_limit"] = int((np.isfinite(x[:, COLUMNS.index("over_limit")]) & moving).sum())
    entered = x[:, COLUMNS.index("crossing_speed")]
    entered = entered[np.isfinite(entered)]
    counts["crossing_standing_share"] = (
        round(float((entered < STANDSTILL_MPS).mean()), 3) if len(entered) else None
    )
    return counts


def scene_bootstrap(
    values: np.ndarray, scenes: np.ndarray, rng: np.random.Generator
) -> list[list[float]] | None:
    """Return each percentile's ``[low, high]`` 95 % interval over scene resamples.

    A scene drawn ``k`` times counts its road users ``k`` times. ``None`` when ``values`` is empty.
    """
    if not len(values):
        return None
    ids, groups = np.unique(scenes, return_inverse=True)
    order = np.argsort(values)
    values, groups = values[order], groups[order]
    targets = np.array(PERCENTILES) / 100
    draws = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        weight = np.cumsum(drawn[groups])
        at = np.searchsorted(weight, targets * weight[-1], side="left")
        draws.append(values[np.minimum(at, len(values) - 1)])
    low, high = np.percentile(np.array(draws), (2.5, 97.5), axis=0)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(low, high, strict=True)]


def scene_files(paths: Sequence[str | Path]) -> list[Path]:
    """Return the scene shards, row files and raw WOMD shards among ``paths``."""
    return [
        f
        for f in scene_inputs(paths)
        if f.name.endswith((SCENE_SHARD_SUFFIX, ROWS_SUFFIX)) or RAW_MARK in f.name
    ]


def _shard(item: tuple[str, int]) -> list[tuple]:
    """Return one row per eligible road user in a shard."""
    path, min_frames = item
    return [row for scene in _scenes(path) for row in _track_rows(scene, min_frames)]


def _percentiles(values: np.ndarray) -> list[float] | None:
    """Return ``values``' ``PERCENTILES``, rounded; None when there are no values."""
    if not len(values):
        return None
    return [round(float(v), 3) for v in np.percentile(values, PERCENTILES)]


def _scenes(path: str) -> Iterator[Scene]:
    """Yield the scenes of a scene shard, or of a raw shard without writing them."""
    if path.endswith(SCENE_SHARD_SUFFIX):
        yield from iter_scenes(path)
        return
    from .waymo.scenes import load_record
    from .waymo.tfrecord import index_records

    for ref in index_records(Path(path), None):
        yield load_record(ref)


def _track_rows(scene: Scene, min_frames: int) -> list[tuple]:
    """Return one row per eligible road user in ``scene``."""
    rows = []
    anchor = scene.anchor
    lanes = None  # read on demand the first time a road user needs them
    for t in scene.tracks.values():
        if t.track_id == scene.ego_id or t.agent_type not in TYPES:
            continue
        if len(t.valid) - anchor < min_frames or not t.valid[anchor:].all():
            continue
        xy = t.states[anchor:, :2]
        m = measures(xy, scene.dt)
        if m is None:
            continue
        start = start_accel(xy, scene.dt)
        lanes = lanes or VehicleLanes(scene)
        if t.agent_type == "vehicle":
            extra = (np.nan, np.nan, *_vehicle(t, anchor, scene.dt, lanes))
        else:
            extra = (*_crossing(t, scene.dt, lanes), np.nan, np.nan, np.nan)
        rows.append(
            (
                t.agent_type,
                scene.scene_id,
                float(t.size[0]),
                float(t.size[1]),
                float(t.height or 0.0),
                *(m[k] for k in QUANTITIES[FIRST_KINEMATIC:]),
                np.nan if start is None else start,
                *extra,
            )
        )
    return rows


def start_accel(xy: np.ndarray, dt: float) -> float | None:
    """Return the peak speeding up within ``START_WINDOW_S`` after leaving standstill.

    ``None`` when the road user never leaves standstill, or leaves it too early to measure.
    """
    step = np.hypot(*np.diff(xy, axis=0).T) / dt
    still = np.flatnonzero(step < STANDSTILL_MPS)
    if not len(still):
        return None
    moving = np.flatnonzero(step >= MOVING_MPS)
    moving = moving[moving > still[0]]
    if not len(moving):
        return None
    go = int(still[still < moving[0]][-1]) + 1  # first point past the last standstill step
    half = SMOOTH_FRAMES // 2
    if go < half:
        return None
    end = go + frame_of(START_WINDOW_S, dt)  # windows centred before this point count
    found = measures(xy[go - half : end + half], dt)  # the slice holds exactly those windows
    return None if found is None else found["speed_up"]


def _crossing(t: Track, dt: float, lanes: VehicleLanes) -> tuple[float, float]:
    """Return ``(speed, deviation)`` at the first lane entry after ``CROSSING_LEAD_S`` outside.

    Speed is the median over the lead time; deviation is from square across, in degrees.
    """
    lead, half = frame_of(CROSSING_LEAD_S, dt), SMOOTH_FRAMES // 2
    xy, valid = t.states[:, :2], t.valid
    inside = np.zeros(len(xy), bool)
    inside[valid] = lanes.inside(xy[valid])
    for entry in np.flatnonzero(valid & inside):
        first, last = entry - lead - half, entry + half  # windows centred over the lead time
        if first < 0 or last >= len(xy) or not valid[first : last + 1].all():
            continue
        if inside[entry - lead : entry].any():  # not fully outside through the lead time
            continue
        speed = float(np.median(speeds(xy[first : last + 1], dt)[half : half + lead + 1]))
        return speed, _deviation(xy[entry] - xy[entry - lead], xy[entry], lanes)
    return np.nan, np.nan


def _deviation(way: np.ndarray, at: np.ndarray, lanes: VehicleLanes) -> float:
    """Return the angle in degrees between ``way`` and square across the lane nearest ``at``.

    0 is perpendicular and 90 along the lane; ``NaN`` for a short way or a lane without direction.
    """
    _, direction = lanes.nearest(at[None, :])
    length = np.linalg.norm(way)
    if length < MIN_WAY_M or not np.isfinite(direction).all():
        return np.nan
    along = abs(float(way @ direction[0])) / length
    return 90.0 - float(np.degrees(np.arccos(np.clip(along, 0.0, 1.0))))


def _vehicle(t: Track, anchor: int, dt: float, lanes: VehicleLanes) -> tuple[float, float, float]:
    """Return ``(limit_ratio, over_limit, start_jump)`` over a vehicle's future window.

    ``limit_ratio`` is the median speed / limit while moving, ``over_limit`` the largest speed
    above the limit and ``start_jump`` the first step's velocity error (m/s); ``NaN`` if none.
    """
    states = t.states[anchor:]
    v = speeds(states[:, :2], dt)
    limit = lanes.limits_of(lanes.same_way_lanes(states))
    on = np.isfinite(v) & np.isfinite(limit)
    over = float(np.max(v[on] - limit[on])) if on.any() else np.nan
    moving = on & (v > MOVING_MPS)
    ratio = float(np.median(v[moving] / limit[moving])) if moving.any() else np.nan
    jump = np.nan
    if states[0, 3] >= MOVING_MPS and len(states) > 1:
        step = (states[1, :2] - states[0, :2]) / dt
        jump = float(np.linalg.norm(step - states[0, 3] * heading_unit_vector(states[0, 2])))
    return ratio, over, jump
