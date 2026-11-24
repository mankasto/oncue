"""95 % confidence intervals as ``Estimate`` objects, clustered by drive or WOMD log.

The unit is the drive once ``use_drives`` loads a drive map, else the log. Intervals are
percentile intervals from ``scipy.stats.bootstrap``; a zero-event ratio gets an exact bound.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse, stats
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

from ..core.io import SCENE_SHARD_SUFFIX
from ..utils.parallel import run_jobs

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable, Iterable, Sequence

    import numpy.typing as npt

RESAMPLES = 10_000
LEVEL = 0.95
METHOD = "percentile"
SEED = 0
BATCH = 1_000  # resamples held in memory at once
QUANTILE_METHOD = "inverted_cdf"
# Upper bound on ``resamples * (logs or distinct values)`` held at once for a quantile batch.
CELLS_AT_ONCE = 4_000_000
MIN_INTERVAL_UNITS = 2  # fewer units than this leaves nothing to resample
# Zero-event ratios get the exact Clopper-Pearson bound, since every draw would be zero.
EXACT_ALPHA = (1.0 - LEVEL) / 2.0
EXACT_METHOD = "clopper-pearson over the units with a denominator"


def settings() -> dict:
    """Return the bootstrap settings, and any active drive map's, recorded with every output."""
    out = {
        "unit": unit_name(),
        "resamples": RESAMPLES,
        "level": LEVEL,
        "method": METHOD,
        "seed": SEED,
        "quantile_method": QUANTILE_METHOD,
        "exact_alpha": EXACT_ALPHA,
        "zero_events": EXACT_METHOD,
    }
    if _DRIVES:
        drives = _DRIVES[0]
        out["drives"] = {
            "file": drives.path,
            **drives.settings,
            "logs": len(drives.drive_of),
            "drives": len(set(drives.drive_of.values())),
            "dropped_logs": len(drives.dropped),
        }
    return out


# Drive clustering (``cluster_logs_by_drive``).
DRIVE_OVERLAP_M = 0.05
DRIVE_SHARED_FRAMES = 10  # 1 s at 10 Hz
DRIVE_CANDIDATE_M = 2.0  # logs this close to a path's frames are candidates for its drive


def ego_path(scene: dict) -> np.ndarray:
    """Return the ego's world positions ``[frames, 2]`` at its valid frames in an exported scene."""
    from ..utils.geometry import scene_to_world

    ego = scene["tracks"][scene["ego_id"]]
    states = np.asarray(ego["states"], float)[np.asarray(ego["valid"], bool)]
    if not len(states):
        return np.empty((0, 2))
    return scene_to_world(states, scene["notes"]["world_origin"])[:, :2]


def shard_ego_paths(path: str | Path) -> dict[str, np.ndarray]:
    """Return ``{scene_id: ego_path}`` for an exported or raw WOMD shard (raw needs protobuf)."""
    path = Path(path)
    if path.name.endswith(SCENE_SHARD_SUFFIX):
        with gzip.open(path, "rt") as handle:
            return {
                scene["scene_id"]: ego_path(scene) for scene in map(json.loads, handle) if scene
            }
    from ..data.waymo.scenes import parse_scenario
    from ..data.waymo.tfrecord import iter_records

    out = {}
    for _, payload in iter_records(path):
        scenario = parse_scenario(payload)
        states = scenario.tracks[scenario.sdc_track_index].states
        out[str(scenario.scenario_id)] = np.array(
            [(s.center_x, s.center_y) for s in states if s.valid], float
        ).reshape(-1, 2)
    return out


def ego_paths(paths: Sequence[str | Path], workers: int = 1) -> dict[str, np.ndarray]:
    """Return ``{scene_id: ego_path}`` over shards; a scene id in two shards raises."""
    shards = list(run_jobs(shard_ego_paths, paths, workers, ordered=True))
    out: dict[str, np.ndarray] = {}
    for path, shard in zip(paths, shards, strict=True):
        if twice := sorted(shard.keys() & out.keys()):
            raise ValueError(f"{path}: scenes {twice[:3]} already read from another shard")
        out.update(shard)
    return out


def cluster_logs_by_drive(
    paths_by_log: dict[str, np.ndarray],
    overlap_m: float = DRIVE_OVERLAP_M,
    shared_frames: int = DRIVE_SHARED_FRAMES,
) -> dict[str, str]:
    """Return ``{log_id: drive_id}``.

    Two logs share a drive when each has ``shared_frames`` ego frames within ``overlap_m`` of
    the other's path; a drive is a connected group, named by its smallest log id.

    Args:
        paths_by_log: ``{log_id: [frames, 2] world ego positions}``.
    """
    import shapely

    ids = sorted(paths_by_log)
    paths = [np.asarray(paths_by_log[log], float).reshape(-1, 2) for log in ids]
    owner = np.repeat(np.arange(len(ids)), [len(path) for path in paths])
    tree = cKDTree(np.concatenate([np.empty((0, 2)), *paths]))
    lines = [shapely.LineString(path) if len(path) >= shared_frames else None for path in paths]
    first, second, frames = [], [], []
    for log, path in enumerate(paths):
        if lines[log] is None:
            continue
        points = shapely.points(path)
        for other in candidate_logs(tree, owner, path, log, shared_frames):
            if lines[other] is None:
                continue
            on_path = int((shapely.distance(points, lines[other]) <= overlap_m).sum())
            first.append(log)
            second.append(other)
            frames.append(on_path)
    shared = sparse.csr_matrix((frames, (first, second)), shape=(len(ids), len(ids)))
    both_ways = shared.minimum(shared.T) >= shared_frames
    _, drive = csgraph.connected_components(both_ways, directed=False)
    first_log = {}
    # ``ids`` is sorted, so a drive's first log seen is its smallest.
    for log, label in zip(ids, drive, strict=True):
        first_log.setdefault(label, log)
    return {log: first_log[label] for log, label in zip(ids, drive, strict=True)}


def candidate_logs(
    tree: cKDTree, owner: np.ndarray, path: np.ndarray, log: int, shared_frames: int
) -> np.ndarray:
    """Return the logs that could share a drive with ``log`` (``owner`` maps points to logs).

    A candidate comes within ``DRIVE_CANDIDATE_M`` of at least ``shared_frames`` of ``path``.
    """
    near = tree.query_ball_point(path, DRIVE_CANDIDATE_M)
    frame = np.repeat(np.arange(len(path)), [len(points) for points in near])
    other = owner[np.concatenate(near).astype(np.intp)]
    n_logs = int(owner.max()) + 1
    pairs = np.unique((frame * n_logs + other)[other != log])  # count each (frame, log) once
    others, counts = np.unique(pairs % n_logs, return_counts=True)
    return others[counts >= shared_frames]


def shared_drive_logs(drive_of: dict[str, str], reference: Iterable[str]) -> list[str]:
    """Return logs outside ``reference`` whose drive holds a reference log, sorted."""
    reference = set(reference)
    shared = {drive_of[log] for log in reference}
    return sorted(
        log for log, drive in drive_of.items() if drive in shared and log not in reference
    )


@dataclass(frozen=True)
class DriveMap:
    """One drives file: the ``{log_id: drive_id}`` map and the dropped logs."""

    path: str
    drive_of: dict
    dropped: frozenset
    settings: dict


_DRIVES: list[DriveMap] = []  # at most one drive map at a time (managed by ``use_drives``)


def use_drives(path: str | Path | None) -> None:
    """Activate a drives file so resampling clusters by drive; None restores per-log resampling.

    Logs the file drops are left out of every later query (``is_dropped``).
    """
    _DRIVES.clear()
    if path is None:
        return
    record = json.loads(Path(path).read_text())
    dropped = frozenset(record["dropped"])
    _DRIVES.append(DriveMap(str(path), record["drives"], dropped, record["settings"]))


def unit_of(log: str) -> str:
    """Return the resampling unit of ``log``: its drive with a drive map active, else itself.

    Raises:
        ValueError: if the active drive map does not hold the log.
    """
    if not _DRIVES:
        return log
    if log not in _DRIVES[0].drive_of:
        raise ValueError(f"log {log} is not in the drive map {_DRIVES[0].path}")
    return _DRIVES[0].drive_of[log]


def units_of(logs: Sequence[str]) -> np.ndarray:
    """Return the resampling units of ``logs`` as an object array."""
    return np.asarray([unit_of(log) for log in logs], dtype=object)


def is_dropped(log: str) -> bool:
    """Return whether an active drive map drops ``log``."""
    return bool(_DRIVES) and log in _DRIVES[0].dropped


def unit_name() -> str:
    """Return ``"drive"`` with a drive map active, otherwise ``"log"``."""
    return "drive" if _DRIVES else "log"


@dataclass(frozen=True)
class Estimate:
    """A point estimate and its confidence interval (NaN bounds when there is none).

    ``exact`` holds the counts and unit of a zero-event cell.
    """

    value: float
    low: float = float("nan")
    high: float = float("nan")
    exact: dict | None = None

    def scaled(self, factor: float) -> Estimate:
        """Return this estimate with value and bounds multiplied by ``factor``."""
        return Estimate(self.value * factor, self.low * factor, self.high * factor, self.exact)

    def to_json(self) -> dict | float | str:
        """Return ``"na"``, the bare value without an interval, or ``{"value", "lo", "hi"}``.

        A zero-event cell's dict also has ``"exact"``.
        """
        if np.isnan(self.value):
            return "na"
        if np.isnan(self.low):
            return self.value
        out = {"value": self.value, "lo": self.low, "hi": self.high}
        return out if self.exact is None else {**out, "exact": self.exact}


def ratio(numerators: npt.ArrayLike, denominators: npt.ArrayLike) -> Estimate:
    """Return ``sum(numerators) / sum(denominators)`` over per-unit entries as an ``Estimate``.

    A zero-event cell gets the exact interval of ``zero_events``.
    """
    num = np.asarray(numerators, dtype=float)
    den = np.asarray(denominators, dtype=float)
    if den.sum() == 0:
        return Estimate(float("nan"))
    value = float(_ratio(num, den))
    if num.sum() == 0:
        return zero_events(value, den)
    if len(num) < MIN_INTERVAL_UNITS:
        return Estimate(value)
    return Estimate(value, *_interval((num, den), _ratio))


def zero_events(value: float, denominators: np.ndarray) -> Estimate:
    """Return ``[0, upper]`` for a zero-event cell, the exact bound over units with a denominator.

    The unit reads ``"test"`` when, with a drive map, each unit holds one test.
    """
    held = denominators[denominators > 0]
    one_each = bool(_DRIVES) and bool(np.all(held == 1))
    low, high = exact_binomial_interval(0, len(held))
    exact = {
        "events": 0,
        "n": float(held.sum()),
        "units": len(held),
        "unit": "test" if one_each else unit_name(),
    }
    return Estimate(value, low, high, exact)


def exact_binomial_interval(k: int, n: int, alpha: float = EXACT_ALPHA) -> tuple[float, float]:
    """Return the two-sided ``1 - 2 * alpha`` Clopper-Pearson interval for ``k`` of ``n`` trials.

    A 0 / 0 cell gives ``(nan, nan)``.
    """
    if n <= 0:
        return float("nan"), float("nan")
    if k == 0:
        return 0.0, float(1.0 - alpha ** (1.0 / n))
    if k == n:
        return float(alpha ** (1.0 / n)), 1.0
    low = float(stats.beta.ppf(alpha, k, n - k + 1))
    high = float(stats.beta.ppf(1.0 - alpha, k + 1, n - k))
    return low, high


def _ratio(numerators: np.ndarray, denominators: np.ndarray, axis: int = -1):
    return numerators.sum(axis) / denominators.sum(axis)


def _interval(data: tuple, statistic: Callable[..., np.ndarray]) -> tuple[float, float]:
    result = stats.bootstrap(
        data,
        statistic,
        paired=True,
        vectorized=True,
        n_resamples=RESAMPLES,
        batch=BATCH,
        confidence_level=LEVEL,
        method=METHOD,
        # Draws as ``rng=SEED`` does in scipy >= 1.15; scipy 1.13 has no ``rng`` argument.
        random_state=np.random.default_rng(SEED),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def ratio_difference(
    num_a: npt.ArrayLike, den_a: npt.ArrayLike, num_b: npt.ArrayLike, den_b: npt.ArrayLike
) -> Estimate:
    """Return ``ratio(a) - ratio(b)`` as a paired ``Estimate`` over the same units in order."""
    arrays = tuple(np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b))
    if arrays[1].sum() == 0 or arrays[3].sum() == 0:
        return Estimate(float("nan"))
    value = float(_ratio_difference(*arrays))
    if len(arrays[0]) < MIN_INTERVAL_UNITS:
        return Estimate(value)
    return Estimate(value, *_interval(arrays, _ratio_difference))


def _ratio_difference(
    num_a: np.ndarray, den_a: np.ndarray, num_b: np.ndarray, den_b: np.ndarray, axis: int = -1
):
    return num_a.sum(axis) / den_a.sum(axis) - num_b.sum(axis) / den_b.sum(axis)


def weighted_ratio(
    numerators: npt.ArrayLike, denominators: npt.ArrayLike, weights: npt.ArrayLike
) -> Estimate:
    """Return a stratum-weighted rate from ``[strata, units]`` arrays as an ``Estimate``.

    The fixed weights are renormalised over the strata with a denominator, in every draw too.
    """
    num, den = (np.asarray(x, dtype=float) for x in (numerators, denominators))
    weights = np.asarray(weights, dtype=float)
    if not (den.sum(axis=1) * weights).sum():
        return Estimate(float("nan"))
    value = float(_weighted(num.sum(axis=1), den.sum(axis=1), weights))
    if num.sum() == 0:
        return zero_events(value, den.sum(axis=0))
    if num.shape[1] < MIN_INTERVAL_UNITS:
        return Estimate(value)
    strata = len(num)

    def statistic(*rows: np.ndarray, axis: int = -1) -> np.ndarray:
        return _weighted(_stacked(rows[:strata], axis), _stacked(rows[strata:], axis), weights)

    return Estimate(value, *_interval((*num, *den), statistic))


def weighted_ratio_difference(
    num_a: npt.ArrayLike,
    den_a: npt.ArrayLike,
    num_b: npt.ArrayLike,
    den_b: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> Estimate:
    """Return ``weighted_ratio(a) - weighted_ratio(b)`` as a paired ``Estimate``."""
    arrays = [np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b)]
    weights = np.asarray(weights, dtype=float)
    if not (arrays[1].sum(axis=1) * weights).sum() or not (arrays[3].sum(axis=1) * weights).sum():
        return Estimate(float("nan"))
    sums = [array.sum(axis=1) for array in arrays]
    value = float(_weighted(*sums[:2], weights) - _weighted(*sums[2:], weights))
    if arrays[0].shape[1] < MIN_INTERVAL_UNITS:
        return Estimate(value)
    strata = len(arrays[0])

    def statistic(*rows: np.ndarray, axis: int = -1) -> np.ndarray:
        a_num, a_den, b_num, b_den = (
            _stacked(rows[i * strata : (i + 1) * strata], axis) for i in range(4)
        )
        return _weighted(a_num, a_den, weights) - _weighted(b_num, b_den, weights)

    rows = tuple(row for array in arrays for row in array)
    return Estimate(value, *_interval(rows, statistic))


def _stacked(rows: Sequence[np.ndarray], axis: int) -> np.ndarray:
    """Sum each row over ``axis`` (units) and stack the results into ``[strata, ...]``."""
    return np.stack([row.sum(axis) for row in rows])


def _weighted(numerators: np.ndarray, denominators: np.ndarray, weights: np.ndarray):
    """Return the weighted mean rate over the strata with a denominator."""
    held = denominators > 0
    rates = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=held)
    w = weights.reshape((-1,) + (1,) * (numerators.ndim - 1)) * held
    return (w * rates).sum(axis=0) / w.sum(axis=0)


def quantile(values: npt.ArrayLike, logs: npt.ArrayLike, q: float) -> Estimate:
    """Return the ``q`` quantile of ``values`` pooled over units.

    ``logs[i]`` is the log of ``values[i]``. The ``inverted_cdf`` method takes the lower middle
    value as an even count's median.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return Estimate(float("nan"))
    value = float(np.quantile(values, q, method=QUANTILE_METHOD))
    n_units, distinct, (per_unit,) = _value_counts_by_unit([values], logs)
    if n_units < MIN_INTERVAL_UNITS:
        return Estimate(value)
    pooled = _in_batches(
        n_units, len(distinct), lambda rows: drawn_quantiles(rows, per_unit, distinct, q)
    )
    return Estimate(value, *_interval((np.arange(n_units),), pooled))


def _draw_counts(drawn: np.ndarray) -> np.ndarray:
    """Return ``[draws, units]``: how many times each row of ``drawn`` picked each unit."""
    n_draws, n_units = drawn.shape
    offsets = np.arange(n_draws)[:, None] * n_units
    counts = np.bincount((drawn + offsets).ravel(), minlength=n_draws * n_units)
    return counts.reshape(n_draws, n_units)


def drawn_quantiles(
    drawn: np.ndarray, per_unit: sparse.csr_array, distinct: np.ndarray, q: float
) -> np.ndarray:
    """Return a ``q`` quantile per row of ``drawn`` unit indices, values weighted by draws.

    This is numpy's ``inverted_cdf`` rule written out on counts.
    """
    counts = _draw_counts(drawn)
    weights = (per_unit.T @ counts.T).T  # each draw's weight of each distinct value
    share = np.cumsum(weights, axis=1, dtype=np.float64)
    share /= share[:, -1:]
    share[share == 0] = -1
    first = np.minimum((share < q).sum(axis=1), len(distinct) - 1)
    return distinct[first]


def wasserstein(
    values: npt.ArrayLike,
    logs: npt.ArrayLike,
    reference: npt.ArrayLike,
    reference_logs: npt.ArrayLike,
) -> Estimate:
    """Return the Wasserstein distance between ``values`` and ``reference`` as an ``Estimate``.

    Both sides are pooled over their units; a draw that leaves a side empty gives NaN.
    """
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(values) == 0 or len(reference) == 0:
        return Estimate(float("nan"))
    value = float(stats.wasserstein_distance(values, reference))
    n_units, distinct, per_unit = _value_counts_by_unit(
        [values, reference], [*logs, *reference_logs]
    )
    if n_units < MIN_INTERVAL_UNITS:
        return Estimate(value)
    pooled = _in_batches(
        n_units, len(distinct), lambda rows: drawn_distances(rows, *per_unit, distinct)
    )
    return Estimate(value, *_interval((np.arange(n_units),), pooled))


def wasserstein_difference(
    values_a: npt.ArrayLike,
    logs_a: npt.ArrayLike,
    values_b: npt.ArrayLike,
    logs_b: npt.ArrayLike,
    reference: npt.ArrayLike,
    reference_logs: npt.ArrayLike,
) -> Estimate:
    """Return ``wasserstein(a) - wasserstein(b)`` against the same reference, as a paired estimate.

    A draw takes the same units on both sides and in the reference.
    """
    sides = [np.asarray(x, dtype=float) for x in (values_a, values_b, reference)]
    if not all(len(side) for side in sides):
        return Estimate(float("nan"))
    a, b, pool = sides
    value = float(stats.wasserstein_distance(a, pool) - stats.wasserstein_distance(b, pool))
    n_units, distinct, (per_a, per_b, per_pool) = _value_counts_by_unit(
        sides, [*logs_a, *logs_b, *reference_logs]
    )
    if n_units < MIN_INTERVAL_UNITS:
        return Estimate(value)

    def difference(rows: np.ndarray) -> np.ndarray:
        drawn_a = drawn_distances(rows, per_a, per_pool, distinct)
        return drawn_a - drawn_distances(rows, per_b, per_pool, distinct)

    pooled = _in_batches(n_units, len(distinct), difference)
    return Estimate(value, *_interval((np.arange(n_units),), pooled))


def _value_counts_by_unit(
    sides: Sequence[np.ndarray], logs: Sequence[str]
) -> tuple[int, np.ndarray, tuple[sparse.csr_array, ...]]:
    """Return ``(n_units, distinct_values, per_side_counts)``, counts sparse ``[units, values]``.

    ``logs`` names the log of each value of each side in turn.
    """
    names, owner = np.unique(units_of(logs), return_inverse=True)
    distinct, which = np.unique(np.concatenate(sides), return_inverse=True)
    shape = (len(names), len(distinct))
    bounds = np.cumsum([0, *(len(side) for side in sides)])
    per_side = tuple(
        sparse.csr_array((np.ones(high - low), (owner[low:high], which[low:high])), shape=shape)
        for low, high in pairwise(bounds)
    )
    return len(names), distinct, per_side


def _in_batches(
    n_units: int, n_values: int, statistic: Callable[[np.ndarray], np.ndarray]
) -> Callable[..., np.ndarray]:
    """Wrap ``statistic`` on ``[draws, units]`` rows for scipy's bootstrap, in bounded batches."""
    rows_at_once = max(1, CELLS_AT_ONCE // max(n_units, n_values))

    def pooled(drawn: np.ndarray, axis: int = -1) -> np.ndarray:
        drawn = np.moveaxis(np.asarray(drawn, dtype=np.intp), axis, -1)
        rows = drawn.reshape(-1, n_units)
        out = np.concatenate(
            [
                statistic(rows[start : start + rows_at_once])
                for start in range(0, len(rows), rows_at_once)
            ]
        )
        return out.reshape(drawn.shape[:-1])

    return pooled


def drawn_distances(
    drawn: np.ndarray, values: sparse.csr_array, reference: sparse.csr_array, distinct: np.ndarray
) -> np.ndarray:
    """Return the Wasserstein distance per row of ``drawn`` unit indices, NaN for an empty side.

    It is the area between the draw-weighted CDFs, as scipy computes it.
    """
    counts = _draw_counts(drawn).astype(np.float64)
    weight_values = (values.T @ counts.T).T  # each draw's weight of each distinct value
    weight_reference = (reference.T @ counts.T).T
    total_values = weight_values.sum(axis=1, keepdims=True)
    total_reference = weight_reference.sum(axis=1, keepdims=True)
    empty = (total_values[:, 0] == 0) | (total_reference[:, 0] == 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cdf_gap = np.cumsum(
            weight_values / total_values - weight_reference / total_reference, axis=1
        )
    out = np.abs(cdf_gap[:, :-1]) @ np.diff(distinct)
    out[empty] = np.nan
    return out
