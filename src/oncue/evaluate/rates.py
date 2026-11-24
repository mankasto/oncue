"""What a closed-loop run counts as (``run_flags``) and the rates, with their intervals, that the
tables report (``rate``, ``weighted_rate``, ``closed_loop_cell``)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..closedloop.scene import MODES as RUN_MODES
from ..closedloop.scene import ROUNDED_TIME_TOL_S, run_name
from ..core.cutoff import LEFT_ROUTE, RUN_END, TARGET_UNOBSERVED
from . import bootstrap
from .outputs import Population, group_by

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable, Iterable, Sequence

    from ..core.record_types import RunRecord, TestRow

# Values of ``occlusion_outcome``; the ``UNDECIDED`` ones mean the runs cannot tell.
CAUSED, NOT_CAUSED = "caused", "not_caused"

UNDECIDED = ("masked_invalid_scene", "see_incomplete", "see_invalid_scene")

MIN_LOGS = 30  # minimum independent logs per main reporting unit; drives the ``low_logs`` flag

PERCENT = 100.0
W_SPREAD = (0.1, 0.5, 0.9)  # actual-W quantiles reported (P10, median, P90); not an interval

MODES = tuple(RUN_MODES)  # "masked", "see"

RUN_RATES = (
    "hit",
    "hit_moving",
    "hit_whole_run",
    "clean",
    "stalled",
    "incomplete",
    "bad_scene",
    "other",
    "rear",
    "contact",
    "failed",
)

PAIRED_RATES = ("hit", "clean")  # reported as masked minus unoccluded

# Key of a W-balanced list's total weighted to all tests (the plain list average has none).
WEIGHTED_TO_ALL_TESTS = "weighted_to_all_tests"

# Stand-in for a planned run the outputs are missing: counts as a failure in ``N``.
MISSING_RUN = {"error": "no run"}

# Why a test ends (``core.cutoff``); a row or run without a reason is ``UNRECORDED_REASON``.
UNRECORDED_REASON = "unrecorded"

CUTOFF_REASONS = (LEFT_ROUTE, TARGET_UNOBSERVED, RUN_END, UNRECORDED_REASON)


def run_flags(run: RunRecord) -> dict[str, bool]:
    """Return the boolean classification of one closed-loop run; every run rate counts these.

    ``complete``: started, on time, observed to the contact, in a valid scene. ``clean``:
    complete, no failing contact, never off the road or against traffic, within the vehicle
    limits. ``hit`` is a target contact before the cutoff; ``hit_whole_run`` adds later ones.
    """
    stalled = bool(run.get("stalled")) or bool(run.get("late"))
    incomplete = not stalled and bool(run.get("incomplete"))
    valid = run.get("scene_ok") is True
    contact = bool(run.get("hit")) or bool(run.get("other_contact"))
    complete = not stalled and not incomplete and valid
    out_of_road = run.get("out_of_road_s") is not None
    wrong_way = run.get("wrong_way_s") is not None
    beyond_limits = bool(run.get("beyond_limits"))
    return {
        "hit": bool(run.get("hit")),
        # ego at closedloop.records.STANDING_CONTACT_MPS or faster
        "hit_moving": bool(run.get("hit_moving")),
        "hit_whole_run": bool(run.get("hit")) or bool(run.get("after_cutoff")),
        "stalled": stalled,
        "incomplete": incomplete,
        "valid": valid,
        "bad_scene": run.get("scene_ok") is False,
        "scene_flaw_known": run.get("scene_flaw_known") is True,
        "other": bool(run.get("other_contact")),
        "rear": bool(run.get("excused")),
        "contact": contact,
        "complete": complete,
        "road_judged": "out_of_road_s" in run,
        "out_of_road": out_of_road,
        "wrong_way": wrong_way,
        # Off the map's extent, so whether the ego is off the road is unknown.
        "road_unknown": bool(run.get("road_unknown")),
        "beyond_limits": beyond_limits,
        "clean": complete
        and not contact
        and not out_of_road
        and not wrong_way
        and not beyond_limits,
        "failed": "error" in run,
    }


def occlusion_outcome(masked: dict, see: dict) -> str:
    """Return whether the occlusion caused a test's masked hit, from both of its runs.

    ``CAUSED``: a masked hit in a valid scene, and a complete unoccluded run in a valid scene
    without one. ``NOT_CAUSED``: no masked hit, or an unoccluded hit too. Else one of
    ``UNDECIDED``.
    """
    if not run_flags(masked)["hit"]:
        return NOT_CAUSED
    why = masked_undecided(masked)
    if why is not None:
        return why
    see_flags = run_flags(see)
    if see_flags["hit"]:
        return NOT_CAUSED
    if see_flags["incomplete"]:
        return "see_incomplete"
    if not see_flags["valid"]:
        return "see_invalid_scene"
    return CAUSED


def masked_undecided(masked: dict) -> str | None:
    """Return ``"masked_invalid_scene"`` for a masked hit in an invalid scene, else None."""
    flags = run_flags(masked)
    return "masked_invalid_scene" if flags["hit"] and not flags["valid"] else None


def run_window(run: RunRecord) -> float | None:
    """Return the run's W: seconds from the planner first seeing the target to the contact.

    None when the run stalled, was late, ended before the contact, or saw the target only after.
    """
    if "meet_s" not in run or run.get("stalled") or run.get("late"):
        return None
    observed = run.get("hit") or run.get("observed_s", -1.0) + ROUNDED_TIME_TOL_S >= run["meet_s"]
    seen = run.get("seen_s")
    if not observed or seen is None or seen > run["meet_s"]:
        return None
    return run["meet_s"] - seen


def deceleration_onset_lead(run: RunRecord) -> float | None:
    """Return seconds from the run's deceleration onset to the contact (negative: after it).

    None without a ``run_window`` or an onset.
    """
    if run_window(run) is None or run.get("response_s") is None:
        return None
    return run["meet_s"] - run["response_s"]


def start_mode(x: dict) -> str:
    """Return ``"triggered"`` or ``"fixed"`` for how this row's target started.

    A row without ``start`` reads as triggered when one of its runs records ``stalled``.
    """
    if x.get("start"):
        return x["start"]
    runs = [r for r in x.get("closed_loop", {}).values() if isinstance(r, dict)]
    return "triggered" if any("stalled" in r for r in runs) else "fixed"


# ---- what the tables share --------------------------------------------------------------


def counts(rows: list[tuple[str, dict]]) -> dict:
    """Return the logs, drives (resampling units), sites and rows behind a reported number."""
    logs = {scene_id for scene_id, _ in rows}
    sites = {(scene_id, row.get("site")) for scene_id, row in rows}
    return {
        "n_logs": len(logs),
        "n_drives": len({bootstrap.unit_of(log) for log in logs}),
        "n_sites": len(sites),
        "n_rows": len(rows),
        "low_logs": len(logs) < MIN_LOGS,
    }


def rate(rows: list[tuple[str, dict]], is_counted: Callable[[dict], bool]) -> bootstrap.Estimate:
    """Return the share (%) of ``rows`` with ``is_counted(row)``, bootstrapped by drive or log."""
    numerators, denominators = Counter(), Counter()
    for scene_id, row in rows:
        unit = bootstrap.unit_of(scene_id)
        denominators[unit] += 1
        numerators[unit] += bool(is_counted(row))
    units = sorted(denominators)
    share = bootstrap.ratio(
        [numerators[unit] for unit in units], [denominators[unit] for unit in units]
    )
    return share.scaled(PERCENT)


def rate_difference(
    pairs: list[tuple[str, dict, dict]], is_counted: Callable[[dict], bool]
) -> bootstrap.Estimate:
    """Return the percentage-point difference of ``is_counted`` between ``row_a`` and ``row_b``."""
    return mean_difference(pairs, lambda row: float(bool(is_counted(row)))).scaled(PERCENT)


def mean_difference(
    pairs: list[tuple[str, dict, dict]], value_of: Callable[[dict], float]
) -> bootstrap.Estimate:
    """Return the mean of ``value_of(row_a) - value_of(row_b)``, each pair drawn together."""
    first, second, denominators = Counter(), Counter(), Counter()
    for scene_id, row_a, row_b in pairs:
        unit = bootstrap.unit_of(scene_id)
        denominators[unit] += 1
        first[unit] += value_of(row_a)
        second[unit] += value_of(row_b)
    units = sorted(denominators)
    den = [denominators[unit] for unit in units]
    return bootstrap.ratio_difference(
        [first[unit] for unit in units], den, [second[unit] for unit in units], den
    )


def planners_in(rows: list[tuple[str, dict]]) -> list[str]:
    """Return the sorted planners that have at least one run across these rows."""
    return sorted({planner_of(name) for row in rows for name in row["closed_loop"]})


def planner_of(name: str) -> str:
    """Return the planner part of a closed-loop run name (the inverse of ``run_name``)."""
    return name.rsplit("_", 1)[0]


def closed_loop_cell(rows: list[tuple[str, dict]], planner: str, mode: str) -> dict:
    """Return one planner's cell at ``mode`` for a group of samples.

    Every sample counts; a run the outputs lack counts as failed and in ``missing_runs``.
    """
    name = run_name(planner, mode)
    flagged = [
        (
            scene_id,
            {**run_flags(row["closed_loop"].get(name, MISSING_RUN)), "site": row.get("site")},
        )
        for scene_id, row in rows
    ]
    return {
        "samples": len(rows),
        "missing_runs": sum(name not in row["closed_loop"] for _, row in rows),
        **counts(flagged),
        **{f"{flag}_pct": rate(flagged, lambda f, flag=flag: f[flag]) for flag in RUN_RATES},
        "out_of_road_pct": road_rate(flagged, "out_of_road"),
        "wrong_way_pct": road_rate(flagged, "wrong_way"),
        "cutoff_reasons": cutoff_reasons(
            row["closed_loop"].get(name, MISSING_RUN) for _, row in rows
        ),
    }


def scenes_valid(row: TestRow, planners: Sequence[str]) -> bool:
    """Return whether every run of ``planners`` in both modes is in a valid scene.

    A missing run never counts as valid.
    """
    return all(
        run_flags(row["closed_loop"].get(run_name(planner, mode), MISSING_RUN))["valid"]
        for planner in planners
        for mode in MODES
    )


def road_rate(flagged: list[tuple[str, dict]], flag: str) -> bootstrap.Estimate:
    """Return the share (%) of all runs with road flag ``flag``.

    NaN unless every run that did not fail was judged for roads.
    """
    judged = [f for _, f in flagged if not f["failed"]]
    if not judged or not all(f["road_judged"] for f in judged):
        return bootstrap.Estimate(float("nan"))
    return rate(flagged, lambda f, flag=flag: f[flag])


def method_difference(
    cells: dict[str, list[tuple[str, dict]]],
    a: str,
    b: str,
    value_of: Callable[[dict], float | None],
) -> dict:
    """Return ``mean(value_of(a) - value_of(b))`` over the tests both methods have.

    ``cells`` is ``{method: [(log, row)]}`` with rows keyed by ``test_id``.
    """
    shared, only = shared_tests(cells, a, b)
    pairs = [
        (log, row_a, row_b)
        for log, row_a, row_b in shared
        if value_of(row_a) is not None and value_of(row_b) is not None
    ]
    return {
        "pairs": len(pairs),
        "lacking": len(shared) - len(pairs),
        **only,
        "difference": mean_difference(pairs, value_of),
    }


def shared_tests(
    cells: dict[str, list[tuple[str, dict]]], a: str, b: str
) -> tuple[list[tuple[str, dict, dict]], dict]:
    """Return the ``(log, row_a, row_b)`` tests both methods have, and counts of the others."""
    rows_a = {row["test_id"]: (log, row) for log, row in cells[a]}
    rows_b = {row["test_id"]: row for _, row in cells[b]}
    shared = [(*rows_a[t], rows_b[t]) for t in sorted(rows_a.keys() & rows_b.keys())]
    only = {
        "only_first": len(rows_a.keys() - rows_b.keys()),
        "only_second": len(rows_b.keys() - rows_a.keys()),
    }
    return shared, only


def planner_difference(
    rows: list[tuple[str, dict]],
    planner_a: str,
    planner_b: str,
    mode: str,
    flag: str,
    population: Population | None = None,
) -> bootstrap.Estimate:
    """Return percentage points between two planners' ``flag`` rates in ``mode``, paired.

    With ``population`` it is weighted to all tests. A missing run counts as failed.
    """
    name_a, name_b = run_name(planner_a, mode), run_name(planner_b, mode)
    pairs = [
        (
            scene_id,
            run_flags(row["closed_loop"].get(name_a, MISSING_RUN)),
            run_flags(row["closed_loop"].get(name_b, MISSING_RUN)),
        )
        for scene_id, row in rows
    ]
    if population is None:
        return rate_difference(pairs, lambda flags: flags[flag])
    keys = [population.key(row) for _, row in rows]
    return weighted_rate_difference(pairs, keys, lambda flags: flags[flag], population)


def weighted_rate(
    rows: list[tuple[str, dict]], is_counted: Callable[[dict], bool], population: Population
) -> bootstrap.Estimate:
    """Return the share (%) of ``rows`` with ``is_counted(row)``, weighted to all tests."""
    keys = [population.key(row) for _, row in rows]
    pairs = [(scene_id, row, row) for scene_id, row in rows]
    num, _, den, weights = _strata_sums(pairs, keys, is_counted, population)
    return bootstrap.weighted_ratio(num, den, weights).scaled(PERCENT)


def weighted_rate_difference(
    pairs: list[tuple[str, dict, dict]],
    keys: Sequence[tuple],
    is_counted: Callable[[dict], bool],
    population: Population,
) -> bootstrap.Estimate:
    """Return the percentage-point difference of ``row_a`` and ``row_b``, weighted to all tests.

    ``keys[i]`` is the stratum of ``pairs[i]``.
    """
    num_a, num_b, den, weights = _strata_sums(pairs, keys, is_counted, population)
    return bootstrap.weighted_ratio_difference(num_a, den, num_b, den, weights).scaled(PERCENT)


def _strata_sums(
    pairs: list[tuple[str, dict, dict]],
    keys: Sequence[tuple],
    is_counted: Callable[[dict], bool],
    population: Population,
) -> tuple:
    """Return ``[strata, units]`` counts of ``row_a``, ``row_b`` and pairs, and the weights."""
    weights = population.weights(keys)
    strata = sorted(weights, key=str)
    units = sorted({bootstrap.unit_of(scene_id) for scene_id, _, _ in pairs})
    row_of, column_of = {k: i for i, k in enumerate(strata)}, {u: i for i, u in enumerate(units)}
    num_a, num_b, den = (np.zeros((len(strata), len(units))) for _ in range(3))
    for (scene_id, row_a, row_b), key in zip(pairs, keys, strict=True):
        at = row_of[key], column_of[bootstrap.unit_of(scene_id)]
        den[at] += 1
        num_a[at] += bool(is_counted(row_a))
        num_b[at] += bool(is_counted(row_b))
    return num_a, num_b, den, np.array([weights[key] for key in strata])


def stratum_cells(
    rows: list[tuple[str, dict]], rates: dict[str, Callable[[dict], bool]], population: Population
) -> list[dict]:
    """Return one cell per stratum: its fields, counts, weight share (%) and every rate."""
    by_stratum = group_by(rows, lambda item: population.key(item[1]))
    weights = population.weights([population.key(row) for _, row in rows])
    total = sum(weights.values())
    return [
        {
            **dict(zip(population.by, key, strict=True)),
            **counts(members),
            "tests": population.tests[key],
            "chosen": population.chosen[key],
            "weight_pct": PERCENT * weights[key] / total,
            **{name: rate(members, is_counted) for name, is_counted in rates.items()},
        }
        for key, members in by_stratum.items()
    ]


def mode_difference(rows: list[tuple[str, dict]], planner: str) -> dict:
    """Return masked-minus-unoccluded percentages for ``planner`` on samples with both runs."""
    masked, see = run_name(planner, "masked"), run_name(planner, "see")
    pairs = [
        (scene_id, run_flags(row["closed_loop"][masked]), run_flags(row["closed_loop"][see]))
        for scene_id, row in rows
        if masked in row["closed_loop"] and see in row["closed_loop"]
    ]
    return {
        "pairs": len(pairs),
        "both_valid": sum(m["valid"] and t["valid"] for _, m, t in pairs),
        **{
            f"{flag}_pct_points": rate_difference(pairs, lambda f, flag=flag: f[flag])
            for flag in PAIRED_RATES
        },
    }


# Tests grouped by where they end.
def cutoff_reasons(records: Iterable[dict]) -> dict[str, int]:
    """Return per-``CUTOFF_REASONS`` counts for rows or runs; an unknown reason raises."""
    found = Counter(record.get("cutoff_reason") or UNRECORDED_REASON for record in records)
    if unknown := sorted(found.keys() - set(CUTOFF_REASONS)):
        raise ValueError(f"cutoff reasons {unknown}, not among {CUTOFF_REASONS}")
    return {reason: found[reason] for reason in CUTOFF_REASONS}
