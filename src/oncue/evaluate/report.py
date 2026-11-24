"""Formatting and writing of experiment tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.io import write_json
from . import bootstrap
from .rates import MIN_LOGS

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Sequence

# Columns ``python -m experiments analyze`` prints per table; the JSON keeps every field.
SHOWN = {
    "generation": (
        "run",
        "generator",
        "group",
        "Q",
        "realizable",
        "n_logs",
        "G",
        "fallback",
        "yield_pct",
        "realizable_yield_pct",
        "realizable_on_time_pct",
        "on_time_share_pct",
        "realizable_fallback_pct",
        "A_yield_pct",
        "A_share_pct",
        "W.abs_error_median_s",
    ),
    "raw_candidates": (
        "run",
        "generator",
        "group",
        "Q",
        "n_logs",
        "candidates",
        "computable",
        "hit_pct",
        "hit_of_all_pct",
        "request_hit_pct",
        "uncomputable",
        "W.abs_error_median_s",
    ),
    "paired": (
        "first",
        "second",
        "group",
        "n_rows",
        "n_logs",
        "yield_pct_points",
        "realizable_yield_pct_points",
        "realizable_on_time_pct_points",
        "A_yield_pct_points",
        "realizable_same_budget_yield_pct_points",
    ),
    "closedloop": (
        "group",
        "planner",
        "masked.n_rows",
        "masked.n_logs",
        "masked.hit_pct",
        "see.hit_pct",
        "masked.hit_moving_pct",
        "see.hit_moving_pct",
        "masked_minus_see.hit_pct_points",
        "masked.clean_pct",
        "see.clean_pct",
        "masked.out_of_road_pct",
        "see.out_of_road_pct",
        "valid_scenes.masked.n_rows",
        "valid_scenes.masked.hit_pct",
        "valid_scenes.see.hit_pct",
        "times.masked.W_run_s.median",
        "times.see.W_run_s.median",
        "times.masked.response_lead_s.median",
        "times.see.response_lead_s.median",
        "distance_8s_m",
    ),
    "brake_labels": ("group", "from", "to", "transitions", "other", "n_logs"),
    "brake_closedloop": (
        "jerk",
        "group",
        "planner",
        "masked.n_rows",
        "masked.n_logs",
        "masked.hit_pct",
        "see.hit_pct",
    ),
    "budget": (
        "group",
        "from",
        "to",
        "transitions",
        "label_failed",
        "n_logs",
        "anticipation_kept_pct",
        "newly_safe",
        "revealed_newly_safe_pct",
        "work.revealed.executed",
        "work.revealed.seconds",
        "work.early.executed",
        "work.early.seconds",
    ),
    # Main table columns first, then the supplement's.
    "baselines": (
        "name",
        "n_tests",
        "n_logs",
        "evaluated",
        "skipped",
        "rule_fallbacks",
        "attack_success_pct",
        "on_requested_W_pct",
        "Behav",
        "Kine",
        "Crash_Obj",
        "Cross_Line",
        "WD_acceleration",
        "WD_speed",
        "WD_yaw_rate",
        "anticipation_pct",
        "occlusion_caused_pct",
        "occlusion_undecided",
        "adversarial_reward",
        "pdm_masked_hit_pct",
        "pdm_see_hit_pct",
        "Behav_high_turn",
        "Behav_stop_and_turn",
        "Crash_Obj_standing",
        "Crash_Obj_moving",
        "rows_lacking",
    ),
    "other_kinds": (
        "kind",
        "generator",
        "W_req",
        "samples",
        "n_logs",
        "Behav",
        "Kine",
        "Crash_Obj",
        "WD_acceleration",
        "WD_speed",
        "WD_yaw_rate",
        "WD_acceleration_crossing",
        "WD_speed_crossing",
        "WD_yaw_rate_crossing",
    ),
    "baselines_panel_b": (
        "name",
        "n_tests",
        "n_logs",
        "attack_success_pct",
        "adversarial_reward",
        "masked_runs",
        "runs_lacking_reward",
    ),
    "baselines_by_W": (
        "name",
        "group",
        "n_rows",
        "n_logs",
        "seen",
        "Behav",
        "Kine",
        "Crash_Obj",
        "Cross_Line",
        "WD_acceleration",
        "WD_speed",
        "WD_yaw_rate",
        "WD_acceleration_crossing",
        "WD_speed_crossing",
        "WD_yaw_rate_crossing",
        "rows_lacking",
    ),
    # Fixed target timetable (experiments/e6_closed_loop/fixed_timing.py).
    "fixed_timing": (
        "group",
        "planner",
        "n_rows",
        "n_logs",
        "masked.triggered_hit_pct",
        "masked.fixed_hit_pct",
        "masked.hit_pct_points",
        "masked.stalled",
        "see.triggered_hit_pct",
        "see.fixed_hit_pct",
        "masked_minus_see.triggered.hit_pct_points",
        "masked_minus_see.fixed.hit_pct_points",
    ),
}

NOT_IN_CELL = "na"  # placeholder used in the text output when a cell has no value for a field


def text(table: str, cells: list[dict]) -> str:
    """Render ``cells`` as one line each with the table's ``SHOWN`` columns."""
    lines = []
    for cell in cells:
        parts = [f"{path} {shown(at_path(cell, path))}" for path in SHOWN[table]]
        low = cell.get("low_logs") or (cell.get("masked") or {}).get("low_logs")
        lines.append("; ".join(parts) + (f" (< {MIN_LOGS} logs)" if low else ""))
    return "\n".join(lines)


def shown(value: object) -> str:
    """Return the text form of one cell value: an estimate, a dict of flags, or ``str``."""
    if isinstance(value, bootstrap.Estimate):
        if np.isnan(value.value):
            return "na"
        if np.isnan(value.low):
            return f"{value.value:.2f}"
        return f"{value.value:.2f} [{value.low:.2f}, {value.high:.2f}]"
    if isinstance(value, dict):
        return " ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def at_path(cell: dict, path: str) -> object:
    """Look up a dotted ``path`` such as ``"masked.hit_pct"`` in ``cell``, or ``NOT_IN_CELL``."""
    for key in path.split("."):
        if not isinstance(cell, dict) or key not in cell:
            return NOT_IN_CELL
        cell = cell[key]
    return cell


def write(out: str | Path, table: str, result: object, inputs: Sequence[str | Path]) -> Path:
    """Write ``result`` as JSON with its inputs and the bootstrap settings."""
    return write_json(
        out,
        {
            "table": table,
            "result": as_json(result),
            "bootstrap": bootstrap.settings(),
            "min_logs": MIN_LOGS,
            "inputs": [str(p) for p in inputs],
        },
        indent=1,
    )


def as_json(value: object) -> object:
    """Convert a nested result to JSON values; estimates use ``to_json`` and NaN becomes "na"."""
    if isinstance(value, bootstrap.Estimate):
        return value.to_json()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [as_json(v) for v in value]
    if isinstance(value, float) and np.isnan(value):
        return "na"
    return value
