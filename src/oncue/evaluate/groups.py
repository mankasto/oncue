"""Grouping rows for the tables: by flaw group and severity, and by what hides the target."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..generation.checks.realism import (
    FLAW_GROUPS,
    SEVERITY_GROUPS,
    flaw_group,
    found_flaws,
    severity_groups,
)
from .rates import rate, start_mode

if TYPE_CHECKING:  # annotations only
    from ..core.record_types import TestRow

# What hides a test: road users alone, or with the walls; ``unknown`` when a row does not say.
UNKNOWN_HIDING = "unknown"

HIDDEN_BY = ("road_users", "walls", UNKNOWN_HIDING)

# ---- reading a closed-loop run -------------------------------------------------------------


def closed_loop_fields(row: TestRow) -> dict:
    """Return the grouping fields a closed-loop row contributes to table rows."""
    return {
        "kind": row.get("kind"),
        "hidden_by": row_hidden_by(row),
        "generator": row.get("generator") or "model",
        "test": row.get("test"),
        "braking": row.get("braking_suffices"),
        "start": start_mode(row),
        "W_req": row.get("W_req"),
        "flaw_group": row_flaw_group(row),
    }


# Tables report both all tests together and each of the three flaw groups separately.
def flaw_group_rows(rows: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    """Return rows grouped by flaw group, one entry per ``FLAW_GROUPS`` key."""
    out: dict[str, list[tuple[str, dict]]] = {group: [] for group in FLAW_GROUPS}
    for log, row in rows:
        out[row_flaw_group(row)].append((log, row))
    return out


def row_flaw_group(row: TestRow) -> str:
    """Return a row's flaw group, counting a baseline row's ``flaws_unknown``."""
    flaws = row.get("flaws")
    return flaw_group(flaws if isinstance(flaws, dict) else None, row.get("flaws_unknown") or ())


def row_severity_groups(row: TestRow) -> list[str]:
    """Return the severity groups a row's found flaws fall into; empty when the row has no flaws."""
    flaws = row.get("flaws")
    return severity_groups(found_flaws(flaws if isinstance(flaws, dict) else None))


def severity_cell(rows: list[tuple[str, dict]]) -> dict:
    """Return per-severity-group counts and shares (%); a row counts in each group it is in."""
    return {
        group: {
            "n_rows": sum(group in row_severity_groups(row) for _, row in rows),
            "pct": rate(rows, lambda row, g=group: g in row_severity_groups(row)),
        }
        for group in SEVERITY_GROUPS
    }


# Tests split by what hides them.
def row_hidden_by(row: TestRow) -> str:
    """Return what hides this test, one of ``HIDDEN_BY``; any other value raises."""
    hiding = row.get("hidden_by") or UNKNOWN_HIDING
    if hiding not in HIDDEN_BY:
        raise ValueError(
            f"sample {row.get('sample_id')}: hidden_by {hiding!r}, not one of {HIDDEN_BY}"
        )
    return hiding


def hidden_by_rows(rows: list[tuple[str, dict]]) -> dict[str, list[tuple[str, dict]]]:
    """Return rows bucketed by what hides them, one entry per ``HIDDEN_BY`` value."""
    out: dict[str, list[tuple[str, dict]]] = {hiding: [] for hiding in HIDDEN_BY}
    for log, row in rows:
        out[row_hidden_by(row)].append((log, row))
    return out


def hidden_by_counts(rows: list[tuple[str, dict]]) -> dict[str, int]:
    """Return how many rows are hidden each way, keyed by ``HIDDEN_BY``."""
    return {hiding: len(of) for hiding, of in hidden_by_rows(rows).items()}
