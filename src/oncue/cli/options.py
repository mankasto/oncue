"""Shared helpers for CLI options: comma-separated values, defaults for ``--workers``, etc."""

from __future__ import annotations

# default --workers of the scene-by-scene commands; the functions themselves default to 1
WORKERS = 8


def comma_list(text: str | None) -> list[str] | None:
    """Split a comma-separated option value; return None when the option was not given."""
    return None if text is None else text.split(",")


def given(**options: object) -> dict:
    """Return the options that were set (not None), so the callee keeps its own defaults."""
    return {name: value for name, value in options.items() if value is not None}


def chosen(text: str) -> tuple[str, ...] | None:
    """Split a comma-separated choice; return None for ``"all"``."""
    return None if text == "all" else tuple(text.split(","))
