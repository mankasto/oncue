"""What a run records about the pipeline next to its output (`run_settings`), and the check that
an upstream output was made under the same settings (`check_upstream`)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.config import SiteConfig
from ..core.io import jsonable
from ..core.runs import input_digests, model_info, settings_path
from .reference import reference_driver

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterable
    from typing import Any

    from ..core.config import Config
    from .targets import TargetSpec


def run_settings(
    command: str,
    spec_name: str,
    spec: TargetSpec,
    config: Config,
    inputs: Iterable[str | Path],
    model: str | Path | None = None,
    **extra: Any,
) -> dict:
    from ..oracle.responses import candidate_list, list_digest

    driver = reference_driver()
    cands = candidate_list(config.oracle.candidates)
    inputs = [str(path) for path in inputs]
    return jsonable(
        {
            "command": command,
            "spec_name": spec_name,
            "spec": spec,
            "config": config,
            "driver": driver,
            "sites": SiteConfig(),
            "candidates": {
                "digest": list_digest(cands),
                "entries": len(cands),
                "core": sum(c.core for c in cands),
            },
            "constants": constants(),
            "model": model_info(model),
            "inputs": inputs,
            "input_sha256": input_digests(inputs),
            "created": datetime.now().isoformat(timespec="seconds"),
            **extra,
        }
    )


def constants() -> dict:
    """Return the pipeline's module-level parameters that are not already in a config class."""
    # imported here: closedloop imports this module
    from ..closedloop import trigger
    from ..closedloop.records import STANDING_CONTACT_MPS
    from ..core import config, kinematics, rules
    from ..data import map_geometry
    from .checks import realism, validity
    from .motion import starts
    from .requests import build, sites

    return {
        "requests": {
            "CONTACT_AFTER_ANCHOR_S": build.CONTACT_AFTER_ANCHOR_S,
        },
        "starts": {
            k: getattr(starts, k)
            for k in (
                "STAND_PLACE_STEP_M",
                "QUEUE_BACK_M",
                "COMMONNESS_STEP",
            )
        },
        "sites": {
            **{
                k: getattr(map_geometry, k)
                for k in (
                    "LANE_UPSTREAM_M",
                    "LANE_DOWNSTREAM_M",
                    "LANE_MIN_AHEAD_M",
                    "LANE_AHEAD_M",
                )
            },
            "STOP_HOLD": sites.STOP_HOLD,
            "WALK_STEP_M": sites.WALK_STEP_M,
            "QUEUE_ALIGN_DEG": sites.QUEUE_ALIGN_DEG,
        },
        "checks": {
            "START_TURN_M": validity.START_TURN_M,
            "START_TURN_TOL_RAD": validity.START_TURN_TOL_RAD,
            "SIGNAL_REACH_M": realism.SIGNAL_REACH_M,
            "EGO_GREEN": realism.EGO_GREEN,
            "DRIVING_TYPES": realism.DRIVING_TYPES,
            "OWN_SIGNAL_AHEAD_M": realism.OWN_SIGNAL_AHEAD_M,
            "EGO_TURN_DEG": realism.EGO_TURN_DEG,
            "EGO_HEADING_CHORD_M": realism.EGO_HEADING_CHORD_M,
        },
        "kinematics": {
            "SMOOTH_FRAMES": kinematics.SMOOTH_FRAMES,
            "SMOOTH_ORDER": kinematics.SMOOTH_ORDER,
            "CURVATURE_MIN_MPS": kinematics.CURVATURE_MIN_MPS,
        },
        "limits": {
            "VEHICLE": asdict(config.VEHICLE_LIMITS),
            "PEDESTRIAN": asdict(config.PEDESTRIAN_LIMITS),
            "CYCLIST": asdict(config.CYCLIST_LIMITS),
            "PHANTOM_RANGE_M": config.PHANTOM_RANGE_M,
        },
        "closedloop": {
            "REF_SPEED_TOL": trigger.REF_SPEED_TOL,
            "LATE_TOL_S": trigger.LATE_TOL_S,
            "STANDING_CONTACT_MPS": STANDING_CONTACT_MPS,
        },
        "rules": {"DEFAULT_SPEED_LIMIT": rules.DEFAULT_SPEED_LIMIT},
    }


def check_upstream(upstream: str | Path, spec_name: str, spec: TargetSpec, config: Config) -> None:
    """Exit unless the upstream run recorded the same target spec, replay settings and
    reference driver."""
    record = settings_path(upstream)
    if not record.exists():
        raise SystemExit(f"{upstream}: no {record.name}; its target spec cannot be checked")
    settings = json.loads(record.read_text())
    made = settings.get("spec") or {}
    given = json.loads(json.dumps(jsonable(spec)))  # as the record holds it
    if made != given:
        diff = sorted(k for k in set(made) | set(given) if made.get(k) != given.get(k))
        raise SystemExit(
            f"{upstream} was made with another target spec than {spec_name!r} (differs in {diff})"
        )
    made = (settings.get("config") or {}).get("replay")
    given = json.loads(json.dumps(jsonable(config.replay)))
    if made != given:
        raise SystemExit(
            f"{upstream} was made on another replayed scene (replay settings {made}, given {given})"
        )
    made = settings.get("driver") or {}
    given = json.loads(json.dumps(jsonable(reference_driver())))
    if made != given:
        diff = sorted(k for k in set(made) | set(given) if made.get(k) != given.get(k))
        why = f"driver settings differ in {diff}" if made else "no driver settings recorded"
        raise SystemExit(
            f"{upstream}: the driver settings recorded with it are not this code's; make it again "
            f"with this code ({why})"
        )
