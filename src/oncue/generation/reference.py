"""The protocol reference ego the generator times its waypoints against: it drives the logged
path at the lane speed limit, obeys the rules and never slows for the target or for occlusion."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import shapely

from ..core.config import DriverConfig
from ..core.rules import lane_speed_limit, logged_path, traffic_rules
from ..core.types import Scene
from ..data.map_geometry import lane_continuation
from ..planners.driver import Drive, drive

# gap the reference ego yields with to crossing and oncoming vehicles
YIELD_GAP_S = 1.5


def reference_settings(driver: DriverConfig) -> DriverConfig:
    """Return the shared driver settings plus the reference ego's yield gap and ignore_behind."""
    return replace(driver, yield_gap_s=YIELD_GAP_S, ignore_behind=True)


def reference_driver() -> DriverConfig:
    """Return the reference ego's driver settings, the same for every target type."""
    return reference_settings(DriverConfig())


def protocol_ego(
    scene: Scene, target_id: str | None, cfg: DriverConfig, speed_limit: float | None = None
) -> tuple[Scene, Drive]:
    """Return (scene with the ego replaced by the reference ego, its drive).

    The reference ego never perceives `target_id` (None: no target). Frames past its run are
    invalid.
    """
    cfg = reference_settings(cfg)
    limit = lane_speed_limit(scene) if speed_limit is None else speed_limit
    rules = traffic_rules(scene, limit, cfg)
    ends_at = log_end_route(scene, cfg)
    run = drive(
        scene, target_id, cfg, v_des=limit, rules=rules, ignore_target=True, ends_at=ends_at
    )
    ego = scene.ego
    valid = ego.valid.copy()
    valid[scene.anchor : run.valid_until] = True
    valid[run.valid_until :] = False
    states = run.states.copy()
    states[~valid] = 0.0
    return replace(
        scene, tracks={**scene.tracks, ego.track_id: ego.with_states(states, valid)}
    ), run


def log_end_route(scene: Scene, cfg: DriverConfig) -> shapely.LineString | None:
    """Return the logged path when no lane continues it past the log's end, else None."""
    ego = scene.ego
    last = ego.states[np.flatnonzero(ego.valid)[-1]]
    if lane_continuation(scene, last[:2], float(last[2]), cfg.extension_m) is not None:
        return None
    return logged_path(scene)
