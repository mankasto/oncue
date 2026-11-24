"""Realism labels (flaws) of a candidate that passes every hard check (`sample_labels`). Flaws
never reject a candidate; the search picks the sample with the fewest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

from ...core.kinematics import exceeded, speeds
from ...core.rules import chain_stop_lines, held_phase
from ...core.types import Scene, Track
from ...core.visibility import occluders, visibility
from ...planners.idm.planner import STOP_SIGN_NEAR_M, STOPPED_MPS
from ...utils.geometry import (
    BOX_CORNERS,
    box_corners,
    boxes_overlap,
    heading_unit_vector,
    line_heading,
    wrap_angle,
)
from ...utils.tracks import frame_of
from .scene import scene_rules

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Container, Sequence

    from numpy.typing import ArrayLike

    from ...core.config import VisibilityConfig
    from ...core.road import VehicleLanes
    from ..context import SceneContext
    from ..requests.sites import Route
    from ..targets import NearMissRule, TargetSpec
    from .scene import SceneRules

FLAWS = (
    "moving_collision",
    "near_miss",
    "red_light",
    "signal_unknown_conflict",
    "stop_sign",
    "assumed_against_signal",
    "wrong_way",
    "curb",
    "speeding",
    "rare_motion",
)
# flaws inferred rather than read off a known signal
ASSUMED_FLAWS = ("signal_unknown_conflict", "assumed_against_signal")
# report groups: no flaw and nothing unjudged; only assumed or unknown flaws; a known flaw
CLEAN, ASSUMED_OR_UNKNOWN, KNOWN = "clean", "assumed_or_unknown", "known_flaws"
FLAW_GROUPS = (CLEAN, ASSUMED_OR_UNKNOWN, KNOWN)
SEVERITY_GROUPS = {
    "contact": ("moving_collision", "near_miss"),
    "rule": ("red_light", "stop_sign", "wrong_way", "curb", "speeding"),
    "assumed": ASSUMED_FLAWS,
    "motion": ("rare_motion",),
}
if sorted(sum(SEVERITY_GROUPS.values(), ())) != sorted(FLAWS):
    raise ValueError("SEVERITY_GROUPS must hold each of FLAWS exactly once")
# flaw names in earlier files, and the current ones
RENAMED_FLAWS = {"against_signal": "assumed_against_signal"}
EGO_GREEN = "go"  # the ego's phase that counts as "crossing against the signal"
# the ego's signal at a crossing: its path's last stop line within this far before it
SIGNAL_REACH_M = 40.0
# an ego turning more than this from its stop line to the crossing gives crossers walk
EGO_TURN_DEG = 45.0
EGO_HEADING_CHORD_M = 1.0  # chord either side when measuring the ego path's heading
# target types that drive along lanes, so can drive the wrong way
DRIVING_TYPES = ("vehicle", "cyclist")
OWN_SIGNAL_AHEAD_M = 1.0  # a signal of its own lanes counts at most this far past its crossing
ALLOWED_PHASES = ("go", "caution")  # phases where its own signal lets it cross
OWN_GO = "go"  # `own_signal` when its own signal lets it cross
OWN_STOP = "stop"  # `own_signal` when its own lane is red as it reaches the stop point


def sample_labels(
    ctx: SceneContext,
    route: Route,
    spec: TargetSpec,
    window_scene: Scene,
    peaks: dict | None,
    moving: list,
    required_speed: float | None = None,
) -> dict:
    """Return the sample's labels over the test's window.

    Args:
        window_scene: the scene with the target valid only inside the window.
        moving: [[id, frame]] of moving road users the target touches, sorted by frame.
        required_speed: the request's mean speed (m/s) for rare motion's peak-speed bounds;
            None for a baseline row.

    Returns:
        {"flaws", "signal", "measures", "over_limit", "required_speed", "curb_depth_m"}; each
        flaw is None when not found.
    """
    rules = scene_rules(ctx)
    target = window_scene.tracks[spec.track_id]
    states, frames, size = target.states, np.flatnonzero(target.valid), spec.size
    vis = ctx.config.visibility
    vehicle = spec.agent_type == "vehicle"
    crossing = band_entry(rules.band, states, frames, size)
    ego = None if crossing is None else ego_phase(rules, route, crossing)
    turn = None if ego is None else ego_turn_deg(ctx.path, rules, route)
    own = own_signal(ctx.scene, route, states, target.valid, size) if vehicle else None
    touched = {track_id for track_id, _ in moving}
    driving = spec.agent_type in DRIVING_TYPES
    excess = limit_excess(rules.vehicle_lanes, states, frames, ctx.dt) if vehicle else None
    flaws = {
        "moving_collision": moving[0] if moving else None,
        "near_miss": near_miss(
            window_scene, target, spec.near_miss, frames[-1], vis, ctx.static, touched
        ),
        **signal_flaws(vehicle, own, ego, crossing, turn is not None and turn > EGO_TURN_DEG),
        "stop_sign": stop_sign_run(rules, route, states, frames, size) if vehicle else None,
        "wrong_way": (
            first_found_frame(frames, rules.directions.opposing(states[frames][:, None, :]))
            if driving
            else None
        ),
        "curb": (
            first_found_frame(frames, rules.edges.offroad(states[frames], size))
            if spec.check_road_edge
            else None
        ),
        "speeding": speeding(excess, spec.speeding_tol_mps),
        "rare_motion": exceeded(peaks, spec.kinematics, required_speed=required_speed) or None,
    }
    measured = None if peaks is None else {k: round(float(v), 3) for k, v in peaks.items()}
    ordered = {name: flaws[name] for name in FLAWS}
    signal = {"own": own, "ego": ego, "ego_turn_deg": None if turn is None else round(turn, 1)}
    required = None if required_speed is None else round(float(required_speed), 3)
    return {
        "flaws": ordered,
        "signal": signal,
        "measures": measured,
        "over_limit": None if excess is None else excess["over_mps"],
        "required_speed": required,
        "curb_depth_m": curb_depth(rules, states, frames, size) if spec.check_road_edge else None,
    }


def curb_depth(
    rules: SceneRules, states: np.ndarray, frames: np.ndarray, size: ArrayLike
) -> float | None:
    """Return the deepest the box reaches past a road edge over the window (m; <= 0 on the road,
    None unknown)."""
    with np.errstate(invalid="ignore"):
        depth = rules.edges.depth(states[frames], size)
    return None if not np.isfinite(depth).any() else round(float(np.nanmax(depth)), 3)


def signal_flaws(
    vehicle: bool, own: str | None, ego: str | None, crossing: int | None, ego_turns: bool
) -> dict:
    """Return the signal flaws of a target crossing the ego's path at `crossing`."""
    green = ego == EGO_GREEN
    red = vehicle and own == OWN_STOP
    unknown = vehicle and green and own not in (OWN_GO, OWN_STOP)
    against = not vehicle and green and not ego_turns
    return {
        "red_light": {"frame": crossing, "own": own, "ego": ego} if red else None,
        "signal_unknown_conflict": {"frame": crossing, "own": own, "ego": ego} if unknown else None,
        "assumed_against_signal": {"frame": crossing, "ego": ego} if against else None,
    }


def limit_excess(
    lanes: VehicleLanes, states: np.ndarray, frames: np.ndarray, dt: float
) -> dict | None:
    """Return {"frame", "over_mps", "limit_mps"} where a vehicle target goes furthest above its
    lane's posted limit, or None if no frame can be judged.

    `frames` must be one contiguous valid run.
    """
    if len(frames) and frames[-1] - frames[0] + 1 != len(frames):
        raise ValueError("the test's window is not one valid run")
    speed = speeds(states[frames, :2], dt)
    if speed is None:
        return None
    lane = lanes.same_way_lanes(states[frames])
    limit = lanes.limits_of(lane)  # NaN marks frames without a recorded limit
    over = speed - limit
    if not np.isfinite(over).any():
        return None
    k = int(np.nanargmax(over))
    return {
        "frame": int(frames[k]),
        "over_mps": round(float(over[k]), 3),
        "limit_mps": round(float(limit[k]), 3),
    }


def speeding(excess: dict | None, tol_mps: float | None) -> dict | None:
    """Return the speeding flaw record when the excess is above `tol_mps`, else None."""
    if excess is None or tol_mps is None or excess["over_mps"] <= tol_mps:
        return None
    return excess


def current_flaws(flaws: dict | None) -> dict:
    """Return a row's flaws under the current names (earlier files use ``RENAMED_FLAWS``)."""
    out = {RENAMED_FLAWS.get(name, name): found for name, found in (flaws or {}).items()}
    red = out.get("red_light")
    if "signal_unknown_conflict" not in out and isinstance(red, dict):
        known = red.get("own") == OWN_STOP
        out["red_light"], out["signal_unknown_conflict"] = (red, None) if known else (None, red)
    return out


def found_flaws(flaws: dict | None) -> list[str]:
    """Return the names of the flaws set on a row (neither None nor False)."""
    found = current_flaws(flaws).items()
    return [name for name, value in found if value is not None and value is not False]


def flaw_count(flaws: dict | None) -> int:
    """Return how many of a sample's flaws were found."""
    return len(found_flaws(flaws))


def flaw_group(flaws: dict | None, unknown: Sequence[str] = ()) -> str:
    """Return the report group of a test; `unknown` lists the flaws that could not be judged."""
    found = found_flaws(flaws)
    if set(found) - set(ASSUMED_FLAWS):
        return KNOWN
    return ASSUMED_OR_UNKNOWN if found or unknown else CLEAN


def severity_groups(found: Container[str]) -> list[str]:
    """Return every severity group that holds one of the flaws in `found`."""
    return [group for group, flaws in SEVERITY_GROUPS.items() if any(f in found for f in flaws)]


def first_found_frame(frames: np.ndarray, found: np.ndarray) -> int | None:
    """Return the first frame in `frames` where `found` is True, or None."""
    return int(frames[np.argmax(found)]) if found.any() else None


def band_entry(
    band: shapely.Geometry, states: np.ndarray, frames: np.ndarray, size: ArrayLike
) -> int | None:
    """Return the first frame the target's box enters the ego's route band, or None."""
    boxes = shapely.polygons(box_corners(states[frames, :3], size))
    return first_found_frame(frames, shapely.intersects(boxes, band))


def ego_stop_line(rules: SceneRules, route: Route) -> float | None:
    """Return the station of the ego's last stop line within SIGNAL_REACH_M before the
    crossing, or None."""
    crossing = route.conflict_station
    near = [st for st, _ in rules.ego_signals if crossing - SIGNAL_REACH_M <= st <= crossing]
    return max(near) if near else None


def ego_phase(rules: SceneRules, route: Route, frame: int) -> str | None:
    """Return the ego's signal phase at the crossing at `frame`; green if any of its lanes is."""
    line = ego_stop_line(rules, route)
    if line is None:
        return None
    phases = {held[min(frame, len(held) - 1)] for st, held in rules.ego_signals if st == line}
    return EGO_GREEN if EGO_GREEN in phases else sorted(phases)[0]


def ego_turn_deg(path: shapely.LineString, rules: SceneRules, route: Route) -> float | None:
    """Return how far (degrees) the ego turns from its stop line to the crossing, or None."""
    line = ego_stop_line(rules, route)
    if line is None:
        return None
    stations = np.array([line, route.conflict_station])
    at_line, at_crossing = line_heading(
        path, stations, back=EGO_HEADING_CHORD_M, ahead=EGO_HEADING_CHORD_M
    )
    return abs(float(np.degrees(wrap_angle(at_crossing - at_line))))


def route_arcs(route: Route, xy: np.ndarray) -> np.ndarray:
    """Return the arc of each point [N, 2] along `route`, negative before its start."""
    arcs = shapely.line_locate_point(route.line, shapely.points(xy))
    back = (xy - route.point(0.0)) @ heading_unit_vector(route.heading(0.0))
    return np.where((arcs <= 0) & (back < 0), back, arcs)


def own_signal(
    scene: Scene, route: Route, states: np.ndarray, valid: np.ndarray, size: ArrayLike
) -> str | None:
    """Return a lane route's own signal as its box reaches each stop point: "stop", "go",
    "unknown", or None when it reaches no signal."""
    owner = route.meta.get("owner")
    frames = np.flatnonzero(valid)
    if not owner or not len(frames):
        return None
    signals = {s.lane_id: s for s in scene.signals}
    allowed = controlled = False  # controlled: a signal of its own lanes before the crossing
    corners = box_corners(states[frames, :3], size).reshape(-1, 2)
    # per frame, the farthest point of its box along the route
    front = route_arcs(route, corners).reshape(len(frames), BOX_CORNERS).max(axis=1)
    for lane_id in dict.fromkeys(owner):
        signal = signals.get(lane_id)
        if signal is None:
            continue
        stop_station = route.line.project(shapely.Point(*signal.stop_point))
        passed = frames[front >= stop_station]
        if not len(passed) or front[0] >= stop_station:
            continue  # never reaches it, or already past it at the first frame
        phase = held_phase(signal)[min(int(passed[0]), len(signal.states) - 1)]
        if phase == "stop":
            return OWN_STOP
        before_crossing = stop_station <= route.s_centre + OWN_SIGNAL_AHEAD_M
        controlled |= before_crossing
        allowed |= before_crossing and phase in ALLOWED_PHASES
    if allowed:
        return OWN_GO
    return "unknown" if controlled else None


def stop_sign_run(
    rules: SceneRules, route: Route, states: np.ndarray, frames: np.ndarray, size: ArrayLike
) -> dict | None:
    """Return {"station", "frame"} of the first stop line the target runs without stopping, or
    None; "stopped" follows the reference ego's rule."""
    points, owner, chain = (route.meta.get(k) for k in ("points", "owner", "chain"))
    if points is None or not owner or not chain or not len(frames):
        return None
    lines = chain_stop_lines(points, owner, chain, rules.stop_lanes, rules.lanes)
    front = route_arcs(route, states[frames, :2]) + size[0] / 2
    for station in lines:
        past = front >= station
        if past[0] or not past.any():
            continue
        k = int(np.argmax(past))
        stopped = (front[:k] >= station - STOP_SIGN_NEAR_M) & (states[frames[:k], 3] < STOPPED_MPS)
        if not stopped.any():
            return {"station": round(float(station), 2), "frame": int(frames[k])}
    return None


def near_miss(
    with_target: Scene,
    target: Track,
    rule: NearMissRule,
    end: int,
    vis: VisibilityConfig,
    static: shapely.Geometry | None,
    skip: Container[str] = (),
) -> list | None:
    """Return [id, frame, PET] of the first road user that has seen the target and passes it
    implausibly close, up to frame `end`; road users in `skip` are not judged."""
    valid_frames = np.flatnonzero(target.valid)
    if not len(valid_frames):
        return None
    frames = np.arange(valid_frames[0], min(end, len(with_target.timestamps) - 1) + 1)
    blockers = None
    for o in with_target.others(target.track_id):
        if o.track_id in skip:
            continue
        found = _pet(o, target, frames, with_target.dt, rule.pet_s, rule.speed_mps)
        if found is None:
            continue
        close = np.array(sorted(found))
        if blockers is None:  # when seen from a road user, the ego also blocks
            blockers = occluders(with_target, target.track_id, vis)
            if with_target.ego.agent_type in vis.occluder_types:
                blockers = [*blockers, with_target.ego]
        values = visibility(
            o, target, [b for b in blockers if b.track_id != o.track_id], vis, static
        )
        seen = np.maximum.accumulate(np.isfinite(values) & (values >= vis.threshold))
        hit = close[seen[close]]
        if len(hit):
            f = int(hit[0])
            return [o.track_id, f, found[f]]
    return None


def _pet(o: Track, target: Track, frames: np.ndarray, dt: float, max_s: float, speed_mps: float):
    """Return {frame of the second occupation: PET (s)} for every place both occupy within
    `max_s`, or None."""
    ft = frames[target.valid[frames]]
    fo = frames[o.valid[frames] & (o.states[frames, 3] >= speed_mps)]
    if not len(ft) or not len(fo):
        return None
    reach = (np.hypot(*target.size) + np.hypot(*o.size)) / 2
    d = np.linalg.norm(target.states[ft, None, :2] - o.states[None, fo, :2], axis=-1)
    if d.min() > reach:
        return None
    in_o = np.zeros(len(o.valid), bool)
    in_o[fo] = True
    events: dict[int, float] = {}
    for lag in range(1, frame_of(max_s, dt) + 1):
        for sign in (1, -1):  # o after the target, then o before it
            j = ft + sign * lag
            ok = (j >= 0) & (j < len(in_o))
            i, j = ft[ok], j[ok]
            i, j = i[in_o[j]], j[in_o[j]]
            if not len(i):
                continue
            hit = boxes_overlap(target.states[i], target.size, o.states[j], o.size)
            for a, b in zip(i[hit], j[hit], strict=True):
                events.setdefault(int(max(a, b)), round(lag * dt, 2))  # lags rise: smallest kept
    return events or None
