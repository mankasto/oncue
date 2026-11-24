"""Phantom vehicles for the occlusion-aware IDM planner; start at ``crossing_conflicts`` and
``phantoms``.

A phantom stands only on a lane whose traffic has the right of way over the ego, within
``phantom_range_m`` of the crossing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ...core.config import DriverConfig
from ...core.rules import (
    DEFAULT_SPEED_LIMIT,
    STOP_LINE_MATCH_M,
    chain_stop_lines,
    held_phase,
    phase_at,
    stop_sign_lanes,
)
from ...core.types import FeatureKind, Scene
from ...core.vehicle import VEHICLE_HEIGHT_M, VEHICLE_SIZE_M
from ...data.map_geometry import chain_speed_limit, lane_crossings
from ...utils.geometry import MIN_LINE_POINTS, line_heading
from .path import HEADING_SAMPLE_M, DriverPath, heading_gap, inclusive_range

if TYPE_CHECKING:
    from ...core.rules import TrafficRules
    from ...core.types import SignalTrack, Track
    from ...data.map_geometry import LaneCrossing
    from ..perception import Beliefs, Perception

HEADING_MATCH_DEG = 30.0  # a heading within this of a way counts as going that way
PARKED_MPS = 0.5  # slower vehicles count as standing: they never arrive
# Phantom-vehicle layout.
PHANTOM_STEP_M = 1.0  # candidate phantom fronts this far apart upstream of a crossing
PHANTOM_LANE_M = 1.5  # a known vehicle within this of a lane's centreline, going its way, is on it
HOLD_TOL_M = 1.0  # a stop line up to this past the crossing still holds the lane before it
GO_PHASES = ("go", "caution")  # a signal showing one of these lets its lane through
# Turning rule (``_yields_to_ego``).
TURN_SPAN_M = 20.0  # an approach heading is the chord over this before the crossing
OPPOSITE_DEG = 135.0  # a lane approaching this far off the ego's approach comes the opposite way


@dataclass(frozen=True, eq=False)
class Conflict:
    """One crossing-lane conflict used by phantom-vehicle placement."""

    station: float  # crossing along the ego's path
    zone: tuple[float, float]  # ego stations where its corridor meets a vehicle on the lane
    edge: float  # how far before the crossing, along the lane, a vehicle front reaches the corridor
    speed: float  # phantom speed
    boxes: np.ndarray  # [K, 4] candidate phantom vehicles along the lane
    fronts: np.ndarray  # [K] each phantom front's distance upstream of the crossing
    line: shapely.LineString
    lane_station: float  # crossing along the lane
    holds: tuple  # stop lines before the crossing: (distance upstream, phase per frame or None)
    # Phase per frame of the ego's signal whose junction lane the crossing lies on, or None.
    ego_phase: np.ndarray | None
    yields_to_ego: bool  # the lane's traffic turns across the ego's path from the opposite way


def crossing_conflicts(
    scene: Scene,
    path: DriverPath,
    corridor_half_width: float,
    cfg: DriverConfig,
    rules: TrafficRules | None = None,
) -> list[Conflict]:
    """Return the lanes crossing the ego's path ahead; build once per drive.

    ``rules`` carries the ego's signals (``None``: it has none).
    """
    lanes = {
        feature.feature_id: feature
        for feature in scene.map_features
        if feature.kind is FeatureKind.LANE and len(feature.points) >= MIN_LINE_POINTS
    }
    stop_lanes = stop_sign_lanes(scene)
    signals: dict[str, list] = {}
    for signal in scene.signals:
        signals.setdefault(signal.lane_id, []).append(signal)
    length, width = VEHICLE_SIZE_M
    out = []
    for crossing, reason in lane_crossings(lanes, path.line, scene.ego):
        if reason is not None:
            continue  # excluded (``lane_crossings`` says why)
        angle = np.radians(crossing.angle_deg)
        sin_angle, cos_angle = np.sin(angle), abs(np.cos(angle))
        limit = (
            chain_speed_limit(lanes, crossing.chain, crossing.lane.feature_id)
            or DEFAULT_SPEED_LIMIT
        )
        speed = cfg.phantom_speed_factor * limit
        fronts, boxes = _phantom_boxes(crossing, length, speed, cfg)
        zone_half = (width / 2 + corridor_half_width * cos_angle) / sin_angle
        out.append(
            Conflict(
                station=crossing.cross_station,
                zone=(crossing.cross_station - zone_half, crossing.cross_station + zone_half),
                edge=(corridor_half_width + width / 2 * cos_angle) / sin_angle,
                speed=speed,
                boxes=boxes,
                fronts=fronts,
                line=crossing.line,
                lane_station=crossing.lane_station,
                holds=_holds(crossing, stop_lanes, lanes, signals),
                ego_phase=_ego_phase(crossing.cross_station, rules, lanes),
                yields_to_ego=_yields_to_ego(crossing, path.line),
            )
        )
    return out


def _ego_phase(station: float, rules: TrafficRules | None, lanes: dict) -> np.ndarray | None:
    """Return the phase per frame of the ego's signal governing ``station``, or ``None``."""
    if rules is None:
        return None
    for (stop, phase), lane_id in zip(rules.signals, rules.signal_lanes, strict=True):
        lane = lanes.get(lane_id)
        if lane is None:
            continue
        length = shapely.LineString(lane.points).length
        if stop - HOLD_TOL_M <= station <= stop + length + HOLD_TOL_M:
            return phase
    return None


def _yields_to_ego(crossing: LaneCrossing, path: shapely.LineString) -> bool:
    """Return whether the lane comes from the opposite way and turns more than the ego's path."""
    lane_before = line_heading(
        crossing.line, crossing.lane_station - TURN_SPAN_M, back=0.0, ahead=HEADING_SAMPLE_M
    )
    lane_at = line_heading(crossing.line, crossing.lane_station, back=0.0, ahead=HEADING_SAMPLE_M)
    path_before = line_heading(
        path, crossing.cross_station - TURN_SPAN_M, back=0.0, ahead=HEADING_SAMPLE_M
    )
    path_at = line_heading(path, crossing.cross_station, back=0.0, ahead=HEADING_SAMPLE_M)
    if heading_gap(lane_before, path_before) <= np.radians(OPPOSITE_DEG):
        return False
    return heading_gap(lane_at, lane_before) > heading_gap(path_at, path_before)


def _phantom_boxes(crossing: LaneCrossing, length: float, speed: float, cfg: DriverConfig):
    """Return ``(fronts, boxes)`` of candidate phantom vehicles upstream on the crossing lane."""
    fronts = inclusive_range(0.0, cfg.phantom_range_m, PHANTOM_STEP_M)
    fronts = fronts[crossing.lane_station - fronts >= length]
    center = crossing.lane_station - fronts - length / 2
    xy = shapely.get_coordinates(shapely.line_interpolate_point(crossing.line, center))
    heading = line_heading(crossing.line, center, back=0.0, ahead=HEADING_SAMPLE_M)
    return fronts, np.column_stack((xy, heading, np.full(len(xy), speed)))


def _holds(
    crossing: LaneCrossing,
    stop_lanes: set[str],
    lanes: dict,
    signals: dict[str, list[SignalTrack]],
) -> tuple:
    """Return the stop lines holding the crossing lane before the crossing.

    Each is ``(distance upstream, phase per frame)``; the phase is ``None`` for a stop sign.
    """
    upstream = crossing.chain[: crossing.chain.index(crossing.lane.feature_id) + 1]
    holds = [
        (max(crossing.lane_station - at, 0.0), None)
        for at in chain_stop_lines(crossing.points, crossing.owner, upstream, stop_lanes, lanes)
        if at <= crossing.lane_station + HOLD_TOL_M
    ]
    for lane_id in upstream:
        for signal in signals.get(lane_id, ()):
            stop = shapely.Point(*signal.stop_point)
            at = crossing.line.project(stop)
            if (
                crossing.line.distance(stop) <= STOP_LINE_MATCH_M
                and at <= crossing.lane_station + HOLD_TOL_M
            ):
                holds.append((max(crossing.lane_station - at, 0.0), held_phase(signal)))
    return tuple(holds)


def phantoms(
    conflicts: list[Conflict],
    beliefs: Beliefs,
    eye: np.ndarray,
    scene: Scene,
    perception: Perception,
) -> list:
    """Return ``(conflict, arrival seconds)`` for each conflict that has a phantom vehicle.

    The phantom is the first candidate short of any holding stop line that could be unseen from
    ``eye`` and that no known vehicle takes. It never arrives before a known vehicle ahead of it.
    """
    obs_frame, believed = beliefs.frame, beliefs.states
    rows, spans = [], []
    for conflict in conflicts:
        held_from = 0.0 if conflict.yields_to_ego else _held_from(conflict, obs_frame)
        keep = np.flatnonzero(conflict.fronts < held_from)
        spans.append((len(rows), keep))
        rows.extend(conflict.boxes[keep])
    if not rows:
        return []
    boxes = np.array(rows, float)
    unseen = perception.unseen_vehicles(
        eye, boxes, VEHICLE_SIZE_M, VEHICLE_HEIGHT_M, beliefs, scene
    )
    length = VEHICLE_SIZE_M[0]
    out = []
    for conflict, (start, keep) in zip(conflicts, spans, strict=True):
        if not len(keep):
            continue
        on_lane = _known_on_lane(conflict, believed, scene.tracks)
        for n, front in enumerate(conflict.fronts[keep]):
            if any(
                front < other + other_length and front + length > other
                for other, other_length, _ in on_lane
            ):
                continue  # a known vehicle is there
            if not unseen[start + n]:
                continue
            arrival_s = max(front - conflict.edge, 0.0) / conflict.speed
            arrival_s = max(
                [arrival_s] + [other_s for other, _, other_s in on_lane if other <= front]
            )
            out.append((conflict, arrival_s))
            break
    return out


def _held_from(conflict: Conflict, obs_frame: int) -> float:
    """Return how far upstream of the crossing its lane is held at ``obs_frame``, or ``inf``.

    While the ego's signal shows go or caution, a lane whose own signal shows neither counts as
    red, as WOMD mostly leaves cross streets unrecorded.
    """
    held_from, signal_stops, lets_through = np.inf, [], False
    for distance, phase in conflict.holds:
        now = None if phase is None else phase_at(phase, obs_frame)
        if now is None or now == "stop":
            held_from = min(held_from, distance)
        if now is not None:
            signal_stops.append(distance)
            lets_through = lets_through or now in GO_PHASES
    ego = conflict.ego_phase
    if ego is not None and phase_at(ego, obs_frame) in GO_PHASES and not lets_through:
        held_from = min(held_from, min(signal_stops, default=0.0))
    return held_from


def _known_on_lane(
    conflict: Conflict, believed: dict[str, np.ndarray], tracks: dict[str, Track]
) -> list:
    """Return ``(front, length, arrival)`` for every known vehicle on the conflict's lane.

    Only vehicles going the lane's way and not yet clear of the corridor are listed.
    """
    on_lane = []
    for road_user, state in believed.items():
        track = tracks[road_user]
        if track.agent_type != "vehicle":
            continue
        point = shapely.Point(*state[:2])
        if conflict.line.distance(point) > PHANTOM_LANE_M:
            continue
        along = conflict.line.project(point)
        lane_heading = line_heading(
            conflict.line, along, back=HEADING_SAMPLE_M, ahead=HEADING_SAMPLE_M
        )
        if heading_gap(lane_heading, state[2]) > np.radians(HEADING_MATCH_DEG):
            continue
        front = conflict.lane_station - along - track.size[0] / 2
        if front + track.size[0] < -conflict.edge:
            continue  # its rear is past the corridor: it has cleared the conflict
        arrival_s = max(front - conflict.edge, 0.0) / state[3] if state[3] >= PARKED_MPS else np.inf
        on_lane.append((front, track.size[0], arrival_s))
    return on_lane
