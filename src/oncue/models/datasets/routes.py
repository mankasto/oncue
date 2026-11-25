"""`route_chunks` gives the trajectory generator's route input, the agent's route as polyline chunks
(numpy only, shared by feature extraction and generation); at training the route is the one a
logged road user took (`logged_route`, with `scene_lanes`).

  route [R, Q, 3]  the route in the agent frame, 1 m spacing, R chunks of Q points from the point
                   nearest the agent; at training: a vehicle's lanes its logged future follows
                   (`vehicle_route`), a pedestrian's or cyclist's future path simplified to 0.5 m;
                   at generation: the inserted target's lane chain or crossing line
The signal phases of the map segments are `waymo_dataset`'s (SIGNAL_TIMES, PHASE here).
"""

from __future__ import annotations

import numpy as np
import shapely
import shapely.ops

from ...core.types import FeatureKind, Scene
from ...utils.geometry import MIN_LINE_POINTS, cumulative_arc_length, rotate, wrap_angle

ROUTE_CHUNKS, ROUTE_POINTS = 8, 16
ROUTE_SPACING_M = 1.0  # the route's points are resampled this far apart
SIGNAL_TIMES = 9  # the anchor and every 1 s
PHASE = {"unknown": 0, "stop": 1, "caution": 2, "go": 3}
# a pedestrian's or cyclist's route: its future path simplified to PATH_SIMPLIFY_M; none if it
# moves PATH_MIN_MOVE_M or less
PATH_SIMPLIFY_M, PATH_MIN_MOVE_M = 0.5, 1.0
# a vehicle follows a lane at a future point (every FOLLOW_EVERY_FRAMES frames, 0.5 s) within
# FOLLOW_LANE_M of it and FOLLOW_LANE_DEG of its heading; the lane's heading there from its points
# LANE_HEADING_REACH_M behind and ahead
FOLLOW_EVERY_FRAMES = 5
FOLLOW_LANE_M = 1.5
FOLLOW_LANE_DEG = 45
LANE_HEADING_REACH_M = 1
LANE_PIECE_MIN_M = 1e-6  # a followed lane's piece shorter than this is left out
JOIN_SAME_M = 0.5  # a lane piece's first point this near the route's last one is that point
ROUTE_JUMP_M = 20.0  # a route with a step longer than this, or turning back, is no route
ROUTE_TURN_DEG = 120.0
TURN_STEP_M = 0.2  # a turn back counts between two steps longer than this (shorter: no heading)
LANE_EXITS = 2  # where a `scene_lanes` entry (centreline, points, exit lanes) has the exit lanes


def route_chunks(
    points_world: np.ndarray | None, origin: np.ndarray, h0: float
) -> tuple[np.ndarray, int]:
    """([R, Q, 3] chunks of a world polyline in the agent frame from its point nearest the agent,
    1 if usable)."""
    out = np.zeros((ROUTE_CHUNKS, ROUTE_POINTS, 3))
    if points_world is None or len(points_world) < MIN_LINE_POINTS:
        return out, 0
    points = _resample(np.asarray(points_world, float))
    local = rotate(points - origin, -h0)  # in the agent's frame
    local = local[int(np.argmin(np.linalg.norm(local, axis=1))) :]
    for chunk in range(ROUTE_CHUNKS):
        piece = local[chunk * (ROUTE_POINTS - 1) : chunk * (ROUTE_POINTS - 1) + ROUTE_POINTS]
        if len(piece) < MIN_LINE_POINTS:
            break
        out[chunk, : len(piece), :2] = piece
        out[chunk, : len(piece), 2] = 1
    return out, int(out[0, :, 2].sum() >= MIN_LINE_POINTS)


def _resample(points: np.ndarray, step: float = ROUTE_SPACING_M) -> np.ndarray:
    """The polyline's points every `step` along it from its start (its two ends if shorter)."""
    arc = cumulative_arc_length(points)
    if arc[-1] < step:
        return points[[0, -1]] if len(points) >= MIN_LINE_POINTS else points
    stations = np.arange(0, arc[-1], step)
    return np.stack(
        [np.interp(stations, arc, points[:, 0]), np.interp(stations, arc, points[:, 1])], -1
    )


def scene_lanes(scene: Scene) -> dict:
    """lane id -> (centreline, points, exit lanes) of the scene's lanes."""
    return {
        feature.feature_id: (
            shapely.LineString(feature.points),
            np.asarray(feature.points, float),
            feature.exit_lanes,
        )
        for feature in scene.map_features
        if feature.kind is FeatureKind.LANE and len(feature.points) >= MIN_LINE_POINTS
    }


def logged_route(
    agent_type: str, world_fut: np.ndarray, lanes: dict, headings: np.ndarray
) -> np.ndarray | None:
    """The route a logged road user took (training): its lanes for a vehicle (`headings`: its
    recorded headings at the future points), else its future path simplified to 0.5 m (None if it
    moves 1 m or less)."""
    if agent_type == "vehicle":
        return vehicle_route(world_fut, lanes, headings)
    if np.linalg.norm(world_fut[-1] - world_fut[0]) <= PATH_MIN_MOVE_M:
        return None
    return np.array(shapely.LineString(world_fut).simplify(PATH_SIMPLIFY_M).coords)


def vehicle_route(fut_xy: np.ndarray, lanes: dict, headings: np.ndarray) -> np.ndarray | None:
    """Centrelines of the lanes a vehicle's logged future follows (within 1.5 m and 45 degrees of
    its heading, every 0.5 s: `_followed_lanes`), in visiting order, each cut where the vehicle was
    on it (a lane change leaves the first lane where it changes and joins the next where it
    arrives, not each lane whole, which turned back); a lane followed into its exit lane is kept to
    its end and the exit from its start (`_joined_lanes`). None if it follows none, or if the
    joined route jumps or turns back (`_jumps_or_turns_back`): no reliable route."""
    order, arcs = _followed_lanes(fut_xy, lanes, headings)
    if not order:
        return None
    route = _joined_lanes(order, arcs, lanes)
    if len(route) < MIN_LINE_POINTS or _jumps_or_turns_back(route):
        return None  # no reliable route
    return route


def _followed_lanes(fut_xy: np.ndarray, lanes: dict, headings: np.ndarray) -> tuple[list, dict]:
    """The lanes a vehicle's future follows in visiting order, and per lane the arcs where it was
    matched: every FOLLOW_EVERY_FRAMES frames the nearest lane running its way (`_lane_at`)."""
    lane_ids = list(lanes)
    lines = np.array([lanes[lane_id][0] for lane_id in lane_ids], dtype=object)
    recorded_headings = np.asarray(headings, float)  # recorded, at the future points
    order, arcs = [], {}  # lanes in visiting order; per lane the arcs where it was matched
    for frame in range(0, len(fut_xy) - 1, FOLLOW_EVERY_FRAMES):
        best = _lane_at(lines, lane_ids, shapely.Point(*fut_xy[frame]), recorded_headings[frame])
        if best:
            if not order or order[-1] != best[1]:
                order.append(best[1])
            arcs.setdefault(best[1], []).append(best[2])
    return order, arcs


def _lane_at(
    lines: np.ndarray, lane_ids: list, point: shapely.Point, heading: float
) -> tuple | None:
    """(distance, lane id, arc) of the nearest lane within FOLLOW_LANE_M of `point` that runs
    within FOLLOW_LANE_DEG of `heading` (the lane's heading from its points LANE_HEADING_REACH_M
    behind and ahead), or None."""
    # every lane at once
    distances = shapely.distance(lines, point) if len(lane_ids) else np.zeros(0)
    best = None
    for lane in np.flatnonzero(distances <= FOLLOW_LANE_M):
        line = lines[lane]
        station = line.project(point)
        behind = np.array(line.interpolate(max(station - LANE_HEADING_REACH_M, 0)).coords[0])
        ahead = np.array(
            line.interpolate(min(station + LANE_HEADING_REACH_M, line.length)).coords[0]
        )
        lane_heading = np.arctan2(*(ahead - behind)[::-1])
        if abs(wrap_angle(lane_heading - heading)) > np.radians(FOLLOW_LANE_DEG):
            continue
        if best is None or distances[lane] < best[0]:
            best = (distances[lane], lane_ids[lane], station)
    return best


def _joined_lanes(order: list, arcs: dict, lanes: dict) -> np.ndarray:
    """The followed lanes' centrelines joined in visiting order: each from its first matched arc
    (from its start if it is the first or entered from the previous lane's exits) to its last (to
    its end if it is the last or left into one of its exits)."""
    points = []
    for position, lane_id in enumerate(order):
        line, exits = lanes[lane_id][0], _exits(lanes, lane_id)
        entered = position > 0 and lane_id in _exits(lanes, order[position - 1])
        start = 0.0 if position == 0 or entered else min(arcs[lane_id])
        next_lane = order[position + 1] if position + 1 < len(order) else None
        end = line.length if next_lane is None or next_lane in exits else max(arcs[lane_id])
        if end - start < LANE_PIECE_MIN_M:
            continue
        piece = np.asarray(shapely.ops.substring(line, start, end).coords)
        if points and np.linalg.norm(piece[0] - points[-1]) < JOIN_SAME_M:
            piece = piece[1:]
        points.extend(piece)
    return np.array(points)


def _exits(lanes: dict, lane_id: str) -> tuple[str, ...]:
    """The lane's exit lanes (none in an entry without them)."""
    return lanes[lane_id][LANE_EXITS] if len(lanes[lane_id]) > LANE_EXITS else ()


def _jumps_or_turns_back(route: np.ndarray) -> bool:
    """Whether the joined route has a step longer than ROUTE_JUMP_M, or turns by more than
    ROUTE_TURN_DEG between two steps longer than TURN_STEP_M."""
    steps = np.diff(route, axis=0)
    length = np.hypot(steps[:, 0], steps[:, 1])
    step_heading = np.arctan2(steps[:, 1], steps[:, 0])
    turn = np.abs(wrap_angle(np.diff(step_heading)))
    real = (length[1:] > TURN_STEP_M) & (length[:-1] > TURN_STEP_M)
    return bool((length > ROUTE_JUMP_M).any() or ((turn > np.radians(ROUTE_TURN_DEG)) & real).any())
