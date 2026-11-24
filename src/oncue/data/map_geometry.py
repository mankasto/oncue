"""Map geometry shared by several modules: drivable area, occluders, elevation and lane chains.

WOMD has no road polygons, so ``drivable_region`` approximates the drivable area from lane
boundaries and road edges; good for containment checks, not as map ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import shapely
from shapely.errors import GEOSException
from shapely.geometry import LineString, Polygon, box

from ..core.types import (
    BoundarySegment,
    FeatureKind,
    LaneType,
    MapFeature,
    Scene,
    Track,
    height_of,
)
from ..utils.geometry import (
    MIN_DIVISOR,
    MIN_LINE_POINTS,
    MIN_POLYGON_POINTS,
    ZERO_LENGTH_M,
    box_corners,
    cumulative_arc_length,
    heading_unit_vector,
    line_heading,
    union_all,
    wrap_angle,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import ArrayLike

MIN_HALF_WIDTH_M = 0.4
# levels (an overpass): roads this far apart in elevation are on different ones
LEVEL_GAP_M = 3.0
LEVEL_EGO_M = 5.0  # ego's own level is read from its recorded elevation when it passes this near
LEVEL_LANE_M = 1.5  # same-level lanes are read within this of the crossing point
LEVEL_LANE_ALIGN_RAD = np.pi / 6  # and within 30 degrees of the path's heading
# lane chains crossing a path (``lane_crossings``)
LANE_UPSTREAM_M, LANE_DOWNSTREAM_M = 120.0, 30.0  # lane-chain reach around the crossing lane
LANE_MIN_AHEAD_M = 10.0  # a crossing lane must lie at least this far along the ego path
JOINT_M = 1e-3  # a lane end within this of the path meets it there (map cut on the path)
OFFSET_CHORD_M = 0.5  # chord ahead used to classify which side of the path a point lies on
# caps on a chain's length in lanes; its reach is set by distance
CHAIN_UPSTREAM_LANES = 40
CHAIN_DOWNSTREAM_LANES = 20
CHAIN_JOINT_M = 0.5  # a lane's first point this near the chain's last point is that point
# angles (degrees) that count as crossing; outside them a lane merges in or out
CROSSING_ANGLES_DEG = (45.0, 135.0)
# drivable area (``drivable_region``)
MAX_HALF_WIDTH_M = 8.0  # a lane's largest half-width
ROAD_EDGE_SEARCH_M = 16.0  # reach for road edges along a normal
REGION_MARGIN_M = 25.0  # margin around the ego's logged extent
MIN_CELL_AREA_M2 = 1e-6  # drivable cells smaller than this are degenerate
ROAD_EDGE_BARRIER_M = 0.02  # buffer road edges by this to use them as cut-out barriers
NORMAL_REACH_SLACK_M = 1e-3  # segments this far past a normal's reach are still considered
PARALLEL_TOL = 1e-8  # a normal this near parallel to a segment never meets it
SEGMENT_END_TOL = 1e-6  # a normal meeting a segment this far past an end still counts
# lane width (``lane_width``): boundary reach, and the range outside which the default applies
LANE_WIDTH_SEARCH_M = 4.0
MIN_LANE_WIDTH_M = 2.5
MAX_LANE_WIDTH_M = 6.0
LANE_WIDTH_MATCH_M = 1.5  # the lane a width is read from: centreline this close to the position
LANE_WIDTH_ALIGN_RAD = np.pi / 4  # and running within 45 degrees of the heading
OCCLUDER_RADIUS_M = 150.0  # static occluders are built this far around the ego's logged extent
OCCLUDER_GRID_M = 0.1  # geometry grid for static occluders
MIN_SWEPT_FRAMES = 2  # a road user's swept area counts as open space from this many frames


def drivable_region(
    scene: Scene,
    *,
    margin: float = REGION_MARGIN_M,
) -> shapely.Geometry:
    """Return the approximate drivable area around the ego's logged extent.

    The union of per-car-lane cells (``_lane_cells``) with road edges cut out as barriers.
    """
    ego = scene.ego.states[scene.ego.valid, :2]
    low, high = ego.min(axis=0) - margin, ego.max(axis=0) + margin
    by_id = {f.feature_id: f for f in scene.map_features}
    edges = [f for f in scene.map_features if f.kind is FeatureKind.ROAD_EDGE]
    search = _SideSearch(by_id, _segments(edges))
    cells = []
    for lane in scene.map_features:
        if lane.kind is not FeatureKind.LANE or lane.lane_type is LaneType.BIKE_LANE:
            continue
        if len(lane.points) < MIN_LINE_POINTS:
            continue
        cells += _lane_cells(lane, low, high, search)
    region = union_all(cells).intersection(box(*low, *high))
    barriers = union_all(
        [
            LineString(f.points).buffer(ROAD_EDGE_BARRIER_M)
            for f in edges
            if len(f.points) >= MIN_LINE_POINTS
        ]
    )
    return region.difference(barriers)


@dataclass(frozen=True, eq=False)
class _SideSearch:
    """Where ``drivable_region`` looks for a lane's sides: boundaries first, then road edges."""

    by_id: dict[str, MapFeature]
    edge_segments: tuple[np.ndarray, np.ndarray]


def _lane_cells(
    lane: MapFeature, low: np.ndarray, high: np.ndarray, search: _SideSearch
) -> list[Polygon]:
    """Return one lane's drivable cells within ``MAX_HALF_WIDTH_M`` of ``[low, high]``.

    One quadrilateral per segment where both sides were found.
    """
    points = lane.points
    # segment bounds, not vertices: a sparse lane can cross the box with no vertex in it
    seg_lo = np.minimum(points[:-1], points[1:])
    seg_hi = np.maximum(points[:-1], points[1:])
    near = ((seg_hi >= low - MAX_HALF_WIDTH_M) & (seg_lo <= high + MAX_HALF_WIDTH_M)).all(axis=1)
    if not near.any():
        return []
    tangent = np.gradient(points, axis=0)
    normals = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), MIN_DIVISOR)
    sides = []
    for segments, sign in ((lane.left_boundaries, 1), (lane.right_boundaries, -1)):
        offset = _lane_side(points, normals, segments, sign, search)
        sides.append(points + offset[:, None] * normals)
    left, right = sides
    cells = []
    for i in np.flatnonzero(near):
        quad = np.array([left[i], left[i + 1], right[i + 1], right[i]])
        if np.isfinite(quad).all():
            cell = Polygon(quad)
            if cell.is_valid and cell.area > MIN_CELL_AREA_M2:
                cells.append(cell)
    return cells


def _lane_side(
    points: np.ndarray,
    normals: np.ndarray,
    segments: tuple[BoundarySegment, ...],
    sign: int,
    search: _SideSearch,
) -> np.ndarray:
    """Return the signed offset of one lane side (``sign`` 1: left) along each point's normal.

    Boundaries come first, then the nearest road edge; ``NaN`` where neither is found.
    """
    offset = np.full(len(points), np.nan)
    for seg in segments:
        feature = search.by_id.get(seg.feature_id)
        if feature is None:
            continue
        idx = np.arange(max(0, seg.start), min(len(points) - 1, seg.end) + 1)
        hit = _normal_offsets(
            points[idx], normals[idx], _segments([feature]), sign, MAX_HALF_WIDTH_M
        )
        offset[idx] = np.where(np.isnan(offset[idx]), hit, offset[idx])
    missing = np.isnan(offset)
    if missing.any():
        offset[missing] = _normal_offsets(
            points[missing], normals[missing], search.edge_segments, sign, ROAD_EDGE_SEARCH_M
        )
    return offset


def _segments(features: list[MapFeature]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` of every segment of each feature's polyline."""
    parts = [f.points for f in features if len(f.points) >= MIN_LINE_POINTS]
    if not parts:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.concatenate([p[:-1] for p in parts]), np.concatenate([p[1:] for p in parts])


def _normal_offsets(
    points: np.ndarray,
    normals: np.ndarray,
    segments: tuple[np.ndarray, np.ndarray],
    side: int,
    max_distance: float,
) -> np.ndarray:
    """Return the signed distance along each normal to its nearest segment on ``side``, or NaN."""
    starts, ends = segments
    out = np.full(len(points), np.nan)
    if not len(starts) or not len(points):
        return out
    # A hit lies within ``max_distance`` of a point, so segments past that reach cannot qualify.
    reach = max_distance + NORMAL_REACH_SLACK_M
    low, high = points.min(axis=0) - reach, points.max(axis=0) + reach
    near = ((np.maximum(starts, ends) >= low) & (np.minimum(starts, ends) <= high)).all(axis=1)
    starts, ends = starts[near], ends[near]
    if not len(starts):
        return out
    d = (ends - starts)[None]  # [1,S,2]
    q = starts[None] - points[:, None]  # [P,S,2]
    n = normals[:, None]  # [P,1,2]
    denom = n[..., 0] * d[..., 1] - n[..., 1] * d[..., 0]
    ok = np.abs(denom) > PARALLEL_TOL
    safe = np.where(ok, denom, 1.0)
    t = (q[..., 0] * d[..., 1] - q[..., 1] * d[..., 0]) / safe  # along normal
    u = (q[..., 0] * n[..., 1] - q[..., 1] * n[..., 0]) / safe  # along segment
    ok &= (u >= -SEGMENT_END_TOL) & (u <= 1 + SEGMENT_END_TOL)
    ok &= (side * t >= MIN_HALF_WIDTH_M) & (side * t <= max_distance)
    distance = np.where(ok, np.abs(t), np.inf).min(axis=1)
    found = np.isfinite(distance)
    out[found] = side * distance[found]
    return out


def lane_width(scene: Scene, position: np.ndarray, heading: float, default: float) -> float:
    """Return the lane width at ``position`` with the given ``heading``, else ``default``."""
    best = None
    by_id = {f.feature_id: f for f in scene.map_features}
    for lane in scene.map_features:
        if lane.kind is not FeatureKind.LANE or len(lane.points) < MIN_LINE_POINTS:
            continue
        i = int(np.argmin(np.linalg.norm(lane.points - position, axis=1)))
        gap = float(np.linalg.norm(lane.points[i] - position))
        tangent = lane.points[min(i + 1, len(lane.points) - 1)] - lane.points[max(i - 1, 0)]
        turn = np.arctan2(tangent[1], tangent[0]) - heading
        aligned = abs(wrap_angle(turn)) < LANE_WIDTH_ALIGN_RAD
        if gap < LANE_WIDTH_MATCH_M and aligned and (best is None or gap < best[0]):
            best = (gap, lane, i, tangent / np.linalg.norm(tangent))
    if best is None:
        return default
    _, lane, i, tangent = best
    normal = np.array([[-tangent[1], tangent[0]]])
    sides = []
    for segments, sign in ((lane.left_boundaries, 1), (lane.right_boundaries, -1)):
        features = [
            by_id[s.feature_id] for s in segments if s.feature_id in by_id and s.start <= i <= s.end
        ]
        hit = _normal_offsets(
            lane.points[i : i + 1], normal, _segments(features), sign, LANE_WIDTH_SEARCH_M
        )
        sides.append(abs(hit[0]))
    width = float(sum(sides))
    return (
        width if np.isfinite(width) and MIN_LANE_WIDTH_M <= width <= MAX_LANE_WIDTH_M else default
    )


def static_occluder_region(
    scene: Scene, setback_m: float, radius_m: float = OCCLUDER_RADIUS_M
) -> shapely.Geometry:
    """Approximate static occluders (buildings, walls, vegetation) within ``radius_m`` of the ego.

    The area farther than ``setback_m`` from open space: the drivable region, crosswalks and
    any area a logged road user swept.
    """
    parts = [drivable_region(scene, margin=radius_m)]
    parts += [swept_area(t) for t in scene.tracks.values() if t.valid.sum() >= MIN_SWEPT_FRAMES]
    parts += [
        shapely.Polygon(f.points).buffer(0)
        for f in scene.map_features
        if f.kind is FeatureKind.CROSSWALK and len(f.points) >= MIN_POLYGON_POINTS
    ]
    parts = [shapely.make_valid(p) for p in parts if p is not None and not p.is_empty]
    union = shapely.union_all(parts, grid_size=OCCLUDER_GRID_M)
    ego = scene.ego.states[scene.ego.valid, :2]
    low, high = ego.min(axis=0) - radius_m, ego.max(axis=0) + radius_m
    try:
        region = grid_difference(box(*low, *high), union.buffer(setback_m))
    except GEOSException:
        # GEOS's buffer can return self-intersecting rings on rare inputs, which both overlays
        # reject; the same set piece by piece: the union of buffers is the buffer of the union
        buffers = [piece.buffer(setback_m) for piece in shapely.get_parts(union)]
        region = grid_difference(box(*low, *high), shapely.union_all(buffers))
    shapely.prepare(region)
    return region


def grid_difference(a: shapely.Geometry, b: shapely.Geometry) -> shapely.Geometry:
    """Return ``a`` minus ``b`` on the occluder grid.

    GEOS's fixed-grid overlay has no fallback and fails on rare inputs ("Ring edge missing");
    then the difference is taken in floating point, where GEOS retries with snapping, and
    rounded to the grid.
    """
    try:
        return shapely.difference(a, b, grid_size=OCCLUDER_GRID_M)
    except GEOSException:
        return shapely.set_precision(shapely.difference(a, b), OCCLUDER_GRID_M)


def swept_area(track: Track) -> shapely.Geometry:
    """Return the area a track's footprint covers, including between consecutive observed frames."""
    frames = np.flatnonzero(track.valid)
    if not len(frames):
        return shapely.Polygon()
    corners = box_corners(track.states[frames], track.size)  # [F, 4, 2]
    joined = np.flatnonzero(np.diff(frames) == 1)
    hulls = shapely.convex_hull(
        shapely.multipoints(np.concatenate((corners[joined], corners[joined + 1]), axis=1))
    )
    return union_all(np.concatenate((shapely.polygons(corners), hulls)))


# ---------------------------------------------------------------- elevation
def lane_z_at(feature: MapFeature, point: shapely.Point) -> float | None:
    """Return a lane's elevation at the point nearest ``point``, or ``None`` when not recorded."""
    if feature.z is None or len(feature.z) != len(feature.points):
        return None
    arcs = cumulative_arc_length(feature.points)
    return float(np.interp(shapely.LineString(feature.points).project(point), arcs, feature.z))


def ego_level(ego: Track | None, point: shapely.Point) -> float | None:
    """Return the road surface under the logged ego at the point nearest ``point``.

    ``None`` without recorded elevation or when the ego never passes within ``LEVEL_EGO_M``.
    """
    if ego is None or ego.z is None:
        return None
    seen = ego.valid & np.isfinite(ego.z)
    if not seen.any():
        return None
    d = np.linalg.norm(ego.states[seen, :2] - np.array([point.x, point.y]), axis=1)
    k = int(np.argmin(d))
    if d[k] > LEVEL_EGO_M:
        return None
    return float(ego.z[seen][k] - height_of(ego) / 2)


# ---------------------------------------------------------------- lane continuation of a path
LANE_MATCH_M = 2.0  # a lane continuation starts on a lane this close to the path's end
LANE_HOPS = 500  # a lane continuation follows at most this many lanes
LANE_SAME_M = 0.1  # lanes this much farther than the nearest from the path's end still tie
# lanes are compared over this length: WOMD junction lanes share their first ~0.5 m
LANE_AHEAD_M = 20.0
LANE_FADE_M = 10.0  # offset from the path fades over this many metres of continuation
LANE_START_ALIGN_RAD = np.pi / 6  # the starting lane must run within 30 degrees of the heading


def lane_continuation(
    scene: Scene, point: np.ndarray, heading: float, length: float
) -> np.ndarray | None:
    """Continue a path that ends at ``point`` with ``heading`` along the lane graph.

    Follows the successor bending least at each choice and goes straight on past the graph's
    end. ``None`` when no lane matches at the start.
    """
    lanes = {
        f.feature_id: f
        for f in scene.map_features
        if f.kind is FeatureKind.LANE and len(f.points) >= MIN_LINE_POINTS
    }
    start = _start_lane(lanes, point, heading)
    if start is None:
        return None
    out, total = _walk_lanes(lanes, start, point, heading, length)
    if not out:
        return None
    out = _faded(out, start, point)
    if total < length:  # the lane graph ends: continue straight
        out.append(_straight_on(out, point, heading, length - total))
    return np.asarray(out)


class _LaneStart(NamedTuple):
    """The lane a continuation starts on (``_start_lane``)."""

    lane: MapFeature
    arcs: np.ndarray  # arc lengths at its points
    at: float  # path end's station on this lane


def _start_lane(lanes: dict, point: np.ndarray, heading: float) -> _LaneStart | None:
    """Return the nearest lane running along ``heading`` at ``point``, the straightest on a tie."""
    at_point = shapely.Point(*point)
    found = []
    for lane in lanes.values():
        points = lane.points
        if _step_direction(points) is None:
            continue
        line = shapely.LineString(points)
        distance = line.distance(at_point)
        if distance >= LANE_MATCH_M:
            continue
        at = line.project(at_point)
        arcs = cumulative_arc_length(points)
        k = int(np.clip(np.searchsorted(arcs, at, side="right") - 1, 0, len(points) - 2))
        # Move off a zero-length segment to find the one actually holding ``at``.
        while k < len(points) - 2 and arcs[k + 1] - arcs[k] <= ZERO_LENGTH_M:
            k += 1
        while k > 0 and arcs[k + 1] - arcs[k] <= ZERO_LENGTH_M:
            k -= 1
        direction = np.arctan2(*(points[k + 1] - points[k])[::-1])
        if abs(wrap_angle(direction - heading)) < LANE_START_ALIGN_RAD:
            found.append((distance, _bend(line, at, heading), lane, arcs, at))
    if not found:
        return None
    near = min(f[0] for f in found)
    _, _, lane, arcs, at = min((f for f in found if f[0] <= near + LANE_SAME_M), key=lambda f: f[1])
    return _LaneStart(lane, arcs, at)


def _step_direction(pts: ArrayLike, last: bool = False) -> float | None:
    """Return the direction of a polyline's first (or last) nonzero step, or ``None``."""
    steps = np.diff(np.asarray(pts, float), axis=0)
    moving = np.flatnonzero(np.linalg.norm(steps, axis=1) > ZERO_LENGTH_M)
    if not len(moving):
        return None
    dx, dy = steps[moving[-1] if last else moving[0]]
    return float(np.arctan2(dy, dx))


def _bend(line: shapely.LineString, at: float, heading: float) -> float:
    """Return how far the ``LANE_AHEAD_M`` chord of ``line`` from ``at`` turns from ``heading``."""
    here = np.array(line.interpolate(at).coords[0])
    ahead = np.array(line.interpolate(min(at + LANE_AHEAD_M, line.length)).coords[0])
    chord = _step_direction(np.array([here, ahead]))
    return 0.0 if chord is None else abs(wrap_angle(chord - heading))


def _walk_lanes(
    lanes: dict, start: _LaneStart, point: np.ndarray, heading: float, length: float
) -> tuple[list, float]:
    """Return ``(points, length)`` walking the lane graph, taking the least-bending successor."""
    out, total, prev_dir, lane = [], 0.0, heading, start.lane
    points = lane.points[start.arcs > start.at + ZERO_LENGTH_M]
    last = point
    for _ in range(LANE_HOPS):
        for next_point in points:
            total += float(np.linalg.norm(next_point - last))
            out.append(next_point)
            last = next_point
            if total >= length:
                break
        if total >= length:
            break
        if len(out) >= MIN_LINE_POINTS:
            moved = _step_direction(out, last=True)
            prev_dir = prev_dir if moved is None else moved
        nexts = _successors(lanes, lane)
        if not nexts:
            break
        lane = min(nexts, key=lambda f: _bend(shapely.LineString(f.points), 0.0, prev_dir))
        points = lane.points[1:]
    return out, total


def _successors(lanes: dict, lane: MapFeature) -> list[MapFeature]:
    """Return a lane's exit lanes in order, a zero-length lane standing for its own exits."""
    out, todo, seen = [], list(lane.exit_lanes), set()
    while todo:
        exit_id = todo.pop(0)
        if exit_id in seen or exit_id not in lanes:
            continue
        seen.add(exit_id)
        if _step_direction(lanes[exit_id].points) is None:
            todo.extend(lanes[exit_id].exit_lanes)
        else:
            out.append(lanes[exit_id])
    return out


def _faded(out: list, start: _LaneStart, point: np.ndarray) -> list:
    """Shift the walked points by ``point``'s offset from the lane, fading over ``LANE_FADE_M``."""
    here = np.array(shapely.LineString(start.lane.points).interpolate(start.at).coords[0])
    walked = cumulative_arc_length(np.vstack([here, out]))[1:]
    return list(
        np.asarray(out)
        + (np.asarray(point) - here) * np.maximum(0, 1 - walked / LANE_FADE_M)[:, None]
    )


def _straight_on(out: list, point: np.ndarray, heading: float, remaining: float) -> np.ndarray:
    """Return a point ``remaining`` metres on along the walk's last step (else ``heading``)."""
    end = _step_direction(np.vstack([point, out]), last=True)
    end = heading if end is None else end
    return out[-1] + heading_unit_vector(end) * remaining


# ---------------------------------------------------------------- lane chains crossing a path
def lane_crossings(
    lanes: dict, path: shapely.LineString, ego: Track | None = None
) -> Iterator[tuple[LaneCrossing | None, str | None]]:
    """Yield every lane chain crossing ``path`` ahead on the same level, once per crossing.

    Each item is ``(LaneCrossing, None)``, or ``(None, why)`` for a meeting point that is not a
    crossing: ``few_points``, ``same_crossing``, ``touches_only``, ``not_crossing`` or
    ``level_separated``. ``ego`` gives the ego's own level.
    """
    ids = list(lanes)
    lines = (ids, np.array([shapely.LineString(lanes[k].points) for k in ids], dtype=object))
    seen = set()
    for f, x in lane_meeting_points(lanes.values(), path):
        # The path's station and the lane chain's arc are both read at this crossing point.
        cross_station = path.project(x)
        pts, owner, chain = lane_chain(lanes, f.feature_id, shapely.LineString(f.points).project(x))
        if len(pts) < MIN_LINE_POINTS:
            yield None, "few_points"
            continue
        line = shapely.LineString(pts)
        lane_station = line.project(x)
        if (tuple(chain), round(lane_station)) in seen:
            yield None, "same_crossing"
            continue  # same crossing reached through another lane in the chain
        seen.add((tuple(chain), round(lane_station)))
        angle = crossing_angle(line, lane_station, path, cross_station)
        if angle is None:
            yield None, "touches_only"
            continue  # meets the path without crossing it
        if not CROSSING_ANGLES_DEG[0] <= angle <= CROSSING_ANGLES_DEG[1]:
            yield None, "not_crossing"
            continue  # merging into or out of the path, not crossing it
        gap = _level_gap(lanes, lines, f, x, path, cross_station, ego)
        if gap is not None and gap > LEVEL_GAP_M:
            yield None, "level_separated"
            continue
        yield (
            LaneCrossing(
                lane=f,
                point=x,
                cross_station=float(cross_station),
                chain=chain,
                points=pts,
                owner=owner,
                line=line,
                lane_station=float(lane_station),
                angle_deg=angle,
                level_gap_m=gap,
            ),
            None,
        )


def lane_meeting_points(
    lanes: Iterable[MapFeature], path: shapely.LineString
) -> Iterator[tuple[MapFeature, shapely.Point]]:
    """Yield ``(lane, point)`` where a lane meets ``path`` past ``LANE_MIN_AHEAD_M``.

    A point shared by two lanes (a map cut on the path) comes once, from the lane starting there.
    """
    lanes = list(lanes)
    ids = {f.feature_id for f in lanes}
    for f in lanes:
        line = shapely.LineString(f.points)
        points = []
        if line.intersects(path):
            hit = line.intersection(path)
            points = [g for g in getattr(hit, "geoms", [hit]) if g.geom_type == "Point"]
        start = shapely.Point(f.points[0])
        found = any(x.distance(start) <= JOINT_M for x in points)
        if path.distance(start) <= JOINT_M and not found:
            points.append(path.interpolate(path.project(start)))  # starts on the path (rounding)
        for x in points:
            if path.project(x) <= LANE_MIN_AHEAD_M:
                continue
            at_end = shapely.Point(f.points[-1]).distance(x) <= JOINT_M
            if at_end and any(e in ids for e in f.exit_lanes):
                continue  # the lane that starts at this point gives it
            yield f, x


def lane_chain(
    lanes: dict, lane_id: str, lane_station: float = 0.0
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(points [N, 2], owner lane per point, lane ids)`` for a chain through ``lane_id``.

    Extended through the straightest entry and exit lanes until ``LANE_UPSTREAM_M`` lies before
    the crossing at arc ``lane_station`` and ``LANE_DOWNSTREAM_M`` beyond the lane's end.
    """

    def length(ids: Iterable[str]) -> float:
        return sum(np.linalg.norm(np.diff(lanes[i].points, axis=0), axis=1).sum() for i in ids)

    chain = [lane_id]

    def short() -> bool:  # upstream lanes so far: chain[:-1]
        before = length(chain[:-1]) + lane_station  # arc before the crossing
        # Also while the first lane alone, or the upstream lanes alone, are shorter than the reach.
        return (
            before < LANE_UPSTREAM_M or max(length(chain[:1]), length(chain[:-1])) < LANE_UPSTREAM_M
        )

    while short() and len(chain) < CHAIN_UPSTREAM_LANES:
        f = lanes[chain[0]]
        opts = [
            e
            for e in f.entry_lanes
            if e in lanes and e not in chain and len(lanes[e].points) >= MIN_LINE_POINTS
        ]
        if not opts:
            break
        # Looking upstream from the joint: both lanes reversed.
        chain.insert(0, min(opts, key=lambda e: _joint_bend(f.points[::-1], lanes[e].points[::-1])))
    head = chain.index(lane_id)
    while (
        length(chain[head + 1 :]) < LANE_DOWNSTREAM_M
        and len(chain) - head - 1 < CHAIN_DOWNSTREAM_LANES
    ):
        f = lanes[chain[-1]]
        opts = [
            e
            for e in f.exit_lanes
            if e in lanes and e not in chain and len(lanes[e].points) >= MIN_LINE_POINTS
        ]
        if not opts:
            break
        chain.append(min(opts, key=lambda e: _joint_bend(f.points, lanes[e].points)))
    pts, owner = [], []
    for i in chain:
        p = lanes[i].points
        if pts and np.linalg.norm(p[0] - pts[-1]) < CHAIN_JOINT_M:
            p = p[1:]
        pts.extend(p)
        owner.extend([i] * len(p))
    return np.array(pts), owner, chain


def _joint_bend(points: np.ndarray, next_points: np.ndarray) -> float:
    """Return how far a lane starting where ``points`` ends bends away from it.

    For a lane ending where another starts, pass both reversed.
    """
    heading = _step_direction(points, last=True)
    return 0.0 if heading is None else _bend(shapely.LineString(next_points), 0.0, heading)


def crossing_angle(
    line: shapely.LineString, lane_station: float, path: shapely.LineString, station: float
) -> float | None:
    """Return the angle in degrees at which ``line`` crosses ``path``, or ``None`` if it does not.

    Measured over 1 m of each, around ``lane_station`` and ``station``.
    """
    if lane_station < 1 or lane_station + 1 > line.length:
        return None
    lane_before = np.array(line.interpolate(lane_station - 1).coords[0])
    lane_after = np.array(line.interpolate(lane_station + 1).coords[0])
    if np.sign(path_offset(path, lane_before)[1]) == np.sign(path_offset(path, lane_after)[1]):
        return None  # both ends on one side: the lane does not cross here
    path_here = np.array(path.interpolate(station).coords[0])
    path_ahead = np.array(path.interpolate(station + 1).coords[0])
    lane_step, path_step = lane_after - lane_before, path_ahead - path_here
    cross = lane_step[0] * path_step[1] - lane_step[1] * path_step[0]
    return abs(np.degrees(np.arctan2(cross, lane_step @ path_step)))


def path_offset(path: shapely.LineString, xy: ArrayLike) -> tuple[float, float]:
    """Return ``(station, lateral offset)`` of ``xy`` relative to ``path`` (left positive)."""
    station = path.project(shapely.Point(*xy))
    p = np.array(path.interpolate(station).coords[0])
    q = np.array(path.interpolate(min(station + OFFSET_CHORD_M, path.length)).coords[0])
    d = (q - p) / max(np.linalg.norm(q - p), MIN_DIVISOR)
    r = np.asarray(xy) - p
    return float(station), float(d[0] * r[1] - d[1] * r[0])


def _level_gap(
    lanes: dict,
    lines: tuple[list[str], np.ndarray],
    crossing: MapFeature,
    x: shapely.Point,
    path: shapely.LineString,
    station: float,
    ego: Track | None = None,
) -> float | None:
    """Return the elevation gap (m) between lane ``crossing`` at ``x`` and the ego's level.

    ``None`` means unknown (flat ground is assumed). The reference is the same-way lane nearest
    the ego's recorded level, not the smallest gap, so an overpass is not taken for the road.
    """
    z_x = lane_z_at(crossing, x)
    if z_x is None:
        return None
    # 1 m either side, as core.rules' speed-limit match
    heading = line_heading(path, station, back=1.0, ahead=1.0)
    ids, geoms = lines
    zs = []
    for k in np.flatnonzero(shapely.distance(geoms, x) <= LEVEL_LANE_M):
        f = lanes[ids[k]]
        if f is crossing:
            continue
        lane_heading = line_heading(geoms[k], geoms[k].project(x), back=1.0, ahead=1.0)
        if abs(wrap_angle(lane_heading - heading)) > LEVEL_LANE_ALIGN_RAD:
            continue
        z = lane_z_at(f, x)
        if z is not None:
            zs.append(z)
    own = ego_level(ego, x)
    if own is not None:
        ref = min(zs, key=lambda z: abs(z - own), default=own)
        return abs((ref if abs(ref - own) <= LEVEL_GAP_M / 2 else own) - z_x)
    if not zs or max(zs) - min(zs) > LEVEL_GAP_M:
        return None
    return abs(float(np.median(zs)) - z_x)


@dataclass(frozen=True, eq=False)
class LaneCrossing:
    """A lane chain (``lane_chain`` through ``lane``) crossing a path at ``point``.

    ``angle_deg`` is 45 to 135; ``level_gap_m`` is ``None`` when elevations are not recorded.
    """

    lane: object
    point: shapely.Point
    cross_station: float
    chain: list
    points: np.ndarray
    owner: list
    line: shapely.LineString
    lane_station: float
    angle_deg: float
    level_gap_m: float | None = None


def chain_speed_limit(lanes: dict[str, MapFeature], chain: list[str], lane_id: str) -> float | None:
    """Return the median posted limit (m/s) of ``chain`` up to ``lane_id``, or ``None``."""
    upto = chain[: chain.index(lane_id) + 1]
    limits = [lanes[i].speed_limit_mps for i in upto if lanes[i].speed_limit_mps]
    return float(np.median(limits)) if limits else None
