"""Where a box sits on the road: against traffic, spanning lanes, or off-road; plus lane lookup.

Off-road follows Waymax: a box corner right of its nearest road edge is off the road. Outside
the map's extent the answer is unknown, not off-road.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..utils.geometry import (
    MIN_LINE_POINTS,
    ZERO_LENGTH_M,
    box_corners,
    heading_unit_vector,
    wrap_angle,
)
from .types import FeatureKind, LaneType, Scene

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .types import MapFeature

# Lanes this far off the heading run the other way (cyclists included).
OPPOSING_DEG = 159.6
OPPOSING_RADIUS_M = 1.8  # direction is read from the lanes this close to the sample
# A same-way lane this much farther out still counts.
WRONG_WAY_TOL_M = 0.412
# WOMD lanes carry no width of their own.
DEFAULT_LANE_WIDTH_M = 3.6
IN_LANE_TOL_M = 0.3
SAME_WAY_DEG = 45.0  # a lane running the box's way lies within this many degrees
LANE_HALF_WIDTH_M = DEFAULT_LANE_WIDTH_M / 2  # half-width used when no lane width is given
LANE_CHORD_M = 0.5  # lane direction is read over this chord before and after the point
OFFROAD_TOL_M = 0.34  # a corner this far past a road edge still counts as on the road
MAP_MARGIN_M = 5.0  # the map's extent reaches this far past its features
# Shorter edges are dropped: driveway cuts and fragments whose sides are not the road's.
MIN_EDGE_M = 2.0
# An edge is cut out this near a vehicle-lane crossing, where the 2D map draws an overpass.
CROSSED_EDGE_M = DEFAULT_LANE_WIDTH_M


class LaneDirections:
    """Lane centreline segments of a scene, keyed for against-traffic checks.

    Segments, not vertices: the middle of a long, sparsely sampled segment still counts.
    """

    def __init__(self, scene: Scene):
        segments, headings = [], []
        for lane in scene.map_features:
            if lane.kind is FeatureKind.LANE and len(lane.points) >= MIN_LINE_POINTS:
                d = np.diff(lane.points, axis=0)
                keep = np.hypot(d[:, 0], d[:, 1]) > ZERO_LENGTH_M
                ends = np.stack([lane.points[:-1], lane.points[1:]], axis=1)[keep]
                segments.append(ends)
                headings.append(np.arctan2(d[keep, 1], d[keep, 0]))
        segments = np.concatenate(segments) if segments else np.empty((0, 2, 2))
        self.headings = np.concatenate(headings) if headings else np.empty(0)
        self.tree = shapely.STRtree(shapely.linestrings(segments)) if len(segments) else None

    def opposing(self, states: np.ndarray) -> np.ndarray:
        """Return ``[R]``: does any sample of ``states [R, S, >= 3]`` drive against traffic?

        It does when an opposing lane lies within ``OPPOSING_RADIUS_M`` and no same-way lane
        within ``OPPOSING_RADIUS_M + WRONG_WAY_TOL_M``.
        """
        if self.tree is None:
            return np.zeros(len(states), bool)
        flat = states.reshape(-1, states.shape[-1])
        points = shapely.points(flat[:, :2])
        sample, lane = self.tree.query(
            points, predicate="dwithin", distance=OPPOSING_RADIUS_M + WRONG_WAY_TOL_M
        )
        against = (
            np.abs(np.degrees(wrap_angle(self.headings[lane] - flat[sample, 2]))) > OPPOSING_DEG
        )
        close = shapely.distance(points[sample], self.tree.geometries[lane]) <= OPPOSING_RADIUS_M
        against_close = np.bincount(sample, weights=against & close, minlength=len(flat)) > 0
        with_traffic = np.bincount(sample, weights=~against, minlength=len(flat)) > 0
        return (against_close & ~with_traffic).reshape(states.shape[:2]).any(axis=1)

    def spans_lanes(self, states: np.ndarray, width: float) -> np.ndarray:
        """Return ``[N]``: does each box of ``width`` straddle two same-way lanes?

        It does not when a same-way centreline lies within
        ``(DEFAULT_LANE_WIDTH_M - width) / 2 + IN_LANE_TOL_M`` of its centre, or none is nearby.
        """
        states = np.atleast_2d(states)
        if self.tree is None:
            return np.zeros(len(states), bool)
        reach = max((DEFAULT_LANE_WIDTH_M - width) / 2, 0.0) + IN_LANE_TOL_M
        points = shapely.points(states[:, :2])
        sample, lane = self.tree.query(points, predicate="dwithin", distance=DEFAULT_LANE_WIDTH_M)
        turn = np.degrees(wrap_angle(self.headings[lane] - states[sample, 2]))
        same = np.abs(turn) < SAME_WAY_DEG
        sample, lane = sample[same], lane[same]
        near = shapely.distance(points[sample], self.tree.geometries[lane]) <= reach
        has_lane = np.bincount(sample, minlength=len(states)) > 0
        inside = np.bincount(sample, weights=near, minlength=len(states)) > 0
        return has_lane & ~inside


class RoadEdges:
    """Oriented WOMD road edges (road on the left), cut into pieces at least ``MIN_EDGE_M`` long.

    A corner whose nearest point lies past a piece's open end counts as on the road.
    """

    def __init__(self, scene: Scene):
        lanes = _vehicle_lane_tree(scene)
        starts, ends, points, first, last = [], [], [], [], []
        for f in scene.map_features:
            if len(f.points):
                points.append(np.asarray(f.points, float)[:, :2])
            if f.kind is not FeatureKind.ROAD_EDGE or len(f.points) < MIN_LINE_POINTS:
                continue
            for piece_starts, piece_ends in edge_pieces(np.asarray(f.points, float)[:, :2], lanes):
                starts.append(piece_starts)
                ends.append(piece_ends)
                n = len(piece_starts)
                first.append(np.arange(n) == 0)  # a piece's open ends
                last.append(np.arange(n) == n - 1)
        self.starts = np.concatenate(starts) if starts else np.empty((0, 2))
        self.ends = np.concatenate(ends) if ends else np.empty((0, 2))
        self.first = np.concatenate(first) if first else np.empty(0, bool)  # each piece's first ...
        self.last = np.concatenate(last) if last else np.empty(0, bool)  # ... and last segment
        self.tree = (
            shapely.STRtree(shapely.linestrings(np.stack([self.starts, self.ends], axis=1)))
            if len(self.starts)
            else None
        )
        every = np.concatenate(points) if points else np.empty((0, 2))
        self.low = every.min(axis=0) - MAP_MARGIN_M if len(every) else None
        self.high = every.max(axis=0) + MAP_MARGIN_M if len(every) else None

    def depth(self, states: np.ndarray, size: ArrayLike) -> np.ndarray:
        """Return how far each box reaches past the road: <= 0 on the road, NaN where unknown."""
        states = np.asarray(states, float)
        corners = box_corners(states[..., :3], size).reshape(-1, 2)
        return self.point_depth(corners).reshape(*states.shape[:-1], 4).max(axis=-1)

    def point_depth(self, xy: np.ndarray) -> np.ndarray:
        """Return ``depth`` for single points ``xy [N, 2]``."""
        corners = np.asarray(xy, float)
        out = np.full(len(corners), np.nan)
        if self.tree is None:
            return out
        known = np.all((corners >= self.low) & (corners <= self.high), axis=1)
        index = np.flatnonzero(known)
        points = shapely.points(corners[index])
        # At a vertex both segments are nearest; the road side of either counts as on the road.
        which, nearest = self.tree.query_nearest(points, all_matches=True)
        a, b, p = self.starts[nearest], self.ends[nearest], corners[index[which]]
        side = (b - a)[:, 0] * (p - a)[:, 1] - (b - a)[:, 1] * (p - a)[:, 0]  # > 0: left, on-road
        along = np.einsum("ki,ki->k", p - a, b - a) / np.einsum("ki,ki->k", b - a, b - a)
        past_end = (self.first[nearest] & (along < 0)) | (self.last[nearest] & (along > 1))
        gap = shapely.distance(points[which], self.tree.geometries[nearest])
        on_road = np.zeros(len(index), bool)
        np.logical_or.at(on_road, which, (side >= 0) | past_end)
        nearest_gap = np.zeros(len(index))
        nearest_gap[which] = gap
        out[index] = np.where(on_road, -nearest_gap, nearest_gap)
        return out

    def offroad(
        self, states: np.ndarray, size: ArrayLike, tol_m: float = OFFROAD_TOL_M
    ) -> np.ndarray:
        """Return whether each box is off the road by more than ``tol_m``; unknown counts as not."""
        with np.errstate(invalid="ignore"):
            return self.depth(states, size) > tol_m


class VehicleLanes:
    """Vehicle lanes of a scene with their posted limits (NaN where none is recorded)."""

    def __init__(self, scene: Scene):
        lanes = vehicle_lane_features(scene)
        self.lines = np.array([shapely.LineString(f.points[:, :2]) for f in lanes], dtype=object)
        self.limits = np.array(
            [np.nan if f.speed_limit_mps is None else f.speed_limit_mps for f in lanes], float
        )
        self.tree = shapely.STRtree(self.lines) if len(lanes) else None

    def inside(self, xy: np.ndarray) -> np.ndarray:
        """Return ``[T]``: is each point within ``LANE_HALF_WIDTH_M`` of a lane?"""
        out = np.zeros(len(xy), bool)
        if self.tree is not None and len(xy):
            hit, _ = self.tree.query(
                shapely.points(xy), predicate="dwithin", distance=LANE_HALF_WIDTH_M
            )
            out[hit] = True
        return out

    def nearest(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the nearest lane and its direction at each of ``xy [T, 2]``.

        Index -1 and direction NaN where no lane lies within ``LANE_HALF_WIDTH_M``.
        """
        which = np.full(len(xy), -1)
        direction = np.full((len(xy), 2), np.nan)
        if self.tree is None or not len(xy):
            return which, direction
        points = shapely.points(xy)
        point, lane = self.tree.query_nearest(points, max_distance=LANE_HALF_WIDTH_M)
        which[point] = lane
        found = which >= 0
        if found.any():
            lines = self.lines[which[found]]
            at = shapely.line_locate_point(lines, points[found])
            ahead = shapely.line_interpolate_point(lines, at + LANE_CHORD_M)
            behind = shapely.line_interpolate_point(lines, np.maximum(at - LANE_CHORD_M, 0.0))
            chord = shapely.get_coordinates(ahead) - shapely.get_coordinates(behind)
            length = np.linalg.norm(chord, axis=1, keepdims=True)
            direction[found] = np.where(length > 0, chord / np.maximum(length, 1e-12), np.nan)
        return which, direction

    def same_way_lanes(self, states: np.ndarray) -> np.ndarray:
        """Return ``[T]`` the nearest lane within ``SAME_WAY_DEG`` of each heading, else -1."""
        which, direction = self.nearest(states[:, :2])
        facing = np.einsum("ti,ti->t", heading_unit_vector(states[:, 2]), direction)
        return np.where(facing >= np.cos(np.radians(SAME_WAY_DEG)), which, -1)

    def limits_of(self, lanes: np.ndarray) -> np.ndarray:
        """Return each lane's posted limit (m/s), NaN for no limit or index -1."""
        out = np.full(len(lanes), np.nan)
        found = lanes >= 0
        out[found] = self.limits[lanes[found]]
        return out


def vehicle_lane_features(scene: Scene) -> list[MapFeature]:
    """Return the scene's vehicle lanes, leaving out bike lanes (they may leave the road)."""
    return [
        f
        for f in scene.map_features
        if f.kind is FeatureKind.LANE
        and f.lane_type is not LaneType.BIKE_LANE
        and len(f.points) >= MIN_LINE_POINTS
    ]


def _vehicle_lane_tree(scene: Scene) -> shapely.STRtree | None:
    """Return an STRtree of the scene's vehicle lane centrelines, or ``None`` if none exist."""
    lines = [shapely.LineString(f.points) for f in vehicle_lane_features(scene)]
    return shapely.STRtree(lines) if lines else None


def edge_pieces(
    xy: np.ndarray, lanes: shapely.STRtree | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(starts, ends)`` of one road edge's pieces, cut out at vehicle-lane crossings."""
    lengths = np.hypot(*np.diff(xy, axis=0).T)
    keep = lengths > ZERO_LENGTH_M
    starts, ends, lengths = xy[:-1][keep], xy[1:][keep], lengths[keep]
    if not len(starts):
        return []
    crossed = np.zeros(len(starts), bool)
    edge = shapely.LineString(xy)
    crossing = [] if lanes is None else lanes.query(edge, predicate="crosses")
    if len(crossing):
        meets = shapely.get_parts(shapely.intersection(edge, lanes.geometries[crossing]))
        segments = shapely.linestrings(np.stack([starts, ends], axis=1))
        crossed = shapely.dwithin(segments[:, None], meets[None, :], CROSSED_EDGE_M).any(axis=1)
    kept = np.flatnonzero(~crossed)
    runs = np.split(kept, np.flatnonzero(np.diff(kept) > 1) + 1) if len(kept) else []
    return [(starts[run], ends[run]) for run in runs if lengths[run].sum() >= MIN_EDGE_M]
