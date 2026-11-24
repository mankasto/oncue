"""Planar geometry: oriented boxes, signed clearance, rays and polylines. A box state is
``(x, y, heading, ...)`` and its size ``(length, width)``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike

Size = np.ndarray | tuple[float, float]  # (length, width)
CORNER_SIGNS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
BOX_CORNERS = 4  # a box has four corners; a shapely box polygon repeats the first
CAP_TOL_M = 1e-6  # tolerance on `box_clearance`'s cap
MIN_EDGE_SQUARED_M2 = 1e-18  # divisor floor for projection onto a zero-length edge
SWEPT_SUBSTEPS = 4  # samples per step of `swept_overlap`
THIN_STEP_M = 0.05  # `thin_path` keeps points at least this far apart
SAME_POINT_M = 1e-6  # points this close are treated as the same point
MIN_SEGMENT_SQUARED_M2 = 1e-12  # divisor floor for projection onto a zero-length ray
RAY_REACH_TOL_M = 1e-6  # tolerance when pruning rays by reach
SLAB_TOL = 1e-9  # slab-test tolerance, box-frame distances and ray fractions alike
# stations this close count as equal; a station this far outside a polyline still lies on it
STATION_TOL_M = 1e-6
MIN_DIVISOR = 1e-9  # divisor floor for a zero-length vector or segment, or a zero speed
ZERO_LENGTH_M = 1e-9  # a segment or step at most this long has no direction
MIN_LINE_POINTS = 2
MIN_POLYGON_POINTS = 3
# GEOS occasionally throws TopologyException on nearly coincident edges; retry on a snapping grid
UNION_FALLBACK_GRID_M = 1e-3


def wrap_angle(angle: ArrayLike) -> np.ndarray | float:
    """Wrap ``angle`` (rad) to ``(-pi, pi]``."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def union_all(geometries: Iterable[shapely.Geometry] | np.ndarray) -> shapely.Geometry:
    """Call ``shapely.union_all``, retrying on a snapping grid if GEOS fails exactly."""
    try:
        return shapely.union_all(geometries)
    except shapely.errors.GEOSException:
        return shapely.union_all(geometries, grid_size=UNION_FALLBACK_GRID_M)


def union(a: shapely.Geometry, b: shapely.Geometry) -> shapely.Geometry:
    """Call ``shapely.union``, retrying on a snapping grid if GEOS fails exactly."""
    try:
        return shapely.union(a, b)
    except shapely.errors.GEOSException:
        return shapely.union(a, b, grid_size=UNION_FALLBACK_GRID_M)


def lerp_states(a: np.ndarray, b: np.ndarray, fraction: np.ndarray | float) -> np.ndarray:
    """Interpolate states ``[..., >= 3]`` ``fraction`` of the way from ``a`` to ``b``.

    Every column is linear; the heading (column 2) goes the short way round. ``fraction``
    broadcasts against the leading axes.
    """
    fraction = np.asarray(fraction, float)
    states = a + fraction[..., None] * (b - a)
    states[..., 2] = a[..., 2] + fraction * wrap_angle(b[..., 2] - a[..., 2])
    return states


def heading_unit_vector(heading: ArrayLike) -> np.ndarray:
    """Return ``[..., 2]`` unit vector ``(cos, sin)`` along ``heading`` (rad)."""
    return np.stack((np.cos(heading), np.sin(heading)), axis=-1)


def scene_to_world(poses: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    """Map scene-frame poses ``[N, >= 3]`` to world ``(x, y, heading)``.

    ``origin`` is the scene frame's placement in the world (``Scene.world_origin``).
    """
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    x, y = poses[:, 0], poses[:, 1]
    return np.column_stack((ox + x * c - y * s, oy + x * s + y * c, wrap_angle(poses[:, 2] + oh)))


def rotate(vectors: np.ndarray, angle: ArrayLike) -> np.ndarray:
    """Rotate ``[..., 2]`` vectors counter-clockwise by ``angle`` (rad).

    To enter a frame of heading h, rotate ``points - origin`` by ``-h``; to leave it, rotate by
    ``h`` and add ``origin``.
    """
    c, s = np.cos(angle), np.sin(angle)
    x, y = vectors[..., 0], vectors[..., 1]
    return np.stack((x * c - y * s, x * s + y * c), axis=-1)


def world_to_scene(poses: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    """Inverse of ``scene_to_world``."""
    ox, oy, oh = origin
    c, s = np.cos(oh), np.sin(oh)
    xy = poses[:, :2] - (ox, oy)
    local = np.column_stack((xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c))
    return np.column_stack((local, wrap_angle(poses[:, 2] - oh)))


def box_axes(heading: ArrayLike) -> np.ndarray:
    """Return ``[..., 2, 2]`` whose rows are the forward and left unit vectors."""
    c, s = np.cos(heading), np.sin(heading)
    return np.stack((np.stack((c, s), -1), np.stack((-s, c), -1)), -2)


def box_corners(states: ArrayLike, size: Size) -> np.ndarray:
    """Return the four corners of each box ``states[..., >= 3]`` with the given ``size``."""
    states = np.asarray(states, float)
    offsets = CORNER_SIGNS * (np.asarray(size, float) / 2)
    return states[..., None, :2] + offsets @ box_axes(states[..., 2])


def _axis_gaps(a: np.ndarray, size_a: Size, b: np.ndarray, size_b: Size) -> np.ndarray:
    """Return the four separating-axis gaps between boxes ``a`` and ``b``; any ``> 0`` separates."""
    axes_a, axes_b = box_axes(a[..., 2]), box_axes(b[..., 2])
    directions = np.concatenate((axes_a, axes_b), axis=-2)  # [..., 4, 2]
    center_gap = np.abs(np.einsum("...i,...ki->...k", b[..., :2] - a[..., :2], directions))
    half_a, half_b = np.asarray(size_a, float) / 2, np.asarray(size_b, float) / 2
    reach_a = np.abs(np.einsum("...ki,...ji->...kj", directions, axes_a)) @ half_a
    reach_b = np.abs(np.einsum("...ki,...ji->...kj", directions, axes_b)) @ half_b
    return center_gap - reach_a - reach_b


def box_clearance(
    states_a: ArrayLike, size_a: Size, states_b: ArrayLike, size_b: Size, cap: float | None = None
) -> np.ndarray:
    """Return signed clearance between broadcast pairs of boxes; ``<= 0`` means contact.

    With ``cap``, pairs farther apart than ``cap`` along an axis return a lower bound instead.
    """
    a, b = np.broadcast_arrays(np.asarray(states_a, float), np.asarray(states_b, float))
    shape = a.shape[:-1]
    a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    result = _axis_gaps(a, size_a, b, size_b).max(axis=1)
    separated = result > 0
    if cap is not None:
        separated &= result < cap + CAP_TOL_M
    if separated.any():
        result[separated] = _corner_distance(
            box_corners(a[separated], size_a), box_corners(b[separated], size_b)
        )
    return result.reshape(shape)


def _corner_distance(ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Return the smallest corner-to-edge distance for pairs of separated quads."""

    def to_edges(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
        # x and y kept apart: equivalent to summing over a trailing length-2 axis
        p0x, p0y = corners[..., 0], corners[..., 1]  # [N, 4] edge starts
        dx, dy = np.roll(p0x, -1, axis=1) - p0x, np.roll(p0y, -1, axis=1) - p0y  # edges
        rx = points[:, :, None, 0] - p0x[:, None, :]  # [N, 4 points, 4 edges]
        ry = points[:, :, None, 1] - p0y[:, None, :]
        dx, dy = dx[:, None, :], dy[:, None, :]
        length2 = np.maximum(dx * dx + dy * dy, MIN_EDGE_SQUARED_M2)
        t = np.clip((rx * dx + ry * dy) / length2, 0.0, 1.0)
        nx, ny = rx - t * dx, ry - t * dy
        return np.sqrt(nx * nx + ny * ny)

    return np.minimum(to_edges(ca, cb).min(axis=(1, 2)), to_edges(cb, ca).min(axis=(1, 2)))


def boxes_overlap(
    states_a: ArrayLike, size_a: Size, states_b: ArrayLike, size_b: Size
) -> np.ndarray:
    """Return whether broadcast pairs of boxes touch (separating-axis test only)."""
    a, b = np.broadcast_arrays(np.asarray(states_a, float), np.asarray(states_b, float))
    return _axis_gaps(a, size_a, b, size_b).max(axis=-1) <= 0


def step_start_frames(frames: np.ndarray | int, anchor: int, *valid: np.ndarray) -> np.ndarray:
    """Return the frame each of ``frames`` steps from: the previous one if it is at or after
    ``anchor`` and valid, else the frame itself."""
    frames = np.asarray(frames)
    before = frames - 1
    stepped = before >= anchor
    for observed in valid:
        stepped = stepped & observed[before]
    return np.where(stepped, before, frames)


def swept_overlap(
    a_from: ArrayLike,
    a_to: ArrayLike,
    size_a: Size,
    b_from: ArrayLike,
    b_to: ArrayLike,
    size_b: Size,
    substeps: int = SWEPT_SUBSTEPS,
) -> np.ndarray:
    """Return whether two boxes touch at any sub-step from the previous frame (``*_from``) to
    this one (``*_to``, included)."""
    a_from, a_to, b_from, b_to = (np.asarray(x, float) for x in (a_from, a_to, b_from, b_to))
    reach = (np.hypot(*size_a) + np.hypot(*size_b)) / 2
    moved = np.linalg.norm(a_to[..., :2] - a_from[..., :2], axis=-1) + np.linalg.norm(
        b_to[..., :2] - b_from[..., :2], axis=-1
    )
    near = np.linalg.norm(a_to[..., :2] - b_to[..., :2], axis=-1) <= reach + moved
    out = np.zeros(near.shape, bool)
    if not near.any():
        return out
    a_from, a_to, b_from, b_to = a_from[near], a_to[near], b_from[near], b_to[near]
    t = (np.arange(1, substeps + 1) / substeps)[:, None]  # last sample hits this frame

    def at(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:  # [S, N, 3]
        return lerp_states(x0[None, :, :3], x1[None, :, :3], t)

    out[near] = boxes_overlap(at(a_from, a_to), size_a, at(b_from, b_to), size_b).any(axis=0)
    return out


def thin_path(points: ArrayLike, min_step: float = THIN_STEP_M) -> np.ndarray:
    """Return indices of ``points [N, 2]`` at least ``min_step`` apart, plus the last if it
    moved."""
    points = np.asarray(points, float)
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) >= min_step:
            keep.append(i)
    end = len(points) - 1
    if end > keep[-1] and np.linalg.norm(points[end] - points[keep[-1]]) > SAME_POINT_M:
        keep.append(end)
    return np.asarray(keep, int)


def ray_box_hits(
    origins: np.ndarray, targets: np.ndarray, boxes: np.ndarray, sizes: ArrayLike, valid: np.ndarray
) -> np.ndarray:
    """Return ``[T, M, P]`` bool: does each ray cross each valid box (slab test, grazing counts).

    Shapes: ``origins [T, 2]``, ``targets [T, P, 2]``, ``boxes [T, M, >= 3]``, ``sizes [M, 2]``,
    ``valid [T, M]``.
    """
    return ray_box_spans(origins, targets, boxes, sizes, valid)[0]


def ray_box_spans(
    origins: np.ndarray, targets: np.ndarray, boxes: np.ndarray, sizes: ArrayLike, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``ray_box_hits`` plus entry and exit fractions along each ray (0 where no hit)."""
    targets = np.asarray(targets, float)
    sizes = np.asarray(sizes, float).reshape(-1, 2)
    hits = np.zeros((targets.shape[0], boxes.shape[1], targets.shape[1]), bool)
    enter, leave = np.zeros(hits.shape), np.zeros(hits.shape)
    if boxes.shape[1] == 0:
        return hits, enter, leave
    center = targets.mean(axis=1)
    reach = np.linalg.norm(targets - center[:, None], axis=-1).max(axis=1)
    segment = center - origins
    relative = boxes[..., :2] - origins[:, None]
    along = (
        np.einsum("tmi,ti->tm", relative, segment)
        / np.maximum(np.einsum("ti,ti->t", segment, segment), MIN_SEGMENT_SQUARED_M2)[:, None]
    )
    distance = np.linalg.norm(
        relative - np.clip(along, 0, 1)[..., None] * segment[:, None], axis=-1
    )
    radius = np.linalg.norm(sizes, axis=1) / 2
    t, m = np.nonzero(valid & (distance <= radius[None] + reach[:, None] + RAY_REACH_TOL_M))
    if len(t):
        hits[t, m], enter[t, m], leave[t, m] = _slab_hits(
            origins[t], targets[t], boxes[t, m], sizes[m] / 2
        )
    return hits, np.where(hits, enter, 0.0), np.where(hits, leave, 0.0)


def _slab_hits(origins: np.ndarray, targets: np.ndarray, boxes: np.ndarray, half: np.ndarray):
    """Slab-test ``[K, P]`` rays against ``[K]`` boxes: hits and clipped entry/exit fractions."""
    axes = box_axes(boxes[:, 2])  # [K,2,2]
    start = np.einsum("kji,ki->kj", axes, origins - boxes[:, :2])[:, None, :]
    delta = np.einsum("kji,kpi->kpj", axes, targets - origins[:, None])
    half = half[:, None, :]
    moving = np.abs(delta) > SLAB_TOL
    safe = np.where(moving, delta, 1.0)
    t1, t2 = (-half - start) / safe, (half - start) / safe
    low = np.where(moving, np.minimum(t1, t2), -np.inf).max(-1)
    high = np.where(moving, np.maximum(t1, t2), np.inf).min(-1)
    parallel_outside = (~moving & (np.abs(start) > half + SLAB_TOL)).any(-1)
    hit = (high >= np.maximum(low, 0) - SLAB_TOL) & (low < 1 - SLAB_TOL) & ~parallel_outside
    return hit, np.clip(low, 0, 1), np.clip(high, 0, 1)


def distinct_points_mask(points: np.ndarray) -> np.ndarray:
    """Return ``[N]`` bool, ``False`` where a point is within ``SAME_POINT_M`` of the one before."""
    return np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > SAME_POINT_M]


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    """Return cumulative arc length ``[N]`` of a polyline ``points [N, D]`` from its first point."""
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def line_heading(
    line: shapely.LineString, station: ArrayLike, *, back: float, ahead: float
) -> np.ndarray | float:
    """Return the heading (rad) of the chord of ``line`` from ``station - back`` to
    ``station + ahead``, clipped onto the line."""
    stations = np.asarray(station, float)
    behind = shapely.line_interpolate_point(line, np.maximum(stations - back, 0.0))
    front = shapely.line_interpolate_point(line, np.minimum(stations + ahead, line.length))
    (x0, y0), (x1, y1) = shapely.get_coordinates(behind).T, shapely.get_coordinates(front).T
    heading = np.arctan2(y1 - y0, x1 - x0)
    return heading if stations.ndim else float(heading[0])


class Polyline:
    """Arc-length parameterized path with linear interpolation; no extrapolation."""

    def __init__(self, points: ArrayLike):
        points = np.asarray(points, float).reshape(-1, 2)
        self.points = points[distinct_points_mask(points)]
        if len(self.points) < MIN_LINE_POINTS:
            raise ValueError("Polyline needs two distinct points")
        self.stations = cumulative_arc_length(self.points)

    @property
    def length(self) -> float:
        return float(self.stations[-1])

    def position(self, station: ArrayLike) -> np.ndarray:
        """Return the point(s) at ``station``; raise if any station lies outside the polyline."""
        station = np.asarray(station, float)
        if (station < -STATION_TOL_M).any() or (station > self.length + STATION_TOL_M).any():
            raise ValueError("Station outside the polyline")
        return np.stack(
            [np.interp(station, self.stations, self.points[:, k]) for k in (0, 1)], axis=-1
        )
