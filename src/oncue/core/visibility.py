"""Geometric visibility of a target from the ego, and persistent reveal events.

Visibility is the share of five rays, from the ego centre to the target's centre and corners,
that no road user box (in 2.5D) or static wall crosses. Start from ``target_visibility``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..utils.geometry import (
    CORNER_SIGNS,
    MIN_LINE_POINTS,
    MIN_SEGMENT_SQUARED_M2,
    box_axes,
    box_corners,
    heading_unit_vector,
    ray_box_spans,
    union_all,
)
from ..utils.tracks import TIME_TOL_S
from .config import VisibilityConfig
from .road import DEFAULT_LANE_WIDTH_M, RoadEdges
from .types import STATE_SIZE, TABLE_NDIM, FeatureKind, RevealEvent, Scene, Track, height_of

if TYPE_CHECKING:  # for annotations only: no new import when running
    from numpy.typing import ArrayLike

MIN_AIM_HEIGHT_M = 0.3  # the sight line aims at least this high on the target
# Float slack when ``occluded_frame_count`` converts seconds to whole frames.
HIDDEN_FRAMES_SLACK = 1e-8

ROAD_CELL_M = 1.0  # cell size (m) for finding the road inside its edges
SIGHT_LINE_MARGIN_M = 0.2  # buffer (m) around a sight-line hull, whose edges the corner rays run


def target_visibility(
    scene: Scene, target_id: str, cfg: VisibilityConfig, static: shapely.Geometry | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(hits, visibility_sequence)`` for one target seen from its ego."""
    target = scene.tracks[target_id]
    hits = blocking(scene.ego, target, occluders(scene, target_id, cfg), cfg, static)
    return hits, visibility_from(hits, scene.ego, target, cfg)


def visibility_values(
    ego: Track,
    target: Track,
    blockers: list[Track],
    cfg: VisibilityConfig,
    static: shapely.Geometry | None = None,
) -> np.ndarray:
    """Return ``visibility_from(blocking(...), ego, target, cfg)`` without the hits array.

    Static occluders are tested only on known frames and on rays no road user blocks.
    """
    hits = blocking(ego, target, blockers, cfg)
    known = _known(ego, target, cfg)
    if static is not None:
        open_rays = known[:, None] & ~hits.any(axis=1)  # [T, 5]
        points = _ray_ends(target.states, target.size)
        hits = np.concatenate(
            (hits, _static_hits(ego.states[:, :2], points, static, open_rays)), axis=1
        )
    values = visible_fraction(hits)
    values[~known] = np.nan
    return values


def blocking(
    ego: Track,
    target: Track,
    blockers: list[Track],
    cfg: VisibilityConfig | None = None,
    static: shapely.Geometry | None = None,
) -> np.ndarray:
    """Return ``[T, M, 5]``: which of the five rays each blocker crosses at each frame.

    With ``static``, an extra column holds the rays the static occluders cross.
    """
    if blockers:
        boxes = np.stack([b.states for b in blockers], axis=1)
        sizes = np.array([b.size for b in blockers])
        valid = np.stack([b.valid for b in blockers], axis=1)
        heights = np.array([height_of(b) for b in blockers])
    else:
        boxes = np.zeros((len(target.states), 0, STATE_SIZE))
        sizes, valid, heights = np.zeros((0, 2)), np.zeros((len(target.states), 0), bool), None
    return blocking_arrays(
        ego.states[:, :2],
        target.states,
        target.size,
        height_of(target),
        boxes,
        sizes,
        valid,
        heights,
        cfg,
        static,
    )


def blocking_arrays(
    origins: np.ndarray,
    target_states: np.ndarray,
    target_size: ArrayLike,
    target_height: ArrayLike,
    boxes: np.ndarray,
    sizes: np.ndarray,
    valid: np.ndarray,
    heights: ArrayLike | None,
    cfg: VisibilityConfig | None,
    static: shapely.Geometry | None,
) -> np.ndarray:
    """Array-input version of ``blocking``, for closed-loop perception.

    Args:
        origins: sensor positions ``[T, 2]``.
        target_states: target states ``[T, 4]``.
        target_size: ``[2]`` or ``[T, 2]`` (one size per frame).
        target_height: scalar or ``[T]``.
        boxes: blocker states ``[T, M, 4]``.
        sizes: blocker sizes ``[M, 2]``.
        valid: blocker validity ``[T, M]``.
        heights: blocker heights ``[M]`` or ``None``.
    """
    points = _ray_ends(target_states, target_size)
    if boxes.shape[1]:
        hits, enter, leave = ray_box_spans(origins, points, boxes, sizes, valid)
        if cfg is not None and cfg.sensor_height_m is not None:
            hits &= _taller_than_sight(enter, leave, heights, target_height, cfg)
    else:
        hits = np.zeros((len(points), 0, points.shape[1]), bool)
    if static is not None:
        hits = np.concatenate((hits, _static_hits(origins, points, static)), axis=1)
    return hits


def _taller_than_sight(
    enter: np.ndarray,
    leave: np.ndarray,
    heights: ArrayLike,
    target_height: ArrayLike,
    cfg: VisibilityConfig,
) -> np.ndarray:
    """Return ``[T, M, 5]``: where each blocker reaches the sight line's lowest point over its box.

    The sight line is straight, so that point is at ``enter`` or ``leave``.
    """
    aim = np.maximum(np.asarray(target_height, float) - cfg.visible_margin_m, MIN_AIM_HEIGHT_M)
    aim = aim[:, None, None] if aim.ndim else aim
    lowest = cfg.sensor_height_m + np.where(aim < cfg.sensor_height_m, leave, enter) * (
        aim - cfg.sensor_height_m
    )
    return np.asarray(heights)[None, :, None] >= lowest


def _ray_ends(target_states: np.ndarray, target_size: ArrayLike) -> np.ndarray:
    """Return ``[T, 5, 2]`` endpoints of the five rays: the target centre and its box corners.

    ``target_size`` is ``[2]`` for a fixed size, or ``[T, 2]`` for one size per row.
    """
    if np.ndim(target_size) == TABLE_NDIM:  # a size per row [T, 2]
        offsets = CORNER_SIGNS[None] * (np.asarray(target_size, float)[:, None, :] / 2)
        corners = target_states[:, None, :2] + offsets @ box_axes(target_states[:, 2])
    else:
        corners = box_corners(target_states, target_size)
    return np.concatenate((target_states[:, None, :2], corners), axis=1)


def _static_hits(
    origins: np.ndarray,
    points: np.ndarray,
    static: shapely.Geometry,
    rays: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``[T, 1, 5]``: whether the static occluders cross each ray.

    Only the rays marked in ``rays [T, 5]`` are tested, when given.
    """
    out = np.zeros((points.shape[0], 1, points.shape[1]), bool)
    if static.is_empty:
        return out
    frame, ray = np.nonzero(np.ones(points.shape[:2], bool) if rays is None else rays)
    if not len(frame):
        return out
    segments = shapely.linestrings(np.stack((origins[frame], points[frame, ray]), axis=1))
    out[frame, 0, ray] = shapely.intersects(segments, static)
    return out


def occluders(scene: Scene, target_id: str, cfg: VisibilityConfig) -> list[Track]:
    """Return the scene's tracks that can occlude the given target."""
    return [t for t in scene.others(target_id) if t.agent_type in cfg.occluder_types]


def visibility_from(
    hits: np.ndarray, ego: Track, target: Track, cfg: VisibilityConfig | None = None
) -> np.ndarray:
    """Return the per-frame visible fraction ``[T]``, NaN where ego or target is unobserved.

    With ``cfg``, frames where the target is beyond the perception range are NaN too.
    """
    values = visible_fraction(hits)
    values[~_known(ego, target, cfg)] = np.nan
    return values


def _known(ego: Track, target: Track, cfg: VisibilityConfig | None) -> np.ndarray:
    """Return ``[T]`` marking frames where the visibility value is defined."""
    known = ego.valid & target.valid
    if cfg is not None:
        known &= in_range(ego.states, target.states, cfg.perception_range_m)
    return known


def in_range(eyes: ArrayLike, points: ArrayLike, range_m: float) -> np.ndarray:
    """Return whether each point is strictly within ``range_m`` of its eye."""
    gap = np.asarray(points, float)[..., :2] - np.asarray(eyes, float)[..., :2]
    return np.linalg.norm(gap, axis=-1) < range_m


def visible_fraction(hits: np.ndarray) -> np.ndarray:
    """Return the fraction of the five rays no occluder crosses from ``blocking``'s output."""
    return (~hits.any(axis=1)).mean(axis=1)


def reveal_events(
    scene: Scene, target_id: str, cfg: VisibilityConfig, static: shapely.Geometry | None = None
) -> tuple[np.ndarray, list[RevealEvent]]:
    """Return ``(visibility_sequence, reveal_events)`` for one target seen from its ego."""
    _, values = target_visibility(scene, target_id, cfg, static)
    return values, find_reveals(values, cfg, scene.dt)


def visibility(
    ego: Track,
    target: Track,
    blockers: list[Track],
    cfg: VisibilityConfig | None = None,
    static: shapely.Geometry | None = None,
) -> np.ndarray:
    """Return a target's visibility from any viewer, with no perception-range filter."""
    return visibility_from(blocking(ego, target, blockers, cfg, static), ego, target)


def find_reveals(values: np.ndarray, cfg: VisibilityConfig, dt: float) -> list[RevealEvent]:
    """Scan a visibility sequence for occluded-then-persistently-visible events."""
    min_hidden = occluded_frame_count(cfg.min_occluded_s, dt)
    events: list[RevealEvent] = []
    hidden = visible = 0
    start = onset = None
    armed = False
    for i, v in enumerate(values):
        if not np.isfinite(v):
            hidden = visible = 0
            armed = False
        elif v < cfg.threshold:
            if hidden == 0:
                start = i
            hidden += 1
            visible = 0
            armed = hidden >= min_hidden
        else:
            if visible == 0:
                onset = i
            visible += 1
            hidden = 0
            if armed and visible == cfg.min_visible_frames:
                events.append(RevealEvent(start, onset, i))
                armed = False
    return events


def occluded_frame_count(seconds: float, dt: float) -> int:
    """Return the number of hidden samples needed to cover ``seconds`` (``n * dt >= seconds``)."""
    return max(1, int(np.ceil(seconds / dt - HIDDEN_FRAMES_SLACK)))


# ---------------------------------------------------------------- static occluders
def static_region(scene: Scene, cfg: VisibilityConfig) -> shapely.Geometry | None:
    """Return the scene's static occluders, or None when ``cfg`` disables them.

    That is the ``walls`` minus the ``sight_lines`` the logged ego saw through; not cached.
    """
    if cfg.static_setback_m is None:
        return None
    region = walls(scene, cfg.static_setback_m)
    seen_through = list(sight_lines(scene, cfg, region).values())
    if seen_through:
        region = _cut_out(region, union_all(seen_through))
    shapely.prepare(region)
    return region


def sight_lines(
    scene: Scene, cfg: VisibilityConfig, near: shapely.Geometry | None = None
) -> dict[str, shapely.Geometry]:
    """Return, per road user, the region where the logged ego saw it.

    The hulls of the ego centre and the road user's box over its tracked frames within range;
    with ``near``, only hulls meeting it.
    """
    ego, out = scene.ego, {}
    for track in scene.others():
        seen = tracked_frames(track) & ego.valid
        seen &= in_range(ego.states, track.states, cfg.perception_range_m)
        frames = np.flatnonzero(seen)
        if not len(frames):
            continue
        corners = box_corners(track.states[frames], track.size)
        hulls = shapely.convex_hull(
            shapely.multipoints(np.concatenate((ego.states[frames, None, :2], corners), axis=1))
        )
        keep = shapely.get_type_id(hulls) == shapely.GeometryType.POLYGON  # not degenerate
        if near is not None:
            keep &= shapely.intersects(near, hulls)
        if keep.any():
            out[track.track_id] = union_all(hulls[keep]).buffer(SIGHT_LINE_MARGIN_M)
    return out


def tracked_frames(track: Track) -> np.ndarray:
    """Return ``[T]`` marking the frames WOMD tracked the road user: valid and not filled in."""
    return track.valid if track.filled is None else track.valid & ~track.filled


def walls(scene: Scene, setback_m: float) -> shapely.Geometry:
    """Return the static occluder region, with the road and logged road users as open space.

    No wall stands within ``setback_m`` of the road, a lane or a logged road user's box.
    """
    from ..data.map_geometry import static_occluder_region  # here: data.map_geometry imports core

    region = static_occluder_region(scene, setback_m)
    lanes = [
        shapely.LineString(np.asarray(f.points, float)[:, :2])
        for f in scene.map_features
        if f.kind is FeatureKind.LANE and len(f.points) >= MIN_LINE_POINTS
    ]
    open_space = union_all(
        [
            road_inside_edges(scene, region),
            *shapely.buffer(lanes, DEFAULT_LANE_WIDTH_M / 2),
            *logged_boxes(scene),
        ]
    )
    return region if open_space.is_empty else _cut_out(region, open_space.buffer(setback_m))


def logged_boxes(scene: Scene) -> np.ndarray:
    """Return every road user's box (the ego's included) at every frame WOMD tracked it."""
    boxes = [
        shapely.polygons(box_corners(t.states[tracked_frames(t)], t.size))
        for t in scene.tracks.values()
    ]
    return np.concatenate(boxes) if boxes else np.zeros(0, object)


def road_inside_edges(scene: Scene, within: shapely.Geometry) -> shapely.Geometry:
    """Return the road inside its edges within ``within``, as ``ROAD_CELL_M`` grid cells.

    A cell is road when ``core.road.RoadEdges`` puts its centre on the road.
    """
    if within.is_empty:
        return shapely.Polygon()
    low_x, low_y, high_x, high_y = within.bounds
    xs = np.arange(low_x + ROAD_CELL_M / 2, high_x, ROAD_CELL_M)
    ys = np.arange(low_y + ROAD_CELL_M / 2, high_y, ROAD_CELL_M)
    grid_x, grid_y = np.meshgrid(xs, ys)
    looked_at = shapely.contains_xy(within, grid_x, grid_y)
    if not looked_at.any():
        return shapely.Polygon()
    centres = np.stack((grid_x[looked_at], grid_y[looked_at]), axis=1)
    on_road = np.zeros(grid_x.shape, bool)
    with np.errstate(invalid="ignore"):  # NaN depth means unknown, which counts as off-road
        on_road[looked_at] = RoadEdges(scene).point_depth(centres) <= 0
    return union_all(_row_runs(on_road, xs, ys))


def _row_runs(cells: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Return a box per run of ``True`` cells along each row of ``cells [len(ys), len(xs)]``."""
    steps = np.diff(np.pad(cells, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(steps == 1)  # row-major: the k-th end closes the k-th start
    _, ends = np.nonzero(steps == -1)
    half = ROAD_CELL_M / 2
    return shapely.box(xs[starts] - half, ys[rows] - half, xs[ends - 1] + half, ys[rows] + half)


def _cut_out(region: shapely.Geometry, cut: shapely.Geometry) -> shapely.Geometry:
    """Return ``region`` minus ``cut`` on the static occluder grid, keeping only polygons."""
    from ..data.map_geometry import grid_difference  # here: data.map_geometry imports core

    parts = shapely.get_parts(grid_difference(region, cut))
    return shapely.multipolygons(parts[shapely.get_type_id(parts) == shapely.GeometryType.POLYGON])


# ---------------------------------------------------------------- unobserved occluders
def could_block_unobserved(
    blocker: Track,
    frames: np.ndarray,
    origins: np.ndarray,
    points: np.ndarray,
    reach: float,
    times: np.ndarray,
    cfg: VisibilityConfig,
) -> np.ndarray:
    """Return at which of ``frames`` an unobserved ``blocker`` could still block the view.

    That is in a tracking gap, or within ``occluder_linger_s`` of the first or last observation,
    unless extrapolating from an observed side with ``occluder_max_accel`` keeps it out of view.
    """
    frames = np.asarray(frames)
    result = np.zeros(len(frames), bool)
    seen = np.flatnonzero(blocker.valid)
    if not len(seen):
        return result
    unseen = ~blocker.valid[frames]
    if not unseen.any():
        return result
    f = frames[unseen]
    origins, points = np.asarray(origins, float)[unseen], np.asarray(points, float)[unseen]
    segment = points - origins
    after = np.searchsorted(seen, f)  # first observation after each frame
    edge = (after == 0) | (after == len(seen))
    lone = seen[np.clip(after - (after == len(seen)), 0, len(seen) - 1)]
    possible = ~edge | (np.abs(times[f] - times[lone]) <= cfg.occluder_linger_s + TIME_TOL_S)
    for side, has in ((after - 1, after > 0), (after, after < len(seen))):
        if not has.any():
            continue
        k = np.flatnonzero(has)
        observed = seen[side[k]]
        anchor = blocker.states[observed]
        dt = times[f[k]] - times[observed]  # negative when extrapolating backward
        velocity = anchor[:, 3:4] * heading_unit_vector(anchor[:, 2])
        if blocker.velocity is not None:  # recorded velocity may differ from speed * heading
            recorded = blocker.velocity[observed]
            known = np.isfinite(recorded).all(axis=1)
            velocity = np.where(known[:, None], recorded, velocity)
        rel = anchor[:, :2] + velocity * dt[:, None] - origins[k]
        seg = segment[k]
        u = np.einsum("ki,ki->k", rel, seg) / np.maximum(
            (seg**2).sum(axis=1), MIN_SEGMENT_SQUARED_M2
        )
        clearance = np.linalg.norm(rel - np.clip(u, 0, 1)[:, None] * seg, axis=1)
        clearance -= np.linalg.norm(blocker.size) / 2 + reach
        possible[k] &= clearance <= cfg.occluder_max_accel * dt**2 / 2
    result[unseen] = possible
    return result
