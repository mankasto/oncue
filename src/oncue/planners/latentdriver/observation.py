"""LatentDriver's Waymax observation (commit 721bd6e), rebuilt in numpy from an OnCue scene.

Tokens are ``[type, x, y, width, length, yaw (deg), last]``: 20 route boxes, 128 road-user slots
and 40 road-edge boxes, in WOMD world coordinates; zero rows are padding.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ...core.config import DriverConfig
from ...core.rules import driver_path
from ...core.types import FeatureKind, RoadEdgeType, Scene
from ...utils.geometry import scene_to_world

if TYPE_CHECKING:
    from ...core.types import Track

MAX_ROUTE_SEGMENTS = 20  # configs/simulate.yaml data_conf.max_route_segments
MAX_MAP_SEGMENTS = 1000  # data_conf.max_map_segments
MAX_ROADGRAPH_SEGMENTS = 40  # data_conf.max_roadgraph_segments
MAX_OBJECTS = 128  # waymax_conf.max_num_objects
FIRST_OBJECT_ROW = MAX_ROUTE_SEGMENTS  # row layout: route, road users, road edges
FOV_M = (80.0, 20.0)  # data_conf.FOV: window around the ego, length x width
RDP_EPSILON_M = 0.5  # ``rdp_downsample_route``'s ``crdp.rdp(route, 0.5)``
MAX_LENGTH_PER_SEGMENT_M = 10.0  # ``rdp_downsample_route``'s ``max_length_per_seg``
ROAD_EDGE_WIDTH_M = 0.5  # ``get_roadgraph_obs_``'s box width
ROUTE_TYPE, OBJECT_TYPE, ROAD_TYPE, EGO_TYPE = 1.0, 2.0, 3.0, 4.0
ROAD_EDGE_TYPES = (RoadEdgeType.BOUNDARY, RoadEdgeType.MEDIAN)  # ``MapElementIds``' two road edges
MIN_ELEMENT_POINTS = 3  # ``get_roadgraph_obs_``: ``len(road_ob) > 2``


def scene_boxes(
    scene: Scene, driver: DriverConfig | None = None, log_route: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(route [20, 6], road edges [1000, 6])`` boxes in world coordinates.

    The route is ``route_points`` (``driver`` defaults to ``DriverConfig()``), or upstream's
    ``log_route_points`` when ``log_route`` is true.
    """
    origin = scene.world_origin
    ego = scene.ego
    points = log_route_points(scene) if log_route else route_points(scene, driver or DriverConfig())
    path = scene_to_world(np.column_stack((points, np.zeros(len(points)))), origin)[:, :2]
    route = _padded(boxes(path, float(ego.size[1])), MAX_ROUTE_SEGMENTS)
    features = sorted(
        (f for f in scene.map_features if len(f.points)), key=lambda f: int(f.feature_id)
    )
    last_id = features[-1].feature_id if features else None  # the original's loop never reaches it
    road = []
    for f in features:
        if (
            f.feature_id == last_id
            or f.kind is not FeatureKind.ROAD_EDGE
            or f.edge_type not in ROAD_EDGE_TYPES
            or len(f.points) < MIN_ELEMENT_POINTS
        ):
            continue
        points = np.column_stack((f.points, np.zeros(len(f.points))))
        road.extend(boxes(scene_to_world(points, origin)[:, :2], ROAD_EDGE_WIDTH_M))
    return route, _padded(road, MAX_MAP_SEGMENTS)


def route_points(scene: Scene, driver: DriverConfig) -> np.ndarray:
    """Return ``[N, 2]`` the route LatentDriver is given: its logged past, then ``driver_path``.

    Upstream's ``get_route_global`` gives the logged path to the log's end instead, which tells
    the model where the log stops.
    """
    ego, anchor = scene.ego, scene.anchor
    history = ego.states[:anchor][ego.valid[:anchor], :2]
    return np.vstack([history, np.asarray(driver_path(scene, driver).coords)])


def log_route_points(scene: Scene) -> np.ndarray:
    """Return ``[N, 2]`` upstream's route (``get_route_global``): the logged ego path to its end.

    Only for the comparison that shows what the log's end leaks, never for the main run.
    """
    return scene.ego.states[scene.ego.valid, :2]


def boxes(points: np.ndarray, width: float) -> list[list]:
    """Convert a polyline into one box per RDP segment, as ``rdp_downsample_route`` does.

    Each box is ``[x, y, width, length, yaw (deg), index]`` at the segment's middle.
    """
    route = np.round(np.asarray(points, np.float32), 2)
    shortened = route[rdp_keep(route, RDP_EPSILON_M)]
    vectors = shortened[1:] - shortened[:-1]
    midpoints = shortened[:-1] + vectors / 2.0
    norms = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    out = []
    for i, midpoint in enumerate(midpoints):
        # Yaw in double, as the original's NumPy 1.x promoted a float32 scalar times an int.
        box = [midpoint[0], midpoint[1], width, norms[i], float(angles[i]) * 180 / np.pi, i]
        out.extend(_split(box, i) if norms[i] > MAX_LENGTH_PER_SEGMENT_M else [box])
    return out


def rdp_keep(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Return which points Ramer-Douglas-Peucker keeps, matching ``crdp.rdp`` exactly."""
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    keep = np.ones(len(points), bool)
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        inner = np.arange(start + 1, end)
        chord = math.sqrt((y[start] - y[end]) ** 2 + (x[start] - x[end]) ** 2)
        if chord:
            p0, p1 = y[start] - y[end], x[start] - x[end]
            p2 = x[start] * y[end] - y[start] * x[end]
            distance = np.abs(p0 * x[inner] - p1 * y[inner] + p2) / chord
        else:
            distance = np.sqrt((y[start] - y[inner]) ** 2 + (x[start] - x[inner]) ** 2)
        farthest = int(np.argmax(distance))
        if distance[farthest] > epsilon:
            stack.append((start, inner[farthest]))
            stack.append((inner[farthest], end))
        else:
            keep[inner] = False
    return keep


def _split(box: list, start_id: int) -> list[list]:
    """Split a long box into equal parts numbered from ``start_id`` (``split_large_BB``)."""
    x, y, width, length, yaw = (float(v) for v in box[:5])
    half = length / 2
    angle = math.radians(yaw)
    x0, x1 = x - half * math.cos(angle), x + half * math.cos(angle)
    y0, y1 = y - half * math.sin(angle), y + half * math.sin(angle)
    inner = math.ceil(half * 2 / MAX_LENGTH_PER_SEGMENT_M) - 1  # points between the two ends
    xs, ys = np.linspace(x0, x1, inner + 2), np.linspace(y0, y1, inner + 2)
    return [
        [
            (xs[i] + xs[i + 1]) / 2,
            (ys[i] + ys[i + 1]) / 2,
            width,
            length / (inner + 1),
            yaw,
            float(start_id + i),
        ]
        for i in range(len(xs) - 1)
    ]


def _padded(rows: list, count: int) -> np.ndarray:
    """Return the first ``count`` rows as ``[count, 6]``, padded with zeros."""
    table = np.zeros((count, 6))
    rows = rows[:count]
    if rows:
        table[: len(rows)] = np.asarray(rows, float)
    return table


def object_slots(scene: Scene) -> list[str]:
    """Return the initial road-user slots at the anchor.

    The scene's tracks in order; beyond ``MAX_OBJECTS``, those nearest the ego, ego included.
    """
    tracks = list(scene.tracks.values())
    if len(tracks) <= MAX_OBJECTS:
        return [t.track_id for t in tracks]
    anchor = scene.anchor
    here = scene.ego.states[anchor, :2]

    def distance(track: Track) -> float:
        frames = np.flatnonzero(track.valid)
        if not len(frames):
            return np.inf
        nearest = frames[np.argmin(np.abs(frames - anchor))]
        return float(np.linalg.norm(track.states[nearest, :2] - here))

    order = sorted(range(len(tracks)), key=lambda i: distance(tracks[i]))
    kept = set(order[:MAX_OBJECTS])
    return [t.track_id for i, t in enumerate(tracks) if i in kept]


def update_slots(
    slots: list[str], scene: Scene, ego_state: np.ndarray, beliefs: dict[str, np.ndarray]
) -> list[str]:
    """Re-pick slots for this step given the previous step's ``slots``.

    Keeps the ego and the ``MAX_OBJECTS - 1`` believed road users nearest it; one that already
    has a slot keeps it, and newcomers take the freed slots, nearest first.
    """
    here = np.asarray(ego_state, float)[:2]

    def distance(track_id: str) -> float:
        return float(np.linalg.norm(np.asarray(beliefs[track_id], float)[:2] - here))

    believed = [t for t in scene.tracks if t != scene.ego_id and t in beliefs]  # scene order
    nearest = [scene.ego_id, *sorted(believed, key=distance)[: MAX_OBJECTS - 1]]
    held = set(slots)
    newcomers = [t for t in nearest if t not in held]
    if not newcomers:
        return slots
    kept = set(nearest)
    free = [i for i, t in enumerate(slots) if t not in kept]  # never fewer than the newcomers
    out = list(slots)
    for index, track_id in zip(free[: len(newcomers)], newcomers, strict=True):
        out[index] = track_id
    return out


def observation(
    route: np.ndarray,
    road: np.ndarray,
    slots: list[str],
    scene: Scene,
    ego_state: np.ndarray,
    beliefs: dict[str, np.ndarray],
) -> np.ndarray:
    """Return the ego's ``[188, 7]`` float32 observation for this step.

    ``route`` and ``road`` come from ``scene_boxes``, ``slots`` from ``update_slots``. The road
    users are the planner's beliefs, in the scene frame like ``ego_state``.
    """
    origin = scene.world_origin
    pose = _ego_pose(ego_state, origin)
    half_length, half_width = FOV_M[0] // 2, FOV_M[1] // 2
    # Route: every box, wherever it sits.
    route_tokens = _typed(_relative(route, pose), ROUTE_TYPE)
    # Road edges: those in the window (edges included), capped at ``MAX_ROADGRAPH_SEGMENTS``.
    local = _relative(road, pose)
    outside = (np.abs(local[:, 0]) > half_length) | (np.abs(local[:, 1]) > half_width)
    local[outside] = 0.0
    road_tokens = _first_rows(_typed(local, ROAD_TYPE), MAX_ROADGRAPH_SEGMENTS)
    # Road users: one slot each in scene order; known and inside the window, else zero.
    objects = np.zeros((MAX_OBJECTS, 6))
    ego_slot = slots.index(scene.ego_id)
    for slot, track_id in enumerate(slots):
        if track_id == scene.ego_id:
            state, world = np.asarray(ego_state, float), pose
        elif track_id in beliefs:
            state = np.asarray(beliefs[track_id], float)
            world = scene_to_world(state[None], origin)[0]
        else:
            continue
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        dx, dy = world[0] - pose[0], world[1] - pose[1]
        x, y = dx * c - dy * s, dx * s + dy * c
        if abs(x) > half_length or abs(y) > half_width:
            continue
        length, width = scene.tracks[track_id].size
        yaw = (world[2] - pose[2]) * 180 / np.pi
        objects[slot] = (x, y, width, length, yaw, state[3])
    object_tokens = _typed(objects, OBJECT_TYPE)
    object_tokens[ego_slot, 0] = EGO_TYPE
    return np.vstack((route_tokens, object_tokens, road_tokens)).astype(np.float32)


def _ego_pose(ego_state: np.ndarray, origin: tuple[float, float, float]) -> np.ndarray:
    """Return the ego pose in world coordinates, its heading unwrapped as Waymax keeps it.

    The turn is read from the scene-frame heading, so it matches Waymax only within half a turn.
    """
    pose = scene_to_world(np.asarray(ego_state, float)[None], origin)[0]
    pose[2] = origin[2] + float(ego_state[2])
    return pose


def _relative(table: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Move boxes ``[M, 6]`` into the frame of ``pose``, yaws unwrapped as ``transform_yaw``."""
    c, s = math.cos(-pose[2]), math.sin(-pose[2])
    dx, dy = table[:, 0] - pose[0], table[:, 1] - pose[1]
    out = table.copy()
    out[:, 0], out[:, 1] = dx * c - dy * s, dx * s + dy * c
    out[:, 4] = table[:, 4] - pose[2] * 180 / np.pi
    out[table.sum(axis=1) == 0] = 0.0
    return out


def _typed(table: np.ndarray, type_id: float) -> np.ndarray:
    """Prepend ``type_id`` to each row and zero padding rows."""
    out = np.column_stack((np.full(len(table), type_id), table))
    out[table.sum(axis=1) == 0] = 0.0
    return out


def _first_rows(rows: np.ndarray, count: int) -> np.ndarray:
    """Return ``count`` rows in order, as ``drop_zero_on_roadobs`` does.

    Padding uses the first zero rows; with too many non-zero rows, the earliest are dropped.
    """
    keep = rows.sum(axis=1) != 0
    missing = count - int(keep.sum())
    if missing > 0:
        keep[np.flatnonzero(~keep)[:missing]] = True
    elif missing < 0:
        keep[np.flatnonzero(keep)[:-missing]] = False
    return rows[keep]
