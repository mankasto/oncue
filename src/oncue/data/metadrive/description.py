"""An OnCue scene as a MetaDrive scenario description (ScenarioNet's dict format).

Ported from ScenarioNet's Waymo converter (d4acdb5, Apache-2.0); start at
``closed_loop_description`` (MetaDrive 0.4.3) or ``baseline_description`` (0.3.0.1).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ...core.types import (
    BoundarySegment,
    FeatureKind,
    LaneNeighbor,
    LaneSignal,
    MapFeature,
    Scene,
    SignalTrack,
    Track,
    height_of,
)
from ...utils.geometry import heading_unit_vector, rotate, scene_to_world
from ..waymo.scenes import MPS_PER_MPH
from .types import (
    AGENT_TYPE,
    CAT_AGENT_TYPES,
    CAT_MAP_TYPES,
    COORDINATE_WAYMO,
    DATASET_NAME,
    KMH_PER_MPH,
    LANE_TYPE,
    LANE_TYPES_0301,
    MIN_LANE_POINTS_0301,
    POLYGON_TYPE,
    ROAD_EDGE_TYPE,
    ROAD_LINE_TYPE,
    SIGNAL_STATE,
    STOP_SIGN_TYPE,
    TRAFFIC_LIGHT_TYPE,
    VERSION_0301_TAG,
)

# --- parameters ---
BASELINE_PAST_FRAMES = 10  # frames before the anchor written for the baselines
BASELINE_FUTURE_FRAMES = 80  # frames after the anchor written for the baselines
ORIGIN_TOLERANCE_M = 1e-6  # sdc within this of the origin counts as at it (MetaDrive re-centres)


# --- entry points ---
def closed_loop_description(scene: Scene, target_id: str | None) -> dict:
    """Return the closed loop's description for MetaDrive 0.4.3, in scene coordinates.

    The scene must not hold the target ``target_id`` (``None``: a run without a target), since
    the loop drives it. Wrap the result in a ``ScenarioDescription`` for ``set_scenario``.
    """
    # 1 frames and tracks
    if target_id == scene.ego_id:
        raise ValueError(
            f"the target {target_id} is the ego: a run without a target passes target_id None"
        )
    if target_id in scene.tracks:
        raise ValueError(
            f"the scene holds the target {target_id}: remove its track before describing the "
            f"scene for the closed loop, which drives the target itself"
        )
    frames = _written_frames(
        scene, scene.anchor, len(scene.timestamps), "the closed loop's window, the anchor on"
    )
    _check_ego(scene, frames)
    tracks = _written_tracks(scene, frames, cat_types_only=False)
    objects_of_interest = [
        str(track_id)
        for track_id in scene.notes.get("objects_of_interest", ())
        if str(track_id) in tracks
    ]
    # 2 map, 3 signals, 4 tracks, 5 metadata
    description = _describe(scene, frames, tracks, objects_of_interest, for_baselines=False)
    _check_sdc_at_origin(description, scene.ego_id)
    return description


def baseline_description(scene: Scene, target: Track) -> tuple[dict, str]:
    """Return ``(description, target_id)`` for the baselines on MetaDrive 0.3.0.1.

    World coordinates, from ``anchor - 10`` to ``anchor + 80``. The target, not yet in the
    scene, gets a new integer id; map an adversary back by that id.
    """
    # 1 frames and tracks
    if target.track_id in scene.tracks:
        raise ValueError(
            f"the scene already holds a track {target.track_id}: the target would be written "
            f"twice (remove the road user it stands for from the scene first)"
        )
    anchor = scene.anchor
    frames = _written_frames(
        scene,
        anchor - BASELINE_PAST_FRAMES,
        anchor + BASELINE_FUTURE_FRAMES + 1,
        f"the baselines' window, anchor-{BASELINE_PAST_FRAMES} to anchor+{BASELINE_FUTURE_FRAMES} "
        f"(the {BASELINE_PAST_FRAMES + BASELINE_FUTURE_FRAMES + 1} frames CAT, SAGE and SEAL "
        f"index)",
    )
    _check_ego(scene, frames)
    _check_target(scene, target, frames)
    tracks = _written_tracks(scene, frames, cat_types_only=True)
    # one past the largest WOMD id, so no scene track owns it; CAT reads ids as integers
    womd_ids = [int(track_id) for track_id in scene.tracks if _is_womd_id(track_id)]
    target_id = str(max(womd_ids, default=0) + 1)
    tracks[target_id] = replace(target, track_id=target_id)
    # 2 map, 3 signals, 4 tracks, 5 metadata
    description = _describe(scene, frames, tracks, [scene.ego_id, target_id], for_baselines=True)
    return description, target_id


class OutsideScene(ValueError):  # noqa: N818 (the name appears in outputs as a failed row's repr)
    """Raised when a description's window reaches outside the scene."""

    def __init__(self, window: str, start: int, stop: int, length: int):
        super().__init__(
            f"{window}: scene frames {start} to {stop} (stop exclusive) do not fit the scene's "
            f"{length} frames"
        )
        self.start, self.stop, self.length = start, stop, length


# --- steps 2 to 5, shared by both entry points ---
def _describe(
    scene: Scene,
    frames: np.ndarray,
    tracks: dict[str, Track],
    objects_of_interest: list[str],
    *,
    for_baselines: bool,
) -> dict:
    """Return the description of ``tracks`` over ``frames``; for the baselines, what CAT reads."""
    coordinates = _Coordinates(scene.world_origin if for_baselines else None)
    # 2 map
    map_features = {
        feature.feature_id: _map_feature(feature, coordinates) for feature in scene.map_features
    }
    _add_lane_widths(map_features)
    # 3 signals
    lights = {
        signal.lane_id: _traffic_light(signal, frames, coordinates) for signal in scene.signals
    }
    if for_baselines:
        map_features, lights = _readable_by_cat(map_features, lights)
    # 4 tracks
    written_tracks = {
        track_id: _track(track, frames, coordinates) for track_id, track in tracks.items()
    }
    # 5 metadata; the version is the dataset's, or 0.3.0.1's DATA_VERSION as its converter writes
    if for_baselines:
        version = VERSION_0301_TAG
    else:
        version = scene.source.dataset_version if scene.source is not None else ""
    metadata = _metadata(scene, tracks, objects_of_interest, frames)
    return {
        "id": scene.scene_id,
        "version": version,
        "length": len(frames),
        "tracks": written_tracks,
        "dynamic_map_states": lights,
        "map_features": map_features,
        "metadata": metadata,
    }


class _Coordinates:
    """Convert scene-frame poses, points and vectors to WOMD's world frame, given an ``origin``."""

    def __init__(self, origin: tuple[float, float, float] | None):
        self.origin = origin

    def poses(self, poses: np.ndarray) -> np.ndarray:
        """Convert scene-frame ``[N, 3]`` ``(x, y, heading)`` poses."""
        poses = np.asarray(poses, float).reshape(-1, 3)
        return poses if self.origin is None else scene_to_world(poses, self.origin)

    def points(self, points: np.ndarray) -> np.ndarray:
        """Convert scene-frame ``[N, 2]`` ``(x, y)`` points."""
        points = np.asarray(points, float).reshape(-1, 2)
        return self.poses(np.column_stack((points, np.zeros(len(points)))))[:, :2]

    def vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate ``[N, 2]`` velocity vectors with the frame (no translation)."""
        vectors = np.asarray(vectors, float).reshape(-1, 2)
        return vectors if self.origin is None else rotate(vectors, self.origin[2])


# --- 1 frames and tracks ---
def _written_frames(scene: Scene, start: int, stop: int, window: str) -> np.ndarray:
    """Return the frames ``start`` to ``stop`` (exclusive), or raise ``OutsideScene``."""
    if not 0 <= start < stop <= len(scene.timestamps):
        raise OutsideScene(window, start, stop, len(scene.timestamps))
    return np.arange(start, stop)


def _check_ego(scene: Scene, frames: np.ndarray) -> None:
    """Refuse an ego unobserved at any written frame, as MetaDrive replays it."""
    unobserved = frames[~scene.ego.valid[frames]]
    if len(unobserved):
        raise ValueError(
            f"ego {scene.ego_id} unobserved at scene frames {unobserved.tolist()}: it must be "
            f"observed at every written frame"
        )


def _check_target(scene: Scene, target: Track, frames: np.ndarray) -> None:
    """Refuse a target the baselines cannot attack.

    It needs a CAT-known type and must be observed from the first written frame through the
    anchor, since CAT, SAGE and SEAL steer it from the first step.
    """
    if len(target.states) != len(scene.timestamps):
        raise ValueError(f"target {target.track_id}: its frames do not match the scene's")
    if AGENT_TYPE[target.agent_type] not in CAT_AGENT_TYPES:
        raise ValueError(f"target {target.track_id}: CAT cannot read its type {target.agent_type}")
    history = np.arange(frames[0], scene.anchor + 1)
    unobserved = history[~target.valid[history]]
    if len(unobserved):
        raise ValueError(
            f"target {target.track_id} unobserved at scene frames {unobserved.tolist()}: it must "
            f"be observed from the first written frame through the anchor"
        )


def _written_tracks(scene: Scene, frames: np.ndarray, *, cat_types_only: bool) -> dict[str, Track]:
    """Return the scene's tracks observed in ``frames``, in record order.

    With ``cat_types_only``, only the agent types CAT's ``_parse`` knows are kept.
    """
    inserted = [track_id for track_id in scene.tracks if not _is_womd_id(track_id)]
    if inserted:
        raise ValueError(
            f"tracks {inserted} have no WOMD id: a scene holds only logged tracks here (insert a "
            f"target through baseline_description, or remove it for the closed loop)"
        )
    return {
        track_id: track
        for track_id, track in scene.tracks.items()
        if track.valid[frames].any()
        and (not cat_types_only or AGENT_TYPE[track.agent_type] in CAT_AGENT_TYPES)
    }


def _is_womd_id(track_id: str) -> bool:
    """Return whether ``track_id`` is a WOMD id (a non-negative integer)."""
    return track_id.isascii() and track_id.isdigit()


def _check_sdc_at_origin(description: dict, ego_id: str) -> None:
    """Refuse an sdc away from the origin at the first frame, as MetaDrive re-centres on it."""
    first = description["tracks"][ego_id]["state"]["position"][0, :2]
    if np.hypot(first[0], first[1]) > ORIGIN_TOLERANCE_M:
        raise ValueError(
            f"the sdc starts at {first.tolist()}, not at the scene origin: MetaDrive would "
            f"re-centre the map on it, shifting every scene-frame state the closed loop writes"
        )


# --- 2 map ---
def _map_feature(feature: MapFeature, coordinates: _Coordinates) -> dict:
    """Convert one map feature, as ScenarioNet's ``extract_*`` helpers do."""
    # [N, 3] float32 points; z = 0 where not recorded (ScenarioNet's ``extract_poly``).
    xy = coordinates.points(feature.points)
    points = np.zeros((len(xy), 3), np.float32)
    points[:, :2] = xy
    if feature.z is not None:
        points[:, 2] = feature.z
    kind = feature.kind
    if kind is FeatureKind.STOP_SIGN:
        lanes = [int(lane_id) for lane_id in feature.controlled_lanes]
        return {"type": STOP_SIGN_TYPE, "lane": lanes, "position": points[0]}
    if kind in POLYGON_TYPE:
        return {"type": POLYGON_TYPE[kind], "polygon": points}
    womd_type = {
        FeatureKind.LANE: feature.lane_type,
        FeatureKind.ROAD_LINE: feature.line_type,
        FeatureKind.ROAD_EDGE: feature.edge_type,
    }[kind]
    if womd_type is None:
        raise ValueError(
            f"map feature {feature.feature_id}: no {kind.value} type (an older export has none): "
            f"re-export the scene"
        )
    if kind is FeatureKind.LANE:
        return _lane(feature, points)
    type_names = ROAD_LINE_TYPE if kind is FeatureKind.ROAD_LINE else ROAD_EDGE_TYPE
    return {"type": type_names[womd_type], "polyline": points}


def _lane(feature: MapFeature, polyline: np.ndarray) -> dict:
    """Convert a lane centre line (ScenarioNet's ``extract_center``, no interpolation).

    An unrecorded speed limit is written as 0 mph, as WOMD's own reaches ScenarioNet.
    """
    speed_limit_mph = (
        0.0 if feature.speed_limit_mps is None else feature.speed_limit_mps / MPS_PER_MPH
    )
    return {
        "speed_limit_mph": float(speed_limit_mph),
        "speed_limit_kmh": float(speed_limit_mph * KMH_PER_MPH),
        "type": LANE_TYPE[feature.lane_type],
        "polyline": polyline,
        "entry_lanes": [int(lane_id) for lane_id in feature.entry_lanes],
        "exit_lanes": [int(lane_id) for lane_id in feature.exit_lanes],
        "left_boundaries": [_boundary(segment) for segment in feature.left_boundaries],
        "right_boundaries": [_boundary(segment) for segment in feature.right_boundaries],
        "left_neighbor": [_neighbor(neighbor) for neighbor in feature.left_neighbors],
        "right_neighbor": [_neighbor(neighbor) for neighbor in feature.right_neighbors],
    }


def _boundary(segment: BoundarySegment) -> dict:
    """ScenarioNet ``extract_boundaries`` entry (all strings; no ``boundary_type`` kept)."""
    return {
        "lane_start_index": str(segment.start),
        "lane_end_index": str(segment.end),
        "boundary_feature_id": str(segment.feature_id),
    }


def _neighbor(neighbor: LaneNeighbor) -> dict:
    """ScenarioNet ``extract_neighbors`` entry (all strings; no boundaries kept)."""
    return {
        "feature_id": str(neighbor.feature_id),
        "self_start_index": str(neighbor.self_start),
        "self_end_index": str(neighbor.self_end),
        "neighbor_start_index": str(neighbor.neighbor_start),
        "neighbor_end_index": str(neighbor.neighbor_end),
    }


def _add_lane_widths(map_features: dict) -> None:
    """Fill each lane's ``width [N, 2]`` to its left and right boundaries (``compute_width``)."""
    for feature in map_features.values():
        if "LANE" not in feature["type"]:
            continue
        polyline = feature["polyline"][:, :2]
        width = np.zeros((len(polyline), 2), np.float32)
        width[:, 0] = _boundary_distances(map_features, polyline, feature["left_boundaries"])
        width[:, 1] = _boundary_distances(map_features, polyline, feature["right_boundaries"])
        width[width[:, 0] == 0, 0] = width[width[:, 0] == 0, 1]
        width[width[:, 1] == 0, 1] = width[width[:, 1] == 0, 0]
        feature["width"] = width


def _boundary_distances(map_features: dict, polyline: np.ndarray, boundaries: list) -> np.ndarray:
    """Return each centre point's distance to ``boundaries``, 0 where none (``extract_width``)."""
    distances = np.zeros(len(polyline), np.float32)
    for boundary in boundaries:
        lane_start = int(boundary["lane_start_index"])
        segment_length = int(boundary["lane_end_index"]) - lane_start
        boundary_line = map_features[boundary["boundary_feature_id"]]["polyline"][:, :2]
        # walk the boundary from its point nearest the segment's start (``nearest_point``)
        start_offsets = np.square(boundary_line - polyline[lane_start])
        boundary_start = int(np.argmin(np.sqrt(start_offsets[:, 0] + start_offsets[:, 1])))
        boundary_end = min(boundary_start + segment_length, len(boundary_line) - 1)
        count = min(boundary_end - boundary_start, segment_length) + 1
        on_lane = np.arange(lane_start, lane_start + count)  # indexing raises past the end
        on_boundary = np.arange(boundary_start, boundary_start + count)
        offsets = np.square(polyline[on_lane] - boundary_line[on_boundary])
        distances[on_lane] = np.sqrt(offsets[:, 0] + offsets[:, 1])
    return distances


# --- 3 signals ---
def _traffic_light(signal: SignalTrack, frames: np.ndarray, coordinates: _Coordinates) -> dict:
    """ScenarioNet ``extract_dynamic_map_states`` entry for one signal-controlled lane.

    An unreported frame is ``LANE_STATE_UNKNOWN`` (ScenarioNet writes ``None``).
    """
    states = [SIGNAL_STATE[LaneSignal(int(signal.states[frame]))] for frame in frames]
    stop_point = np.zeros(3, np.float32)  # z not kept
    stop_point[:2] = coordinates.points(signal.stop_point)[0]
    return {
        "type": TRAFFIC_LIGHT_TYPE,
        "state": {"object_state": states},
        "lane": int(signal.lane_id),
        "stop_point": stop_point,
        "metadata": {
            "track_length": len(frames),
            "type": TRAFFIC_LIGHT_TYPE,
            "object_id": signal.lane_id,
            "dataset": DATASET_NAME,
        },
    }


def _readable_by_cat(map_features: dict, lights: dict) -> tuple[dict, dict]:
    """Return the map features and signals CAT and MetaDrive 0.3.0.1 can read.

    Other feature types are dropped, and so are the neighbours and signals that refer to lanes
    0.3.0.1 then lacks.
    """
    kept = {
        feature_id: feature
        for feature_id, feature in map_features.items()
        if feature["type"] in CAT_MAP_TYPES
    }
    for feature in kept.values():
        for side in ("left_neighbor", "right_neighbor"):
            if side in feature:
                feature[side] = [
                    neighbor for neighbor in feature[side] if neighbor["feature_id"] in kept
                ]
    built_lanes = {
        feature_id
        for feature_id, feature in kept.items()
        if feature["type"] in LANE_TYPES_0301 and len(feature["polyline"]) >= MIN_LANE_POINTS_0301
    }
    return kept, {lane_id: light for lane_id, light in lights.items() if lane_id in built_lanes}


# --- 4 tracks ---
def _track(track: Track, frames: np.ndarray, coordinates: _Coordinates) -> dict:
    """ScenarioNet ``extract_tracks`` entry for one road user."""
    agent_type = AGENT_TYPE[track.agent_type]
    return {
        "type": agent_type,
        "state": _track_state(track, frames, coordinates),
        "metadata": {
            "track_length": len(frames),
            "type": agent_type,
            "object_id": track.track_id,
            "dataset": DATASET_NAME,
        },
    }


def _track_state(track: Track, frames: np.ndarray, coordinates: _Coordinates) -> dict:
    """Return the per-frame state dict for one track.

    Unobserved frames hold zeros, as MetaDrive's valid check expects. The box size is the
    track's own, as a ``Scene`` has no per-frame size.
    """
    count = len(frames)
    valid = track.valid[frames].copy()
    observed = frames[valid]  # the scene frames written as observed
    position = np.zeros((count, 3), np.float32)
    heading = np.zeros(count, np.float32)
    velocity = np.zeros((count, 2), np.float32)
    poses = coordinates.poses(track.states[observed, :3])
    position[valid, :2], heading[valid] = poses[:, :2], poses[:, 2]
    if track.z is not None:
        position[valid, 2] = track.z[observed]
    velocity[valid] = coordinates.vectors(_velocity(track, observed))
    return {
        "position": position,
        "length": np.where(valid, track.size[0], 0.0).astype(np.float32),
        "width": np.where(valid, track.size[1], 0.0).astype(np.float32),
        "height": np.where(valid, height_of(track), 0.0).astype(np.float32),
        "heading": heading,
        "velocity": velocity,
        "valid": valid,
    }


def _velocity(track: Track, frames: np.ndarray) -> np.ndarray:
    """Return ``[F, 2]`` velocities at ``frames``: the recorded ones, else speed along heading."""
    if track.velocity is None:
        states = track.states[frames]
        return states[:, 3:4] * heading_unit_vector(states[:, 2])
    recorded = track.velocity[frames]
    broken = frames[~np.isfinite(recorded).all(axis=1)]
    if len(broken):
        raise ValueError(
            f"track {track.track_id}: non-finite recorded velocity at scene frames "
            f"{broken.tolist()}"
        )
    return recorded


# --- 5 metadata ---
def _metadata(
    scene: Scene, tracks: dict[str, Track], objects_of_interest: list[str], frames: np.ndarray
) -> dict:
    """Return the ScenarioNet metadata for a WOMD scenario, indexing written frames and tracks.

    Without ``object_summary`` and ``number_summary``, which ScenarioNet's dataset writer adds.
    """
    track_ids = list(tracks)
    tracks_to_predict = {}
    for track_id, difficulty in scene.notes.get("tracks_to_predict", ()):
        track_id = str(track_id)
        if track_id in tracks:  # a track left out is not predicted
            tracks_to_predict[track_id] = {
                "track_index": track_ids.index(track_id),
                "track_id": track_id,
                "difficulty": int(difficulty),
                "object_type": AGENT_TYPE[tracks[track_id].agent_type],
            }
    return {
        "id": scene.scene_id,
        "coordinate": COORDINATE_WAYMO,
        "ts": scene.timestamps[frames].astype(np.float32),
        "metadrive_processed": False,
        "sdc_id": scene.ego_id,
        "dataset": DATASET_NAME,
        "scenario_id": scene.scene_id,
        "source_file": scene.source.shard if scene.source is not None else "",
        "track_length": len(frames),
        "current_time_index": int(scene.anchor - frames[0]),
        "sdc_track_index": track_ids.index(scene.ego_id),
        "objects_of_interest": objects_of_interest,
        "tracks_to_predict": tracks_to_predict,
    }
