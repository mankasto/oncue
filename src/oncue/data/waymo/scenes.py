"""Decode Waymo Open Motion Dataset Scenario records into a ``Scene``; start at ``load_record``.

Decoding needs no TensorFlow. Missing observations stay invalid and are never bridged.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...core.types import (
    MIN_TIMESTAMPS,
    STATE_SIZE,
    BoundarySegment,
    FeatureKind,
    LaneNeighbor,
    LaneType,
    MapFeature,
    RoadEdgeType,
    RoadLineType,
    Scene,
    SceneSource,
    SignalTrack,
    Track,
)
from ...utils.geometry import wrap_angle
from ...utils.tracks import TIME_TOL_S
from .tfrecord import RecordRef, read_record

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

AGENT_TYPES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
DATASET_VERSION = "v1.2.0"  # release of the local development data
MPS_PER_MPH = 0.44704  # exact
GRID_DT_S = 0.1  # uniform grid (WOMD's 10 Hz)
# grid times this far outside the recorded span snap to its end, so all 91 frames survive
RESAMPLE_TOLERANCE_S = GRID_DT_S / 2
SPLITS = {"training": "train", "validation": "validation", "testing": "test"}  # shard name prefix


def load_record(
    ref: RecordRef,
    split: str | None = None,
    dt: float | None = GRID_DT_S,
    dataset_version: str = DATASET_VERSION,
) -> Scene:
    """Return the scene for ``ref``, resampled to a ``dt`` s grid (``None``: the raw frames).

    ``split`` defaults to the shard name's prefix (``SPLITS``), else ``train``.
    """
    payload = read_record(ref)
    name = Path(ref.path).name
    if split is None:
        split = next((v for k, v in SPLITS.items() if name.startswith(k)), "train")
    source = SceneSource(
        dataset="womd",
        dataset_version=dataset_version,
        split=split,
        shard=Path(ref.path).name,
        record_index=ref.index,
        record_offset=ref.offset,
        record_length=ref.length,
        record_sha256=hashlib.sha256(payload).hexdigest(),
    )
    scene = convert(parse_scenario(payload), source)
    return scene if dt is None else resample(scene, dt)


def convert(scenario: Any, source: SceneSource | None = None) -> Scene:
    """Convert a decoded scenario to a ``Scene`` in the ego frame, on its raw timeline.

    Call ``resample`` before any analysis.
    """
    anchor = scenario.current_time_index
    origin = scenario.tracks[scenario.sdc_track_index].states[anchor]
    if not origin.valid:
        raise ValueError("Ego is unobserved at the current time")
    c, s = np.cos(origin.heading), np.sin(origin.heading)
    rotation = np.array([[c, s], [-s, c]])
    offset = np.array([origin.center_x, origin.center_y])

    def to_local(xy: np.ndarray) -> np.ndarray:
        return (xy - offset) @ rotation.T

    tracks = {}
    for raw in scenario.tracks:
        valid = np.array([p.valid for p in raw.states], bool)
        if not valid.any():
            continue
        observed = [p for p in raw.states if p.valid]
        states = np.zeros((len(raw.states), STATE_SIZE))
        states[valid, :2] = to_local(np.array([[p.center_x, p.center_y] for p in observed]))
        states[valid, 2] = wrap_angle(np.array([p.heading for p in observed]) - origin.heading)
        raw_velocity = np.array([[p.velocity_x, p.velocity_y] for p in observed])
        states[valid, 3] = np.hypot(raw_velocity[:, 0], raw_velocity[:, 1])
        velocity = np.zeros((len(raw.states), 2))
        velocity[valid] = raw_velocity @ rotation.T  # rotate into the scene frame (no offset)
        z = np.zeros(len(raw.states))
        z[valid] = [p.center_z for p in observed]
        size = np.median([[p.length, p.width] for p in observed], axis=0)
        if min(size) <= 0:
            continue
        heights = [p.height for p in observed if p.height > 0]
        height = float(np.median(heights)) if heights else None
        tid = str(raw.id)
        agent_type = AGENT_TYPES.get(raw.object_type, "other")
        tracks[tid] = Track(tid, agent_type, size, states, valid, height, velocity, z)
    features = tuple(f for f in (_map_feature(m, to_local) for m in scenario.map_features) if f)
    # traffic signals: one track per controlled lane over the raw timeline (UNKNOWN if unreported)
    n_raw = len(scenario.timestamps_seconds)
    per_lane: dict[str, tuple[np.ndarray, list]] = {}
    for i, frame in enumerate(scenario.dynamic_map_states[:n_raw]):
        for ls in frame.lane_states:
            key = str(ls.lane)
            if key not in per_lane:
                per_lane[key] = (np.zeros(n_raw, np.int8), [ls.stop_point.x, ls.stop_point.y])
            per_lane[key][0][i] = ls.state
    signals = tuple(
        SignalTrack(lane, tuple(to_local(np.asarray([stop], float))[0]), states)
        for lane, (states, stop) in sorted(per_lane.items())
    )
    return Scene(
        scene_id=str(scenario.scenario_id),
        timestamps=np.asarray(scenario.timestamps_seconds, float),
        anchor=anchor,
        ego_id=str(scenario.tracks[scenario.sdc_track_index].id),
        tracks=tracks,
        map_features=features,
        signals=signals,
        source=source,
        notes={
            # ego pose at the anchor in WOMD world coordinates (x, y, heading)
            "world_origin": (origin.center_x, origin.center_y, origin.heading),
            # WOMD's tracks to predict (id, difficulty 0-2) and interacting pair (ids)
            "tracks_to_predict": [
                (str(scenario.tracks[p.track_index].id), int(p.difficulty))
                for p in scenario.tracks_to_predict
            ],
            "objects_of_interest": [str(v) for v in scenario.objects_of_interest],
        },
    )


def _map_feature(feature: Any, to_local: Callable[[np.ndarray], np.ndarray]) -> MapFeature | None:
    """Convert one scenario map feature; return ``None`` when unsupported or empty."""
    kind = feature.WhichOneof("feature_data")
    if kind not in {k.value for k in FeatureKind}:
        return None
    data = getattr(feature, kind)
    if kind == FeatureKind.STOP_SIGN.value:
        points, z = [[data.position.x, data.position.y]], [data.position.z]
    else:
        raw_points = data.polygon if hasattr(data, "polygon") else data.polyline
        points, z = [[p.x, p.y] for p in raw_points], [p.z for p in raw_points]
    if not points:
        return None
    extra = {}
    if kind == FeatureKind.LANE.value:
        extra = {
            "lane_type": LaneType(data.type),
            "speed_limit_mps": data.speed_limit_mph * MPS_PER_MPH
            if data.speed_limit_mph > 0
            else None,
            "entry_lanes": tuple(str(v) for v in data.entry_lanes),
            "exit_lanes": tuple(str(v) for v in data.exit_lanes),
            "left_boundaries": tuple(
                BoundarySegment(str(b.boundary_feature_id), b.lane_start_index, b.lane_end_index)
                for b in data.left_boundaries
            ),
            "right_boundaries": tuple(
                BoundarySegment(str(b.boundary_feature_id), b.lane_start_index, b.lane_end_index)
                for b in data.right_boundaries
            ),
            **{
                side: tuple(
                    LaneNeighbor(
                        str(n.feature_id),
                        n.self_start_index,
                        n.self_end_index,
                        n.neighbor_start_index,
                        n.neighbor_end_index,
                    )
                    for n in getattr(data, side)
                )
                for side in ("left_neighbors", "right_neighbors")
            },
        }
    elif kind == FeatureKind.ROAD_EDGE.value:
        extra = {"edge_type": RoadEdgeType(data.type)}
    elif kind == FeatureKind.ROAD_LINE.value:
        extra = {"line_type": RoadLineType(data.type)}
    elif kind == FeatureKind.STOP_SIGN.value:
        extra = {"controlled_lanes": tuple(str(v) for v in data.lane)}
    extra["z"] = np.asarray(z, float)
    return MapFeature(str(feature.id), kind, to_local(np.asarray(points, float)), **extra)


def parse_scenario(payload: bytes) -> Any:
    """Parse a WOMD Scenario protobuf from ``payload``; raise on a malformed record."""
    scenario = _scenario_class()()
    scenario.ParseFromString(payload)
    if (
        not scenario.scenario_id
        or not scenario.tracks
        or len(scenario.timestamps_seconds) < MIN_TIMESTAMPS
    ):
        raise ValueError("Record is not a WOMD Scenario")
    if not 0 <= scenario.sdc_track_index < len(scenario.tracks):
        raise ValueError("Ego track index out of range")
    return scenario


@lru_cache(maxsize=1)
def _scenario_class():
    """Return the ``Scenario`` message class, built once from the vendored descriptor set."""
    try:
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    except ImportError as exc:
        raise RuntimeError("Install oncue[data] to decode WOMD records") from exc
    descriptors = descriptor_pb2.FileDescriptorSet()
    descriptors.ParseFromString(files("oncue.data").joinpath("schema/scenario.desc").read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for proto in descriptors.file:
        pool.Add(proto)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("waymo.open_dataset.Scenario")
    )


def resample(scene: Scene, dt: float = GRID_DT_S, tolerance: float = RESAMPLE_TOLERANCE_S) -> Scene:
    """Resample ``scene`` onto a uniform ``dt`` grid through the anchor time.

    A grid state is valid only when both raw observations around it are valid.
    """
    t = scene.timestamps
    anchor = t[scene.anchor]
    before = int(np.floor((anchor - t[0] + tolerance) / dt))
    after = int(np.floor((t[-1] - anchor + tolerance) / dt))
    grid = anchor + np.arange(-before, after + 1) * dt
    query = np.clip(grid, t[0], t[-1])
    right = np.clip(np.searchsorted(t, query - TIME_TOL_S), 0, len(t) - 1)
    left = np.where(np.abs(t[right] - query) < TIME_TOL_S, right, np.maximum(right - 1, 0))
    span = t[right] - t[left]
    w = np.divide(query - t[left], span, out=np.zeros_like(query), where=span > 0)[:, None]
    tracks = {}
    for key, track in scene.tracks.items():
        a, b = track.states[left], track.states[right]
        # Not ``lerp_states``: its ``a + w (b - a)`` differs in the last bits from exported scenes.
        states = (1 - w) * a + w * b
        states[:, 2] = a[:, 2] + w[:, 0] * wrap_angle(b[:, 2] - a[:, 2])
        valid = track.valid[left] & track.valid[right]
        states[~valid] = 0.0
        velocity = z = None
        if track.velocity is not None:
            velocity = (1 - w) * track.velocity[left] + w * track.velocity[right]
            velocity[~valid] = 0.0
        if track.z is not None:
            z = (1 - w[:, 0]) * track.z[left] + w[:, 0] * track.z[right]
            z[~valid] = 0.0
        tracks[key] = track.with_states(states, valid, velocity, z)
    # Signal states hold until the next report.
    held = np.clip(np.searchsorted(t, grid + TIME_TOL_S) - 1, 0, len(t) - 1)
    signals = tuple(replace(sig, states=sig.states[held]) for sig in scene.signals)
    return replace(scene, timestamps=grid, anchor=before, tracks=tracks, signals=signals)
