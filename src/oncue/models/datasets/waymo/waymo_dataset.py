"""Scene-level inputs of the trajectory generator, MTR style (features version 5): `encode_scene`,
then `assemble`.

A scene is stored once (`encode_scene`, numpy, scene frame); the inputs of one road user are
assembled from it when a batch is drawn (`assemble`, torch, on the GPU in training), in that road
user's frame. Training and generation use the same two functions, so their inputs agree.

Stored per scene:
  agents [A, T, 4]       x, y, heading, speed of every vehicle, pedestrian and cyclist; valid [A, T]
  agent_static [A, 5]    type (0 vehicle, 1 pedestrian, 2 cyclist), length, width, height, is SDC
  agent_vel [A, T, 2]    velocity vector as recorded (WOMD; made tracks: speed along the heading)
  agent_z [A, T]         box centre elevation as recorded (made tracks: the nearest map point's
                         elevation plus half the height)
  map_pts [C, Q, 3]      map polylines as MTR cuts them: WOMD points (x, y, z) as recorded (no
                         resampling), a polyline split where two points are more than 1 m apart
                         (in x-y), then into segments of Q = 20 points; map_dir [C, Q, 3]: the 3-D
                         unit direction from the previous point of the polyline (0 for its first
                         point); map_n: points per segment; map_type: one of MAP_TYPES (MTR's
                         global polyline types, plus driveways)
  map_sig [C, T]         the segment's lane signal phase at every frame (0 none or unknown, 1 stop,
                         2 caution, 3 go; last report held)
Assembled per road user (its frame: origin and x-axis at its pose at the anchor frame):
  hist [H, 11]           x, y, cos h, sin h, speed, valid, vx, vy, ax, ay, z (H frames ending at
                         the anchor; the recorded velocity, its change per 0.1 s as the
                         acceleration (0 on the first frame after a gap), the first frame
                         repeating the second, as MTR; z relative to its own at the anchor)
  nbr [N, H, 16]         the N road users seen in the history nearest by their last position seen
                         (as MTR, which keeps every road user seen in the history): the same
                         motion, type, length, width at [6:9] (on every frame), velocity and
                         acceleration at [9:13], height, SDC (on every frame), z at 15
  map [P, Q, 9]          the P segments whose centres are nearest the point `map_offset_m` ahead:
                         x, y, type, direction x, y, previous point x, y (MTR's pre_x, pre_y),
                         z, direction z
Elevations are model inputs only: visibility (2.5D) uses heights over flat ground.
  msig [P, 9]            their signal phases at the anchor and every 1 s after
  fut [F, 3], typ, size  its future (x, y, speed), type, length, width, height
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ....core.types import (
    FeatureKind,
    LaneType,
    MapFeature,
    RoadEdgeType,
    RoadLineType,
    Scene,
    Track,
)
from ....utils.geometry import heading_unit_vector
from ..routes import PHASE, SIGNAL_TIMES

if TYPE_CHECKING:  # annotations only (the stored inputs need numpy only)
    import torch

TYPES = {"vehicle": 0, "pedestrian": 1, "cyclist": 2}  # the road users modelled: agent_static[:, 0]
SEGMENT_POINTS = 20
BREAK_M = 1.0
FRAMES_PER_S = 10  # WOMD's frame rate
FRAME_S = 1 / FRAMES_PER_S
# MTR's get_polyline_dir: a direction vector's length clipped to these before dividing by it
DIRECTION_MIN_NORM, DIRECTION_MAX_NORM = 1e-6, 1e9

# the sizes of the last axis (see the module docstring)
STATE_SIZE = 4  # agents: x, y, heading, speed
STATIC_SIZE = 5  # agent_static: type, length, width, height, is SDC
HIST_SIZE = 11
NBR_SIZE = 16
MAP_SIZE = 9
AGENT_FUT_SIZE = 4  # agent_fut: x, y, vx, vy
# agent_static channels: type, length (then width), height, is SDC
STATIC_TYPE, STATIC_LENGTH, STATIC_HEIGHT, STATIC_SDC = 0, 1, 3, 4
# hist channels: speed, valid, velocity (2), acceleration (2), elevation
HIST_SPEED, HIST_VALID, HIST_VELOCITY, HIST_ACCELERATION, HIST_ELEVATION = 4, 5, 6, 8, 10

# MTR's input sizes: the defaults of `FeatureConfig`
HISTORY_FRAMES = 11  # 1.1 s
FUTURE_FRAMES = 80  # 8 s
NEIGHBOURS = 128
MAP_CHUNKS = 768
MAP_OFFSET_M = 30.0  # map segments nearest the point this far ahead of the agent


@dataclass(frozen=True)
class FeatureConfig:
    """MTR's inputs (`assemble`): up to 128 road users, the 768 map segments whose centres are
    nearest the point 30 m ahead; per scene the road users learned from."""

    history: int = HISTORY_FRAMES  # frames, 1.1 s at 10 Hz
    future: int = FUTURE_FRAMES  # frames, 8 s
    neighbours: int = NEIGHBOURS
    map_chunks: int = MAP_CHUNKS
    map_offset_m: float = MAP_OFFSET_M  # map segments nearest the point this far ahead of the agent
    per_scene: int = 64  # agents per scene: every pedestrian / cyclist first, then vehicles
    version: int = 5


# map types (0: padding), MTR's `waymo_types.polyline_type` order, plus driveways
MAP_TYPES = {
    ("lane", LaneType.UNDEFINED): 1,
    ("lane", LaneType.FREEWAY): 2,
    ("lane", LaneType.SURFACE_STREET): 3,
    ("lane", LaneType.BIKE_LANE): 4,
    **{("road_line", t): 5 + int(t) for t in RoadLineType},  # 5 unknown, 6-13 the 8 line types
    ("road_edge", RoadEdgeType.UNKNOWN): 14,
    ("road_edge", RoadEdgeType.BOUNDARY): 15,
    ("road_edge", RoadEdgeType.MEDIAN): 16,
    ("stop_sign", None): 17,
    ("crosswalk", None): 18,
    ("speed_bump", None): 19,
    ("driveway", None): 20,
}
N_MAP_TYPES = 21

# the stored arrays: per road user or per map segment, their shape after that axis ("T": frames)
LAYOUT = {
    "agents": ("agent", ("T", STATE_SIZE), np.float32),
    "agent_valid": ("agent", ("T",), bool),
    "agent_static": ("agent", (STATIC_SIZE,), np.float32),
    "agent_vel": ("agent", ("T", 2), np.float32),
    "agent_z": ("agent", ("T",), np.float32),
    "map_pts": ("map", (SEGMENT_POINTS, 3), np.float32),
    "map_dir": ("map", (SEGMENT_POINTS, 3), np.float32),
    "map_n": ("map", (), np.int16),
    "map_type": ("map", (), np.uint8),
    "map_sig": ("map", ("T",), np.uint8),
}
AGENT_KEYS = tuple(k for k, v in LAYOUT.items() if v[0] == "agent")
FRAME_KEYS = tuple(k for k, v in LAYOUT.items() if "T" in v[1])


# ---------------------------------------------------------------- a stored scene (numpy)
def encode_scene(scene: Scene) -> dict:
    """The scene's stored inputs (numpy, scene frame)."""
    tracks = [track for track in scene.tracks.values() if track.agent_type in TYPES]
    out = encode_agents(tracks, scene.ego_id, ground_points(scene))
    pts, dirs, counts, types, sigs = map_segments(scene)
    out.update(
        map_pts=pts.astype(np.float32),
        map_dir=dirs.astype(np.float32),
        map_n=counts.astype(np.int16),
        map_type=types.astype(np.uint8),
        map_sig=sigs.astype(np.uint8),
    )
    return out


def encode_agents(
    tracks: list[Track], ego_id: str, ground: tuple[np.ndarray, np.ndarray] | None = None
) -> dict:
    """The AGENT_KEYS arrays of road users (vehicles, pedestrians, cyclists); `ground`: map points
    (x-y [M, 2], z [M]) for the elevation of tracks without a recorded one."""
    from ....core.types import height_of

    count = len(tracks)
    static = [
        [TYPES[track.agent_type], *track.size, height_of(track), float(track.track_id == ego_id)]
        for track in tracks
    ]
    elevation = [_elevation(track, height_of(track), ground) for track in tracks]
    return {
        "agents": np.array([track.states for track in tracks], np.float32).reshape(
            count, -1, STATE_SIZE
        ),
        "agent_valid": np.array([track.valid for track in tracks], bool).reshape(count, -1),
        "agent_static": np.array(static, np.float32).reshape(count, STATIC_SIZE),
        "agent_vel": np.array([_velocity(track) for track in tracks], np.float32).reshape(
            count, -1, 2
        ),
        "agent_z": np.array(elevation, np.float32).reshape(count, -1),
        "agent_ids": [track.track_id for track in tracks],
    }


def _elevation(
    track: Track, height: float, ground: tuple[np.ndarray, np.ndarray] | None
) -> np.ndarray:
    """[T] the recorded box centre elevation; a made track's: the elevation of the map point
    nearest each of its positions plus half its height (half its height without map points)."""
    if track.z is not None:
        return np.asarray(track.z, float) * track.valid
    elevation = np.full(len(track.states), height / 2)
    if ground is not None and len(ground[1]) and track.valid.any():
        xy = track.states[track.valid, :2]
        # [T, M]: a few made tracks per scene
        squared_distance = ((xy[:, None] - ground[0][None]) ** 2).sum(-1)
        elevation[track.valid] += ground[1][squared_distance.argmin(1)]
    return elevation * track.valid


def _velocity(track: Track) -> np.ndarray:
    """[T, 2] the recorded velocity (WOMD); a made track's: its speed along its heading."""
    if track.velocity is not None:
        velocity = np.asarray(track.velocity, float)
    else:
        velocity = track.states[:, 3:4] * heading_unit_vector(track.states[:, 2])
    return velocity * track.valid[:, None]


def ground_points(scene: Scene) -> tuple[np.ndarray, np.ndarray] | None:
    """(x-y [M, 2], z [M]) of every map point with a recorded elevation."""
    features = [
        feature for feature in scene.map_features if feature.z is not None and len(feature.points)
    ]
    if not features:
        return None
    return (
        np.concatenate([np.asarray(feature.points, float).reshape(-1, 2) for feature in features]),
        np.concatenate([np.asarray(feature.z, float) for feature in features]),
    )


def map_segments(
    scene: Scene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(points [C, P, 3], directions [C, P, 3], counts [C], types [C], signals [C, T]), P =
    SEGMENT_POINTS: every map feature cut as MTR cuts it (split where points are more than 1 m
    apart, then P points)."""
    frames = len(scene.timestamps)
    phases = _signal_frames(scene)
    pts, dirs, counts, types, sigs = [], [], [], [], []
    for feature in scene.map_features:
        xy = np.asarray(feature.points, float).reshape(-1, 2)
        if not len(xy):
            continue
        elevation = np.zeros(len(xy)) if feature.z is None else np.asarray(feature.z, float)
        points = np.column_stack((xy, elevation))
        kind = FeatureKind(feature.kind).value
        directions = np.zeros_like(points) if kind == "stop_sign" else _direction(points)
        gaps = np.linalg.norm(np.diff(xy, axis=0), axis=1) > BREAK_M
        bounds = np.r_[0, np.flatnonzero(gaps) + 1, len(points)]
        type_code = map_type(feature)
        signal = phases.get(feature.feature_id, np.zeros(frames, np.uint8))
        for piece_start, piece_end in pairwise(bounds):
            for first in range(piece_start, piece_end, SEGMENT_POINTS):
                last = min(first + SEGMENT_POINTS, piece_end)
                segment, segment_directions = (
                    np.zeros((SEGMENT_POINTS, 3)),
                    np.zeros((SEGMENT_POINTS, 3)),
                )
                segment[: last - first], segment_directions[: last - first] = (
                    points[first:last],
                    directions[first:last],
                )
                pts.append(segment)
                dirs.append(segment_directions)
                counts.append(last - first)
                types.append(type_code)
                sigs.append(signal)
    if not pts:
        return (
            np.zeros((0, SEGMENT_POINTS, 3)),
            np.zeros((0, SEGMENT_POINTS, 3)),
            np.zeros(0, int),
            np.zeros(0, int),
            np.zeros((0, frames), np.uint8),
        )
    return np.stack(pts), np.stack(dirs), np.array(counts), np.array(types), np.stack(sigs)


def _signal_frames(scene: Scene) -> dict:
    """lane id -> [T] phase code at every frame (last report held)."""
    frames = len(scene.timestamps)
    from ....core.types import LaneSignal

    out = {}
    for signal in scene.signals:
        held, last = np.zeros(frames, np.uint8), 0
        for frame in range(frames):
            code = (
                PHASE[LaneSignal(int(signal.states[frame])).phase]
                if frame < len(signal.states)
                else 0
            )
            last = code if code else last
            held[frame] = last
        out[signal.lane_id] = held
    return out


def _direction(points: np.ndarray) -> np.ndarray:
    """MTR's get_polyline_dir: unit vector from the previous point (0 for the first)."""
    previous = np.roll(points, 1, axis=0)
    previous[0] = points[0]
    diff = points - previous
    return diff / np.clip(
        np.linalg.norm(diff, axis=-1, keepdims=True), DIRECTION_MIN_NORM, DIRECTION_MAX_NORM
    )


def map_type(feature: MapFeature) -> int:
    """The feature's MAP_TYPES code (a lane, road line or road edge without a subtype: its
    undefined or unknown one)."""
    kind = FeatureKind(feature.kind).value
    subtype = {
        "lane": feature.lane_type,
        "road_line": feature.line_type,
        "road_edge": feature.edge_type,
    }.get(kind)
    if kind == "lane":
        subtype = LaneType(subtype) if subtype is not None else LaneType.UNDEFINED
    elif kind == "road_line":
        subtype = RoadLineType(subtype) if subtype is not None else RoadLineType.UNKNOWN
    elif kind == "road_edge":
        subtype = RoadEdgeType(subtype) if subtype is not None else RoadEdgeType.UNKNOWN
    return MAP_TYPES[(kind, subtype)]


# ---------------------------------------------------------------- stored scenes as one batch
def collate(scenes: list[dict]) -> dict:
    """Scene dicts of `encode_scene` padded into one batch of tensors."""
    import torch

    return {key: torch.from_numpy(value) for key, value in pad(scenes).items()}


def pad(scenes: list[dict]) -> dict:
    """Scene dicts (`encode_scene`, or views of stored scenes) padded into one batch of LAYOUT
    arrays (numpy)."""
    most_agents = max(len(scene["agents"]) for scene in scenes)
    most_frames = max(scene["agents"].shape[1] for scene in scenes)
    most_segments = max(max(len(scene["map_pts"]) for scene in scenes), 1)
    out = {}
    for key, (owner, shape, dtype) in LAYOUT.items():
        rest = tuple(most_frames if size == "T" else size for size in shape)
        array = np.zeros(
            (len(scenes), most_agents if owner == "agent" else most_segments, *rest), dtype
        )
        for index, scene in enumerate(scenes):
            value = scene[key]
            array[(index, *(slice(size) for size in value.shape))] = value
        out[key] = array
    return out


def pad_frames(array: np.ndarray, frames: int) -> np.ndarray:
    """A FRAME_KEYS array [N, T, ...] padded with zeros to `frames` frames (a few scenes have 92
    frames, most 91: stored together, every scene gets the most)."""
    return np.pad(array, [(0, 0), (0, frames - array.shape[1])] + [(0, 0)] * (array.ndim - 2))


# ---------------------------------------------------------------- assembly (torch)
def assemble(
    batch: dict, center: torch.Tensor, anchor: torch.Tensor, feature_cfg: FeatureConfig
) -> dict:
    """Model inputs of road users `center` [B] (agent indices) at frames `anchor` [B] from padded
    scene tensors [B, ...] (`collate`), with the context sizes of `feature_cfg`: see the module
    docstring."""
    frame = _center_frame(batch, center, anchor)
    motion, seen, positions = _history_motion(batch, frame, feature_cfg)
    nbr, order, taken = _nearest_neighbours(batch, frame, motion, seen, positions, feature_cfg)
    fut, agent_fut, agent_fut_valid = _futures(batch, frame, order, taken, feature_cfg)
    map_input, msig = _nearest_map(batch, frame, feature_cfg)
    own = batch["agent_static"][frame.batch_rows, center]
    return {
        "hist": motion[frame.batch_rows, center],
        "nbr": nbr,
        "map": map_input,
        "msig": msig.float(),
        "fut": fut,
        "typ": own[:, STATIC_TYPE],
        "size": own[:, STATIC_LENGTH:STATIC_SDC],  # length, width, height
        "sdc": own[:, STATIC_SDC],
        "agent_fut": agent_fut,  # [B, 1 + N, F, 4]: the center, then nbr's road users
        "agent_fut_valid": agent_fut_valid,
    }


def _center_frame(batch: dict, center: torch.Tensor, anchor: torch.Tensor) -> CenterFrame:
    """The center's frame at the anchor, per sample."""
    import torch

    agents = batch["agents"]
    batch_rows = torch.arange(batch["agent_valid"].shape[0], device=agents.device)
    pose = agents[batch_rows, center, anchor]  # [B, 4]
    origin, h0 = pose[:, :2], pose[:, 2]
    cos, sin = torch.cos(-h0), torch.sin(-h0)
    z0 = batch["agent_z"][batch_rows, center, anchor]  # [B]
    return CenterFrame(center, anchor, batch_rows, origin, h0, cos, sin, z0)


class CenterFrame(NamedTuple):
    """The frame inputs are assembled in, per sample [B]: the center (its agent index) and the
    anchor frame, its batch row, the center's position (origin) and heading (h0) at the anchor,
    the rotation into its frame (cos and sin of -h0), the center's elevation at the anchor (z0,
    elevations are relative to it)."""

    center: torch.Tensor
    anchor: torch.Tensor
    batch_rows: torch.Tensor
    origin: torch.Tensor
    h0: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    z0: torch.Tensor


def _history_motion(
    batch: dict, frame: CenterFrame, feature_cfg: FeatureConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The motion of every road user over the history, in the center's frame: ([B, A, H, 11] as
    hist, [B, A, H] where seen, positions [B, A, H, 2])."""
    import torch

    agents, valid = batch["agents"], batch["agent_valid"]
    dev = agents.device
    batch_size, agent_count, frame_count = valid.shape
    anchor, history = frame.anchor, feature_cfg.history
    origin, h0, cos, sin, z0 = frame.origin, frame.h0, frame.cos, frame.sin, frame.z0
    frames = (anchor[:, None] + torch.arange(-history + 1, 1, device=dev)).clamp(0, frame_count - 1)
    history_index = frames[:, None, :].expand(batch_size, agent_count, history)
    states = torch.gather(  # [B, A, H, 4]
        agents, 2, history_index[..., None].expand(-1, -1, -1, STATE_SIZE)
    )
    seen = torch.gather(valid, 2, history_index) & (
        anchor[:, None, None] - history + 1 + torch.arange(history, device=dev) >= 0
    )
    positions = _rotate(
        states[..., :2] - origin[:, None, None], cos[:, None, None], sin[:, None, None]
    )
    headings = states[..., 2] - h0[:, None, None]
    speed = states[..., 3]
    scene_velocity = torch.gather(
        batch["agent_vel"], 2, history_index[..., None].expand(-1, -1, -1, 2)
    )
    # the recorded velocity, in the center's frame
    velocity = _rotate(scene_velocity, cos[:, None, None], sin[:, None, None])
    acceleration = (velocity - torch.roll(velocity, 1, dims=2)) / FRAME_S
    acceleration[:, :, 1:] *= seen[:, :, :-1, None]  # 0 after a gap (MTR: the speed / 0.1 s there)
    acceleration[:, :, 0] = acceleration[:, :, 1]
    elevation = torch.gather(batch["agent_z"], 2, history_index) - z0[:, None, None]
    seen_mask = seen.float()[..., None]
    motion = (
        torch.cat(
            (
                positions,
                torch.cos(headings)[..., None],
                torch.sin(headings)[..., None],
                speed[..., None],
                seen_mask,
                velocity,
                acceleration,
                elevation[..., None],
            ),
            -1,
        )
        * seen_mask
    )  # [B, A, H, 11]
    return motion, seen, positions


def _rotate(xy: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate [..., 2] by -heading given cos and sin broadcast to xy[..., 0]."""
    import torch

    return torch.stack(
        (xy[..., 0] * cos - xy[..., 1] * sin, xy[..., 0] * sin + xy[..., 1] * cos), -1
    )


def _nearest_neighbours(
    batch: dict,
    frame: CenterFrame,
    motion: torch.Tensor,
    seen: torch.Tensor,
    positions: torch.Tensor,
    feature_cfg: FeatureConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The nearest road users seen in the history (as MTR; by their last position seen), the
    center excluded: (nbr [B, N, H, 16], their agent indices [B, k], whether each is one)."""
    import torch

    center, neighbours = frame.center, feature_cfg.neighbours
    static = batch["agent_static"]
    batch_size, agent_count, history = seen.shape
    dev = batch["agents"].device
    here = seen.any(2) & (torch.arange(agent_count, device=dev)[None] != center[:, None])
    last_seen = (seen * torch.arange(1, history + 1, device=dev)).argmax(2)  # [B, A]
    last_position = torch.gather(positions, 2, last_seen[..., None, None].expand(-1, -1, 1, 2))[
        :, :, 0
    ]
    distance = last_position.norm(dim=-1).masked_fill(~here, float("inf"))
    taken_count = min(neighbours, agent_count)
    order = distance.argsort(dim=1, stable=True)[:, :taken_count]
    taken = torch.gather(here, 1, order)
    taken_static = torch.gather(
        static, 1, order[..., None].expand(-1, -1, STATIC_SIZE)
    )  # [B, k, 5]
    taken_motion = torch.gather(
        motion, 1, order[..., None, None].expand(-1, -1, history, HIST_SIZE)
    )
    per_frame = taken_static[:, :, None, :].expand(-1, -1, history, -1)  # every frame, seen or not
    nbr = torch.cat(
        (
            taken_motion[..., :HIST_VELOCITY],  # position, heading, speed, valid
            per_frame[..., STATIC_TYPE:STATIC_HEIGHT],  # type, length, width
            taken_motion[..., HIST_VELOCITY:HIST_ELEVATION],  # velocity, acceleration
            per_frame[..., STATIC_HEIGHT:STATIC_SIZE],  # height, is SDC
            taken_motion[..., HIST_ELEVATION:],
        ),
        -1,
    )
    nbr = nbr * taken[..., None, None]
    if taken_count < neighbours:
        nbr = torch.cat(
            (nbr, nbr.new_zeros(batch_size, neighbours - taken_count, history, NBR_SIZE)), 1
        )
    return nbr, order, taken


def _futures(
    batch: dict,
    frame: CenterFrame,
    order: torch.Tensor,
    taken: torch.Tensor,
    feature_cfg: FeatureConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(fut [B, F, 3]: the center's future x, y, speed; agent_fut [B, 1 + N, F, 4] and its
    validity: the futures of the center and its neighbours, x, y, vx, vy, the dense future
    prediction's targets as MTR, valid where recorded)."""
    import torch

    center, anchor = frame.center, frame.anchor
    future, neighbours = feature_cfg.future, feature_cfg.neighbours
    agents, valid = batch["agents"], batch["agent_valid"]
    dev = agents.device
    batch_size, _, frame_count = valid.shape
    batch_rows, origin, cos, sin = frame.batch_rows, frame.origin, frame.cos, frame.sin
    taken_count = order.shape[1]
    future_frames = (anchor[:, None] + torch.arange(1, future + 1, device=dev)).clamp(
        max=frame_count - 1
    )
    fut_state = agents[batch_rows[:, None], center[:, None], future_frames]  # [B, F, 4]
    future_xy = _rotate(fut_state[..., :2] - origin[:, None], cos[:, None], sin[:, None])

    future_agents = torch.cat((center[:, None], order), 1)  # [B, 1 + k]
    future_index = future_frames[:, None, :].expand(-1, future_agents.shape[1], -1)
    future_states = torch.gather(
        agents[batch_rows[:, None], future_agents],
        2,
        future_index[..., None].expand(-1, -1, -1, STATE_SIZE),
    )
    future_velocity = torch.gather(
        batch["agent_vel"][batch_rows[:, None], future_agents],
        2,
        future_index[..., None].expand(-1, -1, -1, 2),
    )
    in_scene = (anchor[:, None] + torch.arange(1, future + 1, device=dev)) < frame_count  # [B, F]
    agent_fut_valid = (
        torch.gather(valid[batch_rows[:, None], future_agents], 2, future_index) & in_scene[:, None]
    )
    agent_fut_valid[:, 1:] &= taken[..., None]
    agent_fut = (
        torch.cat(
            (
                _rotate(
                    future_states[..., :2] - origin[:, None, None],
                    cos[:, None, None],
                    sin[:, None, None],
                ),
                _rotate(future_velocity, cos[:, None, None], sin[:, None, None]),
            ),
            -1,
        )
        * agent_fut_valid[..., None]
    )
    if taken_count < neighbours:
        agent_fut = torch.cat(
            (
                agent_fut,
                agent_fut.new_zeros(batch_size, neighbours - taken_count, future, AGENT_FUT_SIZE),
            ),
            1,
        )
        agent_fut_valid = torch.cat(
            (
                agent_fut_valid,
                agent_fut_valid.new_zeros(batch_size, neighbours - taken_count, future),
            ),
            1,
        )
    fut = torch.cat((future_xy, fut_state[..., 3:4]), -1)  # x, y, speed
    return fut, agent_fut, agent_fut_valid


def _nearest_map(
    batch: dict, frame: CenterFrame, feature_cfg: FeatureConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """The map segments nearest the point map_offset_m ahead, in the center's frame: (map
    [B, P, Q, 9], msig [B, P, 9]: their signal phases at the anchor and every 1 s after)."""
    import torch

    anchor = frame.anchor
    map_chunks, map_offset_m = feature_cfg.map_chunks, feature_cfg.map_offset_m
    map_points, point_counts = batch["map_pts"], batch["map_n"].long()  # [B, C, Q, 3], [B, C]
    dev = batch["agents"].device
    batch_size, _, frame_count = batch["agent_valid"].shape
    origin, cos, sin, z0 = frame.origin, frame.cos, frame.sin, frame.z0
    segment_points = map_points.shape[2]
    on = torch.arange(segment_points, device=dev)[None, None] < point_counts[..., None]  # [B, C, Q]
    local = _rotate(
        map_points[..., :2] - origin[:, None, None], cos[:, None, None], sin[:, None, None]
    )
    centre = (local * on[..., None]).sum(2) / point_counts.clamp(min=1)[..., None]
    centre_distance = (centre - centre.new_tensor([map_offset_m, 0.0])).norm(dim=-1)
    centre_distance = centre_distance.masked_fill(point_counts == 0, float("inf"))
    chunk_count = min(map_chunks, centre_distance.shape[1])
    pick = centre_distance.argsort(dim=1, stable=True)[:, :chunk_count]
    has = torch.gather(point_counts, 1, pick) > 0
    local = torch.gather(local, 1, pick[..., None, None].expand(-1, -1, segment_points, 2))
    on = torch.gather(on, 1, pick[..., None].expand(-1, -1, segment_points)) & has[..., None]
    map_elevation = (
        torch.gather(map_points[..., 2], 1, pick[..., None].expand(-1, -1, segment_points))
        - z0[:, None, None]
    )
    directions_3d = torch.gather(
        batch["map_dir"], 1, pick[..., None, None].expand(-1, -1, segment_points, 3)
    )
    directions = _rotate(directions_3d[..., :2], cos[:, None, None], sin[:, None, None])
    types = torch.gather(batch["map_type"].long(), 1, pick)
    previous = torch.roll(local, 1, dims=2)
    previous[:, :, 0] = local[:, :, 0]
    type_channel = types[..., None, None].expand(-1, -1, segment_points, 1).float()
    map_input = (
        torch.cat(
            (
                local,
                type_channel,
                directions,
                previous,
                map_elevation[..., None],
                directions_3d[..., 2:],
            ),
            -1,
        )
        * on[..., None]
    )
    sig_frames = (anchor[:, None] + FRAMES_PER_S * torch.arange(SIGNAL_TIMES, device=dev)).clamp(
        max=frame_count - 1
    )  # every 1 s
    sig_all = torch.gather(batch["map_sig"].long(), 1, pick[..., None].expand(-1, -1, frame_count))
    msig = (
        torch.gather(sig_all, 2, sig_frames[:, None, :].expand(-1, chunk_count, -1))
        * has[..., None]
    )
    if chunk_count < map_chunks:
        map_input = torch.cat(
            (
                map_input,
                map_input.new_zeros(batch_size, map_chunks - chunk_count, segment_points, MAP_SIZE),
            ),
            1,
        )
        msig = torch.cat(
            (msig, msig.new_zeros(batch_size, map_chunks - chunk_count, SIGNAL_TIMES)), 1
        )
    return map_input, msig
