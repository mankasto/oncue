"""Scene data model shared by every stage.

SI units in the scene frame, whose origin and x-axis are the ego pose at ``anchor``. Arrays are
read-only; derive a changed scene with ``Scene.with_track``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

AGENT_TYPES = ("vehicle", "pedestrian", "cyclist", "other")
STATE_SIZE = 4  # columns: x, y, heading, speed
TABLE_NDIM = 2  # per-frame arrays have one row per frame
FOOTPRINT_DIMS = 2  # a footprint: length, width
MIN_TIMESTAMPS = 2  # a scene has at least two timestamps, so `dt` is defined
# Dataclass field metadata key: the field's spelling in written files when it differs from the name.
FILE_KEY = "file_key"
ANCHOR_FILE_KEY = "current_idx"  # `Scene.anchor`'s key in scene files; kept for back-compat
# Heights (m) of road users with no logged height; the generated vehicle target has its own.
UNLOGGED_HEIGHT_M = {"vehicle": 1.6, "pedestrian": 1.7, "cyclist": 1.7, "other": 1.5}


def frozen_array(values: ArrayLike, dtype: DTypeLike = float) -> np.ndarray:
    """Return a read-only ``np.ndarray`` copy of ``values``."""
    array = np.array(values, dtype=dtype)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Track:
    track_id: str
    agent_type: str
    size: tuple[float, float]  # length, width
    states: np.ndarray  # [T, 4]
    valid: np.ndarray  # [T] bool
    height: float | None = None  # m, median of observed heights; None uses the type default
    # [T, 2] recorded velocity vector in the scene frame; None when made or replaced states
    velocity: np.ndarray | None = None
    z: np.ndarray | None = None  # [T] recorded box centre elevation (m); None: not recorded
    # [T] bool: valid states produced by `core.replay.fill_gaps` rather than logged
    filled: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Freeze every per-frame array and validate the fields."""
        states, valid = frozen_array(self.states), frozen_array(self.valid, bool)
        self._freeze_optional(states, valid)
        size = tuple(float(v) for v in self.size)
        if (
            states.ndim != TABLE_NDIM
            or states.shape[1] != STATE_SIZE
            or valid.shape != (len(states),)
        ):
            raise ValueError(f"Track {self.track_id}: expected states [T,4] and valid [T]")
        if len(size) != FOOTPRINT_DIMS or min(size) <= 0:
            raise ValueError(f"Track {self.track_id}: invalid footprint {size}")
        if self.agent_type not in AGENT_TYPES:
            raise ValueError(f"Track {self.track_id}: unknown agent type {self.agent_type}")
        if not np.isfinite(states[valid]).all() or (states[valid, 3] < 0).any():
            raise ValueError(f"Track {self.track_id}: valid states must be finite, speed >= 0")
        if self.height is not None:
            if not np.isfinite(self.height) or self.height <= 0:
                raise ValueError(f"Track {self.track_id}: invalid height {self.height}")
            object.__setattr__(self, "height", float(self.height))
        object.__setattr__(self, "track_id", str(self.track_id))
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "valid", valid)

    def _freeze_optional(self, states: np.ndarray, valid: np.ndarray) -> None:
        """Freeze ``filled``, ``velocity`` and ``z`` when given; each is checked against states."""
        if self.filled is not None:
            filled = frozen_array(self.filled, bool)
            if filled.shape != valid.shape or (filled & ~valid).any():
                raise ValueError(f"Track {self.track_id}: filled [T] must mark valid frames")
            object.__setattr__(self, "filled", filled)
        if self.velocity is not None:
            velocity = frozen_array(self.velocity)
            if velocity.shape != (len(states), 2):
                raise ValueError(f"Track {self.track_id}: expected velocity [T,2]")
            object.__setattr__(self, "velocity", velocity)
        if self.z is not None:
            z = frozen_array(self.z)
            if z.shape != (len(states),):
                raise ValueError(f"Track {self.track_id}: expected z [T]")
            object.__setattr__(self, "z", z)

    def with_states(
        self,
        states: np.ndarray,
        valid: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        z: np.ndarray | None = None,
    ) -> Track:
        """Return a copy with new ``states``; velocity, z and fill marks drop unless given."""
        valid = self.valid if valid is None else valid
        return replace(self, states=states, valid=valid, velocity=velocity, z=z, filled=None)

    def absent_from(self, frame: int) -> Track:
        """Return a copy invalid from ``frame`` on; any fill marks there are dropped too."""
        valid = self.valid.copy()
        valid[frame:] = False
        filled = None if self.filled is None else self.filled & valid
        return replace(self, valid=valid, filled=filled)

    def valid_until(self, start: int, stop: int) -> int:
        """Return the exclusive end of the contiguous valid run starting at ``start``."""
        gaps = np.flatnonzero(~self.valid[start:stop])
        return start + int(gaps[0]) if len(gaps) else stop


def height_of(track: Track) -> float:
    """Return the track's logged height in metres, or the default for its agent type."""
    return track.height if track.height is not None else UNLOGGED_HEIGHT_M[track.agent_type]


class FeatureKind(str, Enum):
    LANE = "lane"
    ROAD_LINE = "road_line"
    ROAD_EDGE = "road_edge"
    CROSSWALK = "crosswalk"
    STOP_SIGN = "stop_sign"
    SPEED_BUMP = "speed_bump"
    DRIVEWAY = "driveway"


class LaneType(IntEnum):  # WOMD LaneCenter.LaneType
    UNDEFINED = 0
    FREEWAY = 1
    SURFACE_STREET = 2
    BIKE_LANE = 3


class RoadEdgeType(IntEnum):  # WOMD RoadEdge.RoadEdgeType
    UNKNOWN = 0
    BOUNDARY = 1
    MEDIAN = 2


class RoadLineType(IntEnum):  # WOMD RoadLine.RoadLineType
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8


@dataclass(frozen=True)
class BoundarySegment:
    """Lane-index interval ``[start, end]`` bounded by another map feature."""

    feature_id: str
    start: int
    end: int


@dataclass(frozen=True)
class LaneNeighbor:
    """A lane change is possible from ``self[start:end]`` into ``neighbor[start:end]``."""

    feature_id: str
    self_start: int
    self_end: int
    neighbor_start: int
    neighbor_end: int


@dataclass(frozen=True, eq=False)
class MapFeature:
    feature_id: str
    kind: FeatureKind
    points: np.ndarray  # [N, 2]; polygons are not closed
    lane_type: LaneType | None = None
    edge_type: RoadEdgeType | None = None
    line_type: RoadLineType | None = None  # None in older exports
    entry_lanes: tuple[str, ...] = ()
    exit_lanes: tuple[str, ...] = ()
    left_boundaries: tuple[BoundarySegment, ...] = ()
    right_boundaries: tuple[BoundarySegment, ...] = ()
    left_neighbors: tuple[LaneNeighbor, ...] = ()
    right_neighbors: tuple[LaneNeighbor, ...] = ()
    controlled_lanes: tuple[str, ...] = ()  # for stop signs, the lanes they control
    speed_limit_mps: float | None = None  # posted limit on lanes, if recorded
    z: np.ndarray | None = None  # [N] point elevations (m); None: not recorded

    def __post_init__(self):
        points = frozen_array(self.points).reshape(-1, 2)
        if not np.isfinite(points).all():
            raise ValueError(f"Map feature {self.feature_id}: nonfinite points")
        object.__setattr__(self, "points", points)
        if self.z is not None:
            z = frozen_array(self.z).reshape(-1)
            if len(z) != len(points):
                raise ValueError(f"Map feature {self.feature_id}: expected z [N]")
            object.__setattr__(self, "z", z)
        object.__setattr__(self, "kind", FeatureKind(self.kind))
        if self.lane_type is not None:
            object.__setattr__(self, "lane_type", LaneType(self.lane_type))
        if self.edge_type is not None:
            object.__setattr__(self, "edge_type", RoadEdgeType(self.edge_type))
        if self.line_type is not None:
            object.__setattr__(self, "line_type", RoadLineType(self.line_type))


class LaneSignal(IntEnum):  # WOMD TrafficSignalLaneState.State
    UNKNOWN = 0
    ARROW_STOP = 1
    ARROW_CAUTION = 2
    ARROW_GO = 3
    STOP = 4
    CAUTION = 5
    GO = 6
    FLASHING_STOP = 7
    FLASHING_CAUTION = 8

    @property
    def phase(self) -> str:
        """Return ``"stop"``, ``"caution"``, ``"go"`` or ``"unknown"``."""
        if self in (LaneSignal.ARROW_STOP, LaneSignal.STOP, LaneSignal.FLASHING_STOP):
            return "stop"
        if self in (LaneSignal.ARROW_CAUTION, LaneSignal.CAUTION, LaneSignal.FLASHING_CAUTION):
            return "caution"
        if self in (LaneSignal.ARROW_GO, LaneSignal.GO):
            return "go"
        return "unknown"


@dataclass(frozen=True, eq=False)
class SignalTrack:
    """Traffic signal controlling one lane: its stop point and its state per scene frame.

    States come from the recording vehicle's perception and may lag the real light.
    """

    lane_id: str
    stop_point: tuple[float, float]
    states: np.ndarray  # [T] int, LaneSignal codes

    def __post_init__(self):
        object.__setattr__(self, "lane_id", str(self.lane_id))
        object.__setattr__(self, "stop_point", tuple(float(v) for v in self.stop_point))
        object.__setattr__(self, "states", frozen_array(self.states, np.int8))


@dataclass(frozen=True)
class SceneSource:
    """Where a scene came from; with the data root, enough to re-read the raw record.

    ``shard`` is the file name only, so a run can move between machines.
    """

    dataset: str
    dataset_version: str
    split: str
    shard: str
    record_index: int
    record_offset: int
    record_length: int
    record_sha256: str


@dataclass(frozen=True, eq=False)
class Scene:
    scene_id: str
    timestamps: np.ndarray  # [T], uniform
    anchor: int = field(metadata={FILE_KEY: ANCHOR_FILE_KEY})  # fixed anchor frame (WOMD's `now`)
    ego_id: str
    tracks: Mapping[str, Track]
    map_features: tuple[MapFeature, ...] = ()
    drivable_area: tuple[np.ndarray, ...] = ()  # nothing populates this; kept for scene files
    signals: tuple[SignalTrack, ...] = ()  # one per signal-controlled lane
    source: SceneSource | None = None
    notes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        t = frozen_array(self.timestamps)
        if (
            t.ndim != 1
            or len(t) < MIN_TIMESTAMPS
            or not np.isfinite(t).all()
            or (np.diff(t) <= 0).any()
        ):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if not 0 <= self.anchor < len(t):
            raise ValueError(f"{ANCHOR_FILE_KEY} outside the timeline")
        if self.ego_id not in self.tracks:
            raise ValueError("Ego track missing")
        for key, track in self.tracks.items():
            if key != track.track_id or len(track.states) != len(t):
                raise ValueError(f"Track {key}: id or length does not match the scene")
        for sig in self.signals:
            if len(sig.states) != len(t):
                raise ValueError(f"Signal of lane {sig.lane_id}: length does not match the scene")
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "timestamps", t)
        object.__setattr__(self, "tracks", dict(self.tracks))
        object.__setattr__(
            self, "drivable_area", tuple(frozen_array(p).reshape(-1, 2) for p in self.drivable_area)
        )

    @property
    def ego(self) -> Track:
        return self.tracks[self.ego_id]

    @property
    def dt(self) -> float:
        return float(self.timestamps[1] - self.timestamps[0])

    @property
    def world_origin(self) -> tuple[float, float, float]:
        """Return the ego pose at the anchor in WOMD world coordinates ``(x, y, heading)``."""
        if "world_origin" not in self.notes:
            raise ValueError("scene has no world origin; re-read it from its record")
        return self.notes["world_origin"]

    def others(self, *exclude: str) -> list[Track]:
        """Return the tracks other than the ego and any ``exclude`` ids."""
        skip = {self.ego_id, *exclude}
        return [t for k, t in self.tracks.items() if k not in skip]

    def with_track(self, track: Track) -> Scene:
        """Return a copy with ``track`` replacing its existing entry by ``track_id``."""
        if track.track_id not in self.tracks:
            raise ValueError(f"Unknown track {track.track_id}")
        return replace(self, tracks={**self.tracks, track.track_id: track})


@dataclass(frozen=True)
class RevealEvent:
    """Persistent reveal: occluded for a while, then visible for K frames.

    ``reveal`` is the first visible frame and ``confirmation`` the K-th, both scene frames.
    """

    occlusion_start: int
    reveal: int
    confirmation: int
