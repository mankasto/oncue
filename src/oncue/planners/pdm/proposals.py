"""PDM-Closed's proposals: every path (route centerline and offsets) driven by every IDM policy.

Ported from tuplan_garage b51d5d04 (Apache-2.0, LICENSE in this folder), planning/simulation/
planner/pdm_planner/, and nuplan-devkit e9241677 where named.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.interpolate import interp1d
from shapely.ops import substring

from ...core.tracker import STATE_SE2, StateIndex
from ...core.types import Scene
from ...utils.geometry import thin_path, wrap_angle

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence

    from .planner import Observation

# nuPlan's Chrysler Pacifica axles (vehicle_parameters.py:125-138); the box is the observed one
PACIFICA_WHEEL_BASE_M = 3.089
PACIFICA_REAR_AXLE_TO_CENTER_M = (4.049 + 1.127) / 2 - 1.127  # half length - rear length: 1.461
CENTERLINE_STEP_M = 0.5  # the centerline's poses this far apart, each heading toward the next
CLIP_MIN_PROGRESS_M = 1e-5  # pdm_path.py:72, interpolation starts just after the path's start
ACCELERATION_EXPONENT = 10  # batch_idm_policy.py:181 (nuPlan's IDMPolicy: 4)
LEADING_AGENT_UPDATE_RATE = 2  # pdm_generator.py:46, the lead re-searched every 2 steps
FIRST_STEP = 1  # the first step unrolled: its lead is searched too
RED_LIGHT_TOKEN = "red_light"  # pdm_observation.py:68

# ---------------------------------------------------------------- states and boxes
# `StateIndex` and its slices live in core.tracker (pdm_enums.py:4-17, 91-107)
POINT = slice(StateIndex.X, StateIndex.Y + 1)  # pdm_enums.py:85-89


class BoxPoint(IntEnum):
    """Points of a box coordinate array (pdm_enums.py:140-147, BBCoordsIndex)."""

    FRONT_LEFT = 0
    REAR_LEFT = 1
    REAR_RIGHT = 2
    FRONT_RIGHT = 3
    CENTER = 4


CORNERS = int(BoxPoint.CENTER)  # a box's four corners come before its centre


@dataclass(frozen=True)
class Vehicle:
    """The ego's box (length, width) and axles: nuPlan's VehicleParameters, as far as PDM reads."""

    length: float
    width: float
    wheel_base: float = PACIFICA_WHEEL_BASE_M
    rear_axle_to_center: float = PACIFICA_REAR_AXLE_TO_CENTER_M

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2


def translate_lon_and_lat(
    centers: np.ndarray, headings: np.ndarray, lon: float, lat: float
) -> np.ndarray:
    """Points moved `lon` along and `lat` across their headings (pdm_geometry_utils.py:39-61)."""
    half_pi = np.pi / 2.0
    translation = np.stack(
        [
            (lat * np.cos(headings + half_pi)) + (lon * np.cos(headings)),
            (lat * np.sin(headings + half_pi)) + (lon * np.sin(headings)),
        ],
        axis=-1,
    )
    return centers + translation


def box_points(
    centers: np.ndarray, headings: np.ndarray, half_length: float, half_width: float
) -> np.ndarray:
    """[..., 5, 2] corners and centre of boxes (pdm_array_representation.py:195-212).

    Not `utils.geometry.box_corners`: nuPlan's formula differs from it in the last bit.
    """
    points = np.zeros((*np.shape(headings), len(BoxPoint), 2), dtype=np.float64)
    points[..., BoxPoint.CENTER, :] = centers
    for point, lon, lat in (
        (BoxPoint.FRONT_LEFT, half_length, half_width),
        (BoxPoint.FRONT_RIGHT, half_length, -half_width),
        (BoxPoint.REAR_LEFT, -half_length, half_width),
        (BoxPoint.REAR_RIGHT, -half_length, -half_width),
    ):
        points[..., point, :] = translate_lon_and_lat(centers, headings, lon, lat)
    return points


def box_coordinates(states: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """[..., 5, 2] ego boxes at rear-axle states (pdm_array_representation.py:165-214)."""
    headings = states[..., StateIndex.HEADING]
    cos, sin = np.cos(headings), np.sin(headings)
    rear_axle_to_center_translate = np.stack(
        [vehicle.rear_axle_to_center * cos, vehicle.rear_axle_to_center * sin], axis=-1
    )
    centers = states[..., POINT] + rear_axle_to_center_translate
    return box_points(centers, headings, vehicle.half_length, vehicle.half_width)


def box_polygons(coordinates: np.ndarray) -> np.ndarray:
    """Shapely polygons of box coordinates [..., 5, 2] (pdm_array_representation.py:217-234)."""
    exterior = coordinates.copy()
    exterior[..., BoxPoint.CENTER, :] = exterior[..., BoxPoint.FRONT_LEFT, :]
    return shapely.polygons(exterior)


def from_center(center_state: np.ndarray, dynamics: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """[11] rear-axle ego state of a box-centre state (pdm_array_representation.py:72-100).

    `dynamics` [11] supplies the other columns, carried by the planner between steps.
    """
    x, y, heading, speed = (float(value) for value in center_state)
    state = np.array(dynamics, dtype=np.float64)
    state[StateIndex.X] = x - vehicle.rear_axle_to_center * np.cos(heading)
    state[StateIndex.Y] = y - vehicle.rear_axle_to_center * np.sin(heading)
    state[StateIndex.HEADING] = heading
    state[StateIndex.VELOCITY_X] = speed
    state[StateIndex.VELOCITY_Y] = 0.0  # the kinematic bicycle has no lateral velocity
    return state


def to_center(state: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    """[4] the box-centre state (x, y, heading, speed) of an ego state at the rear axle."""
    heading = state[StateIndex.HEADING]
    return np.array(
        [
            state[StateIndex.X] + vehicle.rear_axle_to_center * np.cos(heading),
            state[StateIndex.Y] + vehicle.rear_axle_to_center * np.sin(heading),
            heading,
            state[StateIndex.VELOCITY_X],
        ]
    )


# ---------------------------------------------------------------- the centerline and paths
def route_centerline(scene: Scene, path: shapely.LineString, vehicle: Vehicle) -> np.ndarray:
    """[N, 3] poses every CENTERLINE_STEP_M along `path`, with the logged history before it.

    Unlike upstream's lane centre lines (abstract_pdm_planner.py:91-117), it follows where the
    logged ego drove, extended back so the rear axle projects onto it.
    """
    ego, anchor = scene.ego, scene.anchor
    frames = np.flatnonzero(ego.valid)
    history = ego.states[frames[frames < anchor], :2]
    points = np.vstack([history, np.asarray(path.coords)])
    points = points[thin_path(points, CENTERLINE_STEP_M)]
    line = shapely.LineString(points)
    behind = line.project(shapely.Point(*ego.states[anchor, :2]))
    if behind < vehicle.length:
        back = points[0] - points[1]
        back /= np.linalg.norm(back)
        points = np.vstack([points[0] + (vehicle.length - behind) * back, points])
        line = shapely.LineString(points)
    stations = np.append(np.arange(0.0, line.length, CENTERLINE_STEP_M), line.length)
    xy = shapely.get_coordinates(shapely.line_interpolate_point(line, stations))
    return headed_poses(xy)


def headed_poses(points: np.ndarray) -> np.ndarray:
    """[N, 3] the points with headings toward the next point (the last keeps the step into it)."""
    steps = np.diff(points, axis=0)
    headings = np.arctan2(steps[:, 1], steps[:, 0])
    return np.column_stack((points, np.append(headings, headings[-1])))


def calculate_progress(poses: np.ndarray) -> np.ndarray:
    """[N]: the cumulative distance along poses [N, >= 2] (pdm_geometry_utils.py:64-78)."""
    x_diff, y_diff = np.diff(poses[:, 0]), np.diff(poses[:, 1])
    points_diff = np.concatenate(([x_diff], [y_diff]), axis=0, dtype=np.float64)
    progress_diff = np.append(0.0, np.linalg.norm(points_diff, axis=0))
    return np.cumsum(progress_diff, dtype=np.float64)


def parallel_discrete_path(poses: np.ndarray, offset: float) -> np.ndarray:
    """[N, 3]: poses moved `offset` to their left (pdm_geometry_utils.py:21-36)."""
    theta = poses[:, 2] + np.pi / 2
    return np.column_stack(
        (poses[:, 0] + np.cos(theta) * offset, poses[:, 1] + np.sin(theta) * offset, poses[:, 2])
    )


def ramped_offset_path(
    poses: np.ndarray,
    progress: np.ndarray,
    start: float,
    offset_from: float,
    offset_to: float,
    ramp_m: float,
) -> np.ndarray:
    """[N, 3] poses moved left by an offset that eases from `offset_from` to `offset_to`.

    The offset follows a quintic smoothstep over `ramp_m` from progress `start`, and headings turn
    with its slope. Upstream's offset paths start fully offset instead.
    """
    share = np.clip((progress - start) / ramp_m, 0.0, 1.0)
    change = offset_to - offset_from
    offset = offset_from + change * share**3 * (10 - 15 * share + 6 * share**2)
    slope = change * 30 * share**2 * (1 - share) ** 2 / ramp_m  # offset per metre of progress
    theta = poses[:, 2] + np.pi / 2
    return np.column_stack(
        (
            poses[:, 0] + np.cos(theta) * offset,
            poses[:, 1] + np.sin(theta) * offset,
            wrap_angle(poses[:, 2] + np.arctan(slope)),
        )
    )


class Path:
    """Poses [N, 3] interpolated linearly by progress, headings unwrapped (pdm_path.py:24-101)."""

    def __init__(self, poses: np.ndarray):
        self.discrete_path = np.array(poses, dtype=np.float64)  # as given, headings wrapped
        self.poses = self.discrete_path.copy()
        self.poses[:, 2] = np.unwrap(self.poses[:, 2], axis=0)
        self.progress = calculate_progress(poses)
        self.linestring = shapely.linestrings(self.poses[:, :2])
        self._interpolator = interp1d(self.progress, self.poses, axis=0)

    @property
    def length(self) -> float:
        return float(self.progress[-1])

    def project(self, points: np.ndarray) -> np.ndarray:
        """Progress of the points nearest each of `points` (shapely geometries)."""
        return self.linestring.project(points)

    def interpolate(self, distances: np.ndarray | float) -> np.ndarray:
        """[..., 3] poses at `distances` along the path (clipped to it)."""
        clipped_distances = np.clip(distances, CLIP_MIN_PROGRESS_M, self.length)
        poses = self._interpolator(clipped_distances)
        poses[..., 2] = wrap_angle(poses[..., 2])
        poses[np.isnan(poses)] = 0.0  # upstream: a zero-length step interpolates to NaN
        return poses

    def substring(self, start_distance: float, end_distance: float) -> shapely.LineString:
        """The path between two distances: its own points, or shapely's substring if fewer."""
        start_distance = np.clip(start_distance, 0.0, self.length)
        end_distance = np.clip(end_distance, 0.0, self.length)
        in_interval = np.logical_and(start_distance <= self.progress, self.progress <= end_distance)
        coordinates = self.poses[in_interval, :2]
        if len(coordinates) > 1:
            return shapely.LineString(coordinates)
        return substring(self.linestring, start_distance, end_distance)


# ---------------------------------------------------------------- the proposals
class IDMStateIndex(IntEnum):
    """pdm_enums.py:125-129 (StateIDMIndex)."""

    PROGRESS = 0
    VELOCITY = 1


class LeadingAgentIndex(IntEnum):
    """pdm_enums.py:132-137."""

    PROGRESS = 0
    VELOCITY = 1
    LENGTH_REAR = 2


class IDMPolicies:
    """IDM policies, one per speed-limit fraction (batch_idm_policy.py:12-215, BatchIDMPolicy)."""

    def __init__(
        self,
        speed_limit_fractions: Sequence[float],
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
    ):
        self.num_policies = len(speed_limit_fractions)
        self._speed_limit_fractions = np.array(speed_limit_fractions, dtype=np.float64)
        self._min_gap_to_lead_agent = np.full(self.num_policies, min_gap_to_lead_agent)
        self._headway_time = np.full(self.num_policies, headway_time)
        self._accel_max = np.full(self.num_policies, accel_max)
        self._decel_max = np.full(self.num_policies, decel_max)
        self._target_velocities = np.zeros(self.num_policies, dtype=np.float64)

    @property
    def max_target_velocity(self) -> float:
        return float(np.max(self._target_velocities))

    def update(self, speed_limit_mps: float) -> None:
        """Target velocities from the current speed limit (batch_idm_policy.py:124-135)."""
        self._target_velocities = self._speed_limit_fractions * speed_limit_mps

    def propagate(
        self,
        previous_idm_states: np.ndarray,
        leading_agent_states: np.ndarray,
        longitudinal_indices: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """[K, 2]: progress and velocity after one Euler step of `dt`
        (batch_idm_policy.py:137-215)."""
        x_agent = previous_idm_states[:, IDMStateIndex.PROGRESS]
        v_agent = previous_idm_states[:, IDMStateIndex.VELOCITY]
        x_lead = leading_agent_states[:, LeadingAgentIndex.PROGRESS]
        v_lead = leading_agent_states[:, LeadingAgentIndex.VELOCITY]
        l_r_lead = leading_agent_states[:, LeadingAgentIndex.LENGTH_REAR]
        target_velocity = self._target_velocities[longitudinal_indices]
        min_gap_to_lead_agent = self._min_gap_to_lead_agent[longitudinal_indices]
        headway_time = self._headway_time[longitudinal_indices]
        accel_max = self._accel_max[longitudinal_indices]
        decel_max = self._decel_max[longitudinal_indices]
        s_star = (
            min_gap_to_lead_agent
            + v_agent * headway_time
            + (v_agent * (v_agent - v_lead)) / (2 * np.sqrt(accel_max * decel_max))
        )
        s_alpha = np.maximum(x_lead - x_agent - l_r_lead, min_gap_to_lead_agent)  # no zero gap
        x_agent_dot = v_agent
        v_agent_dot = accel_max * (
            1 - (v_agent / target_velocity) ** ACCELERATION_EXPONENT - (s_star / s_alpha) ** 2
        )
        v_agent_dot = np.clip(v_agent_dot, -decel_max, accel_max)
        next_idm_states = np.zeros(
            (len(longitudinal_indices), len(IDMStateIndex)), dtype=np.float64
        )
        next_idm_states[:, IDMStateIndex.PROGRESS] = x_agent + dt * x_agent_dot
        next_idm_states[:, IDMStateIndex.VELOCITY] = v_agent + dt * v_agent_dot
        return next_idm_states


@dataclass(frozen=True)
class Proposal:
    """One lateral path driven by one IDM policy (pdm_proposal.py:12-29)."""

    lateral_index: int
    longitudinal_index: int
    path: Path


def proposal_grid(paths: list[Path], policies: IDMPolicies) -> list[Proposal]:
    """Every path with every policy, the path index major (pdm_proposal.py:35-63)."""
    return [
        Proposal(lateral_index, longitudinal_index, path)
        for lateral_index, path in enumerate(paths)
        for longitudinal_index in range(policies.num_policies)
    ]


class ProposalGenerator:
    """Unrolls the proposals in the forecast (pdm_generator.py:39-432), without the 8 s output.

    `trajectory_poses` still sets the IDM corridor's reach.
    """

    def __init__(self, proposal_poses: int, trajectory_poses: int, interval_s: float):
        self._proposal_poses = proposal_poses
        self._trajectory_poses = trajectory_poses
        self._interval_s = interval_s

    def generate(
        self,
        rear_axle_state: np.ndarray,
        vehicle: Vehicle,
        observation: Observation,
        proposals: list[Proposal],
        policies: IDMPolicies,
    ) -> np.ndarray:
        """[P, proposal_poses + 1, 11] rear-axle poses of the proposals, other columns 0 as
        upstream (pdm_generator.py:81-108)."""
        self._vehicle, self._observation = vehicle, observation
        self._proposals, self._policies = proposals, policies
        size = (len(proposals), self._proposal_poses + 1)
        self._state_array = np.zeros((*size, len(StateIndex)), dtype=np.float64)
        self._state_idm_array = np.zeros((*size, len(IDMStateIndex)), dtype=np.float64)
        self._leading_agent_array = np.zeros((*size, len(LeadingAgentIndex)), dtype=np.float64)
        self._driving_corridor_cache: dict[int, shapely.Polygon] = {}
        lateral_batches: dict[int, list[int]] = {}
        for proposal_index, proposal in enumerate(proposals):
            lateral_batches.setdefault(proposal.lateral_index, []).append(proposal_index)
        for batch in lateral_batches.values():
            self._initialize_states(batch, rear_axle_state)
            for time_index in range(1, self._proposal_poses + 1):
                self._update_leading_agents(batch, time_index)
                self._update_idm_states(batch, time_index)
                self._update_states_se2(batch, time_index)
        return self._state_array

    def _initialize_states(self, batch: list[int], rear_axle_state: np.ndarray) -> None:
        """The ego's progress, speed and pose on the path at step 0 (pdm_generator.py:198-225)."""
        path = self._proposals[batch[0]].path
        ego_progress = path.project(shapely.Point(*rear_axle_state[:2]))
        self._state_idm_array[batch, 0, IDMStateIndex.PROGRESS] = ego_progress
        self._state_idm_array[batch, 0, IDMStateIndex.VELOCITY] = rear_axle_state[3]
        self._state_array[batch, 0, STATE_SE2] = path.interpolate([ego_progress])[0]

    def _update_states_se2(self, batch: list[int], time_index: int) -> None:
        """The poses on the path at the new progress (pdm_generator.py:227-243)."""
        progress = self._state_idm_array[batch, time_index, IDMStateIndex.PROGRESS]
        self._state_array[batch, time_index, STATE_SE2] = self._proposals[
            batch[0]
        ].path.interpolate(progress)

    def _update_idm_states(self, batch: list[int], time_index: int) -> None:
        """One IDM step of each policy (pdm_generator.py:245-262)."""
        longitudinal_indices = [self._proposals[index].longitudinal_index for index in batch]
        self._state_idm_array[batch, time_index] = self._policies.propagate(
            self._state_idm_array[batch, time_index - 1],
            self._leading_agent_array[batch, time_index],
            longitudinal_indices,
            self._interval_s,
        )

    def _update_leading_agents(self, batch: list[int], time_index: int) -> None:
        """Each proposal's lead: the nearest object or red light ahead, else the path's end.

        Searched at step 1 and every LEADING_AGENT_UPDATE_RATE steps (pdm_generator.py:264-353).
        """
        # upstream copies step 0's unset (zero) lead at step 1 too; fixed here
        if time_index != FIRST_STEP and time_index % LEADING_AGENT_UPDATE_RATE != 0:
            self._leading_agent_array[batch, time_index] = self._leading_agent_array[
                batch, time_index - 1
            ]
            return
        path = self._proposals[batch[0]].path
        occupancy = self._observation[time_index]
        leading_agent_array = np.zeros(len(LeadingAgentIndex), dtype=np.float64)
        object_progress = {
            token: path.project(occupancy[token].centroid)
            for token in occupancy.intersects(self._driving_corridor(batch[0]))
            if token not in self._observation.collided_track_ids
        }
        for proposal_index in batch:
            ego_progress = self._state_idm_array[
                proposal_index, time_index - 1, IDMStateIndex.PROGRESS
            ]
            agents_ahead = {
                token: progress
                for token, progress in object_progress.items()
                if progress > ego_progress
            }
            if agents_ahead:  # a red light, object or agent ahead
                ego_pose = self._state_array[proposal_index, time_index - 1, STATE_SE2]
                ego_polygon = box_polygons(box_coordinates(ego_pose, self._vehicle))
                relative_distances = [
                    ego_polygon.distance(occupancy[token]) for token in agents_ahead
                ]
                nearest = int(np.argmin(relative_distances))
                nearest_agent = list(agents_ahead)[nearest]
                leading_agent_array[LeadingAgentIndex.PROGRESS] = (
                    ego_progress + relative_distances[nearest]
                )
                if RED_LIGHT_TOKEN not in nearest_agent:
                    leading_agent_array[LeadingAgentIndex.VELOCITY] = self._leading_agent_velocity(
                        ego_pose[2], nearest_agent
                    )
            else:  # nothing ahead: the path's end stands
                leading_agent_array[LeadingAgentIndex.PROGRESS] = self._proposals[
                    proposal_index
                ].path.length
                leading_agent_array[LeadingAgentIndex.LENGTH_REAR] = self._vehicle.length / 2
            # upstream quirk: the same vector goes on to the next proposal
            self._leading_agent_array[proposal_index, time_index] = leading_agent_array

    def _leading_agent_velocity(self, ego_heading: float, token: str) -> float:
        """An agent's speed along the ego's heading; 0 for a static object
        (pdm_generator.py:355-373)."""
        tracked = self._observation.unique_objects[token]
        if not tracked.is_agent:
            return 0.0
        return tracked.speed * np.cos(wrap_angle(tracked.heading - ego_heading))

    def _driving_corridor(self, proposal_index: int) -> shapely.Polygon:
        """The path ahead as far as the fastest policy drives in the output horizon, as wide as
        the ego, cached per path (pdm_generator.py:388-415)."""
        lateral_index = self._proposals[proposal_index].lateral_index
        if lateral_index not in self._driving_corridor_cache:
            ego_distance = self._state_idm_array[proposal_index, 0, IDMStateIndex.PROGRESS]
            trajectory_distance = (
                ego_distance
                + abs(self._policies.max_target_velocity)
                * self._trajectory_poses
                * self._interval_s
            )
            linestring_ahead = self._proposals[proposal_index].path.substring(
                ego_distance, trajectory_distance
            )
            self._driving_corridor_cache[lateral_index] = linestring_ahead.buffer(
                self._vehicle.width / 2, cap_style="square"
            )
        return self._driving_corridor_cache[lateral_index]
