"""PDM-Closed's scorer, `Scorer.score`: nuPlan's closed-loop metrics on the simulated proposals.

Ported from tuplan_garage b51d5d04 (Apache-2.0, LICENSE in this folder), planning/simulation/
planner/pdm_planner/scoring/, and nuplan-devkit e9241677 where named.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.signal import savgol_filter

from .proposals import (
    CORNERS,
    RED_LIGHT_TOKEN,
    STATE_SE2,
    BoxPoint,
    Path,
    StateIndex,
    Vehicle,
    box_coordinates,
    box_polygons,
)

if TYPE_CHECKING:  # annotations only
    import numpy.typing as npt

    from .planner import Observation, RouteMap, TrackedObject

# pdm_scorer.py:47-56
WEIGHTED_METRICS_WEIGHTS = {"progress": 5.0, "ttc": 5.0, "comfortable": 2.0}
DRIVING_DIRECTION_COMPLIANCE_THRESHOLD = 2.0  # m against the traffic: still compliant
DRIVING_DIRECTION_VIOLATION_THRESHOLD = 6.0  # m: not compliant (in between: 0.5)
STOPPED_SPEED_THRESHOLD = 5e-03  # m/s, the ego stands for the TTC
PROGRESS_DISTANCE_THRESHOLD = 0.1  # m, below it every proposal's progress counts as full
TTC_FUTURE_STEPS = np.arange(0, 10, 3)  # pdm_scorer.py:402, the ego moved on 0, 0.3, 0.6, 0.9 s
EGO_STOPPED_SPEED = 5e-02  # m/s, pdm_scorer_utils.py:22 (collision type)
TRACK_STOPPED_SPEED = 5e-02  # m/s, nuPlan's is_track_stopped default
AHEAD_DEG = 30.0  # nuPlan's is_agent_ahead: within this of the ego's heading
BEHIND_DEG = 150.0  # nuPlan's is_agent_behind: beyond this
AGENT_COLLISION_SCORE = 0.0  # pdm_scorer.py:373-377: an at-fault collision with an agent
OBJECT_COLLISION_SCORE = 0.5  # with a static object
# pdm_comfort_metrics.py:13-30
MAX_ABS_MAG_JERK = 8.37  # m/s^3
MAX_ABS_LAT_ACCEL = 4.89  # m/s^2
MAX_LON_ACCEL = 2.40  # m/s^2
MIN_LON_ACCEL = -4.05  # m/s^2
MAX_ABS_YAW_ACCEL = 1.93  # rad/s^2
MAX_ABS_LON_JERK = 4.13  # m/s^3
MAX_ABS_YAW_RATE = 0.95  # rad/s


class EgoArea(IntEnum):
    """pdm_enums.py:150-155 (EgoAreaIndex)."""

    MULTIPLE_LANES = 0
    NON_DRIVABLE_AREA = 1
    ONCOMING_TRAFFIC = 2


class MultiMetric(IntEnum):
    """pdm_enums.py:158-163 (MultiMetricIndex)."""

    NO_COLLISION = 0
    DRIVABLE_AREA = 1
    DRIVING_DIRECTION = 2


class WeightedMetric(IntEnum):
    """pdm_enums.py:166-171 (WeightedMetricIndex)."""

    PROGRESS = 0
    TTC = 1
    COMFORTABLE = 2


class CollisionType(IntEnum):
    """nuplan-devkit metrics/utils/collision_utils.py:25-32."""

    STOPPED_EGO_COLLISION = 0
    STOPPED_TRACK_COLLISION = 1
    ACTIVE_FRONT_COLLISION = 2
    ACTIVE_REAR_COLLISION = 3
    ACTIVE_LATERAL_COLLISION = 4


WEIGHTS = np.array([WEIGHTED_METRICS_WEIGHTS[metric.name.lower()] for metric in WeightedMetric])


class Scorer:
    """Scores simulated proposals (pdm_scorer.py:59-586, PDMScorer)."""

    def __init__(self, interval_s: float, side_at_fault: bool = False):
        self._interval_s = interval_s
        self._side_at_fault = side_at_fault  # `PDMConfig.side_at_fault`

    def score(
        self,
        states: np.ndarray,
        vehicle: Vehicle,
        observation: Observation,
        centerline: Path,
        route_map: RouteMap,
    ) -> np.ndarray:
        """[P] score of each simulated proposal [P, T, 11] (pdm_scorer.py:112-158)."""
        self._states, self._observation, self._route_map = states, observation, route_map
        self._num_proposals, self._num_times = states.shape[:2]
        self._ego_coords = box_coordinates(states, vehicle)
        self._ego_polygons = box_polygons(self._ego_coords)
        self._multi_metrics = np.zeros((len(MultiMetric), self._num_proposals), dtype=np.float64)
        self._weighted_metrics = np.zeros(
            (len(WeightedMetric), self._num_proposals), dtype=np.float64
        )
        self._collision_time_indices = np.full(self._num_proposals, np.inf)
        self._ttc_time_indices = np.full(self._num_proposals, np.inf)
        self._ego_areas = self._calculate_ego_area()
        self._calculate_no_at_fault_collision()
        self._calculate_driving_direction_compliance()
        self._calculate_drivable_area_compliance()
        self._progress_raw = self._calculate_progress(centerline)
        self._calculate_ttc()
        self._calculate_is_comfortable()
        return self._aggregate_scores()

    def _calculate_ego_area(self) -> np.ndarray:
        """[P, T, 3] `EgoArea` flags of the ego's box (pdm_scorer.py:257-316).

        Points on an edge count as outside, as upstream.
        """
        boxes = self._num_proposals * self._num_times
        corners = self._ego_coords[:, :, :CORNERS].reshape(-1, 2)
        centers = self._ego_coords[:, :, BoxPoint.CENTER].reshape(-1, 2)
        lane_tree, num_lanes = self._route_map.lane_tree, len(self._route_map.lane_ids)
        corner_index, lane_index = lane_tree.query(shapely.points(corners), predicate="within")
        corners_in_lane = np.zeros((boxes, num_lanes), dtype=int)
        np.add.at(corners_in_lane, (corner_index // CORNERS, lane_index), 1)
        areas = np.zeros((boxes, len(EgoArea)), dtype=np.bool_)
        areas[:, EgoArea.MULTIPLE_LANES] = ((corners_in_lane > 0).sum(axis=-1) > 1) & np.all(
            corners_in_lane != CORNERS, axis=-1
        )
        drivable = shapely.contains_xy(self._route_map.drivable_area, corners[:, 0], corners[:, 1])
        areas[:, EgoArea.NON_DRIVABLE_AREA] = (
            drivable.reshape(boxes, CORNERS).sum(axis=-1) < CORNERS
        )
        center_index, lane_index = lane_tree.query(shapely.points(centers), predicate="within")
        on_route = np.zeros(boxes, dtype=np.bool_)
        on_route[center_index[self._route_map.on_route[lane_index]]] = True
        areas[:, EgoArea.ONCOMING_TRAFFIC] = ~on_route
        return areas.reshape(self._num_proposals, self._num_times, len(EgoArea))

    def _in_multiple_lanes_or_nondrivable_area(self, proposal: int, time_index: int) -> bool:
        return bool(
            self._ego_areas[proposal, time_index, EgoArea.MULTIPLE_LANES]
            or self._ego_areas[proposal, time_index, EgoArea.NON_DRIVABLE_AREA]
        )

    def _calculate_no_at_fault_collision(self) -> None:
        """nuPlan's at-fault collisions (pdm_scorer.py:318-388).

        At fault: with a standing object, with the front bumper, or sideways while in multiple
        lanes or off the drivable area. Any other collision ignores that object from then on.
        """
        no_collision_scores = np.ones(self._num_proposals, dtype=np.float64)
        collided = [set(self._observation.collided_track_ids) for _ in range(self._num_proposals)]
        for time_index in range(self._num_times):
            occupancy = self._observation[time_index]
            hits = occupancy.query(self._ego_polygons[:, time_index], predicate="intersects")
            for proposal, geometry_index in zip(*hits, strict=True):
                token = occupancy.tokens[geometry_index]
                if RED_LIGHT_TOKEN in token or token in collided[proposal]:
                    continue
                tracked = self._observation.unique_objects[token]
                kind = collision_type(
                    self._states[proposal, time_index],
                    self._ego_polygons[proposal, time_index],
                    tracked,
                    occupancy[token],
                )
                if self._side_at_fault and kind == CollisionType.ACTIVE_LATERAL_COLLISION:
                    kind = CollisionType.ACTIVE_FRONT_COLLISION
                at_fault = kind in (
                    CollisionType.ACTIVE_FRONT_COLLISION,
                    CollisionType.STOPPED_TRACK_COLLISION,
                ) or (
                    self._in_multiple_lanes_or_nondrivable_area(proposal, time_index)
                    and kind == CollisionType.ACTIVE_LATERAL_COLLISION
                )
                if at_fault:
                    score = AGENT_COLLISION_SCORE if tracked.is_agent else OBJECT_COLLISION_SCORE
                    no_collision_scores[proposal] = min(no_collision_scores[proposal], score)
                    self._collision_time_indices[proposal] = min(
                        time_index, self._collision_time_indices[proposal]
                    )
                else:
                    collided[proposal].add(token)
        self._multi_metrics[MultiMetric.NO_COLLISION] = no_collision_scores

    def _calculate_driving_direction_compliance(self) -> None:
        """1, 0.5 or 0 by the longest stretch in oncoming traffic (pdm_scorer.py:537-586)."""
        center_coordinates = self._ego_coords[:, :, BoxPoint.CENTER]
        cum_progress = np.zeros((self._num_proposals, self._num_times), dtype=np.float64)
        cum_progress[:, 1:] = ((center_coordinates[:, 1:] - center_coordinates[:, :-1]) ** 2.0).sum(
            axis=-1
        ) ** 0.5
        oncoming_traffic_masks = self._ego_areas[:, :, EgoArea.ONCOMING_TRAFFIC]
        cum_progress[~oncoming_traffic_masks] = 0.0
        scores = np.ones(self._num_proposals, dtype=np.float64)
        for proposal in range(self._num_proposals):
            # one stretch per run of steps in (or out of) oncoming traffic
            splits = np.split(
                cum_progress[proposal],
                np.where(np.diff(oncoming_traffic_masks[proposal]))[0] + 1,
            )
            max_oncoming_traffic_progress = max(split.sum() for split in splits)
            if max_oncoming_traffic_progress < DRIVING_DIRECTION_COMPLIANCE_THRESHOLD:
                scores[proposal] = 1.0
            elif max_oncoming_traffic_progress < DRIVING_DIRECTION_VIOLATION_THRESHOLD:
                scores[proposal] = 0.5
            else:
                scores[proposal] = 0.0
        self._multi_metrics[MultiMetric.DRIVING_DIRECTION] = scores

    def _calculate_drivable_area_compliance(self) -> None:
        """0 if the box ever leaves the drivable area (pdm_scorer.py:524-535)."""
        off_road = self._ego_areas[:, :, EgoArea.NON_DRIVABLE_AREA].any(axis=-1)
        self._multi_metrics[MultiMetric.DRIVABLE_AREA] = np.where(off_road, 0.0, 1.0)

    def _calculate_progress(self, centerline: Path) -> np.ndarray:
        """[P] forward progress of the box centre along the centerline (pdm_scorer.py:493-509)."""
        starts = shapely.points(self._ego_coords[:, 0, BoxPoint.CENTER])
        ends = shapely.points(self._ego_coords[:, -1, BoxPoint.CENTER])
        progress = centerline.project(ends) - centerline.project(starts)
        return np.clip(progress, a_min=0, a_max=None)

    def _calculate_ttc(self) -> None:
        """nuPlan's time to collision (pdm_scorer.py:390-491).

        0 if the ego, moved on for 0 to 0.9 s, meets an object ahead, or one not behind while in
        multiple lanes, off the drivable area or in an intersection.
        """
        ttc_scores = np.ones(self._num_proposals, dtype=np.float64)
        collided = [set(self._observation.collided_track_ids) for _ in range(self._num_proposals)]
        coords_exterior = self._ego_coords.copy()
        coords_exterior[:, :, BoxPoint.CENTER] = coords_exterior[:, :, BoxPoint.FRONT_LEFT]
        coords_exterior_time_steps = np.repeat(
            coords_exterior[:, :, None], len(TTC_FUTURE_STEPS), axis=2
        )
        speeds = np.hypot(
            self._states[..., StateIndex.VELOCITY_X], self._states[..., StateIndex.VELOCITY_Y]
        )
        dxy_per_s = np.stack(
            [
                np.cos(self._states[..., StateIndex.HEADING]) * speeds,
                np.sin(self._states[..., StateIndex.HEADING]) * speeds,
            ],
            axis=-1,
        )
        for index, future_step in enumerate(TTC_FUTURE_STEPS):
            delta_t = float(future_step) * self._interval_s
            coords_exterior_time_steps[:, :, index] = (
                coords_exterior_time_steps[:, :, index] + dxy_per_s[:, :, None] * delta_t
            )
        polygons = shapely.polygons(coords_exterior_time_steps)
        for time_index in range(self._num_times):
            for step_index, future_step in enumerate(TTC_FUTURE_STEPS):
                occupancy = self._observation[time_index + future_step]
                hits = occupancy.query(polygons[:, time_index, step_index], predicate="intersects")
                for proposal, geometry_index in zip(*hits, strict=True):
                    token = occupancy.tokens[geometry_index]
                    if (
                        RED_LIGHT_TOKEN in token
                        or token in collided[proposal]
                        or speeds[proposal, time_index] < STOPPED_SPEED_THRESHOLD
                    ):
                        continue
                    rear_axle = self._states[proposal, time_index, STATE_SE2]
                    centroid = occupancy[token].centroid
                    object_pose = (
                        centroid.x,
                        centroid.y,
                        self._observation.unique_objects[token].heading,
                    )
                    sideways_counts = (
                        self._in_multiple_lanes_or_nondrivable_area(proposal, time_index)
                        or self._in_intersection(rear_axle)
                    ) and not is_agent_behind(rear_axle, object_pose)
                    if is_agent_ahead(rear_axle, object_pose) or sideways_counts:
                        ttc_scores[proposal] = min(ttc_scores[proposal], 0.0)
                        self._ttc_time_indices[proposal] = min(
                            time_index, self._ttc_time_indices[proposal]
                        )
                    else:
                        collided[proposal].add(token)
        self._weighted_metrics[WeightedMetric.TTC] = ttc_scores

    def _in_intersection(self, rear_axle: np.ndarray) -> bool:
        """nuPlan's `is_in_layer(rear axle, INTERSECTION)` on our intersection stand-in."""
        return bool(shapely.contains_xy(self._route_map.intersection, rear_axle[0], rear_axle[1]))

    def _calculate_is_comfortable(self) -> None:
        """Whether every comfort metric holds (pdm_scorer.py:511-522)."""
        time_points_s = np.arange(0, self._num_times).astype(np.float64) * self._interval_s
        is_comfortable = ego_is_comfortable(self._states, time_points_s)
        self._weighted_metrics[WeightedMetric.COMFORTABLE] = np.all(is_comfortable, axis=-1)

    def _aggregate_scores(self) -> np.ndarray:
        """Multiplicative metrics times the weighted mean (pdm_scorer.py:160-188).

        Progress is normalized by the best proposal's.
        """
        multiplicative_scores = self._multi_metrics.prod(axis=0)
        raw_progress = self._progress_raw * multiplicative_scores
        max_raw_progress = np.max(raw_progress)
        if max_raw_progress > PROGRESS_DISTANCE_THRESHOLD:
            normalized_progress = raw_progress / max_raw_progress
        else:
            normalized_progress = np.ones(len(raw_progress), dtype=np.float64)
            normalized_progress[multiplicative_scores == 0.0] = 0.0
        self._weighted_metrics[WeightedMetric.PROGRESS] = normalized_progress
        weighted_scores = (self._weighted_metrics * WEIGHTS[..., None]).sum(axis=0)
        weighted_scores /= WEIGHTS.sum()
        return multiplicative_scores * weighted_scores

    def time_to_at_fault_collision(self, proposal_index: int) -> float:
        """Seconds to the proposal's first at-fault collision (inf: none)."""
        return self._collision_time_indices[proposal_index] * self._interval_s


# ---------------------------------------------------------------- collisions
def collision_type(
    state: np.ndarray,
    ego_polygon: shapely.Polygon,
    tracked: TrackedObject,
    object_polygon: shapely.Polygon,
) -> CollisionType:
    """The kind of the ego's collision with an object (pdm_scorer_utils.py:17-71)."""
    ego_speed = np.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])
    center = object_polygon.centroid
    object_center = (center.x, center.y, tracked.heading)
    if float(ego_speed) <= EGO_STOPPED_SPEED:
        return CollisionType.STOPPED_EGO_COLLISION
    if not tracked.is_agent or tracked.speed <= TRACK_STOPPED_SPEED:  # is_track_stopped
        return CollisionType.STOPPED_TRACK_COLLISION
    if is_agent_behind(state[STATE_SE2], object_center):
        return CollisionType.ACTIVE_REAR_COLLISION
    exterior = ego_polygon.exterior.coords
    if shapely.LineString([exterior[0], exterior[3]]).intersects(object_polygon):  # front bumper
        return CollisionType.ACTIVE_FRONT_COLLISION
    return CollisionType.ACTIVE_LATERAL_COLLISION


def agent_relative_angle(ego_pose: npt.ArrayLike, agent_pose: npt.ArrayLike) -> float:
    """Angle between the ego's heading and the direction to the agent (idm/utils.py:19-30).

    NaN, without a warning, where the positions coincide, as upstream.
    """
    agent_vector = np.array([agent_pose[0] - ego_pose[0], agent_pose[1] - ego_pose[1]])
    ego_vector = np.array([np.cos(ego_pose[2]), np.sin(ego_pose[2])])
    with np.errstate(invalid="ignore", divide="ignore"):
        dot_product = np.dot(ego_vector, agent_vector / np.linalg.norm(agent_vector))
        return float(np.arccos(dot_product))


def is_agent_ahead(ego_pose: npt.ArrayLike, agent_pose: npt.ArrayLike) -> bool:
    return bool(agent_relative_angle(ego_pose, agent_pose) < np.deg2rad(AHEAD_DEG))


def is_agent_behind(ego_pose: npt.ArrayLike, agent_pose: npt.ArrayLike) -> bool:
    return bool(agent_relative_angle(ego_pose, agent_pose) > np.deg2rad(BEHIND_DEG))


# ---------------------------------------------------------------- comfort
def ego_is_comfortable(states: np.ndarray, time_points_s: np.ndarray) -> np.ndarray:
    """[P, 6] whether each comfort metric holds (pdm_comfort_metrics.py:341-368)."""
    n_batch, n_time, n_states = states.shape
    if n_time != len(time_points_s) or n_states != len(StateIndex):
        raise ValueError(f"states {states.shape} for {len(time_points_s)} time points")
    results = np.zeros((n_batch, len(COMFORT_METRICS)), dtype=np.bool_)
    for index, metric in enumerate(COMFORT_METRICS):
        results[:, index] = metric(states, time_points_s)
    return results


def _compute_lon_acceleration(states: np.ndarray, time_steps_s: np.ndarray):
    lon_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate="x", window_length=states.shape[1]
    )
    return _within_bound(lon_acceleration, min_bound=MIN_LON_ACCEL, max_bound=MAX_LON_ACCEL)


def _compute_lat_acceleration(states: np.ndarray, time_steps_s: np.ndarray):
    lat_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate="y", window_length=states.shape[1]
    )
    return _within_bound(
        lat_acceleration, min_bound=-MAX_ABS_LAT_ACCEL, max_bound=MAX_ABS_LAT_ACCEL
    )


def _compute_jerk_metric(states: np.ndarray, time_steps_s: np.ndarray):
    jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="magnitude",
        time_steps_s=time_steps_s,
        window_length=states.shape[1],
    )
    return _within_bound(jerk_metric, min_bound=-MAX_ABS_MAG_JERK, max_bound=MAX_ABS_MAG_JERK)


def _compute_lon_jerk_metric(states: np.ndarray, time_steps_s: np.ndarray):
    lon_jerk_metric = _extract_ego_jerk(
        states,
        acceleration_coordinate="x",
        time_steps_s=time_steps_s,
        window_length=states.shape[1],
    )
    return _within_bound(lon_jerk_metric, min_bound=-MAX_ABS_LON_JERK, max_bound=MAX_ABS_LON_JERK)


def _compute_yaw_accel(states: np.ndarray, time_steps_s: np.ndarray):
    yaw_accel_metric = _extract_ego_yaw_rate(
        states, time_steps_s, deriv_order=2, poly_order=3, window_length=states.shape[1]
    )
    return _within_bound(
        yaw_accel_metric, min_bound=-MAX_ABS_YAW_ACCEL, max_bound=MAX_ABS_YAW_ACCEL
    )


def _compute_yaw_rate(states: np.ndarray, time_steps_s: np.ndarray):
    yaw_rate_metric = _extract_ego_yaw_rate(states, time_steps_s, window_length=states.shape[1])
    return _within_bound(yaw_rate_metric, min_bound=-MAX_ABS_YAW_RATE, max_bound=MAX_ABS_YAW_RATE)


COMFORT_METRICS = (  # pdm_comfort_metrics.py:354-361, in upstream's order
    _compute_lon_acceleration,
    _compute_lat_acceleration,
    _compute_jerk_metric,
    _compute_lon_jerk_metric,
    _compute_yaw_accel,
    _compute_yaw_rate,
)


def _extract_ego_acceleration(
    states: np.ndarray,
    acceleration_coordinate: str,
    decimals: int = 8,
    poly_order: int = 2,
    window_length: int = 8,
) -> np.ndarray:
    """[P, T] filtered acceleration along x, y or its magnitude (pdm_comfort_metrics.py:33-76)."""
    n_time = states.shape[1]
    if acceleration_coordinate == "x":
        acceleration = states[..., StateIndex.ACCELERATION_X]
    elif acceleration_coordinate == "y":
        acceleration = states[..., StateIndex.ACCELERATION_Y]
    elif acceleration_coordinate == "magnitude":
        acceleration = np.hypot(
            states[..., StateIndex.ACCELERATION_X], states[..., StateIndex.ACCELERATION_Y]
        )
    else:
        raise ValueError(f"acceleration_coordinate {acceleration_coordinate}: x, y or magnitude")
    acceleration = savgol_filter(
        acceleration, polyorder=poly_order, window_length=min(window_length, n_time), axis=-1
    )
    return np.round(acceleration, decimals=decimals)


def _extract_ego_jerk(
    states: np.ndarray,
    acceleration_coordinate: str,
    time_steps_s: np.ndarray,
    decimals: int = 8,
    deriv_order: int = 1,
    poly_order: int = 2,
    window_length: int = 15,
) -> np.ndarray:
    """[P, T] the jerk of the filtered acceleration (pdm_comfort_metrics.py:79-111)."""
    n_time = states.shape[1]
    ego_acceleration = _extract_ego_acceleration(
        states, acceleration_coordinate=acceleration_coordinate
    )
    jerk = _approximate_derivatives(
        ego_acceleration,
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
        window_length=min(window_length, n_time),
    )
    return np.round(jerk, decimals=decimals)


def _extract_ego_yaw_rate(
    states: np.ndarray,
    time_steps_s: np.ndarray,
    deriv_order: int = 1,
    poly_order: int = 2,
    decimals: int = 8,
    window_length: int = 15,
) -> np.ndarray:
    """[P, T] yaw rate or its derivative (pdm_comfort_metrics.py:114-140).

    `window_length` is unused, as upstream.
    """
    ego_yaw_rate = _approximate_derivatives(
        _phase_unwrap(states[..., StateIndex.HEADING]),
        time_steps_s,
        deriv_order=deriv_order,
        poly_order=poly_order,
    )
    return np.round(ego_yaw_rate, decimals=decimals)


def _phase_unwrap(headings: np.ndarray) -> np.ndarray:
    """Headings unwrapped to steps of at most pi (pdm_comfort_metrics.py:143-163)."""
    two_pi = 2.0 * np.pi
    adjustments = np.zeros_like(headings)
    adjustments[..., 1:] = np.cumsum(np.round(np.diff(headings, axis=-1) / two_pi), axis=-1)
    return headings - two_pi * adjustments


def _approximate_derivatives(
    values: np.ndarray,
    times: np.ndarray,
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> np.ndarray:
    """Savitzky-Golay derivative over evenly spaced `times` (pdm_comfort_metrics.py:166-207)."""
    window_length = min(window_length, len(times))
    if not poly_order < window_length:
        raise ValueError(f"{poly_order} < {window_length} does not hold!")
    time_steps = np.diff(times, axis=-1)
    if not (time_steps > 0).all():
        raise RuntimeError("times are not increasing")
    return savgol_filter(
        values,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=time_steps.mean(),
        axis=axis,
    )


def _within_bound(
    metric: np.ndarray, min_bound: float | None = None, max_bound: float | None = None
):
    """[P] every value strictly within the bounds (pdm_comfort_metrics.py:210-226).

    A bound of 0 counts as none, as upstream.
    """
    min_bound = min_bound if min_bound else float(-np.inf)
    max_bound = max_bound if max_bound else float(np.inf)
    metric_values = np.array(metric)
    return np.all((metric_values > min_bound) & (metric_values < max_bound), axis=-1)
