"""PDM-Closed as a closed-loop planner, `PDMClosedPlanner`, with its observation and map stand-in.

Ported from tuplan_garage b51d5d04 (Apache-2.0, LICENSE in this folder), planning/simulation/
planner/pdm_planner/. `PDMConfig()` is upstream's configuration.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import shapely

from ...core.config import DriverConfig, VehicleConfig
from ...core.rules import DEFAULT_SPEED_LIMIT, TrafficRules, driver_path, phase_at
from ...core.tracker import simulate
from ...core.types import FeatureKind, LaneType, MapFeature, Scene
from ...core.vehicle import execute, initial_steering
from ...data.map_geometry import drivable_region, lane_crossings, lane_width
from ...utils.geometry import MIN_LINE_POINTS, box_corners, union_all, wrap_angle
from ..perception import Beliefs
from ..step_records import StepRecords
from .proposals import (
    RED_LIGHT_TOKEN,
    STATE_SE2,
    BoxPoint,
    IDMPolicies,
    Path,
    ProposalGenerator,
    StateIndex,
    Vehicle,
    box_coordinates,
    box_points,
    box_polygons,
    from_center,
    headed_poses,
    parallel_discrete_path,
    proposal_grid,
    ramped_offset_path,
    route_centerline,
    to_center,
)
from .scorer import Scorer

# The scene's frame step must be PDM's interval_s within this (WOMD's 0.1 s up to rounding).
INTERVAL_TOLERANCE_S = 1e-9
DEFAULT_LANE_WIDTH_M = 3.6  # for a lane whose boundaries are not mapped
ROUTE_LANE_DEG = 45.0  # a lane the route passes through runs along it within this angle
# the centerline runs this far past the route's end, so PDM's IDM does not stop at it as a lead
ROUTE_BEYOND_M = 240.0
OFFSET_RAMP_S = 2.0  # offset paths ease in over this much travel
OFFSET_RAMP_MIN_M = 10.0  # and over at least this distance
# pdm_object_manager.py:20-26: the nearest ones of each type kept
MAX_DYNAMIC_OBJECTS = {"vehicle": 50, "pedestrian": 25, "cyclist": 10}
MAX_STATIC_OBJECTS = 50
AGENT_TYPES = tuple(MAX_DYNAMIC_OBJECTS)  # nuPlan's AGENT_TYPES (vehicle, pedestrian, bicycle)
STATIC = "static"  # nuPlan's static objects: WOMD's "other"
OBSERVATION_SAMPLE_RESOLUTION = 2  # pdm_observation.py:36, one occupancy map per 2 steps
STR_TREE_NODE_CAPACITY = 10  # pdm_occupancy_map.py:16
# pdm_emergency_brake.py:28-32
TIME_TO_INFRACTION_THRESHOLD = 2.0  # s, an at-fault collision this soon
MAX_LONG_ACCEL = 2.40  # m/s^2 (the speed and deceleration limits are PDMConfig's)
MOVING_SPEED = 0.2  # m/s, pdm_emergency_brake.py:105: above it, a P controller on the speed
MOVING_GAINS = (10.0, 0.0)  # (proportional, derivative), pdm_emergency_brake.py:106-107
STANDING_GAINS = (4.0, 1.0)  # pdm_emergency_brake.py:117-118
OVERSHOOT = 11 / 10  # pdm_emergency_brake.py:114


@dataclass(frozen=True)
class PDMConfig:
    """Upstream's PDM-Closed parameters at its values (pdm_closed_planner.yaml).

    The exception: an unrecorded speed limit falls back to OnCue's DEFAULT_SPEED_LIMIT.
    """

    trajectory_poses: int = 80  # the output trajectory: 8 s (it sets the IDM corridor's reach)
    proposal_poses: int = 40  # the proposals: 4 s
    interval_s: float = 0.1
    speed_limit_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
    min_gap_to_lead_agent: float = 1.0  # m
    headway_time: float = 1.5  # s
    accel_max: float = 1.5  # m/s^2
    decel_max: float = 3.0  # m/s^2
    # the emergency brake (pdm_emergency_brake.py:28-32)
    emergency_max_speed: float | None = 5.0  # m/s, acts only this slow (None: at any speed)
    emergency_decel: float = 4.05  # m/s^2, its deceleration limit
    lateral_offsets: tuple[float, ...] = (-1.0, 1.0)  # m, besides the centerline itself
    map_radius: float = 50.0  # m
    # ours: a sideways collision is at fault even while the ego keeps its lane
    side_at_fault: bool = False
    # ours: simulate the proposals with the shared executor, not nuPlan's LQR on the Pacifica
    simulate_with_executor: bool = False
    # ours: offset paths ease in over this much travel (s); None: fully offset, as upstream
    offset_ramp_s: float | None = None


class PDMClosedPlanner(StepRecords):
    """The PDM-Closed driver of one run (pdm_closed_planner.py:32-97) on the IDM planners' route.

    Red signals in `rules` stop it. `config` defaults to upstream's parameters.
    """

    def __init__(
        self,
        scene: Scene,
        driver: DriverConfig,
        rules: TrafficRules,
        config: PDMConfig | None = None,
    ):
        config = PDMConfig() if config is None else config
        if abs(scene.dt - config.interval_s) > INTERVAL_TOLERANCE_S:
            raise ValueError(f"PDM samples every {config.interval_s} s, the scene {scene.dt} s")
        super().__init__(driver_path(scene, driver), scene.dt)
        self.config, self.rules = config, rules
        self.anchor_state = scene.ego.states[scene.anchor].copy()
        # acceleration, steering and yaw rate carried between steps, as nuPlan's EgoState does
        self.dynamics = np.zeros(len(StateIndex))
        self.executor_vehicle: VehicleConfig | None = None  # set: simulate with the executor
        if config.simulate_with_executor:
            self.executor_vehicle = driver.vehicle
            self.vehicle = Vehicle(
                *scene.ego.size, driver.vehicle.wheelbase, driver.vehicle.center_to_rear_axle
            )
            # the executor's start (`closedloop.execution.ExecutedPlanner`)
            self.dynamics[StateIndex.STEERING_ANGLE] = initial_steering(
                scene.ego, scene.anchor, scene.dt, driver.vehicle, scene.anchor
            )
        else:
            self.vehicle = Vehicle(*scene.ego.size)
        beyond = replace(driver, extension_m=driver.extension_m + ROUTE_BEYOND_M)
        self.map = cached_route_map(
            scene, driver_path(scene, beyond), rules, self.vehicle, config.map_radius
        )
        self.centerline = self.map.centerline
        # the proposals' paths, the centerline first (abstract_pdm_closed_planner.py:112-134)
        offsets = [
            Path(parallel_discrete_path(self.centerline.discrete_path, offset))
            for offset in config.lateral_offsets
        ]
        self.policies = IDMPolicies(
            config.speed_limit_fractions,
            config.min_gap_to_lead_agent,
            config.headway_time,
            config.accel_max,
            config.decel_max,
        )
        self.proposals = proposal_grid([self.centerline, *offsets], self.policies)
        self.generator = ProposalGenerator(
            config.proposal_poses, config.trajectory_poses, config.interval_s
        )
        self.observation = Observation(config.proposal_poses, config.interval_s, config.map_radius)
        self.scorer = Scorer(config.interval_s, config.side_at_fault)
        # the last step's chosen trajectory when simulating with the executor, else None
        self.planned_states: np.ndarray | None = None
        self.target_brake_s: float | None = None  # PDM names no road user it brakes for

    def state(self) -> np.ndarray:
        """The ego's state at the anchor (x, y, heading, speed): the logged one."""
        return self.anchor_state.copy()

    def step(
        self, frame: int, ego_states: np.ndarray, scene: Scene, beliefs: Beliefs
    ) -> np.ndarray:
        """The ego's state at the next frame (abstract_pdm_closed_planner.py:136-189).

        `ego_states` [T, 4] is indexed by scene frame.
        """
        config = self.config
        state = from_center(ego_states[frame], self.dynamics, self.vehicle)
        center = ego_states[frame, :2]
        footprint = box_polygons(
            box_points(
                center, ego_states[frame, 2], self.vehicle.half_length, self.vehicle.half_width
            )
        )
        # 1. forecast the known road users, red lights standing
        self.observation.update(
            center, footprint, beliefs.states, scene, self._red_lights(beliefs.frame)
        )
        # 2. the target speeds from the starting lane's limit
        self.policies.update(self.map.speed_limit(state[STATE_SE2], center, footprint))
        # 3. unroll, 4. simulate and 5. score the proposals
        if config.offset_ramp_s is not None:
            self.proposals = proposal_grid(
                [self.centerline, *self._ramped_offsets(state)], self.policies
            )
        proposals = self.generator.generate(
            state, self.vehicle, self.observation, self.proposals, self.policies
        )
        simulated = self._simulate(proposals[..., STATE_SE2], state, ego_states[frame])
        scores = self.scorer.score(
            simulated, self.vehicle, self.observation, self.centerline, self.map
        )
        # 6. brake if an at-fault collision is expected soon, else take the best proposal
        best = int(np.argmax(scores))
        if brake_needed(state, self.scorer, best, config.emergency_max_speed):
            poses = brake_poses(
                state,
                self.vehicle,
                config.proposal_poses,
                config.interval_s,
                config.emergency_decel,
            )
            executed = self._simulate(poses[None], state, ego_states[frame])[0]
        else:
            poses = proposals[best][:, STATE_SE2]
            executed = simulated[best]
        if self.executor_vehicle is not None:
            self.planned_states = center_states(poses, self.vehicle, config.interval_s)
        next_state = executed[1]
        if next_state[StateIndex.VELOCITY_X] < 0:  # ours: OnCue's egos never reverse
            next_state = standing(state)
        self.dynamics = next_state
        next_center = to_center(next_state, self.vehicle)
        self.record_step(ego_states[frame], next_center)
        return next_center

    def _ramped_offsets(self, state: np.ndarray) -> list[Path]:
        """Offset paths easing from the ego's offset to their own over `offset_ramp_s` of travel."""
        start = float(self.centerline.project(shapely.Point(*state[:2])))
        x, y, heading = self.centerline.interpolate(start)
        ego_offset = (state[1] - y) * np.cos(heading) - (state[0] - x) * np.sin(heading)
        speed = max(float(state[StateIndex.VELOCITY_X]), 0.0)
        ramp_m = max(speed * self.config.offset_ramp_s, OFFSET_RAMP_MIN_M)
        poses, progress = self.centerline.discrete_path, self.centerline.progress
        return [
            Path(ramped_offset_path(poses, progress, start, ego_offset, offset, ramp_m))
            for offset in self.config.lateral_offsets
        ]

    def _simulate(self, poses: np.ndarray, state: np.ndarray, center: np.ndarray) -> np.ndarray:
        """[P, N, 11] rear-axle states tracking rear-axle poses [P, N, 3] from the ego's `state`.

        Tracked by the shared executor with `simulate_with_executor`, else by nuPlan's LQR.
        `center` is the ego's box-centre state [4].
        """
        interval = self.config.interval_s
        if self.executor_vehicle is None:
            return simulate(poses, state, self.vehicle.wheel_base, interval)
        desired = center_states(poses, self.vehicle, interval)
        accel = None if self.planned_states is None else state[StateIndex.ACCELERATION_X]
        execution = execute(
            desired,
            interval * np.arange(poses.shape[1]),
            state[StateIndex.STEERING_ANGLE],
            self.executor_vehicle,
            accel0=accel,
            start=np.repeat(center[None], len(poses), axis=0),
        )
        return execution.bicycle_states

    def _red_lights(self, obs_frame: int) -> dict[str, shapely.Polygon]:
        """Polygons of the route's signal lanes showing red ("stop") at `obs_frame`.

        Caution and unknown do not count as red, as upstream.
        """
        red = {}
        for (_, phase), lane in zip(self.rules.signals, self.rules.signal_lanes, strict=True):
            if phase_at(phase, obs_frame) == "stop":
                red[lane] = self.map.signal_polygons[lane]
        return red


# ---------------------------------------------------------------- the map stand-in
# a scene's runs come one after another, so a few cached route maps cover them
ROUTE_MAP_CACHE_SIZE = 4
_route_maps: OrderedDict[tuple, RouteMap] = OrderedDict()


def cached_route_map(
    scene: Scene,
    path: shapely.LineString,
    rules: TrafficRules,
    vehicle: Vehicle,
    map_radius: float,
) -> RouteMap:
    """`route_map`, cached for the ROUTE_MAP_CACHE_SIZE latest distinct inputs."""
    ego = scene.ego
    logged = hashlib.blake2b(ego.states.tobytes() + ego.valid.tobytes()).hexdigest()
    key = (
        scene.scene_id,
        scene.anchor,
        logged,
        ego.size,
        shapely.to_wkb(path),
        tuple(rules.signal_lanes),
        vehicle,
        map_radius,
    )
    if key in _route_maps:
        _route_maps.move_to_end(key)
    else:
        _route_maps[key] = route_map(scene, path, rules, vehicle, map_radius)
        if len(_route_maps) > ROUTE_MAP_CACHE_SIZE:
            _route_maps.popitem(last=False)
    return _route_maps[key]


def route_map(
    scene: Scene,
    path: shapely.LineString,
    rules: TrafficRules,
    vehicle: Vehicle,
    map_radius: float,
) -> RouteMap:
    """The map layers within `map_radius` of `path`, built once (upstream queries every step)."""
    centerline = Path(route_centerline(scene, path, vehicle))
    lanes = _nearby_lanes(scene, centerline.linestring, map_radius)
    lane_polygons = np.array([lane_polygon(scene, lane) for lane in lanes], dtype=object)
    lane_poses = tuple(headed_poses(lane.points) for lane in lanes)
    lane_by_id = {
        feature.feature_id: feature
        for feature in scene.map_features
        if feature.kind is FeatureKind.LANE and len(feature.points) >= MIN_LINE_POINTS
    }
    return RouteMap(
        centerline=centerline,
        lane_ids=tuple(lane.feature_id for lane in lanes),
        lane_polygons=lane_polygons,
        lane_tree=shapely.STRtree(lane_polygons),
        lane_poses=lane_poses,
        speed_limits=tuple(lane.speed_limit_mps for lane in lanes),
        on_route=route_lanes(lanes, lane_polygons, lane_poses, centerline),
        drivable_area=drivable_area(scene, centerline, lane_polygons, map_radius),
        intersection=intersection_area(scene, path, lane_by_id),
        signal_polygons={
            lane: lane_polygon(scene, lane_by_id[lane]) for lane in rules.signal_lanes
        },
    )


@dataclass(frozen=True, eq=False)
class RouteMap:
    """The map layers of one run; the per-lane fields cover the lanes near the route."""

    centerline: Path  # the route the proposals follow
    lane_ids: tuple[str, ...]
    lane_polygons: np.ndarray
    lane_tree: shapely.STRtree  # over lane_polygons
    lane_poses: tuple[np.ndarray, ...]  # each lane's points with headings
    speed_limits: tuple[float | None, ...]  # None: not recorded
    on_route: np.ndarray  # nuPlan's route lanes
    drivable_area: shapely.Geometry
    intersection: shapely.Geometry  # nuPlan's intersection layer
    signal_polygons: dict[str, shapely.Polygon]  # signal lanes on the route (red-light obstacles)

    def speed_limit(
        self, rear_axle_pose: np.ndarray, center: np.ndarray, footprint: shapely.Polygon
    ) -> float:
        """The speed limit of the ego's starting lane (abstract_pdm_planner.py:119-194).

        An unrecorded limit gives DEFAULT_SPEED_LIMIT (upstream: 15 m/s).
        """
        lane = self._starting_lane(rear_axle_pose, center, footprint)
        limit = self.speed_limits[lane]
        return DEFAULT_SPEED_LIMIT if limit is None else limit

    def _starting_lane(
        self, rear_axle_pose: np.ndarray, center: np.ndarray, footprint: shapely.Polygon
    ) -> int:
        holding = [
            lane
            for lane in self.lane_tree.query(shapely.Point(*rear_axle_pose[:2]), "intersects")
            if self.on_route[lane]
        ]
        if holding:
            heading_errors = []
            for lane in holding:
                poses = self.lane_poses[lane]
                nearest = np.argmin(
                    (((rear_axle_pose[None, :2] - poses[:, :2]) ** 2).sum(axis=-1)) ** 0.5
                )
                heading_errors.append(np.abs(wrap_angle(poses[nearest, 2] - rear_axle_pose[2])))
            return int(holding[int(np.argmin(np.abs(heading_errors)))])
        closest_distance, starting_lane = np.inf, None
        for lane in np.flatnonzero(self.on_route):
            if self.lane_polygons[lane].contains(shapely.Point(*center[:2])):
                return int(lane)
            distance = self.lane_polygons[lane].distance(footprint)
            if distance < closest_distance:
                starting_lane, closest_distance = int(lane), distance
        if starting_lane is None:
            raise ValueError("no route lane in the map")
        return starting_lane


def _nearby_lanes(scene: Scene, route: shapely.LineString, map_radius: float) -> list[MapFeature]:
    """The vehicle lanes within `map_radius` of the route (nuPlan's lanes and lane connectors)."""
    return [
        feature
        for feature in scene.map_features
        if feature.kind is FeatureKind.LANE
        and feature.lane_type is not LaneType.BIKE_LANE
        and len(feature.points) >= MIN_LINE_POINTS
        and route.distance(shapely.LineString(feature.points)) <= map_radius
    ]


def lane_polygon(scene: Scene, lane: MapFeature) -> shapely.Polygon:
    """nuPlan's lane polygon: the centre line buffered by half the lane width at its middle.

    DEFAULT_LANE_WIDTH_M where boundaries are not mapped, as in most junctions.
    """
    middle = len(lane.points) // 2
    toward = lane.points[min(middle + 1, len(lane.points) - 1)] - lane.points[max(middle - 1, 0)]
    width = lane_width(
        scene, lane.points[middle], np.arctan2(toward[1], toward[0]), DEFAULT_LANE_WIDTH_M
    )
    return shapely.LineString(lane.points).buffer(width / 2, cap_style="flat")


def route_lanes(
    lanes: list[MapFeature],
    lane_polygons: np.ndarray,
    lane_poses: tuple[np.ndarray, ...],
    centerline: Path,
) -> np.ndarray:
    """[L] whether each lane is on the route (nuPlan's route lanes).

    The lanes the centerline passes through along their heading, plus their neighbours.
    """
    on_route = np.zeros(len(lanes), bool)
    poses = centerline.poses
    pose_index, lane_index = shapely.STRtree(lane_polygons).query(
        shapely.points(poses[:, :2]), predicate="within"
    )
    for pose, lane in zip(pose_index, lane_index, strict=True):
        points = lane_poses[lane]
        nearest = np.argmin(np.linalg.norm(points[:, :2] - poses[pose, :2], axis=1))
        if abs(wrap_angle(points[nearest, 2] - poses[pose, 2])) <= np.radians(ROUTE_LANE_DEG):
            on_route[lane] = True
    index_of = {lane.feature_id: index for index, lane in enumerate(lanes)}
    todo = list(np.flatnonzero(on_route))
    while todo:  # the neighbours, theirs and so on
        lane = lanes[todo.pop()]
        for neighbor in (*lane.left_neighbors, *lane.right_neighbors):
            index = index_of.get(neighbor.feature_id)
            if index is not None and not on_route[index]:
                on_route[index] = True
                todo.append(index)
    return on_route


def drivable_area(
    scene: Scene, centerline: Path, lane_polygons: np.ndarray, map_radius: float
) -> shapely.Geometry:
    """Stand-in for nuPlan's drivable area, which always contains the centerline.

    The union of the lane polygons, `data.map_geometry.drivable_region` and the logged ego's boxes,
    since each alone leaves gaps.
    """
    logged = scene.ego.states[scene.ego.valid]
    low, high = logged[:, :2].min(axis=0), logged[:, :2].max(axis=0)
    beyond_log = max(
        float((low - centerline.poses[:, :2]).max()),
        float((centerline.poses[:, :2] - high).max()),
        0.0,
    )
    region = drivable_region(scene, margin=beyond_log + map_radius)
    logged_boxes = shapely.polygons(box_corners(logged, scene.ego.size))
    area = union_all([region, *lane_polygons, *logged_boxes])
    shapely.prepare(area)
    return area


def intersection_area(
    scene: Scene, path: shapely.LineString, lane_by_id: dict[str, MapFeature]
) -> shapely.Geometry:
    """nuPlan's intersection layer: the polygons of the lanes crossing `path`."""
    crossing_lanes = {
        crossing.lane.feature_id
        for crossing, reason in lane_crossings(lane_by_id, path, scene.ego)
        if reason is None
    }
    region = union_all([lane_polygon(scene, lane_by_id[lane]) for lane in sorted(crossing_lanes)])
    shapely.prepare(region)
    return region


# ---------------------------------------------------------------- the observation
class Observation:
    """The occupancy maps of one planning step, by 0.1 s step (pdm_observation.py:28-276).

    `collided_track_ids` lists the objects the ego ever touched; PDM ignores them from then on.
    """

    def __init__(self, proposal_poses: int, interval_s: float, map_radius: float):
        # the proposals' steps and 1 s more for the TTC (pdm_observation.py:52-57)
        self._observation_samples = proposal_poses + int(1 / interval_s)
        self._interval_s = interval_s
        self._map_radius = map_radius
        self._global_to_local = [
            index // OBSERVATION_SAMPLE_RESOLUTION
            for index in range(self._observation_samples + OBSERVATION_SAMPLE_RESOLUTION)
        ]
        self.collided_track_ids: list[str] = []
        self.unique_objects: dict[str, TrackedObject] = {}
        self._occupancy_maps: list[OccupancyMap] = []

    def __getitem__(self, time_index: int) -> OccupancyMap:
        if not 0 <= time_index < len(self._global_to_local):
            raise IndexError(f"observation index {time_index} out of range")
        return self._occupancy_maps[self._global_to_local[time_index]]

    def update(
        self,
        ego_center: np.ndarray,
        ego_polygon: shapely.Polygon,
        beliefs: dict[str, np.ndarray],
        scene: Scene,
        red_lights: dict[str, shapely.Polygon],
    ) -> None:
        """Build the occupancy maps from the beliefs, the red lights standing.

        `beliefs` maps a road user to (x, y, heading, speed), `red_lights` a token to a polygon.
        """
        static, dynamic = self._objects(ego_center, beliefs, scene)
        static_tokens, static_coordinates, _ = static
        dynamic_tokens, dynamic_coordinates, dynamic_velocities = dynamic
        static_polygons = box_polygons(static_coordinates)
        red_light_tokens = [f"{RED_LIGHT_TOKEN}_{lane}" for lane in red_lights]
        red_light_polygons = np.array(list(red_lights.values()), dtype=object)
        self._occupancy_maps = []
        for sample in np.arange(
            0,
            self._observation_samples + OBSERVATION_SAMPLE_RESOLUTION,
            OBSERVATION_SAMPLE_RESOLUTION,
        ):
            delta_t = float(sample) * self._interval_s
            moved = dynamic_coordinates + delta_t * dynamic_velocities[:, None]
            all_polygons = np.concatenate(
                [static_polygons, box_polygons(moved), red_light_polygons], axis=0
            )
            self._occupancy_maps.append(
                OccupancyMap(static_tokens + dynamic_tokens + red_light_tokens, all_polygons)
            )
        # a red light counts as touched only once the ego is fully within it
        new_collided = []
        for token in self._occupancy_maps[0].intersects(ego_polygon):
            if RED_LIGHT_TOKEN in token and not ego_polygon.within(self._occupancy_maps[0][token]):
                continue
            new_collided.append(token)
        self.collided_track_ids = self.collided_track_ids + new_collided

    def _objects(self, ego_center: np.ndarray, beliefs: dict[str, np.ndarray], scene: Scene):
        """(static objects, agents), each (tokens, boxes [M, 5, 2], velocities [M, 2]).

        The nearest known road users of each type within the map radius, not collided with
        (pdm_observation.py:225-249, pdm_object_manager.py:57-149).
        """
        self.unique_objects = {}
        groups: dict[str, list] = {kind: [] for kind in (*AGENT_TYPES, STATIC)}
        for token, state in beliefs.items():
            if (
                np.linalg.norm(state[:2] - ego_center[:2]) > self._map_radius
                or token in self.collided_track_ids
            ):
                continue
            track = scene.tracks[token]
            is_agent = track.agent_type in AGENT_TYPES
            self.unique_objects[token] = TrackedObject(is_agent, float(state[2]), float(state[3]))
            length, width = track.size
            coordinates = box_points(state[:2], state[2], length / 2, width / 2)
            velocity = float(state[3]) * np.array([np.cos(state[2]), np.sin(state[2])])
            groups[track.agent_type if is_agent else STATIC].append((token, coordinates, velocity))
        static = _nearest(groups[STATIC], ego_center[:2], MAX_STATIC_OBJECTS)
        agents = [
            _nearest(groups[kind], ego_center[:2], MAX_DYNAMIC_OBJECTS[kind])
            for kind in AGENT_TYPES
        ]
        dynamic = (
            [token for tokens, _, _ in agents for token in tokens],
            np.concatenate([coordinates for _, coordinates, _ in agents], axis=0),
            np.concatenate([velocities for _, _, velocities in agents], axis=0),
        )
        return static, dynamic


@dataclass(frozen=True)
class TrackedObject:
    """What PDM reads of a known road user."""

    is_agent: bool
    heading: float
    speed: float


def _nearest(objects: list, position: np.ndarray, limit: int):
    """The `limit` objects nearest `position` by box centre (pdm_object_manager.py:184-249)."""
    tokens = [token for token, _, _ in objects]
    coordinates = np.array([coordinates for _, coordinates, _ in objects], dtype=np.float64)
    coordinates = coordinates.reshape(len(objects), len(BoxPoint), 2)
    velocities = np.array([velocity for _, _, velocity in objects], dtype=np.float64)
    velocities = velocities.reshape(len(objects), 2)
    if tokens:
        distance = ((coordinates[..., BoxPoint.CENTER, :] - position[None]) ** 2.0).sum(axis=-1)
        order = np.argsort(distance**0.5)
        tokens = [tokens[index] for index in order][:limit]
        coordinates, velocities = coordinates[order][:limit], velocities[order][:limit]
    return tokens, coordinates, velocities


class OccupancyMap:
    """Polygons with their tokens in a shapely STR-tree (pdm_occupancy_map.py:10-99)."""

    def __init__(self, tokens: list[str], geometries: np.ndarray):
        if len(tokens) != len(geometries):
            raise ValueError(f"{len(tokens)} tokens for {len(geometries)} geometries")
        self.tokens = tokens
        self.token_to_index = {token: index for index, token in enumerate(tokens)}
        self.geometries = geometries
        self._str_tree = shapely.STRtree(geometries, STR_TREE_NODE_CAPACITY)

    def __getitem__(self, token: str):
        return self.geometries[self.token_to_index[token]]

    def __len__(self) -> int:
        return len(self.tokens)

    def intersects(self, geometry: shapely.Geometry) -> list[str]:
        """The tokens of the geometries intersecting `geometry`."""
        return [self.tokens[index] for index in self.query(geometry, predicate="intersects")]

    def query(
        self, geometry: shapely.Geometry | np.ndarray, predicate: str | None = None
    ) -> np.ndarray:
        """shapely's STR-tree query: indices, or [2, K] (input, tree) pairs for an array."""
        return self._str_tree.query(geometry, predicate=predicate)


def center_states(poses: np.ndarray, vehicle: Vehicle, interval_s: float) -> np.ndarray:
    """[..., N, 4] box-centre states of rear-axle poses [..., N, 3], as the executor tracks them.

    Each speed is the centre's distance to the next pose over `interval_s`; the last is held.
    """
    centers = box_coordinates(poses, vehicle)[..., BoxPoint.CENTER, :]
    speeds = np.linalg.norm(np.diff(centers, axis=-2), axis=-1) / interval_s
    speeds = np.concatenate((speeds, speeds[..., -1:]), axis=-1)
    return np.concatenate(
        (centers, poses[..., StateIndex.HEADING, None], speeds[..., None]), axis=-1
    )


def standing(state: np.ndarray) -> np.ndarray:
    """The rear-axle state `state` standing where it is: no velocity, acceleration or yaw rate."""
    out = state.copy()
    moving = (
        StateIndex.VELOCITY_X,
        StateIndex.VELOCITY_Y,
        StateIndex.ACCELERATION_X,
        StateIndex.ACCELERATION_Y,
        StateIndex.ANGULAR_VELOCITY,
        StateIndex.ANGULAR_ACCELERATION,
    )
    out[list(moving)] = 0.0
    return out


# ---------------------------------------------------------------- the emergency brake
def brake_needed(state: np.ndarray, scorer: Scorer, best: int, max_speed: float | None) -> bool:
    """Whether the best proposal collides at fault soon, the ego at most `max_speed` fast.

    `max_speed` None means any speed; the speed is the rear axle's (pdm_emergency_brake.py:61-91).
    """
    ego_speed = np.hypot(state[StateIndex.VELOCITY_X], state[StateIndex.VELOCITY_Y])
    slow_enough = max_speed is None or ego_speed <= max_speed
    return bool(
        scorer.time_to_at_fault_collision(best) <= TIME_TO_INFRACTION_THRESHOLD and slow_enough
    )


def brake_poses(
    state: np.ndarray, vehicle: Vehicle, num_poses: int, interval_s: float, decel: float
) -> np.ndarray:
    """[num_poses + 1, 3] rear-axle poses braking from the rear-axle `state` [11].

    The controller's deceleration is clipped at `decel` (pdm_emergency_brake.py:93-152; centre
    acceleration as dynamic_car_state.py:13-44, 150-167).
    """
    yaw_rate = state[StateIndex.ANGULAR_VELOCITY]
    current_velocity = state[StateIndex.VELOCITY_X]
    current_acceleration = (
        state[StateIndex.ACCELERATION_X]
        + vehicle.rear_axle_to_center * yaw_rate**2
        + vehicle.rear_axle_to_center * state[StateIndex.ANGULAR_ACCELERATION]
    )
    target_velocity = 0.0
    if current_velocity > MOVING_SPEED:
        proportional_gain, derivative_gain = MOVING_GAINS
        error = -current_velocity
        dt_error = -current_acceleration
        control = proportional_gain * error + derivative_gain * dt_error
        error = max(min(control, MAX_LONG_ACCEL), -decel)
        correcting_velocity = OVERSHOOT * (current_velocity + error)
    else:
        proportional_gain, derivative_gain = STANDING_GAINS
        error = target_velocity - current_velocity
        dt_error = -current_acceleration
        control = proportional_gain * error + derivative_gain * dt_error
        correcting_velocity = max(min(control, MAX_LONG_ACCEL), -decel)
    heading = state[StateIndex.HEADING]
    cos, sin = np.cos(heading), np.sin(heading)
    center = state[:2] + vehicle.rear_axle_to_center * np.array([cos, sin])
    ahead = correcting_velocity * (interval_s * np.arange(num_poses + 1))
    centers = center[None] + ahead[:, None] * np.array([cos, sin])[None]
    rear_axles = centers - vehicle.rear_axle_to_center * np.array([cos, sin])[None]
    return np.column_stack((rear_axles, np.full(num_poses + 1, np.arctan2(sin, cos))))
