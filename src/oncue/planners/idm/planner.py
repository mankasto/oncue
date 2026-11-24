"""IDM planner for the closed loop, ``IDMPlanner``: unoccluded or masked perception.

The ego keeps its logged path and controls only its speed: each frame it takes the IDM
acceleration under road users, signals, stop signs, crossing vehicles and phantom bounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely
import shapely.affinity
import shapely.ops

from ...core.config import DriverConfig
from ...core.rules import TrafficRules, phase_at
from ...core.types import Scene
from ...utils.geometry import BOX_CORNERS, box_corners, heading_unit_vector
from ..perception import Beliefs, Perception
from .crossings import HEADING_MATCH_DEG, PARKED_MPS, crossing_conflicts, phantoms
from .path import ROUNDING_M, DriverPath, heading_gap, inclusive_range

if TYPE_CHECKING:
    from ...core.types import Track
    from .crossings import Conflict

PREDICT_STEP_S = 0.25  # time step for predicted road-user positions
STOPPED_MPS = 0.3  # under this counts as stopped (a road user, or the ego at a stop sign)
ENTRY_ETA_FLOOR_MPS = 0.5  # the ego's time-to-point ahead uses at least this speed
ENTRY_BEFORE_S = 1.0  # a road user entering the corridor this long before the ego gets there ...
ENTRY_AFTER_S = 0.5  # ... to this long after is an obstacle there
GAP_FLOOR_M = 0.1  # IDM gap floor (an obstacle at the ego's front: hardest braking)
PASSED_M = 0.5  # a stop line this far behind the ego's front is passed
STOP_SIGN_NEAR_M = 4.0  # the ego stopped within this of a stop sign waits there
DESIRED_FLOOR_MPS = 0.5  # the rules' speed cap is at least this
SPEED_FLOOR_MPS = 0.1  # floor for speeds that divide into IDM ratios
CAUTION_STOP_M = 1.0  # the ego stays able to stop this far before a phantom pedestrian's spot
TARGET_BRAKE_MPS2 = 1.0  # braking harder than this for the target counts as braking for it
# road users the ego keeps yielding to until they have crossed
REMEMBERED_TYPES = ("pedestrian",)
IDM_EXPONENT = 4  # IDM acceleration exponent (delta)
RAMP_PARTS = 20  # parts the step's acceleration ramp is integrated in (``ramp_step``)
STOP_BISECTIONS = 20  # bisection depth of ``_stop_bound``; final error ~(jerk_max dt * 2) / 2^20
# ``sync_to`` searches the vehicle's progress this far either side of the planner's station
SYNC_WINDOW_M = 10.0
# Gap-acceptance tunables (``DriverConfig.yield_gap_s``).
CROSSING_MPS = 1.0  # a vehicle slower than this is not crossing; no gap to accept
YIELD_ETA_FLOOR_MPS = 2.0  # the ego's reach/clear times use at least this speed
YIELD_STOP_M = 1.0  # the ego yields this far before the point a crossing vehicle takes
# Phantom-vehicle tunables.
PHANTOM_STOP_M = 1.0  # the ego stops this far before a conflict zone it may not enter
CONFLICT_PAST_M = 1.0  # a crossing this far behind the ego's front counts as past


class IDMPlanner:
    """One-drive driver, one step per frame (``v_des`` and ``rules`` from ``driver.drive``)."""

    # plans from its own station and speed, synced to the executed vehicle by ``sync_to``
    plans_from_own_state = True

    def __init__(
        self,
        scene: Scene,
        target_id: str | None,
        cfg: DriverConfig,
        perception: Perception,
        *,
        v_des: float,
        rules: TrafficRules | None = None,
    ):
        ego, anchor = scene.ego, scene.anchor
        self.target_id, self.cfg, self.perception = target_id, cfg, perception
        self.v_des, self.rules = v_des, rules
        self.anchor, self.dt = anchor, scene.dt
        self.path = DriverPath(scene, cfg)
        self.ego_size = ego.size
        self.corridor_half_width = ego.size[1] / 2 + cfg.corridor_margin_m
        self.half_length = ego.size[0] / 2
        self.stopped_at: dict[float, int] = {}  # stop line -> frames the ego has stopped there
        self.cleared: set[float] = set()  # stop lines passed or already waited at
        self.yielding: set[str] = set()  # road users of REMEMBERED_TYPES the ego has yielded to
        self.station, self.speed = 0.0, float(ego.states[anchor, 3])  # along the path
        self.accel = 0.0  # last step's acceleration, used as the jerk-limit base
        self.conflicts = (
            crossing_conflicts(scene, self.path, self.corridor_half_width, cfg, rules)
            if cfg.phantom_brake is not None
            else []
        )
        self.stations: list[float] = []  # per-step records (mirroring ``Drive``)
        self.speeds: list[float] = []
        self.accels: list[float] = []
        self.target_brake_s: float | None = None

    def state(self) -> np.ndarray:
        """Return the ego's current state ``(x, y, heading, speed)`` at its station on the path."""
        position, heading = self.path.pose(self.station)
        return np.array([position[0], position[1], heading, self.speed])

    def sync_to(self, executed: np.ndarray) -> None:
        """Take the executed vehicle's progress and speed as the planner's own.

        The closed-loop executor calls this before each step. Progress is searched within
        ``SYNC_WINDOW_M`` of the station, so a path passing near itself never makes it jump.
        """
        first = max(self.station - SYNC_WINDOW_M, 0.0)
        last = min(self.station + SYNC_WINDOW_M, self.path.length)
        window = shapely.ops.substring(self.path.line, first, last)
        self.station = first + float(window.project(shapely.Point(executed[:2])))
        self.speed = float(executed[3])

    def step(
        self, frame: int, ego_states: np.ndarray, scene: Scene, beliefs: Beliefs
    ) -> np.ndarray:
        """Return the ego's state at the next frame, given its beliefs at ``frame``.

        Call once per frame from the anchor on, after ``Perception.observe``. ``ego_states
        [T, 4]`` is indexed by scene frame; of ``scene`` only the map and road-user types and
        sizes are read.
        """
        obs_frame, believed = beliefs.frame, beliefs.states
        ego_heading = ego_states[obs_frame][2]
        constraints = self._road_user_constraints(believed, ego_heading, scene.tracks)
        constraints += self._signal_constraints(obs_frame)
        constraints += self._stop_sign_constraints()
        constraints += self._yield_constraints(believed, ego_heading, scene.tracks)
        desired, pedestrian_bound = self._phantom_pedestrians(
            believed, scene.tracks, self._desired_speed()
        )
        vehicle_bound = self._phantom_vehicles(beliefs, ego_states[obs_frame], scene, desired)
        accel, binding = self._accel(desired, constraints, vehicle_bound, pedestrian_bound)
        accel = self._jerk_limited(accel)
        self._record(frame, accel, binding)
        self.station, self.speed = self._advance(accel)
        self.accel = accel if self.speed > 0 else 0.0  # a stopped ego carries no acceleration
        return self.state()

    def _advance(self, accel: float) -> tuple[float, float]:
        """Return ``(station, speed)`` one step on, ramping from the last step's acceleration."""
        return ramp_step(self.station, self.speed, self.accel, accel, self.dt)

    def _jerk_limited(self, accel: float) -> float:
        """Clip ``accel`` to ``cfg.jerk_max`` per second from the last step's acceleration."""
        change = self.cfg.jerk_max * self.dt
        return min(max(accel, self.accel - change), self.accel + change)

    # ------------------------------------------------ stopping within a distance
    def _stops_within(self, accel: float, room: float, braking: float) -> bool:
        """Return whether a step at ``accel`` then braking at ``braking`` fits in ``room``."""
        moved, speed = ramp_step(0.0, self.speed, self.accel, accel, self.dt)
        after = accel if speed > 0 else 0.0  # a stopped ego carries no acceleration (``step``)
        stopping = stop_distance(speed, after, braking, self.cfg.jerk_max, self.dt)
        return moved + stopping <= room

    def _allowed_accels(self) -> tuple[float, float]:
        """Return this step's lowest and highest acceleration under the bounds and jerk limit."""
        change = self.cfg.jerk_max * self.dt
        low = max(self.accel - change, -self.cfg.b_max)
        return low, max(min(self.accel + change, self.cfg.a_max), low)

    def _can_stop(self, room: float, braking: float) -> bool:
        """Return whether the ego can still stop within ``room`` braking at ``braking``."""
        return self._stops_within(self._allowed_accels()[0], room, braking)

    def _stop_bound(self, room: float, braking: float) -> float:
        """Return the largest acceleration this step that still fits a stop in ``room``.

        Found by bisection; the lowest allowed one when even that is too late.
        """
        low, high = self._allowed_accels()
        if self._stops_within(high, room, braking):
            return high
        if not self._stops_within(low, room, braking):
            return low
        for _ in range(STOP_BISECTIONS):
            middle = (low + high) / 2
            if self._stops_within(middle, room, braking):
                low = middle
            else:
                high = middle
        return low

    def _accel_to(self, speed: float) -> float:
        """Return the acceleration that ends this step at ``speed``, ramping from the last one."""
        return 2 * (speed - self.speed) / self.dt - self.accel

    # ------------------------------------------------ constraints: (gap, lead speed, owner)
    def _road_user_constraints(
        self, believed: dict[str, np.ndarray], ego_heading: float, tracks: dict[str, Track]
    ) -> list:
        """Return known road users as leaders in the corridor or obstacles where they enter it."""
        constraints = []
        for road_user, state in believed.items():
            track = tracks[road_user]
            box = shapely.Polygon(box_corners(state[None], track.size)[0])
            ego_heading_gap = heading_gap(state[2], ego_heading)
            leader = self._leader(road_user, state, box, ego_heading_gap)
            if leader is not None:
                constraints.append(leader)
                continue
            if ego_heading_gap < np.radians(HEADING_MATCH_DEG) or state[3] < STOPPED_MPS:
                continue  # same-way traffic beside the corridor, or standing beside it
            entry = self._entry(road_user, track.agent_type, state, box)
            if entry is not None:
                constraints.append(entry)
        return constraints

    def _leader(
        self, road_user: str, state: np.ndarray, box: shapely.Polygon, ego_heading_gap: float
    ):
        """Return the road user as a leader at its path speed if it is in the corridor ahead."""
        line = self.path.line
        corners = [line.project(shapely.Point(c)) for c in box.exterior.coords[:BOX_CORNERS]]
        if line.distance(box) <= self.corridor_half_width and max(corners) > self.station:
            return (
                max(min(corners) - self.station - self.half_length, GAP_FLOOR_M),
                max(state[3] * np.cos(ego_heading_gap), 0.0),
                road_user,
            )
        return None

    def _entry(self, road_user: str, agent_type: str, state: np.ndarray, box: shapely.Polygon):
        """Return the road user as an obstacle where it enters the corridor as the ego arrives.

        One of ``REMEMBERED_TYPES`` stays an obstacle while it is predicted into the corridor.
        """
        entry = self._first_entry(state, box)
        if entry is None:
            self.yielding.discard(road_user)
            return None
        ahead_s, station = entry
        eta = (station - self.station) / max(self.speed, ENTRY_ETA_FLOOR_MPS)
        if road_user not in self.yielding and not (
            eta - ENTRY_BEFORE_S <= ahead_s <= eta + ENTRY_AFTER_S
        ):
            return None
        if agent_type in REMEMBERED_TYPES:
            self.yielding.add(road_user)
        return (max(station - self.station - self.half_length, GAP_FLOOR_M), 0.0, road_user)

    def _first_entry(self, state: np.ndarray, box: shapely.Polygon):
        """Return ``(time ahead, station)`` where the road user first enters the corridor ahead.

        ``None`` when that is behind the ego or beyond ``predict_s``.
        """
        speed, heading, line = state[3], state[2], self.path.line
        for ahead_s in inclusive_range(PREDICT_STEP_S, self.cfg.predict_s, PREDICT_STEP_S):
            moved = shapely.affinity.translate(
                box, speed * np.cos(heading) * ahead_s, speed * np.sin(heading) * ahead_s
            )
            if line.distance(moved) > self.corridor_half_width:
                continue
            station = min(
                line.project(shapely.Point(c)) for c in moved.exterior.coords[:BOX_CORNERS]
            )
            return None if station <= self.station else (ahead_s, station)
        return None

    def _signal_constraints(self, obs_frame: int) -> list:
        """Return each applicable signal as a stopped obstacle at its stop line.

        Red always applies; yellow only while the ego can still stop at ``b_comf``.
        """
        if self.rules is None:
            return []
        out = []
        for stop_line, phase in self.rules.signals:
            gap = stop_line - self.station - self.half_length
            if gap < -PASSED_M:
                continue
            light = phase_at(phase, obs_frame)
            stoppable = gap > self.speed * self.speed / (2 * self.cfg.b_comf)
            if light == "stop" or (light == "caution" and stoppable):
                out.append((max(gap, GAP_FLOOR_M), 0.0, None))
        return out

    def _stop_sign_constraints(self) -> list:
        """Return each stop sign as a stopped obstacle until cleared.

        A stop sign clears once the ego has stood near it for ``stop_sign_wait_s`` or passed it.
        """
        if self.rules is None:
            return []
        out = []
        for stop_line in self.rules.stop_signs:
            if stop_line in self.cleared:
                continue
            gap = stop_line - self.station - self.half_length
            if gap < -PASSED_M:
                self.cleared.add(stop_line)
                continue
            if gap < STOP_SIGN_NEAR_M and self.speed < STOPPED_MPS:
                self.stopped_at[stop_line] = self.stopped_at.get(stop_line, 0) + 1
                if self.stopped_at[stop_line] * self.dt >= self.cfg.stop_sign_wait_s:
                    self.cleared.add(stop_line)
                    continue
            out.append((max(gap, GAP_FLOOR_M), 0.0, None))
        return out

    def _yield_constraints(
        self, believed: dict[str, np.ndarray], ego_heading: float, tracks: dict[str, Track]
    ) -> list:
        """Return gap-acceptance constraints for crossing vehicles (with ``yield_gap_s`` set).

        The ego stops before where a vehicle takes the path if it would be there within
        ``yield_gap_s`` of the ego's passage, unless it is too late to stop.
        """
        cfg, line, station, speed = self.cfg, self.path.line, self.station, self.speed
        half_length = self.half_length
        if cfg.yield_gap_s is None:
            return []
        out = []
        for road_user, state in believed.items():
            track = tracks[road_user]
            if track.agent_type != "vehicle" or state[3] < CROSSING_MPS:
                continue
            if heading_gap(state[2], ego_heading) < np.radians(HEADING_MATCH_DEG):
                continue
            corners = box_corners(state[None], track.size)[0]
            crossing_speed, heading = state[3], state[2]
            # Its box at every time ahead at once, as translated corners.
            ahead_s = inclusive_range(0.0, cfg.yield_horizon_s, PREDICT_STEP_S)
            shift = np.stack(
                (
                    crossing_speed * np.cos(heading) * ahead_s,
                    crossing_speed * np.sin(heading) * ahead_s,
                ),
                axis=1,
            )
            moved = corners[None] + shift[:, None]  # [time, 4, 2]
            distance = shapely.distance(line, shapely.polygons(moved))
            inside = np.flatnonzero(distance <= self.corridor_half_width)
            if not len(inside):
                continue
            entry = float(shapely.line_locate_point(line, shapely.points(moved[inside[0]])).min())
            enter_s, leave_s = ahead_s[inside[0]], ahead_s[inside[-1]]
            if entry <= station + half_length:
                continue  # already at or past the ego's front
            gap = entry - station - half_length
            if gap < speed * speed / (2 * cfg.b_max):
                continue  # too late to stop; commit
            pace = max(speed, YIELD_ETA_FLOOR_MPS)
            reach_s, clear_s = gap / pace, (gap + 2 * half_length + track.size[1]) / pace
            if enter_s < clear_s + cfg.yield_gap_s and leave_s > reach_s - cfg.yield_gap_s:
                out.append((max(gap - YIELD_STOP_M, GAP_FLOOR_M), 0.0, road_user))
        return out

    # ------------------------------------------------ the desired speed and its lower bounds
    def _desired_speed(self) -> float:
        """Return ``v_des`` under the rules' speed cap, scaled by ``desired_speed_factor``."""
        desired = self.v_des
        if self.rules is not None:
            desired = min(desired, max(self.rules.speed_cap(self.station), DESIRED_FLOOR_MPS))
        return self.cfg.desired_speed_factor * desired

    def _phantom_pedestrian_spots(
        self, believed: dict[str, np.ndarray], tracks: dict[str, Track]
    ) -> list:
        """Return both ends of every parked vehicle in range (occlusion-aware masked mode only)."""
        spots = []
        if self.cfg.occlusion_aware and self.perception.masked:
            for road_user, state in believed.items():
                track = tracks[road_user]
                if track.agent_type == "vehicle" and state[3] < PARKED_MPS:
                    direction = heading_unit_vector(state[2])
                    spots += [
                        state[:2] + track.size[0] / 2 * direction,
                        state[:2] - track.size[0] / 2 * direction,
                    ]
        return spots

    def _phantom_pedestrians(
        self,
        believed: dict[str, np.ndarray],
        tracks: dict[str, Track],
        desired: float,
    ):
        """Return ``(desired, accel bound)`` for phantom-pedestrian spots beside the path.

        The desired speed drops to what stops at ``b_caution`` before each spot, never below
        ``caution_v_min``; the bound is needed as IDM's free term only eases toward it.
        """
        cfg, line = self.cfg, self.path.line
        bound = None
        for xy in self._phantom_pedestrian_spots(believed, tracks):
            point = shapely.Point(*xy)
            station = line.project(point)
            lateral = line.distance(point) - self.ego_size[1] / 2
            if lateral > cfg.caution_lateral_m or station <= self.station:
                continue
            room = max(station - self.station - self.half_length - CAUTION_STOP_M, 0.0)
            slow = max(np.sqrt(2 * cfg.b_caution * room), cfg.caution_v_min)
            desired = min(desired, slow)
            accel = max(self._stop_bound(room, cfg.b_caution), self._accel_to(cfg.caution_v_min))
            if bound is None or accel < bound:
                bound = accel
        return desired, bound

    def _phantom_vehicles(self, beliefs: Beliefs, eye: np.ndarray, scene: Scene, desired: float):
        """Return the acceleration bound that keeps the ego safe from phantom vehicles, or None."""
        if not self.conflicts:
            return None
        front, top_speed = self.station + self.half_length, max(desired, self.speed)
        near = self._near_conflicts(front, top_speed)
        bound = None
        for conflict, arrival_s in phantoms(near, beliefs, eye, scene, self.perception):
            accel = self._phantom_stop(conflict, front, top_speed, arrival_s)
            if accel is not None:
                bound = accel if bound is None else min(bound, accel)
        return bound

    def _near_conflicts(self, front: float, top_speed: float) -> list:
        """Return the conflicts ahead within a bound on the ego's stopping distance."""
        cfg, near = self.cfg, []
        turn_s = (cfg.a_max + cfg.phantom_brake) / cfg.jerk_max
        stop = top_speed * (self.dt + turn_s) + top_speed * top_speed / (2 * cfg.phantom_brake)
        for conflict in self.conflicts:
            if front > conflict.station + CONFLICT_PAST_M:
                continue  # past it
            if conflict.station - front < stop + PHANTOM_STOP_M + (
                conflict.station - conflict.zone[0]
            ):
                near.append(conflict)
        return near

    def _phantom_stop(self, conflict: Conflict, front: float, top_speed: float, arrival_s: float):
        """Return the acceleration bound for one phantom-vehicle conflict, or ``None``.

        Keeps the ego able to stop ``PHANTOM_STOP_M`` before the zone at ``phantom_brake``, unless
        it is in the zone, cannot stop in time, or clears the zone well before the phantom.
        """
        cfg = self.cfg
        to_zone = conflict.zone[0] - front
        if to_zone < -ROUNDING_M:
            return None  # in the zone: clearing it beats stopping in it
        room = max(to_zone - PHANTOM_STOP_M, 0.0)
        if not self._can_stop(room, cfg.b_max):
            return None
        clear = _time_to_cover(
            conflict.zone[1] + self.half_length - self.station, self.speed, cfg.a_max, top_speed
        )
        if clear + cfg.phantom_margin_s <= arrival_s:
            return None
        return self._stop_bound(room, cfg.phantom_brake)

    # ------------------------------------------------ the speed update
    def _accel(
        self,
        desired: float,
        constraints: list,
        vehicle_bound: float | None,
        pedestrian_bound: float | None,
    ):
        """Return ``(accel, binding)`` under every constraint and bound.

        ``binding`` is the road user it is for, or ``None`` for a phantom or a rule.
        """
        cfg, speed = self.cfg, self.speed
        free = self._free_road_accel(desired)
        accel, binding = free, None
        for gap, lead, constraint_owner in constraints:
            desired_gap = (
                cfg.s0_m
                + speed * cfg.t_head_s
                + speed * (speed - lead) / (2 * np.sqrt(cfg.a_max * cfg.b_comf))
            )
            constrained = free - cfg.a_max * (max(desired_gap, 0.0) / gap) ** 2
            if constrained < accel:
                accel, binding = constrained, constraint_owner
        if vehicle_bound is not None and vehicle_bound < accel:
            accel, binding = vehicle_bound, None
        if pedestrian_bound is not None and pedestrian_bound < accel:
            accel, binding = pedestrian_bound, None
        return float(np.clip(accel, -cfg.b_max, cfg.a_max)), binding

    def _free_road_accel(self, desired: float) -> float:
        """Return IDM's free-road term toward ``desired``.

        With ``comfort_free_brake`` and the speed above ``desired``, the improved-IDM term
        (Treiber and Kesting 2013, sec. 11.3.7) slows the ego at most at ``b_comf``.
        """
        cfg, speed = self.cfg, self.speed
        desired = max(desired, SPEED_FLOOR_MPS)
        if cfg.comfort_free_brake and speed > desired:
            exponent = cfg.a_max * IDM_EXPONENT / cfg.b_comf
            return -cfg.b_comf * (1 - (desired / speed) ** exponent)
        return cfg.a_max * (1 - (speed / desired) ** IDM_EXPONENT)

    def _record(self, frame: int, accel: float, binding: str | None) -> None:
        braking = accel < -TARGET_BRAKE_MPS2 and self.target_brake_s is None
        if self.target_id is not None and binding == self.target_id and braking:
            self.target_brake_s = (frame - self.anchor) * self.dt
        self.stations.append(self.station)
        self.speeds.append(self.speed)
        self.accels.append(accel)


def ramp_step(
    station: float, speed: float, accel_from: float, accel_to: float, dt: float
) -> tuple[float, float]:
    """Return ``(station, speed)`` one step on with a linearly ramping acceleration.

    Integrated as the oracle's ``longitudinal_profile``; a stop within the step stays put.
    """
    part = dt / RAMP_PARTS
    for k in range(RAMP_PARTS):
        a = accel_from + (accel_to - accel_from) * (k + 0.5) / RAMP_PARTS  # midpoint of the part
        after = speed + a * part
        if after <= 0:  # stops within this part
            return station + (speed * speed / (-2 * a) if a < 0 else 0.0), 0.0
        station, speed = station + (speed + after) / 2 * part, after
    return station, speed


def stop_distance(speed: float, accel: float, braking: float, jerk: float, dt: float) -> float:
    """Return the distance to a stop, turning toward ``-braking`` at most ``jerk`` per second."""
    distance, change = 0.0, jerk * dt
    while speed > 0:
        goal = min(max(-braking, accel - change), accel + change)
        distance, speed = ramp_step(distance, speed, accel, goal, dt)
        accel = goal
    return distance


def _time_to_cover(distance: float, speed: float, accel: float, top_speed: float) -> float:
    """Return the time to cover ``distance`` from ``speed`` at ``accel`` up to ``top_speed``."""
    if distance <= 0:
        return 0.0
    top_speed = max(top_speed, speed, SPEED_FLOOR_MPS)
    speeding_up_s = (top_speed - speed) / accel
    speeding_up_m = (speed + top_speed) / 2 * speeding_up_s
    if distance <= speeding_up_m:
        return float((-speed + np.sqrt(speed * speed + 2 * accel * distance)) / accel)
    return float(speeding_up_s + (distance - speeding_up_m) / top_speed)
