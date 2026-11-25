"""A minimal outside planner for OnCue's closed loop, as a template for your own.

It drives along the test's route at the desired speed and brakes hard while a road user it knows
is in its lane ahead. Importing this module registers it as ``cautious``:

    python -m oncue closedloop labels.jsonl scenes --spec pedestrian \\
        --planner-module my_planner --planners cautious,pdm_closed --out closedloop.jsonl

Run the command from this folder (or put it on PYTHONPATH) so ``my_planner`` can be imported.

The planner contract (``planners.driver.drive``): the planner is built as
``Planner(scene, target_id, cfg, perception, v_des=, rules=)`` once per run; ``state()`` returns
the ego's state ``(x, y, heading, speed)`` at the test's start; ``step(frame, ego_states, scene,
beliefs)`` returns the state it wants one frame later, which a vehicle model and tracker then
execute; ``stations``, ``speeds``, ``accels`` and ``target_brake_s`` are read after the run.
``beliefs.states`` holds what the planner knows: under masked observation only road users the ego
has seen.
"""

from __future__ import annotations

import numpy as np
import shapely

from oncue.closedloop.execution import executed
from oncue.closedloop.registry import register, run_planner
from oncue.core.rules import driver_path
from oncue.planners.step_records import StepRecords
from oncue.utils.geometry import line_heading

ACCEL_MPS2 = 1.5  # speeding up to the desired speed
BRAKE_MPS2 = 6.0  # braking for a road user in the lane ahead
LOOKAHEAD_M = 25.0  # how far ahead along the route a road user counts
LANE_HALF_WIDTH_M = 2.0  # how far beside the route a road user counts
HEADING_CHORD_M = 1.0  # chord for the route's heading at a station
TARGET_BRAKE_MPS2 = 1.0  # braking at least this hard for the target counts as braking for it


class CautiousPlanner:
    """Route following at the desired speed; hard braking for a known road user in the lane."""

    def __init__(self, scene, target_id, cfg, perception, v_des, rules):
        self.route = driver_path(scene, cfg)  # the logged route the test was built on
        self.dt, self.anchor, self.target_id, self.v_des = scene.dt, scene.anchor, target_id, v_des
        self.start = np.array(scene.ego.states[scene.anchor][:4], float)
        self.records = StepRecords(self.route, scene.dt)  # stations, speeds and accels per step
        self.target_brake_s: float | None = None  # first braking the target causes

    @property
    def stations(self) -> list[float]:
        return self.records.stations

    @property
    def speeds(self) -> list[float]:
        return self.records.speeds

    @property
    def accels(self) -> list[float]:
        return self.records.accels

    def state(self) -> np.ndarray:
        return self.start

    def step(self, frame: int, ego_states: np.ndarray, scene, beliefs) -> np.ndarray:
        now = ego_states[frame]
        station = self.route.project(shapely.Point(*now[:2]))
        blocker = self.blocker(station, beliefs)
        if blocker is None:
            speed = min(now[3] + ACCEL_MPS2 * self.dt, self.v_des)
        else:
            speed = max(now[3] - BRAKE_MPS2 * self.dt, 0.0)
            if (
                blocker == self.target_id
                and self.target_brake_s is None
                and now[3] - speed >= TARGET_BRAKE_MPS2 * self.dt
            ):
                self.target_brake_s = (frame - self.anchor) * self.dt
        ahead = min(station + (now[3] + speed) / 2 * self.dt, self.route.length)
        point = self.route.interpolate(ahead)
        heading = line_heading(self.route, ahead, back=HEADING_CHORD_M, ahead=HEADING_CHORD_M)
        planned = np.array([point.x, point.y, heading, speed])
        self.records.record_step(now, planned)
        return planned

    def blocker(self, station: float, beliefs) -> str | None:
        """The nearest known road user in the lane ahead, or None."""
        nearest, found = LOOKAHEAD_M, None
        for road_user, state in beliefs.states.items():
            point = shapely.Point(*state[:2])
            gap = self.route.project(point) - station
            if 0.0 < gap < nearest and self.route.distance(point) < LANE_HALF_WIDTH_M:
                nearest, found = gap, road_user
        return found


def drive_cautious(scene, target_id, cfg, **options):
    """One closed-loop run of ``CautiousPlanner``, executed like the built-in planners."""
    return run_planner(scene, target_id, cfg, executed(CautiousPlanner), **options)


register("cautious", drive_cautious)
