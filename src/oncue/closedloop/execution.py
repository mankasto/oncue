"""Executor that moves every closed-loop planner's ego with ``core.vehicle.execute``.

The planner proposes, nuPlan's LQR tracker drives, and the world sees the executed ego.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.tracker import TRACKING_HORIZON
from ..core.types import STATE_SIZE
from ..core.vehicle import execute, initial_steering
from ..utils.geometry import ZERO_LENGTH_M, wrap_angle

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable
    from typing import Any

    from ..core.config import DriverConfig, VehicleConfig
    from ..core.types import Scene
    from ..planners.perception import Beliefs, Perception

# Time constant (s) over which the ego's sideways offset from a planner's own path decays.
REJOIN_S = 1.0
STEERING_COLUMN = 4  # index of the steering angle in an executed state: after x, y, heading, speed


def executed(build: Callable[..., Any]) -> Callable[..., ExecutedPlanner]:
    """Wrap a planner builder so its steps run through ``core.vehicle.execute``.

    ``build`` has the signature ``planners.driver.drive`` expects.
    """

    def build_executed(
        scene: Scene,
        target_id: str | None,
        cfg: DriverConfig,
        perception: Perception,
        **options: Any,
    ) -> ExecutedPlanner:
        planner = build(scene, target_id, cfg, perception, **options)
        return ExecutedPlanner(planner, scene, cfg.vehicle)

    return build_executed


class ExecutedPlanner:
    """A planner whose proposals ``core.vehicle.execute`` carries out; other attributes pass on."""

    def __init__(self, planner, scene: Scene, vehicle: VehicleConfig):
        self.planner, self.vehicle, self.dt = planner, vehicle, scene.dt
        anchor = scene.anchor
        self.steering = initial_steering(scene.ego, anchor, scene.dt, vehicle, anchor)
        # Acceleration applied on the last executed step; None means use the first step's own.
        self.accel: float | None = None

    def __getattr__(self, name: str):
        return getattr(self.planner, name)

    def state(self) -> np.ndarray:
        """Return the ego's state at the anchor (the planner's own state)."""
        return self.planner.state()

    def step(
        self, frame: int, ego_states: np.ndarray, scene: Scene, beliefs: Beliefs
    ) -> np.ndarray:
        """Return the executed state ``[x, y, heading, speed]`` one frame after ``frame``.

        Tracks the planner's ``planned_states`` when it has them, else its proposed step held.
        """
        now = ego_states[frame]
        sync = getattr(self.planner, "sync_to", None)
        if sync is not None:  # IDM-style: plan from the vehicle's progress and speed
            sync(now)
        own = self.planner.state() if getattr(self.planner, "plans_from_own_state", False) else now
        aim = np.array(self.planner.step(frame, ego_states, scene, beliefs), float)
        aim[:2] += (now[:2] - own[:2]) * (1.0 - self.dt / REJOIN_S)
        planned = getattr(self.planner, "planned_states", None)
        if planned is None:
            desired, start = held_step(now, aim, self.dt, TRACKING_HORIZON), None
        else:
            desired, start = planned, now[None, :STATE_SIZE]
        times = self.dt * np.arange(len(desired))
        execution = execute(
            desired[None], times, self.steering, self.vehicle, accel0=self.accel, start=start
        )
        reached = execution.states[0, self.vehicle.substeps]  # one frame on
        self.steering = float(reached[STEERING_COLUMN])
        # The ego never reverses: hold still rather than roll back.
        if reached[3] < 0:
            self.accel = 0.0
            return np.array([*now[:3], 0.0])
        self.accel = float((reached[3] - now[3]) / self.dt)
        return reached[:STATE_SIZE].copy()


def held_step(now: np.ndarray, aim: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """Return ``[steps + 1, 4]`` desired states every ``dt``: ``now``, ``aim``, then that step held.

    The step's acceleration and turn rate are held with speed floored at 0. A step that covers
    no ground keeps its heading, so the states never spin in place.
    """
    aim = np.array(aim[:STATE_SIZE], float)
    if np.hypot(*(aim[:2] - now[:2])) <= ZERO_LENGTH_M:
        aim[2] = now[2]
    accel, turn = (aim[3] - now[3]) / dt, float(wrap_angle(aim[2] - now[2]))
    out = [np.asarray(now, float)[:STATE_SIZE], aim]
    for _ in range(steps - 1):
        x, y, heading, speed = out[-1]
        after = max(speed + accel * dt, 0.0)
        covered = (speed + after) / 2 * dt
        step_turn = turn if covered > 0 else 0.0
        middle = heading + step_turn / 2
        out.append(
            np.array(
                [
                    x + covered * np.cos(middle),
                    y + covered * np.sin(middle),
                    heading + step_turn,
                    after,
                ]
            )
        )
    return np.stack(out)
