"""Per-step records for planners that move the ego themselves (PDM-Closed, LatentDriver)."""

from __future__ import annotations

import numpy as np
import shapely


class StepRecords:
    """Per-step station, speed and speed change (m/s^2) for ``planners.driver.drive``."""

    def __init__(self, route: shapely.LineString, dt: float):
        self.route, self.dt = route, dt
        self.stations: list[float] = []
        self.speeds: list[float] = []
        self.accels: list[float] = []

    def record_step(self, state: np.ndarray, next_state: np.ndarray) -> None:
        """Record one step from ``state`` to ``next_state`` (``(x, y, heading, speed)``)."""
        self.stations.append(float(self.route.project(shapely.Point(*state[:2]))))
        self.speeds.append(float(state[3]))
        self.accels.append(float((next_state[3] - state[3]) / self.dt))
