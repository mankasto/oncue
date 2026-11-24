"""The IDM planner's path, plus small angle/range helpers shared across its modules."""

from __future__ import annotations

import numpy as np

from ...core.config import DriverConfig
from ...core.rules import driver_path
from ...core.types import Scene
from ...utils.geometry import heading_unit_vector, wrap_angle

HEADING_SAMPLE_M = 0.5  # chord length for reading a path's or lane's heading
ROUNDING_M = 1e-6  # points this close count as the same
RANGE_END_TOL = 1e-9  # ``inclusive_range`` reaches its stop despite float error in the steps


def inclusive_range(start: float, stop: float, step: float) -> np.ndarray:
    """Return ``[start, start + step, ..., stop]`` including the endpoint."""
    return np.arange(start, stop + RANGE_END_TOL, step)


def heading_gap(heading: float, other: float) -> float:
    """Return the absolute heading difference in ``[0, pi]``."""
    return abs(wrap_angle(heading - other))


class DriverPath:
    """The path the ego drives (``core.rules.driver_path``); also used for the curvature cap."""

    def __init__(self, scene: Scene, cfg: DriverConfig):
        self.line = driver_path(scene, cfg)
        self.length = self.line.length
        self.start_heading = float(scene.ego.states[scene.anchor, 2])

    def pose(self, station: float) -> tuple[np.ndarray, float]:
        """Return ``(position, heading)`` at ``station``, clamped to the path.

        Where the path gives no heading (at its end), the start heading is used.
        """
        station = min(max(station, 0.0), self.length)
        position = np.array(self.line.interpolate(station).coords[0])
        ahead = min(station + HEADING_SAMPLE_M, self.length)
        toward = np.array(self.line.interpolate(ahead).coords[0])
        if np.linalg.norm(toward - position) < ROUNDING_M:
            start = heading_unit_vector(self.start_heading)
            toward = position + HEADING_SAMPLE_M * start
        return position, float(np.arctan2(toward[1] - position[1], toward[0] - position[0]))
