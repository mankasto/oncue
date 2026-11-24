"""Where a test ends: the first frame from contact on where the target leaves the route band.

A frame where the target is unobserved also ends it, with its own reason. Every method, planner
and perception mode shares this cutoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import shapely

from ..utils.geometry import box_corners
from .config import DriverConfig

if TYPE_CHECKING:  # annotations only
    from numpy.typing import ArrayLike

# Why a test ended (``Cutoff.reason``).
LEFT_ROUTE = "left_route"
TARGET_UNOBSERVED = "target_unobserved"
RUN_END = "run_end"


class Cutoff(NamedTuple):
    """The first frame a test does not judge, and why."""

    frame: int
    reason: str


def route_band(route_xy: np.ndarray, ego_width: float) -> shapely.Geometry:
    """Return ``route_xy`` [N, 2] widened by ``ego_width / 2`` plus the corridor margin
    (``DriverConfig.corridor_margin_m``) each side, flat-ended."""
    line = shapely.LineString(np.asarray(route_xy, float)[:, :2])
    return line.buffer(ego_width / 2 + DriverConfig.corridor_margin_m, cap_style="flat")


def cutoff_frame(
    band: shapely.Geometry,
    states: np.ndarray,
    valid: np.ndarray,
    size: ArrayLike,
    contact_frame: int,
    end_frame: int,
) -> Cutoff:
    """Return the first frame in ``[contact_frame, end_frame)`` where the target is off ``band``.

    An unobserved frame also counts; with neither, the cutoff is ``end_frame`` (``RUN_END``).
    """
    frames = np.arange(contact_frame, end_frame)
    observed = np.asarray(valid, bool)[frames]
    boxes = shapely.polygons(box_corners(np.asarray(states, float)[frames, :3], size))
    off = np.flatnonzero(~(observed & shapely.intersects(boxes, band)))
    if not len(off):
        return Cutoff(end_frame, RUN_END)
    first = off[0]
    return Cutoff(int(frames[first]), LEFT_ROUTE if observed[first] else TARGET_UNOBSERVED)
