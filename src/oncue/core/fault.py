"""Fault rule for ego contacts with background road users, from nuPlan's at-fault collisions.

Shared by the oracle and the closed loop; target contacts always count against the ego.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..utils.geometry import box_corners, wrap_angle

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

STOPPED_MPS = 0.05  # nuPlan `pdm_scorer_utils.is_track_stopped`
BEHIND_DEG = 150.0  # nuPlan `is_agent_behind`


class ContactKind(str, Enum):
    STOPPED_EGO = "stopped_ego"
    STOPPED_OTHER = "stopped_other"
    REAR = "rear"
    FRONT = "front"
    LATERAL = "lateral"


def contact_kind(
    ego: np.ndarray,
    ego_speed: float,
    ego_size: ArrayLike,
    other: np.ndarray,
    other_speed: float,
    other_size: ArrayLike,
) -> ContactKind:
    """Classify the contact starting between ``ego`` and ``other``.

    Poses are ``(x, y, heading, ...)``; sizes are ``(length, width)``.
    """
    if ego_speed <= STOPPED_MPS:
        return ContactKind.STOPPED_EGO
    if other_speed <= STOPPED_MPS:
        return ContactKind.STOPPED_OTHER
    if _bearing_deg(ego, other) > BEHIND_DEG:
        return ContactKind.REAR
    front = shapely.LineString(_front_edge(np.asarray(ego[:3], float), ego_size))
    if front.intersects(shapely.Polygon(box_corners(np.asarray(other[:3], float), other_size))):
        return ContactKind.FRONT
    return ContactKind.LATERAL


def _bearing_deg(ego: np.ndarray, other: np.ndarray) -> float:
    """Return how far off the ego's heading ``other`` lies, in degrees (0 to 180)."""
    to_other = np.asarray(other[:2], float) - np.asarray(ego[:2], float)
    return abs(np.degrees(wrap_angle(np.arctan2(to_other[1], to_other[0]) - ego[2])))


def ego_at_fault(kind: ContactKind, spans_lanes_or_offroad: bool) -> bool:
    """Return whether a background road user's contact of ``kind`` counts against the ego."""
    if kind in (ContactKind.STOPPED_OTHER, ContactKind.FRONT):
        return True
    return kind is ContactKind.LATERAL and spans_lanes_or_offroad


# How the other road user reached the ego. A same-direction rear-end (the ego cut in) does not
# count as the other attacking the ego.
APPROACH_AHEAD = "ahead"
APPROACH_CROSSING = "crossing"
APPROACH_SAME_DIRECTION = "same_direction"
APPROACH_HEADING_GAP_DEG = 90.0
FRONT_HALF_DEG = 90.0  # a road user within this bearing of the ego's heading is ahead of it


def approach_at_contact(ego: np.ndarray, other: np.ndarray) -> str:
    """Return how ``other`` reached ``ego`` at a contact, one of the ``APPROACH_*`` values."""
    off = _bearing_deg(ego, other)
    heading_gap = abs(np.degrees(wrap_angle(float(other[2]) - float(ego[2]))))
    if off > BEHIND_DEG and heading_gap <= APPROACH_HEADING_GAP_DEG:
        return APPROACH_SAME_DIRECTION
    if off <= FRONT_HALF_DEG and heading_gap > APPROACH_HEADING_GAP_DEG:
        return APPROACH_AHEAD
    return APPROACH_CROSSING


def _front_edge(pose: np.ndarray, size) -> np.ndarray:
    """Return the two front corners of a box at ``pose`` (centre, heading) and ``size``."""
    half_length, half_width = np.asarray(size, float) / 2
    forward = np.array([np.cos(pose[2]), np.sin(pose[2])])
    left = np.array([-forward[1], forward[0]])
    tip = pose[:2] + half_length * forward
    return np.stack([tip + half_width * left, tip - half_width * left])
