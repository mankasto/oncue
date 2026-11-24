"""Checks on executed oracle candidates: dynamics, road containment, clearance, lane direction.

``evaluate.Oracle._check`` runs them in that order on all candidates' executed states at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..core.config import OracleConfig
from ..core.fault import contact_kind, ego_at_fault
from ..core.road import LaneDirections, RoadEdges
from ..core.rules import lane_speed_limit
from ..core.types import STATE_SIZE, Scene
from ..core.vehicle import Execution
from ..utils.geometry import box_clearance, box_corners
from ..utils.tracks import track_at

if TYPE_CHECKING:  # for annotations only: no new import when running
    from numpy.typing import ArrayLike

CLEARANCE_CAP_M = 10.0
CLEARANCE_TOL_M = 1e-6  # clearances within this tolerance are treated as equal


def samples(execution: Execution, vehicle_substeps: int, per_frame: int) -> np.ndarray:
    """Return indices into the dense execution timeline used for checks (frames included)."""
    step = max(1, vehicle_substeps // per_frame)
    return np.arange(0, len(execution.times), step)


def dynamics_violations(execution: Execution, cfg: OracleConfig) -> list[tuple[str, ...]]:
    """Return per-row dynamics violations, each a tuple from {``"tracking"``, ``"speed"``}.

    A direct steer has no path (NaN tracking error), so it never reports ``"tracking"``.
    """
    speed = execution.states[..., 3]
    reasons = []
    for r in range(len(speed)):
        found = []
        error = execution.tracking_error_m[r]
        if np.isfinite(error) and error > cfg.max_tracking_error_m:
            found.append("tracking")
        if speed[r].max() > cfg.max_speed:
            found.append("speed")
        reasons.append(tuple(found))
    return reasons


def outside_region(states: np.ndarray, size: ArrayLike, region: shapely.Geometry) -> np.ndarray:
    """Return per row whether any sampled footprint leaves ``region``, a prepared geometry."""
    footprints = shapely.polygons(box_corners(states, size))
    return ~shapely.covers(region, footprints).all(axis=1)


def clearances(
    states: np.ndarray,
    size: ArrayLike,
    times: np.ndarray,
    scene: Scene,
    roads: Roads | None = None,
    never_excused: tuple[str, ...] = (),
) -> Clearances:
    """Return clearances between every candidate ``[R, S, >=4]`` and every other road user.

    With ``roads``, contacts the ego is not at fault for are excused, except with road users in
    ``never_excused``.
    """
    per_actor, excused, first_contact = {}, {}, {}
    ego_radius = np.linalg.norm(size) / 2
    xy = states[..., :2].reshape(-1, 2)
    low_xy, high_xy = xy.min(axis=0), xy.max(axis=0)  # every candidate position
    for other in scene.others():
        if not other.valid.any():
            continue
        pose, valid = track_at(other, scene.timestamps, times)
        radius = ego_radius + np.linalg.norm(other.size) / 2
        if not valid.any():
            continue
        # Skip a road user that stays beyond the cap from every candidate position.
        seen = pose[valid, :2]
        apart = np.maximum(np.maximum(low_xy - seen.max(axis=0), seen.min(axis=0) - high_xy), 0)
        if np.hypot(*apart) - radius >= CLEARANCE_CAP_M + CLEARANCE_TOL_M:
            continue
        gap = np.linalg.norm(states[..., :2] - pose[None, :, :2], axis=-1) - radius
        near = valid[None] & (gap < CLEARANCE_CAP_M)
        if not near.any():
            continue
        exact = np.full(near.shape, CLEARANCE_CAP_M)
        # The circle gap is a lower bound: box-test each candidate's nearest pair first, then
        # only pairs that could still beat that minimum or touch.
        rows = np.flatnonzero(near.any(axis=1))
        first = np.where(near, gap, np.inf)[rows].argmin(axis=1)
        _fill(exact, states, size, pose, other.size, rows, first)
        bound = exact.min(axis=1)[:, None]
        todo = near & ((gap < bound + CLEARANCE_TOL_M) | (gap <= CLEARANCE_TOL_M))
        todo[rows, first] = False
        _fill(exact, states, size, pose, other.size, *np.nonzero(todo))
        contact = exact <= 0
        first = np.where(contact.any(axis=1), contact.argmax(axis=1), -1)
        cleared = np.zeros(contact.shape, bool)  # marks the samples of excused contacts
        if roads is not None and other.track_id not in never_excused:
            cleared = _not_at_fault(states, size, pose, other.size, contact, roads)
        per_actor[other.track_id] = np.where(cleared, CLEARANCE_CAP_M, exact).min(axis=1)
        excused[other.track_id] = cleared.any(axis=1)
        first_contact[other.track_id] = first
    return Clearances(per_actor, excused, first_contact)


@dataclass(frozen=True, eq=False)
class Roads:
    """Scene geometry for the fault, traffic-direction and speeding checks."""

    lanes: LaneDirections
    edges: RoadEdges
    speed_limit: float  # m/s, measured along the ego's path

    @classmethod
    def of(cls, scene: Scene) -> Roads:
        """Build a ``Roads`` from the scene's lanes, edges and posted limit."""
        return cls(LaneDirections(scene), RoadEdges(scene), lane_speed_limit(scene))

    def speeding(self, speeds: np.ndarray, tol_mps: float) -> np.ndarray:
        """Return per-row flags saying whether the max speed exceeds the limit by ``tol_mps``."""
        return speeds.max(axis=1) > self.speed_limit + tol_mps

    def spans_lanes_or_offroad(self, states: np.ndarray, size: ArrayLike) -> np.ndarray:
        """Return per-box flags for "not within one lane running its way" or "leaves the road"."""
        return self.lanes.spans_lanes(states, size[1]) | self.edges.offroad(states, size)


@dataclass(frozen=True, eq=False)
class Clearances:
    """Per-road-user clearance results for a batch of candidates."""

    # Per road user, each ``[R]``:
    per_actor: dict[str, np.ndarray]  # minimum signed distance, capped; excused contacts ignored
    excused: dict[str, np.ndarray]  # whether any contact was excused
    first_contact: dict[str, np.ndarray]  # first sample in contact, excused ones too; -1: none

    def nearest(self, rows: int, skip: str | None = None) -> tuple[np.ndarray, list[str | None]]:
        """Return the smallest clearance per response and the id of the road user it is to."""
        best = np.full(rows, CLEARANCE_CAP_M)
        who: list[str | None] = [None] * rows
        for track_id, low in self.per_actor.items():
            if track_id == skip:
                continue
            for r in np.flatnonzero(low < best):
                best[r], who[r] = low[r], track_id
        return best, who

    def excused_by(self, r: int) -> list[str]:
        """Return the sorted road-user ids whose contact with response ``r`` was excused."""
        return sorted(k for k, hit in self.excused.items() if hit[r])


def _fill(
    exact: np.ndarray,
    states: np.ndarray,
    size: ArrayLike,
    pose: np.ndarray,
    other_size: ArrayLike,
    r: np.ndarray,
    s: np.ndarray,
) -> None:
    """Fill ``exact[r, s]`` with the capped clearance of response ``r`` at sample ``s``."""
    exact[r, s] = np.minimum(
        box_clearance(states[r, s, :STATE_SIZE], size, pose[s], other_size, CLEARANCE_CAP_M),
        CLEARANCE_CAP_M,
    )


def _not_at_fault(
    states: np.ndarray,
    size: ArrayLike,
    pose: np.ndarray,
    other_size: ArrayLike,
    contact: np.ndarray,
    roads: Roads,
) -> np.ndarray:
    """Return ``[R, S]`` marking contact samples the ego is not at fault for.

    Fault is judged where each contact begins and holds for the whole contact.
    """
    begins = contact & ~np.pad(contact, ((0, 0), (1, 0)))[:, :-1]
    r, s = np.nonzero(begins)
    if not len(r):
        return np.zeros(contact.shape, bool)
    ego, other = states[r, s], pose[s]
    sideways = roads.spans_lanes_or_offroad(ego, size)
    excused = np.zeros(contact.shape, bool)
    excused[r, s] = [
        not ego_at_fault(contact_kind(e, e[3], size, o, o[3], other_size), side)
        for e, o, side in zip(ego, other, sideways, strict=True)
    ]
    # Propagate each contact's verdict from its first sample across the whole run.
    began = np.maximum.accumulate(np.where(begins, np.arange(contact.shape[1]), 0), axis=1)
    return contact & np.take_along_axis(excused, began, axis=1)
