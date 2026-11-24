"""Release a standing target once the tested ego is as near in time as the reference ego was.

Targets already moving at the anchor keep fixed timing; ``lag_at_reveal`` measures the ego's lag.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapely

from ..core.rules import logged_path
from ..utils.geometry import STATION_TOL_M, wrap_angle

if TYPE_CHECKING:  # annotations only
    from ..core.types import Scene
    from ..generation.context import ReferenceSchedule

REF_SPEED_TOL = 0.3  # m/s below the reference speed that still counts as its pace
MOVE_EPS_M = 0.05  # distance from the anchor position at which the target has started
LATE_TOL_S = 0.2  # lag behind the reference at the reveal above which the test did not happen
TRIGGER_ETA_FLOOR_MPS = 0.1  # ego speed floor for the time to the trigger station
LAG_FLOOR_MPS = 1.0  # reference speed floor for turning the reveal gap into a lag
HOLD_SHIFT_S = 1e9  # onset shift meaning "never started"


class TriggeredReplay:
    """The ``target_motion`` callback of ``planners.driver.drive`` for a standing target.

    Called as ``(frame, ego_state)``; returns the target's new states once triggered, else None.
    ``move_frame`` ends the standing phase; ``started_s`` is when this ego triggered it.
    """

    def __init__(
        self, scene: Scene, states: np.ndarray, station: float, reference: ReferenceSchedule
    ):
        self.scene, self.states = scene, np.asarray(states, float)
        anchor, dt = scene.anchor, scene.dt
        # Stand at the last frame near the anchor position, so even a slow creep plays as generated.
        away = (
            np.linalg.norm(self.states[anchor:, :2] - self.states[anchor, :2], axis=1) > MOVE_EPS_M
        )
        self.move_frame = anchor + int(np.argmax(away)) - 1 if away.any() else len(states) - 1
        self.move_frame = max(self.move_frame, anchor)
        self.stand_s = (self.move_frame - anchor) * dt
        self.station, self.reference = station, reference
        self.path = logged_path(scene)
        at = min(int(np.searchsorted(reference.stations, station)), len(reference.frames) - 1)
        self.arrival_frame = reference.frames[at]  # reference ego's front arrives at the station
        self.lead_s = (self.arrival_frame - self.move_frame) * dt  # target-start to that arrival
        self.started_s = None

    def shifted_states(self, start_s: float | None) -> np.ndarray:
        """Return the target's states with its motion starting ``start_s`` after the anchor.

        ``start_s`` None means it never starts.
        """
        states, dt, anchor = self.states, self.scene.dt, self.scene.anchor
        n_frames = len(states)
        times = np.arange(n_frames) * dt
        end_s, move_s = (n_frames - 1) * dt, self.move_frame * dt
        shift_s = HOLD_SHIFT_S if start_s is None else (anchor * dt + start_s) - move_s
        past_end = times - shift_s > end_s  # it holds its last place
        waiting = times < move_s + shift_s
        source = np.clip(times - shift_s, 0, end_s)
        source = np.where(waiting, move_s, source)
        heading = np.unwrap(states[:, 2])  # across +-pi the short way
        columns = (states[:, 0], states[:, 1], heading, states[:, 3])
        moved = np.stack([np.interp(source, times, column) for column in columns], -1)
        moved[:, 2] = np.where(source <= move_s, heading[self.move_frame], moved[:, 2])
        moved[:, 2] = wrap_angle(moved[:, 2])
        moved[past_end | waiting, 3] = 0.0  # standing: speed matches the held position
        return moved

    def __call__(self, frame: int, ego_state: np.ndarray):
        if self.started_s is not None:
            return None
        eta = self._eta(frame, ego_state)
        if eta > self.lead_s:
            return None
        lag = min(self.lead_s - eta, self.scene.dt)
        self.started_s = (frame - self.scene.anchor) * self.scene.dt - lag
        return self.shifted_states(self.started_s)

    def _eta(self, frame: int, ego_state: np.ndarray) -> float:
        """Return the ego's time (s) to the trigger station, at reference pace unless it lags."""
        front = self.path.project(shapely.Point(*ego_state[:2])) + self.scene.ego.size[0] / 2
        eta = (self.station - front) / max(float(ego_state[3]), TRIGGER_ETA_FLOOR_MPS)
        stations, frames, speeds = self.reference
        if front < stations[-1]:
            here = reference_frame_at(stations, frames, front, frame)
            paced = (self.arrival_frame - here) * self.scene.dt
            reference_speed = float(np.interp(here, frames, speeds))
            eta = paced if ego_state[3] >= reference_speed - REF_SPEED_TOL else max(eta, paced)
        return eta


def reference_frame_at(stations: np.ndarray, frames: np.ndarray, front: float, frame: int) -> float:
    """Return the reference frame at which the reference front was at station ``front``.

    Where the reference stopped there, it is ``frame`` clamped to the stop, so an ego following
    the reference through a stop does not trigger the target early.
    """
    stations, frames = np.asarray(stations, float), np.asarray(frames, float)
    first = float(np.interp(front, stations, frames))  # strictly rising: unique solution
    at = int(np.searchsorted(stations, front, side="left"))
    if at < len(stations) and stations[at] - front <= STATION_TOL_M:  # at a sample: maybe a stop
        last = (
            at + int(np.searchsorted(stations[at:], stations[at] + STATION_TOL_M, side="right")) - 1
        )
        if last > at:
            return float(min(max(frame, frames[at]), frames[last]))
    return first


def lag_at_reveal(
    path: shapely.LineString,
    reference: ReferenceSchedule,
    ego_states: np.ndarray,
    reveal_frame: int,
    ego_size: np.ndarray,
) -> float:
    """Return the tested ego's lag behind the reference at the reveal frame, in seconds."""
    stations, frames, speeds = reference
    reveal_frame = int(min(max(reveal_frame, frames[0]), frames[-1], len(ego_states) - 1))
    reference_front = float(np.interp(reveal_frame, frames, stations))
    reference_speed = max(float(np.interp(reveal_frame, frames, speeds)), LAG_FLOOR_MPS)
    front = path.project(shapely.Point(*ego_states[reveal_frame, :2])) + ego_size[0] / 2
    return round((reference_front - front) / reference_speed, 2)
