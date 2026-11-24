"""What one closed-loop run counts as: its contacts, braking, motion, road use and scene checks.

Every target contact is a failure; a background contact counts only when nuPlan's at-fault rules
blame the ego.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import VEHICLE_LIMITS
from ..core.kinematics import beyond_physical, measures
from ..core.types import Track
from ..generation.checks.realism import near_miss
from ..generation.checks.validity import UNMEASURED, background_contacts
from ..generation.targets import TargetSpec
from ..oracle.checks import Roads, clearances
from ..planners import driver as ego_driver
from ..planners.idm.planner import TARGET_BRAKE_MPS2
from ..utils.geometry import heading_unit_vector
from ..utils.tracks import TIME_TOL_S, frame_of, substep_states

if TYPE_CHECKING:  # annotations only
    from numpy.typing import ArrayLike

    from ..core.config import VisibilityConfig
    from ..core.types import Scene


# Deceleration onset: the first step whose deceleration exceeds this (m/s^2).
DECELERATION_ONSET_MPS2 = TARGET_BRAKE_MPS2

# Contacts with the ego slower than this are reported separately as ``hit_moving``.
STANDING_CONTACT_MPS = 1.0

# Generation flaw each scene check rediscovers (see ``generation.checks.realism.FLAWS``).
SCENE_FLAWS = {"scene_background": "moving_collision", "scene_near_miss": "near_miss"}

# A logged target on its logged schedule gets no scene checks: its interactions are the log's.
LOGGED_TARGET_CHECKS = {
    "scene_background": None,
    "scene_near_miss": None,
    "scene_ok": True,
    "scene_flaw_known": None,
    "scene_unchecked": "logged",
}


def ego_background_contacts(
    scene: Scene, run: ego_driver.Drive, target_id: str | None, substeps: int, roads: Roads
) -> dict[str, list]:
    """Return the run's ego contacts with every road user but the target.

    Maps a road user id to ``[frame, at_fault]``; a contact between two frames counts at the next.
    """
    anchor = scene.anchor
    end = int(run.valid_until) - 1  # last simulated frame
    ego = scene.ego.with_states(run.states, run.ego_valid)
    times, states, sampled = substep_states(ego, scene.timestamps, anchor, end, substeps)
    times = times[sampled]
    background = replace(
        scene, tracks={key: track for key, track in scene.tracks.items() if key != target_id}
    )
    found = clearances(states[sampled][None], ego.size, times, background, roads)
    contacts = {}
    for track_id, first in sorted(found.first_contact.items()):
        if first[0] >= 0:
            frame = int(np.searchsorted(scene.timestamps, times[first[0]] - TIME_TOL_S))
            contacts[track_id] = [frame, bool(found.per_actor[track_id][0] <= 0)]
    return contacts


def background_record(contacts: dict[str, list]) -> dict:
    """Split ``ego_background_contacts`` into at-fault and excused ones for the run record."""
    at_fault = [track_id for track_id, (_, fault) in contacts.items() if fault]
    excused = [track_id for track_id, (_, fault) in contacts.items() if not fault]
    return {
        "background_contacts": contacts,
        "other_contact": bool(at_fault),
        "other_contact_with": at_fault,
        "excused": bool(excused),
        "excused_by": excused,
    }


def drove_into_ego(contacts: dict[str, list]) -> dict[str, int]:
    """Return ``{road_user: frame}`` of the excused contacts: road users that drove into the ego."""
    return {track_id: frame for track_id, (frame, fault) in contacts.items() if not fault}


def contact_record(
    run: ego_driver.Drive,
    target_states: np.ndarray,
    anchor: int,
    dt: float,
    cutoff: int | None = None,
) -> dict:
    """Return how a run met the target.

    ``hit`` is a contact before ``cutoff`` (None: any time); a later one is ``after_cutoff``.
    """
    cutoff_s = None if cutoff is None else round((cutoff - anchor) * dt, 2)
    if run.target_contact_s is None:
        return {
            "hit": False,
            "hit_moving": False,
            "after_cutoff": False,
            "cutoff_s": cutoff_s,
            "contact_s": None,
            "contact_speed": None,
            "rel_speed": None,
        }
    frame = anchor + frame_of(run.target_contact_s, dt)
    hit = cutoff is None or frame < cutoff
    ego, target = run.states[frame], np.asarray(target_states, float)[frame]
    ego_velocity = ego[3] * heading_unit_vector(ego[2])
    target_velocity = target[3] * heading_unit_vector(target[2])
    speed = float(run.target_contact_speed)
    return {
        "hit": hit,
        "hit_moving": hit and speed >= STANDING_CONTACT_MPS,
        "after_cutoff": not hit,
        "cutoff_s": cutoff_s,
        "contact_s": run.target_contact_s,
        "contact_speed": round(speed, 2),
        "rel_speed": round(float(np.linalg.norm(ego_velocity - target_velocity)), 2),
    }


def deceleration_onset_s(run: ego_driver.Drive, anchor: int, dt: float) -> float | None:
    """Return seconds after the anchor of the first hard braking once the target was known.

    Read from the driven states, so slowing for any other reason counts too. None if never.
    """
    if run.target_known_s is None:
        return None
    speeds = run.states[anchor : run.valid_until, 3]
    decelerations = (speeds[:-1] - speeds[1:]) / dt  # over each step, decided at its start
    known = frame_of(run.target_known_s, dt)
    slowed = np.flatnonzero(decelerations[known:] > DECELERATION_ONSET_MPS2)
    return None if len(slowed) == 0 else round((known + int(slowed[0])) * dt, 2)


def ego_measures(
    run: ego_driver.Drive, anchor: int, dt: float, upto: int | None = None
) -> dict | None:
    """Return ``core.kinematics.measures`` of the run's ego from the anchor to ``anchor + upto``."""
    return measures(run.states[anchor : run.valid_until, :2], dt, upto=upto)


def ego_kinematics(run: ego_driver.Drive, anchor: int, dt: float) -> dict:
    """Return the ego's peak measures and the vehicle limits they exceed, for the run record.

    A run too short to measure lists ``UNMEASURED`` so it never counts as a pass.
    """
    found = ego_measures(run, anchor, dt)
    return {
        "ego_peaks": None if found is None else {k: round(v, 3) for k, v in found.items()},
        "beyond_limits": [UNMEASURED] if found is None else beyond_physical(found, VEHICLE_LIMITS),
    }


def road_record(
    run: ego_driver.Drive, roads: Roads, size: ArrayLike, anchor: int, dt: float
) -> dict:
    """Return when the run's ego first left the road or drove against traffic (None: never).

    Judged on the scene's own map in every world; ``road_unknown`` marks leaving the map.
    """
    states = run.states[anchor : run.valid_until]
    return {
        "out_of_road_s": _first_time(roads.edges.offroad(states, size), dt),
        "wrong_way_s": _first_time(roads.lanes.opposing(states[:, None]), dt),
        "road_unknown": bool(np.isnan(roads.edges.depth(states, size)).any()),
    }


def _first_time(flags: np.ndarray, dt: float) -> float | None:
    """Return seconds from frame 0 to the first ``True`` in ``flags``; ``None`` if never."""
    found = np.flatnonzero(flags)
    return None if not len(found) else round(float(found[0]) * dt, 2)


def traffic_key(run: ego_driver.Drive) -> str:
    """Return a full-precision digest of the simulated ego for the scene-check cache."""
    digest = hashlib.sha256(
        np.ascontiguousarray(run.states, np.float64).tobytes()
        + np.asarray(run.ego_valid, bool).tobytes()
    )
    return digest.hexdigest()


def scene_checks(
    scene: Scene,
    target: Track,
    spec: TargetSpec,
    vis: VisibilityConfig,
    static: object,
    until: int | None,
    substeps: int,
    flaws: dict,
    retimed: bool,
) -> dict:
    """Check the target for contacts and near misses with the background, anchor to ``until``.

    A finding invalidates the scene only when the run ``retimed`` the target and the sample does
    not already carry it as a flaw. ``until`` None means the scene's end.
    """
    valid = target.valid.copy()
    valid[: scene.anchor] = False
    if until is not None:
        valid[until:] = False
    target = replace(target, valid=valid)
    with_target = replace(scene, tracks={**scene.tracks, target.track_id: target})
    contacts = background_contacts(with_target, target.track_id, substeps)
    end = len(scene.timestamps) - 1 if until is None else until - 1  # the last frame checked
    touched = {track_id for track_id, _ in contacts}
    near = near_miss(with_target, target, spec.near_miss, end, vis, static, touched)
    found = {
        "scene_background": list(contacts[0]) if contacts else None,
        "scene_near_miss": near,
    }
    seen = [field for field, value in found.items() if value is not None]
    unknown = [field for field in seen if not flaws.get(SCENE_FLAWS[field])]
    known = not unknown if seen else None
    return {**found, "scene_ok": not (retimed and unknown), "scene_flaw_known": known}


def simulated_traffic(
    scene: Scene, run: ego_driver.Drive, gone: dict[str, int] | None = None
) -> Scene:
    """Return the scene with the ego from ``run``; road users in ``gone`` vanish at their frame."""
    tracks = {**scene.tracks, scene.ego_id: scene.ego.with_states(run.states, run.ego_valid)}
    for track_id, frame in (gone or {}).items():
        tracks[track_id] = tracks[track_id].absent_from(frame)
    return replace(scene, tracks=tracks)
