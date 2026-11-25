"""Tunable parameters grouped by stage: frozen, validated, loadable from JSON.

``Config.load`` rejects unknown keys, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_type_hints

RIGHT_ANGLE_DEG = 90


@dataclass(frozen=True)
class PhysicalLimits:
    """What a road user type can physically do; None means no limit and no check.

    Targets, egos, oracle responses and planners all read these, never their own copies.
    """

    speed: float | None = None  # m/s
    accel: float | None = None  # m/s^2, speeding up
    brake: float | None = None  # m/s^2, slowing down (positive)
    brake_jerk: float | None = None  # m/s^3, brake onset
    lateral: float | None = None  # m/s^2
    lateral_jerk: float | None = None  # m/s^3
    curvature: float | None = None  # 1/m


VEHICLE_LIMITS = PhysicalLimits(
    accel=6.0, brake=8.3, brake_jerk=30.0, lateral=6.4, lateral_jerk=16.0, curvature=0.3
)
# Sprint speed and acceleration from standing.
PEDESTRIAN_LIMITS = PhysicalLimits(speed=12.3, accel=10.0)
CYCLIST_LIMITS = PhysicalLimits(brake=8.5)  # bicycle pitch-over
LIMITS_BY_TYPE = {
    "vehicle": VEHICLE_LIMITS,
    "pedestrian": PEDESTRIAN_LIMITS,
    "cyclist": CYCLIST_LIMITS,
}
# How far the ego perceives, a reveal counts and the oracle may use early information.
PERCEPTION_RANGE_M = 80.0
PHANTOM_RANGE_M = 60.0  # phantom vehicles stay inside perception
SPEEDING_TOL_MPS = 0.655  # how far above the posted limit a speed is still no speeding


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(obj: object, *names: str) -> None:
    for name in names:
        value = getattr(obj, name)
        _check(
            not isinstance(value, bool) and math.isfinite(value) and value > 0,
            f"{type(obj).__name__}.{name} must be positive and finite",
        )


@dataclass(frozen=True)
class VisibilityConfig:
    """Visibility rule: ego-centre rays to the target centre and its four corners.

    A blocker blocks a ray only where taller than the sight line from ``sensor_height_m`` to
    ``visible_margin_m`` below the target's top. Static occluders lie farther than
    ``static_setback_m`` from open space. Either one None turns that part off.
    """

    threshold: float = 0.2  # visible fraction of the 5 rays; below = occluded
    min_occluded_s: float = 0.5
    min_visible_frames: int = 3
    occluder_types: tuple[str, ...] = ("vehicle",)
    occluder_max_accel: float = 4.0  # m/s^2, bounds where an occluder could be during a gap
    occluder_linger_s: float = 1.0  # before its first / after its last observation
    sensor_height_m: float | None = 2.3
    visible_margin_m: float = 0.8
    static_setback_m: float | None = 4.0
    perception_range_m: float = PERCEPTION_RANGE_M  # a reveal counts only this near the ego

    def __post_init__(self):
        _positive(
            self,
            "threshold",
            "min_occluded_s",
            "min_visible_frames",
            "occluder_max_accel",
            "occluder_linger_s",
            "perception_range_m",
        )
        _check(self.threshold <= 1, "VisibilityConfig.threshold must be <= 1")
        _check(self.sensor_height_m is None or self.sensor_height_m > 0, "sensor height > 0")
        _check(self.visible_margin_m >= 0, "visible margin >= 0")
        _check(self.static_setback_m is None or self.static_setback_m > 0, "static setback > 0")


# Settings of the archived seed mining, kept because every output's settings record has them.
@dataclass(frozen=True)
class MiningConfig:
    horizon_s: float = 8.0  # search window after the anchor
    target_types: tuple[str, ...] = ("vehicle", "pedestrian", "cyclist")
    lane_match_m: float = 1.25
    heading_tolerance_deg: float = 60.0
    max_route_m: float = 100.0
    max_route_nodes: int = 256
    max_lane_hops: int = 12
    max_join_gap_m: float = 2.0
    routes_per_intent: int = 2
    pedestrian_extension_m: float = 20.0
    lane_change_m: float = 25.0  # length of a lane change into a same-direction neighbour lane
    lane_change_lead_m: float = 3.0  # earliest start ahead of the target
    around_occluder_m: float = 4.0  # occluders this close to a pedestrian get a detour route
    window_station_step_m: float = 2.0  # reveal-window sampling along a route
    window_time_step_s: float = 0.5
    block_stop_before_m: float = 2.0  # blocking is checked up to this far before the conflict
    block_moved_m: float = 1.0  # a blocker moves less than this ...
    block_hold_s: float = 1.0  # ... until this long after the ego reaches the conflict

    def __post_init__(self):
        _positive(
            self,
            "horizon_s",
            "lane_match_m",
            "heading_tolerance_deg",
            "max_route_m",
            "max_route_nodes",
            "max_lane_hops",
            "max_join_gap_m",
            "routes_per_intent",
            "pedestrian_extension_m",
            "lane_change_m",
            "lane_change_lead_m",
            "around_occluder_m",
            "window_station_step_m",
            "window_time_step_s",
            "block_stop_before_m",
            "block_moved_m",
            "block_hold_s",
        )
        _check(
            self.heading_tolerance_deg < RIGHT_ANGLE_DEG,
            "heading tolerance must be below 90 degrees",
        )


@dataclass(frozen=True)
class VehicleConfig:
    """Geometry of the executor's kinematic bicycle; what it can do is in ``VEHICLE_LIMITS``."""

    wheelbase: float = 2.7
    center_to_rear_axle: float = 1.35
    substeps: int = 10  # executed samples per scene frame

    def __post_init__(self):
        _positive(self, "wheelbase", "center_to_rear_axle", "substeps")
        _check(self.center_to_rear_axle < self.wheelbase, "center must lie between the axles")

    @property
    def max_steering(self) -> float:
        """Return the steering angle (rad) at the curvature limit."""
        return math.atan(VEHICLE_LIMITS.curvature * self.wheelbase)


@dataclass(frozen=True)
class Longitudinal:
    """A longitudinal response: ramp to ``accel`` at ``jerk``; braking ends in a jerk-limited stop.

    ``stop_at_conflict`` picks the deceleration within ``accel_range`` that stops ``stop_margin_m``
    before the first conflict with the target instead. A recovery switches to ``then_accel``
    either ``switch_s`` after the response starts or ``wait_s`` after the ego first stands still.
    """

    name: str
    accel: float  # m/s^2
    jerk: float  # m/s^3
    stop_at_conflict: bool = False
    accel_range: tuple[float, float] = (-VEHICLE_LIMITS.brake, -1.0)
    stop_margin_m: float = 1.0
    switch_s: float | None = None
    wait_s: float | None = None
    then_accel: float | None = None
    then_jerk: float = 5.0

    def __post_init__(self):
        _positive(self, "jerk", "stop_margin_m", "then_jerk")
        _check(math.isfinite(self.accel), f"{self.name}: finite acceleration required")
        _check(self.accel_range[0] < self.accel_range[1] < 0, f"{self.name}: braking range")
        timed = (self.switch_s is not None) + (self.wait_s is not None)
        _check(
            timed == (self.then_accel is not None),
            f"{self.name}: a recovery needs then_accel and one of switch_s, wait_s",
        )
        _check(not (timed and self.stop_at_conflict), f"{self.name}: no recovery of a stop")


@dataclass(frozen=True)
class Lateral:
    """A lateral response: a smooth offset from the ego path, or a direct steering command.

    An offset (``offset_m``, lane widths when ``in_lanes``) is laid out over the distance driven
    in ``duration_s``; ``hold`` keeps it, ``return`` goes out and back. ``steer`` commands
    ``steer_rad`` (left positive), back to zero after ``return_s`` (None: held).
    """

    name: str
    offset_m: float = 0.0
    duration_s: float = 0.0
    mode: str = "hold"
    in_lanes: bool = False
    steer_rad: float = 0.0
    return_s: float | None = None

    def __post_init__(self):
        _check(self.mode in ("hold", "return", "steer"), f"{self.name}: hold, return or steer")
        _check(self.offset_m == 0 or self.duration_s > 0, f"{self.name}: duration required")
        steer = self.mode == "steer"
        _check(steer == (self.steer_rad != 0), f"{self.name}: steer_rad only for steer, nonzero")
        _check(steer or self.return_s is None, f"{self.name}: return_s only for steer")
        _check(not steer or self.offset_m == 0, f"{self.name}: a steer has no offset")


def _templates(
    nudges: tuple[float, ...],
    nudge_s: tuple[float, ...],
    swerves: tuple[float, ...],
    swerve_s: tuple[float, ...],
    lane_s: tuple[float, ...],
) -> tuple[Lateral, ...]:
    return (
        Lateral("keep"),
        *(Lateral("nudge", d, t) for d in nudges for t in nudge_s),
        *(Lateral("swerve", d, t, "return") for d in swerves for t in swerve_s),
        *(Lateral("lane_change", d, t, in_lanes=True) for d in (-1.0, 1.0) for t in lane_s),
    )


_MAINTAIN = Longitudinal("maintain", 0.0, 5.0)
_BRAKE = Longitudinal("brake", -6.0, 10.0)
_EMERGENCY = Longitudinal("emergency_brake", -VEHICLE_LIMITS.brake, VEHICLE_LIMITS.brake_jerk)
_STOP = Longitudinal("stop_at_conflict", -1.0, 10.0, stop_at_conflict=True)


@dataclass(frozen=True)
class CandidateConfig:
    """Oracle candidate list (``oracle.responses.candidate_list``).

    Core: ``longitudinal`` x ``lateral``; the fixed longitudinals x the moving laterals started
    ``delay_s`` later; ``recover`` x ``lateral``; ``steer_longitudinal`` x ``steer``. Then the
    refinement: ``refine_longitudinal`` x ``refine_lateral`` and, delayed,
    ``refine_delayed_longitudinal`` x ``refine_delayed_lateral``, minus entries already in the
    core.
    """

    longitudinal: tuple[Longitudinal, ...] = (
        Longitudinal("accelerate", 1.5, 5.0),
        Longitudinal("accelerate_hard", 3.0, 10.0),
        _MAINTAIN,
        Longitudinal("decelerate", -2.0, 5.0),
        Longitudinal("decelerate", -4.0, 5.0),
        _BRAKE,
        _EMERGENCY,
        _STOP,
    )
    lateral: tuple[Lateral, ...] = _templates(
        (-1.0, -0.5, 0.5, 1.0), (2.0,), (-2.5, -1.5, 1.5, 2.5), (4.0,), (2.5, 4.0)
    )
    delay_s: float = 0.5
    recover: tuple[Longitudinal, ...] = (
        Longitudinal("recover", -2.0, 5.0, switch_s=1.0, then_accel=1.5),
        Longitudinal("recover", -4.0, 5.0, switch_s=1.0, then_accel=1.5),
        Longitudinal("recover", -4.0, 5.0, switch_s=2.0, then_accel=1.5),
        Longitudinal("recover", -6.0, 10.0, wait_s=0.5, then_accel=1.5),
    )
    steer: tuple[Lateral, ...] = tuple(
        Lateral("steer", mode="steer", steer_rad=side * a, return_s=back)
        for side in (1.0, -1.0)
        for a in (0.1, 0.2, 0.4)
        for back in (None, 0.8)
    )
    steer_longitudinal: tuple[Longitudinal, ...] = (
        _MAINTAIN,
        Longitudinal("decelerate", -4.0, 5.0),
        _BRAKE,
        _EMERGENCY,
    )
    # The refinement stays inside the core's envelope: no brake onset faster than
    # ``_EMERGENCY``, no offset or duration outside the core's ranges.
    refine_longitudinal: tuple[Longitudinal, ...] = (
        Longitudinal("accelerate", 1.5, 5.0),
        Longitudinal("accelerate_hard", 3.0, 10.0),
        _MAINTAIN,
        *(
            Longitudinal("decelerate", -a, 5.0)
            for a in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
        ),
        *(Longitudinal("brake", -a, 10.0) for a in (6.0, 7.0)),
        _EMERGENCY,
        _STOP,
    )
    refine_lateral: tuple[Lateral, ...] = _templates(
        (-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5),
        (1.0, 2.0),
        (-3.0, -2.75, -2.5, -2.25, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0),
        (3.0, 3.5, 4.0),
        (2.0, 2.5, 3.0, 4.0),
    )
    refine_delayed_longitudinal: tuple[Longitudinal, ...] = (
        _BRAKE,
        Longitudinal("brake", -7.0, 10.0),
        _EMERGENCY,
    )
    refine_delayed_lateral: tuple[Lateral, ...] = (
        *(
            Lateral("swerve", d, t, "return")
            for d in (-3.0, -2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5, 3.0)
            for t in (3.0, 4.0)
        ),
        *(Lateral("lane_change", d, t, in_lanes=True) for d in (-1.0, 1.0) for t in (2.0, 2.5)),
    )
    default_lane_width_m: float = 3.6
    # Offsets are laid out in distance: ``duration x max(takeover speed, this)``.
    min_maneuver_speed: float = 3.0

    def __post_init__(self):
        _check(len(self.longitudinal) > 0 and len(self.lateral) > 0, "empty candidate list")
        _check(all(lat.mode != "steer" for lat in self.lateral), "steers go in `steer`")
        _check(all(lat.mode == "steer" for lat in self.steer), "`steer` holds steers only")
        _positive(self, "default_lane_width_m", "min_maneuver_speed", "delay_s")


def with_emergency_jerk(candidates: CandidateConfig, jerk: float) -> CandidateConfig:
    """Return ``candidates`` with the emergency brake ramping at ``jerk`` (m/s^3) instead."""
    slow = replace(_EMERGENCY, jerk=jerk)
    changed = {}
    for f in fields(candidates):
        value = getattr(candidates, f.name)
        if isinstance(value, tuple) and _EMERGENCY in value:
            changed[f.name] = tuple(slow if entry == _EMERGENCY else entry for entry in value)
    return replace(candidates, **changed)


@dataclass(frozen=True)
class ReplayConfig:
    """Replayed scene rule: fill missing states only where the log leaves no doubt.

    Gaps up to ``bridge_s`` are interpolated. A road user slower than ``still_mps`` at its first
    or last observation stands there for at most ``end_fill_s`` either side, as long as its box
    touches no other road user's logged box; the same across a longer gap whose two standing
    ends are within ``same_place_m``.
    """

    bridge_s: float = 1.0
    still_mps: float = 0.5
    same_place_m: float = 1.0
    end_fill_s: float = 1.0

    def __post_init__(self):
        _check(
            min(self.bridge_s, self.still_mps, self.same_place_m, self.end_fill_s) >= 0,
            "replay settings must be >= 0",
        )


@dataclass(frozen=True)
class OracleConfig:
    """Budgeted oracle: the first ``budget`` candidates per information condition."""

    reaction_delay_s: float = 0.0  # none
    # The log must go on at least this long past the reference contact (not the judged window).
    cover_contact_s: float | None = 1.0
    budget: int = 288  # core of the candidate list
    collision_substeps: int = 5  # clearance samples per frame; divides ``vehicle.substeps``
    max_speed: float = 40.0
    max_tracking_error_m: float = 0.5
    speeding_tol_mps: float = SPEEDING_TOL_MPS
    path_extension: str = "lanes"  # straight | lanes: how the ego path continues past the log
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)

    def __post_init__(self):
        _positive(
            self,
            "budget",
            "collision_substeps",
            "max_speed",
            "max_tracking_error_m",
        )
        _check(self.reaction_delay_s >= 0, "reaction delay must be nonnegative")
        _check(self.speeding_tol_mps >= 0, "speeding tolerance must be nonnegative")
        _check(self.cover_contact_s is None or self.cover_contact_s >= 0, "cover_contact_s >= 0")
        _check(self.path_extension in ("straight", "lanes"), "path_extension: straight | lanes")
        _check(
            self.vehicle.substeps % self.collision_substeps == 0,
            "collision_substeps must divide vehicle.substeps (the same sample times everywhere)",
        )


@dataclass(frozen=True)
class DriverConfig:
    """Rule-based ego driver along its logged path, for the reference ego and the IDM planners.

    An IDM held back by road users in the corridor, traffic rules and a curvature cap. Kept
    outside ``Config`` so run digests do not depend on it.
    """

    a_max: float = 2.5
    b_comf: float = 3.0
    b_max: float = VEHICLE_LIMITS.brake
    jerk_max: float = _EMERGENCY.jerk  # m/s^3, as the oracle's emergency brake onset
    t_head_s: float = 1.2
    s0_m: float = 2.0
    corridor_margin_m: float = 0.4  # added to the ego's half width
    predict_s: float = 3.0
    # Conservative-IDM alpha: scales the desired speed (and rule caps). 1.0 disables.
    desired_speed_factor: float = 1.0
    # Above the desired speed, brake at most at ``b_comf`` (improved-IDM free-road term).
    comfort_free_brake: bool = False
    perception_range_m: float = PERCEPTION_RANGE_M  # both masked and unoccluded
    perception_period_s: float = 0.1
    # The masked planner coasts a road user at constant velocity this long after last seeing it.
    memory_s: float = 1.0
    latency_s: float = 0.0  # extra perception latency, both modes
    # Contact samples per frame step (``swept_overlap``); the closed loop sets the oracle's.
    collision_substeps: int = 5
    # Occlusion-aware (masked only): able to stop at ``b_caution`` for a phantom pedestrian by a
    # stationary vehicle within ``caution_lateral_m`` of the path, never below ``caution_v_min``.
    occlusion_aware: bool = False
    b_caution: float = 3.0
    caution_lateral_m: float = 3.5
    caution_v_min: float = 2.0
    lateral_accel: float = 2.0  # curvature cap
    anticipate_decel: float = 2.0  # braking into the curvature cap
    stop_sign_wait_s: float = 1.0
    # Gap acceptance: stop before a conflict point a known crossing or oncoming vehicle occupies
    # within ``yield_gap_s`` of the ego's passage, looking ``yield_horizon_s`` ahead (None: off).
    yield_gap_s: float | None = None
    yield_horizon_s: float = 6.0
    # Ignore road users behind the ego along the path (a replayed follower never reacts).
    ignore_behind: bool = False
    # Phantom vehicles: an unseen vehicle on each lane crossing ahead, at ``phantom_speed_factor``
    # x its limit. The ego clears the conflict ``phantom_margin_s`` before it arrives or stays
    # able to stop at ``phantom_brake`` (None: off).
    phantom_brake: float | None = None
    phantom_speed_factor: float = 1.1
    phantom_margin_s: float = 1.5
    phantom_range_m: float = PHANTOM_RANGE_M
    horizon_s: float = 8.0
    extension_m: float = 60.0  # path extension beyond the log (``core.rules.driver_path``)
    # The executor's vehicle; the closed loop sets the oracle's (``closedloop.scene.test_driver``).
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    def __post_init__(self):
        _positive(
            self,
            "a_max",
            "b_comf",
            "b_max",
            "jerk_max",
            "t_head_s",
            "predict_s",
            "perception_period_s",
        )
        _check(self.memory_s >= 0 and self.latency_s >= 0, "memory and latency >= 0")
        _check(0 < self.desired_speed_factor <= 1, "desired speed factor in (0, 1]")
        _positive(self, "phantom_speed_factor", "phantom_range_m", "perception_range_m")
        _check(self.phantom_range_m < self.perception_range_m, "phantoms within perception")
        _check(self.phantom_margin_s >= 0, "phantom margin >= 0")
        _check(self.phantom_brake is None or self.phantom_brake > 0, "phantom brake > 0")


@dataclass(frozen=True)
class SiteConfig:
    """Where a hidden target crosses: by a hider standing still beside the ego's path ahead.

    Site kinds are ``kerb``, ``median``, ``lot`` (parked off the road) and ``queue`` (the gap
    between two vehicles stopped in a lane).
    """

    stand_beyond_m: float = 1.0  # kerb: the target stands this far beyond the hider's outer side
    cross_to_m: float = 4.0  # kerb: and crosses to this far past the path
    gap_beyond_m: float = 0.7  # kerb: on a line this far beyond the hider's front end
    min_ahead_m: float = 15.0  # kerb: the hider at least this far ahead along the path
    lateral_range_m: tuple[float, float] = (2.0, 4.5)  # kerb, median: offset from the path
    kerb_edge_m: float = 1.5  # kerb, median: outer side this close to a road edge
    stationary_share: float = 0.9  # kerb: share of its frames slower than 0.5 m/s
    max_gap_s: float = 0.5  # check: longest visibility gap between the reveal and the contact
    stand_edge_m: float = 3.0  # check: a kerbside target stands this close to a road edge
    queue_lateral_m: tuple[float, float] = (2.0, 5.5)  # queue: vehicles' offset from the path
    queue_lane_tolerance_m: float = 1.0  # queue: the two offsets differ by at most this
    queue_gap_m: tuple[float, float] = (1.5, 6.0)  # queue: gap between the two vehicles
    lot_lateral_m: tuple[float, float] = (4.0, 10.0)  # lot: offset from the path
    # The hider stays within ``still_drift_m`` of its anchor position from the target's first
    # frame to ``still_after_contact_s`` past the reference ego's arrival.
    still_drift_m: float = 0.5
    still_after_contact_s: float = 1.0
    # The route runs back as far as a moving start at this x the type's median speed needs.
    lead_in_speed_factor: float = 1.5


@dataclass(frozen=True)
class SearchConfig:
    """Reveal-time search toward the boundary; unused, kept for the settings record."""

    boundary_ma: float = 0.15  # 0 < MA < boundary_ma
    threat_gap_m: float = 2.0  # a sample threatens when it comes this close to the logged ego
    samples_per_frame: int = 3
    max_frames: int = 5
    coarse_stride: int = 1  # frames between the first wave of reveal times (1: all frames)

    def __post_init__(self):
        _positive(self, "boundary_ma", "threat_gap_m", "samples_per_frame", "max_frames")
        _check(self.boundary_ma <= 1, "SearchConfig.boundary_ma must be <= 1")


@dataclass(frozen=True)
class ReferenceConfig:
    """The reference ego's driver settings where they differ from the shared ones.

    None keeps the shared value; only the reference-ego sensitivity runs set them.
    """

    desired_speed_factor: float | None = None
    t_head_s: float | None = None

    def __post_init__(self):
        factor, headway = self.desired_speed_factor, self.t_head_s
        _check(factor is None or 0 < factor <= 1, "reference desired speed factor in (0, 1]")
        _check(headway is None or headway > 0, "reference time headway must be positive")

    def overrides(self) -> dict:
        """Return the settings this section sets, for `dataclasses.replace` on a DriverConfig."""
        return {name: value for name, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class Config:
    """Top-level configuration bundle; the field names also appear in config files."""

    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """Build a Config from a dict, dropping retired keys first."""
        return _build(cls, _drop_retired(data), "config")

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a Config from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# Sections and keys from earlier versions, ignored when an old run's snapshot is read.
RETIRED_SECTIONS = ("generation",)  # unused motion limits
RETIRED = {
    "search": (
        "samples_per_probe",
        "max_rounds",
        "max_probes_per_round",
        "time_step_s",
        "station_step_m",
        "lateral_step_m",
        "max_time_shift_s",
        "max_station_shift_m",
        "max_lateral_shift_m",
        "window_time_s",
        "window_station_m",
        "window_lateral_m",
        "start_lead_s",
    ),
    "oracle": (
        "library",
        "min_background_safe",
        "unique_responses",
        "excuse_target_rear_end",
        "gap_bridge_s",
        "gap_accel_vehicle",
        "gap_accel_pedestrian",
        "gap_accel_cyclist",
        "gap_still_mps",
        "max_accel",  # was the friction circle; replaced by the vehicle limits
        "horizon_s",  # was the min window after the reveal; replaced by the cutoff
    ),
    # Steering-servo settings from before nuPlan's two-stage controller.
    "oracle.vehicle": (
        "max_steering",
        "max_steering_rate",
        "max_steering_accel",
        "actuator_bandwidth",
        "position_gain",
    ),
}


def _drop_retired(data: Any) -> Any:
    """Return ``data`` with retired sections and keys removed."""
    if not isinstance(data, dict):
        return data
    out = {k: v for k, v in data.items() if k not in RETIRED_SECTIONS}
    for section, keys in RETIRED.items():
        out = _without(out, section.split("."), keys)
    return out


def _without(data: dict, path: list[str], keys: tuple[str, ...]) -> dict:
    """Return ``data`` with ``keys`` removed from its nested section at ``path`` (copied)."""
    head, *rest = path
    section = data.get(head)
    if not isinstance(section, dict):
        return data
    if rest:
        return {**data, head: _without(section, rest, keys)}
    return {**data, head: {k: v for k, v in section.items() if k not in keys}}


def _build(kind: type, data: Any, where: str) -> Any:
    """Build a dataclass of type ``kind`` from ``data``, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be an object")
    known = {f.name: f for f in fields(kind)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")
    hints = get_type_hints(kind)
    values = {}
    for name, value in data.items():
        hint, item = hints[name], _tuple_item(hints[name])
        if is_dataclass(hint):
            values[name] = _build(hint, value, f"{where}.{name}")
        elif item is not None and is_dataclass(item):
            values[name] = tuple(
                _build(item, v, f"{where}.{name}[{i}]") for i, v in enumerate(value)
            )
        elif isinstance(value, list):
            values[name] = tuple(value)
        else:
            values[name] = value
    try:
        return kind(**values)
    except TypeError as exc:  # missing required fields
        raise ValueError(f"{where}: {exc}") from None


def _tuple_item(hint: Any) -> Any:
    """Return the element type of ``tuple[X, ...]``, else ``None``."""
    args = getattr(hint, "__args__", ())
    return args[0] if getattr(hint, "__origin__", None) is tuple and args else None
