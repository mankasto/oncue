"""Per-type target specs: one `TargetSpec` per inserted type. Where each value comes from is in
PARAMETERS.md."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from ..core.config import SPEEDING_TOL_MPS
from ..core.kinematics import KinematicLimits
from ..core.vehicle import VEHICLE_HEIGHT_M, VEHICLE_SIZE_M
from ..utils.tracks import frame_of

NO_LIMITS = KinematicLimits(accel=None, jerk=None, lateral=None, yaw_rate=None, peak_speed=None)


def _only_default(instance: object, *names: str) -> None:
    """Raise ValueError if a recorded-only field differs from its default, the only one built."""
    defaults = {f.name: f.default for f in fields(instance)}
    for name in names:
        value = getattr(instance, name)
        if value != defaults[name]:
            raise ValueError(
                f"{type(instance).__name__}.{name} {value!r}: only {defaults[name]!r} is built"
            )


@dataclass(frozen=True)
class NearMissRule:
    """When a logged road user passing the target counts as a near miss.

    It has seen the target and passes it within `pet_s` post-encroachment time while faster than
    `speed_mps`, before the cutoff. The other fields are recorded only; just their defaults are
    built.
    """

    method: str = "pet"
    gap_m: float = 0.5
    pet_s: float = 1.0
    speed_mps: float = 2.0
    until_contact: bool = False  # runs to the test's cutoff, past the contact
    seen: str = "before"
    types: tuple[str, ...] | None = None
    exclude_site: bool = False

    def __post_init__(self):
        _only_default(self, "method", "gap_m", "until_contact", "seen", "types", "exclude_site")


@dataclass(frozen=True)
class ContactRule:
    """Where the target meets the reference ego (the contact waypoint).

    At the reference front's arrival the target's centre lands at each fraction `impact` of the
    ego's front width; `corner_overlap_m` adds a corner contact with the target's leading edge
    that far inside the ego's near side. `mode`, `leads_s` and `at` are recorded only.
    """

    mode: str = "impact"
    leads_s: tuple[float, ...] = ()
    at: str = "band_entry"
    impact: tuple[float, ...] = (0.5,)
    corner_overlap_m: float | None = None  # None: the impact fractions alone

    def __post_init__(self):
        _only_default(self, "mode", "leads_s", "at")


@dataclass(frozen=True)
class LogMotion:
    """How a road user type moves in the logs (`oncue stats`); starts rank by their distance
    from these medians."""

    speed_median_mps: float  # median peak speed of moving road users
    start_accel_median_mps2: float  # median start acceleration from standstill
    start_accels_mps2: tuple[float, ...]  # the start accelerations stand_go and stop_go try
    speed_common_mps: tuple[float, float]  # a mean speed outside this range is atypical
    # the transition rule's starting speeds, and for vehicles also as ratios of the lane limit
    start_speeds_mps: tuple[float, ...] = ()
    start_limit_ratios: tuple[float, ...] = ()


START_RULES = ("mean_speed", "transition")  # TargetSpec.start_rule


@dataclass(frozen=True)
class TargetSpec:
    kind: str  # pedestrian | cyclist | vehicle
    track_id: str  # id of the inserted track
    agent_type: str  # WOMD type of the inserted track
    model_type: int  # recorded only; the model maps `agent_type` (`waymo_dataset.TYPES`)
    size: tuple[float, float]  # length, width (m)
    height: float  # m
    sites: tuple[str, ...]  # hiding sites: kerb | median | lot | queue | lane
    route_kinds: tuple[str, ...]  # straight | sidewalk | oblique | lane
    start_kinds: tuple[str, ...]  # moving | stand_go | stop_go
    log: LogMotion
    # K: the starts a request keeps at most, one model pass each; None keeps every start
    starts_per_request: int | None = 5
    stand_back_m: float = 0.0  # stand_go: at most this far behind the hiding spot
    # every start's history before the anchor, at its anchor speed along its route
    assumed_history_s: float = 1.0
    # whether the reveal checks see the assumed history; False treats it as hidden
    history_visibility: bool = True
    # how a moving start is placed: "transition" changes speed evenly while hidden from a logged
    # starting speed; "mean_speed" keeps the request's mean speed throughout
    start_rule: str = "transition"
    # whether the trajectory generator reads a moving start's assumed history, or the anchor alone
    model_history: bool = True
    # speeding flaw above the posted limit by more than this (m/s); None skips the check
    speeding_tol_mps: float | None = None
    contact: ContactRule = field(default_factory=ContactRule)
    impact_ahead_m: float = 0.0  # impact mode: reference point this far ahead of the centre
    # straight routes' angles with the ego's direction of travel (below 90: slanting its way)
    crossing_angles_deg: tuple[float, ...] = (90.0,)
    oblique_angles_deg: tuple[float, ...] = ()  # oblique routes' cut-out angles
    windows_s: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)  # requested W
    kinematics: KinematicLimits = NO_LIMITS
    kinematics_method: str = "gradient"  # recorded only
    # a moving start's first-step velocity within this of its start velocity (m/s; None skips)
    start_jump_mps: float | None = None
    near_miss: NearMissRule = field(default_factory=NearMissRule)
    timing_tolerance_s: float = 0.2  # a candidate's W within this of the requested W
    timing: str = "window"  # recorded only
    placement: str = "kerbside"  # kerbside: off the road near an edge, not in static | lane
    # record the first frame the target touches a road edge (a label, never a reject)
    check_road_edge: bool = False
    yield_gap_s: float | None = None  # reference ego's yield gap to crossing traffic
    table_step_m: float = 0.25  # visibility table spacing along the route
    prescreen: bool = False  # screen candidates by the table's approximate reveal before checking
    rule_exact_checks: int | None = None  # with prescreen: rule candidates fully checked (closest)
    heading_method: str = "polyline"  # step | polyline (`states_from_positions`)
    # closed loop: a triggered start waits for the tested ego at the contact station; recorded only
    trigger_station: str = "contact"

    def __post_init__(self):
        _only_default(self, "kinematics_method", "timing", "trigger_station")
        if self.start_rule not in START_RULES:
            raise ValueError(f"start_rule {self.start_rule!r}: one of {START_RULES}")

    def first_frame(self, anchor: int, dt: float) -> int:
        """Return the target's first frame, where its assumed history begins before the anchor."""
        return max(anchor - frame_of(self.assumed_history_s, dt), 0)


# ---------------------------------------------------------------- log statistics and cited values
PEDESTRIAN_LOG = LogMotion(
    speed_median_mps=1.466,
    start_accel_median_mps2=0.762,
    start_accels_mps2=(0.494, 0.762, 1.092),
    speed_common_mps=(1.038, 1.759),
    start_speeds_mps=(1.176, 1.369, 1.552),
)
CYCLIST_LOG = LogMotion(
    speed_median_mps=4.96,
    start_accel_median_mps2=1.304,
    start_accels_mps2=(0.563, 1.304, 1.822),
    speed_common_mps=(1.876, 7.461),
    start_speeds_mps=(2.633, 3.779, 5.293),
)
VEHICLE_LOG = LogMotion(
    speed_median_mps=8.496,
    start_accel_median_mps2=1.586,
    start_accels_mps2=(0.9, 1.586, 2.223),
    speed_common_mps=(1.379, 17.44),
    start_speeds_mps=(2.995, 5.564, 9.406),
    start_limit_ratios=(0.19, 0.344, 0.582),
)
UNIFIED_NEAR_MISS = NearMissRule(pet_s=1.0, speed_mps=2.0)
# a lower peak speed is rare motion (vehicles: none)
PEDESTRIAN_MIN_PEAK_SPEED_MPS = 1.021
CYCLIST_MIN_PEAK_SPEED_MPS = 1.577
# a peak speed up to this times the request's mean speed is no flaw
PEDESTRIAN_PACE_RATIO = 1.571
CYCLIST_PACE_RATIO = 1.967
VEHICLE_PACE_RATIO = 2.87
VEHICLE_PEAK_SPEED_MPS = 24.844
# rare-motion limits
PEDESTRIAN_KINEMATICS = KinematicLimits(
    accel=2.488,
    jerk=23.701,
    lateral=1.316,
    yaw_rate=1.013,
    peak_speed=3.722,
    min_peak_speed=PEDESTRIAN_MIN_PEAK_SPEED_MPS,
    pace_ratio=PEDESTRIAN_PACE_RATIO,
)
CYCLIST_KINEMATICS = KinematicLimits(
    accel=2.836,
    jerk=18.649,
    lateral=3.293,
    yaw_rate=0.983,
    peak_speed=10.435,
    min_peak_speed=CYCLIST_MIN_PEAK_SPEED_MPS,
    pace_ratio=CYCLIST_PACE_RATIO,
)
VEHICLE_KINEMATICS = KinematicLimits(
    accel=6.137,
    jerk=44.376,
    lateral=3.669,
    yaw_rate=0.666,
    peak_speed=VEHICLE_PEAK_SPEED_MPS,
    pace_ratio=VEHICLE_PACE_RATIO,
)

STRAIGHT_ANGLES_DEG = (90.0, 60.0)  # square across and 30 deg off it, slanting the ego's way
# corner contact: the target's leading edge this far inside the ego's near side
CORNER_OVERLAP_M = 0.2
HIDING_SITES = ("kerb", "median", "lot", "queue")  # every kind, each reported separately

PEDESTRIAN = TargetSpec(
    kind="pedestrian",
    track_id="ghost_pedestrian",
    agent_type="pedestrian",
    model_type=1,
    size=(0.95, 0.87),  # WOMD median
    height=1.75,  # WOMD median
    sites=HIDING_SITES,
    route_kinds=("straight", "sidewalk"),  # sidewalk: walks along it behind the hider, turns out
    start_kinds=("moving", "stand_go"),
    log=PEDESTRIAN_LOG,
    stand_back_m=2.0,
    contact=ContactRule(impact=(0.25, 0.5, 0.75), corner_overlap_m=CORNER_OVERLAP_M),  # NCAP CPNCO
    crossing_angles_deg=STRAIGHT_ANGLES_DEG,
    kinematics=PEDESTRIAN_KINEMATICS,
    near_miss=UNIFIED_NEAR_MISS,
    placement="kerbside",
    table_step_m=0.25,
    prescreen=True,
    rule_exact_checks=3,
    heading_method="step",
)

CYCLIST = TargetSpec(
    kind="cyclist",
    track_id="ghost_cyclist",
    agent_type="cyclist",
    model_type=2,
    size=(1.87, 0.85),  # WOMD median
    height=1.79,  # WOMD median
    sites=HIDING_SITES,
    route_kinds=("straight", "oblique"),  # oblique: along the parking strip, then cutting out
    start_kinds=("moving", "stand_go"),
    log=CYCLIST_LOG,
    stand_back_m=2.0,
    contact=ContactRule(impact=(0.5, 0.75), corner_overlap_m=CORNER_OVERLAP_M),  # NCAP CBNAO
    crossing_angles_deg=STRAIGHT_ANGLES_DEG,
    oblique_angles_deg=(15.0, 30.0),
    kinematics=CYCLIST_KINEMATICS,
    near_miss=UNIFIED_NEAR_MISS,
    placement="kerbside",
    table_step_m=0.25,
    prescreen=True,
    rule_exact_checks=3,
    heading_method="polyline",
)

VEHICLE = TargetSpec(
    kind="vehicle",
    track_id="ghost_vehicle",
    agent_type="vehicle",
    model_type=0,
    size=VEHICLE_SIZE_M,  # WOMD median
    height=VEHICLE_HEIGHT_M,  # WOMD median
    sites=("lane",),
    route_kinds=("lane",),
    # stop_go at a stop sign's line; a moving start there runs the sign (a flaw)
    start_kinds=("moving", "stop_go"),
    log=VEHICLE_LOG,
    contact=ContactRule(impact=(0.5,)),  # NCAP CCCscp
    impact_ahead_m=VEHICLE_SIZE_M[0] / 4,  # its point 1/4 of the length behind the front
    kinematics=VEHICLE_KINEMATICS,
    start_jump_mps=2.248,  # p99 of the logged first-step jump (`start_jump`)
    speeding_tol_mps=SPEEDING_TOL_MPS,
    near_miss=UNIFIED_NEAR_MISS,
    placement="lane",
    check_road_edge=True,
    yield_gap_s=1.5,
    table_step_m=1.0,
    heading_method="polyline",
)

SPECS = {"pedestrian": PEDESTRIAN, "cyclist": CYCLIST, "vehicle": VEHICLE}
