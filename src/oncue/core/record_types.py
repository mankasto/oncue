"""Typed views of the JSON lines the pipeline writes, one `SceneLine` per scene.

The code passes plain dicts; these types only document the fields. Times are seconds after the
anchor frame, distances metres, speeds m/s.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RunRecord(TypedDict, total=False):
    """One planner's run on one test under one observation mode.

    Keyed by `closedloop.run_name(planner, mode)`; a run that raised has only `error` and `seconds`.
    """

    # The target contact
    hit: bool  # a contact before the test's cutoff
    hit_moving: bool  # a hit with the ego at 1 m/s or faster
    after_cutoff: bool  # the only contact came at or after the cutoff
    contact_s: float | None
    contact_speed: float | None  # the ego's speed at contact
    rel_speed: float | None  # closing speed at contact
    cutoff_s: float
    cutoff_reason: str  # "left_route", "target_unobserved" or "run_end" (core.cutoff)
    meet_s: float  # the test's contact time for this run (shifted by a triggered start)
    # Contacts with other road users, judged by nuPlan's at-fault rules
    background_contacts: dict[str, Any]
    other_contact: bool  # an at-fault contact with a road user other than the target
    other_contact_with: list[str]
    excused: bool  # contacts the rules do not blame on the ego
    excused_by: list[str]
    # When the planner knew, saw and reacted to the target (None: never)
    known_s: float | None
    seen_s: float | None
    brake_s: float | None
    response_s: float | None  # the ego's first hard deceleration
    # How much of the run was simulated and how far the ego got
    observed_s: float
    progress_4s: float | None  # distance in 4 s over the reference ego's
    incomplete: bool  # no hit, but the run ended before the test's contact window did
    # A standing target started by the tested ego
    stalled: bool  # the target never started
    stand_s: float
    start_s: float | None  # when the target started, s after the anchor (None: never)
    shifted: bool  # starting it moved the target off its generated path
    late: bool  # masked, no hit: more than closedloop.trigger.LATE_TOL_S behind at the reveal
    lag_at_reveal_s: float
    # Whether the inserted target's scene stayed valid in this run
    scene_ok: bool
    scene_flaw_known: bool | None  # the problem was already one of the sample's flaws
    scene_background: list[Any] | None
    scene_near_miss: list[Any] | None
    # The ego's own driving
    ego_peaks: dict[str, float] | None  # peak speed, acceleration, jerk and the like
    beyond_limits: list[str]  # vehicle limits the ego exceeded
    out_of_road_s: float | None
    wrong_way_s: float | None
    road_unknown: bool  # the ego left the map's known extent
    seconds: float  # wall time of the run
    error: str


class TestRow(TypedDict, total=False):
    """One sample: a request of a scene and the candidate chosen for it.

    A request no candidate met has `found` false and no motion fields; a failed route has only
    `error`.
    """

    # Identity
    sample_id: str  # "<request id>/<generator>"
    request_id: str
    scene_id: str
    kind: str  # target type: "pedestrian", "cyclist" or "vehicle"
    generator: str  # "model" or "rule" (and the ablations' "contact_only", "free", "grid")
    # Where the target hides and crosses
    site_kind: str  # "kerb", "median", "lot", "queue" or "lane"
    site: str | None
    route_kind: str
    route: list[str] | None  # a vehicle target's lane chain
    lane: str | None
    u_conflict: float  # where the route crosses the ego's path, along that path
    angle: int  # crossing angle in degrees
    # The request
    W_req: float  # requested reaction window
    contact_key: float | str  # contact point across the ego's front (0.25, 0.5, 0.75) or "corner"
    realizable: str  # "ok", or why no hidden start meets the request
    mean_speed: float | None  # mean speed the target needs after the reveal
    atypical_speed: bool  # that speed is uncommon for the type
    limit: float  # speed limit used for the target
    limit_source: str  # "route" or "ego"
    starts: int
    start_drops: dict[str, int]  # why candidate starts were dropped
    first_start: dict[str, Any] | None
    # The chosen start
    profile: str  # speed profile, e.g. "stand_go"
    start_kind: str  # "moving" or "stand_go"
    start_rank: int  # 1: the first start tried
    start_s: float  # start station along the route, m (not a time)
    start_speed: float
    start_v0: float
    start_accel: float | None
    transition_accel: float
    start_back_m: float | None
    start_state: list[float]  # [x, y, heading, speed] at the anchor
    band: None  # no longer set
    f_go: int | None  # frame a standing target starts moving
    wait_s: float  # time a standing target waits after the anchor
    start_lead_s: float | None
    wp_reveal: list[Any]  # [arc length, frame] of the exposure waypoint
    wp_contact: list[Any]  # [arc length, frame] of the contact waypoint
    # The search
    found: bool  # some candidate passed every hard check
    generated: int
    tried: int
    starts_tried: int
    first_pass_tried: int
    pick_rank: int  # rank among the candidates that passed (1: the first)
    rejects: dict[str, int]  # candidates rejected, by hard check
    mode: int  # the model's mode the candidate came from
    score: float | None  # the model's score
    # The chosen motion and its timing
    t_r: float  # exposure
    t_c: float  # contact
    W: float  # t_c - t_r
    f_reveal: int
    f_contact: int
    reveal_frame: int
    on_time: bool  # exposure and contact within tolerance of their requested times
    window_hit: bool  # W within tolerance of W_req
    peak_speed: float
    required_speed: float
    measures: dict[str, Any]  # the motion's measured kinematics
    over_limit: float | None  # by how much the target exceeds the speed limit
    curb_depth_m: float | None
    signal: dict[str, Any]  # the signal states the target and the ego face
    flaws: dict[str, Any]  # flaw name -> what was found (None: no such flaw)
    flaw_count: int
    cutoff_frame: int
    cutoff_reason: str
    # The oracle's label
    test: str  # "reaction", "anticipation" or "unresolved" (Reactive, Anticipatory, ...)
    braking_suffices: bool  # a braking response alone avoids the target
    W0: float  # window from the earlier visibility the oracle granted
    early_visible_frame: int
    evaluation_end_frame: int
    oracle_revealed: dict[str, Any]  # the search after exposure
    oracle_early: dict[str, Any]  # the search with earlier visibility
    budget: int  # candidates per condition
    candidate_list: str  # digest of the response library
    hidden_by: str  # "road_users", or "walls": road users and walls
    core_coverage_complete: bool
    # The closed loop
    closed_loop: dict[str, RunRecord]
    nominal_4s: dict[str, Any]  # each planner's distance in 4 s without the target
    nominal_8s: dict[str, Any]
    start: str  # how the target was started: "fixed" or "triggered"
    # Internal: the states later steps rebuild the target from
    _states: list[list[float]]
    _valid: list[bool]
    _station: float
    _f_hold: int | None
    error: str


class SceneLine(TypedDict, total=False):
    """One scene of an output file: its id, its rows and, from generate and label, its funnel."""

    scene_id: str
    rows: list[TestRow]
    funnel: dict[str, Any]  # how many requests reached each step of the search
