"""One scene's closed-loop runs: every planner on every test row, in each observation mode.

A target standing at the anchor is started by the tested ego; everyone else replays the log.
Start from ``closed_loop_scene``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import DriverConfig
from ..core.cutoff import RUN_END, Cutoff
from ..core.rules import traffic_rules
from ..core.runs import error_fields
from ..core.types import Track
from ..generation.checks.validity import test_cutoff
from ..generation.context import SceneContext, scene_context
from ..generation.reference import YIELD_GAP_S
from ..generation.targets import TargetSpec
from ..oracle.checks import Roads
from ..planners import driver as ego_driver
from ..utils.tracks import frame_of
from .records import (
    LOGGED_TARGET_CHECKS,
    background_record,
    contact_record,
    deceleration_onset_s,
    drove_into_ego,
    ego_background_contacts,
    ego_kinematics,
    road_record,
    scene_checks,
    simulated_traffic,
    traffic_key,
)
from .registry import BUILT_IN_DRIVES, DEFAULT_PLANNERS, PLANNERS
from .trigger import LATE_TOL_S, TriggeredReplay, lag_at_reveal

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence

    from ..core.config import Config
    from ..core.record_types import RunRecord, TestRow
    from ..core.types import Scene
    from ..generation.context import ReferenceSchedule


# Progress of a test run: the distance it covers in this long, divided by the reference's.
PROGRESS_S = 4.0

# Each planner's distance without the target after these seconds, by output field.
NOMINAL_HORIZONS_S = {"nominal_4s": 4.0, "nominal_8s": 8.0}

# A target slower than this at the anchor counts as standing (triggered start).
STANDING_START_MPS = 0.5

# Below this reference distance over ``PROGRESS_S``, no progress ratio is reported.
MIN_REFERENCE_4S_M = 1.0

# A started target farther than this off its generated path was effectively shifted.
SHIFT_TOL_M = 0.05

# Tolerance for a run's rounded seconds.
ROUNDED_TIME_TOL_S = 1e-6

# Field names on a labelled sample for the stored cutoff frame and reason.
CUTOFF_FIELD = "cutoff_frame"
CUTOFF_REASON_FIELD = "cutoff_reason"

# Worlds a run can use; MetaDrive only mirrors the replay world to draw it.
WORLDS = ("replay", "metadrive")

# Profiles that stand at the anchor; earlier files call two of them ``stand`` and ``pullout``.
STANDING_PROFILES = ("stand_go", "stop_go", "stand", "pullout")

# Mode name (also the suffix in a run's key) -> whether perception is masked. Iterated in order.
MODES = {"masked": True, "see": False}

ROW_FIELDS = (
    "sample_id",
    "kind",
    "site",
    "route",
    "profile",
    "band",
    "contact_key",
    "W_req",
    "generator",
    "W",
    "test",
    "braking_suffices",
    "start_lead_s",
    "wait_s",
    "flaws",  # sample's flaw labels; drives the row's report group
    "hidden_by",  # road users alone, or road users and the walls
)


@dataclass(frozen=True)
class RunOptions:
    """How the tests run: the same for every scene of one ``oncue closedloop`` call."""

    planners: tuple[str, ...] = DEFAULT_PLANNERS  # registry names; empty: the defaults
    fixed: bool = False  # fixed timing for every target: no triggered start
    world: str = "replay"  # one of WORLDS
    modes: tuple[str, ...] = tuple(MODES)  # observation modes
    jerk_max: float | None = None  # the planners' jerk limit; None keeps the driver's
    keep_states: bool = False  # also keep the ego's and target's states on every run


DEFAULT_RUN_OPTIONS = RunOptions()  # the default planners, every mode, triggered starts


def world_class(name: str) -> type[ego_driver.ReplayWorld]:
    """Return the world class for ``name``; MetaDrive's is imported only when asked for."""
    if name == "replay":
        return ego_driver.ReplayWorld
    if name == "metadrive":
        from .metadrive_world import MetaDriveWorld

        return MetaDriveWorld
    raise ValueError(f"unknown world {name!r}; one of {WORLDS}")


def checked_modes(modes: Sequence[str]) -> tuple:
    """Return ``modes`` in ``MODES`` order; raise ``ValueError`` if empty or unknown."""
    unknown = sorted(set(modes) - set(MODES))
    if unknown or not modes:
        raise ValueError(f"observation modes {list(modes)}: one or more of {list(MODES)}")
    return tuple(mode for mode in MODES if mode in modes)


def closed_loop_scene(
    scene: Scene,
    rows: list[TestRow],
    spec: TargetSpec,
    config: Config,
    driver: DriverConfig,
    options: RunOptions = DEFAULT_RUN_OPTIONS,
) -> list[TestRow]:
    """Return closed-loop records for the test rows of one scene."""
    driver = test_driver(driver, config)
    ctx = scene_context(scene, config, driver)
    return closed_loop_rows(ctx, rows, spec, driver, options)


def test_driver(driver: DriverConfig, config: Config) -> DriverConfig:
    """Return the driver settings for test runs: a fixed yield gap, the oracle's vehicle."""
    return replace(
        driver,
        yield_gap_s=YIELD_GAP_S,
        collision_substeps=config.oracle.collision_substeps,
        vehicle=config.oracle.vehicle,
    )


def closed_loop_rows(
    ctx: SceneContext,
    rows: list[TestRow],
    spec: TargetSpec,
    driver: DriverConfig,
    options: RunOptions = DEFAULT_RUN_OPTIONS,
    logged_target: bool = False,
) -> list[TestRow]:
    """Return closed-loop records for test rows within a prebuilt scene context.

    ``ctx`` comes from ``test_driver`` then ``scene_context``; ``driver`` is the reference ego's.

    Args:
        logged_target: the targets are logged road users on their logged schedule, so the
            target's scene checks are skipped.
    """
    world_type = world_class(options.world)
    modes = checked_modes(options.modes)
    planners = list(options.planners or DEFAULT_PLANNERS)
    if options.jerk_max is not None:
        driver = replace(driver, jerk_max=options.jerk_max)
    reference = ctx.reference_schedule()
    rules, nominal = _rules_and_nominal(ctx, planners, driver, world_type)
    # The reference's distance over PROGRESS_S, to catch a planner that crawls and never triggers.
    stations, frames, _ = reference
    end = ctx.anchor + frame_of(PROGRESS_S, ctx.dt)
    reference_4s = float(np.interp(end, frames, stations) - np.interp(ctx.anchor, frames, stations))
    setup = _SceneSetup(
        ctx=ctx,
        spec=spec,
        planners=planners,
        driver=driver,
        rules=rules,
        nominal=nominal,
        reference=reference,
        reference_4s=reference_4s,
        fixed=options.fixed,
        keep_states=options.keep_states,
        world_type=world_type,
        modes=modes,
        logged_target=logged_target,
        roads=Roads.of(ctx.scene),
    )
    return [_row_record(setup, row) for row in rows]


def _rules_and_nominal(
    ctx: SceneContext,
    planners: list[str],
    driver: DriverConfig,
    world_type: type[ego_driver.ReplayWorld],
):
    """Return per-planner traffic rules and the distances each covers without the target.

    A planner that fails here appears only under ``nominal_errors``.
    """
    scene = ctx.scene
    rules, errors = {}, {}
    fields = {field: {} for field in NOMINAL_HORIZONS_S}
    for name in planners:
        fn, opts = PLANNERS[name]
        cfg = replace(driver, **opts)
        try:
            rules[name] = traffic_rules(scene, ctx.limit, cfg)
            if fn in BUILT_IN_DRIVES:
                run = fn(
                    scene,
                    None,
                    cfg,
                    masked=True,
                    vis_cfg=ctx.config.visibility,
                    static=ctx.static,
                    v_des=ctx.limit,
                    rules=rules[name],
                    ignore_target=True,
                    world=world_type,
                )
                for field, horizon_s in NOMINAL_HORIZONS_S.items():
                    fields[field][name] = round(run.distance_at(horizon_s, ctx.dt), 2)
        except Exception as e:  # only this planner is affected
            errors[name] = repr(e)
    if errors:
        fields["nominal_errors"] = errors
    return rules, fields


@dataclass(frozen=True, eq=False)
class _SceneSetup:
    """What every run of one scene shares."""

    ctx: SceneContext
    spec: TargetSpec
    planners: list
    driver: DriverConfig  # base planner settings, before each planner's own options
    rules: dict  # planner -> traffic rules (missing where setup failed)
    nominal: dict  # fields ``nominal_4s`` / ``nominal_8s`` plus any ``nominal_errors``
    reference: ReferenceSchedule  # reference ego clock (``SceneContext.reference_schedule``)
    reference_4s: float  # reference distance over PROGRESS_S
    fixed: bool  # pin every target to the reference clock (no triggered start)
    keep_states: bool
    world_type: type  # world class the runs use (``world_class``)
    modes: tuple  # observation modes to run, in ``MODES`` order
    logged_target: bool  # target is a logged road user on its logged schedule (no checks)
    roads: Roads  # scene lanes and road edges (``oracle.checks.Roads``)


def _row_record(setup: _SceneSetup, row: TestRow) -> TestRow:
    """Return the record of one test row; a failed run keeps its error and the rest still run."""
    ctx, spec = setup.ctx, setup.spec
    states, valid = np.asarray(row["_states"], float), np.asarray(row["_valid"], bool)
    record = {k: row.get(k) for k in ROW_FIELDS} | setup.nominal
    cutoff = sample_cutoff(setup, row)
    record[CUTOFF_FIELD], record[CUTOFF_REASON_FIELD] = cutoff
    results = record["closed_loop"] = {}
    triggered = is_triggered(row, setup.fixed, ctx.anchor)
    record["start"] = "triggered" if triggered else "fixed"  # reported separately in outputs
    cache: dict = {}  # scene checks of runs with the same traffic
    for name in setup.planners:
        fn, opts = PLANNERS[name]
        cfg = replace(setup.driver, **opts)
        for mode in setup.modes:
            key, masked = run_name(name, mode), MODES[mode]
            started = time.perf_counter()
            try:
                trigger, target = _run_target(setup, row, states, valid, triggered)
                run = fn(
                    replace(ctx.scene, tracks={**ctx.scene.tracks, spec.track_id: target}),
                    spec.track_id,
                    cfg,
                    masked=masked,
                    vis_cfg=ctx.config.visibility,
                    static=ctx.static,
                    v_des=ctx.limit,
                    rules=setup.rules[name],
                    target_motion=trigger,
                    world=setup.world_type,
                )
                results[key] = _run_record(setup, row, run, target, trigger, masked, cache, cutoff)
            except Exception as e:  # keep the other runs even if this one failed
                results[key] = error_fields(e)
            # Wall time including failed runs; shared setup of the scene is not counted here.
            results[key]["seconds"] = round(time.perf_counter() - started, 2)
    return record


def sample_cutoff(setup: _SceneSetup, row: TestRow) -> Cutoff:
    """Return the row's stored cutoff, or derive it from the reference contact if none is stored.

    Samples from files written before reasons were stored have None.
    """
    if row.get(CUTOFF_FIELD) is not None:
        return Cutoff(int(row[CUTOFF_FIELD]), row.get(CUTOFF_REASON_FIELD))
    ctx = setup.ctx
    states, valid = np.asarray(row["_states"], float), np.asarray(row["_valid"], bool)
    return test_cutoff(ctx, setup.spec, states, valid, _contact_frame(ctx, row))


def _contact_frame(ctx: SceneContext, row: TestRow) -> int:
    """Return the row's reference contact frame, computing it from ``t_c`` if not stored."""
    return int(row.get("f_contact", ctx.anchor + frame_of(row["t_c"], ctx.dt)))


def onset_shift_s(trigger: TriggeredReplay | None, anchor: int, dt: float) -> float | None:
    """Return how many seconds later than generated the target started moving.

    Returns 0 with fixed timing and None if the target never started.
    """
    if trigger is None:
        return 0.0
    if trigger.started_s is None:
        return None
    return trigger.started_s - (trigger.move_frame - anchor) * dt


def is_triggered(row: TestRow, fixed: bool, anchor: int) -> bool:
    """Return whether the tested ego starts the target: not fixed timing, and a standing start."""
    if fixed or row.get("profile") not in STANDING_PROFILES:
        return False
    return float(np.asarray(row["_states"], float)[anchor, 3]) < STANDING_START_MPS


def run_name(planner: str, mode: str) -> str:
    """Return a run's key within a row's ``closed_loop`` record, e.g. ``"pdm_closed_masked"``."""
    return f"{planner}_{mode}"


def _run_target(
    setup: _SceneSetup, row: dict, states: np.ndarray, valid: np.ndarray, triggered: bool
):
    """Return ``(trigger, target_track)`` for one run; ``trigger`` is None with fixed timing."""
    ctx, spec, reference = setup.ctx, setup.spec, setup.reference
    trigger, run_states = None, states
    if triggered:
        contact_frame = ctx.anchor + frame_of(row["t_c"], ctx.dt)
        station = float(np.interp(contact_frame, reference.frames, reference.stations))
        trigger = TriggeredReplay(ctx.scene, states, station, reference)
        run_states = trigger.shifted_states(None)
    return trigger, Track(spec.track_id, spec.agent_type, spec.size, run_states, valid, spec.height)


def _run_record(
    setup: _SceneSetup,
    row: TestRow,
    run: ego_driver.Drive,
    target: Track,
    trigger: TriggeredReplay | None,
    masked: bool,
    cache: dict,
    cutoff: Cutoff,
) -> RunRecord:
    """Return the record of one closed-loop run.

    A triggered start shifts the sample's cutoff with the target; a target that never starts
    has none.
    """
    ctx = setup.ctx
    target_states = target.states
    if trigger is not None and trigger.started_s is not None:
        target_states = trigger.shifted_states(trigger.started_s)
    shift_s = onset_shift_s(trigger, ctx.anchor, ctx.dt)
    run_cutoff = ctx.n if shift_s is None else cutoff.frame + frame_of(shift_s, ctx.dt)
    substeps = ctx.config.oracle.collision_substeps
    background = ego_background_contacts(ctx.scene, run, setup.spec.track_id, substeps, setup.roads)
    record = {
        **contact_record(run, target_states, ctx.anchor, ctx.dt, run_cutoff),
        CUTOFF_REASON_FIELD: cutoff.reason if run_cutoff < ctx.n else RUN_END,
        **background_record(background),
        "known_s": run.target_known_s,
        "seen_s": run.target_seen_s,
        "brake_s": run.target_brake_s,
        # Deceleration onset, under the field name the released runs use.
        "response_s": deceleration_onset_s(run, ctx.anchor, ctx.dt),
        **ego_kinematics(run, ctx.anchor, ctx.dt),
        **road_record(run, setup.roads, ctx.scene.ego.size, ctx.anchor, ctx.dt),
    }
    if setup.keep_states:
        record.update(
            states=run.states, valid_until=int(run.valid_until), target_states=target_states
        )
    # This run's contact time: the sample's, shifted by the trigger lag.
    meet_s = (_contact_frame(ctx, row) - ctx.anchor) * ctx.dt + (shift_s or 0.0)
    record["meet_s"] = round(meet_s, 2)
    if setup.logged_target:
        record.update(LOGGED_TARGET_CHECKS)
    else:
        gone = drove_into_ego(background)
        flaws = row.get("flaws") or {}
        checks = _target_scene_checks(
            setup, run, target, target_states, trigger, cache, gone, run_cutoff, flaws
        )
        record.update(checks)
    record.update(_observed(setup, run, record["hit"], meet_s, record["cutoff_s"]))
    if trigger is None:
        record.update(_lateness(setup, row, run, masked, record["hit"]))
    else:
        record.update(_start(trigger, target_states, target.valid))
    return record


def _target_scene_checks(
    setup: _SceneSetup,
    run: ego_driver.Drive,
    target: Track,
    target_states: np.ndarray,
    trigger: TriggeredReplay | None,
    cache: dict,
    gone: dict[str, int],
    cutoff: int,
    flaws: dict,
):
    """Return the target's scene checks in this run, cached per traffic.

    They run from the anchor to the cutoff, the target contact or the run's end, whichever is
    first, and skip road users in ``gone`` from the frame they drove into the ego.
    """
    ctx = setup.ctx
    anchor, dt = ctx.anchor, ctx.dt
    contact_s = run.target_contact_s
    until = int(run.valid_until) if contact_s is None else anchor + frame_of(contact_s, dt) + 1
    until = min(until, int(run.valid_until), cutoff)
    started = None if trigger is None else trigger.started_s
    retimed = onset_shift_s(trigger, anchor, dt) != 0.0  # target no longer on the sample's timing
    key = (trigger is None, started, until, traffic_key(run))
    if key not in cache:
        cache[key] = scene_checks(
            simulated_traffic(ctx.scene, run, gone),
            target.with_states(target_states),
            setup.spec,
            ctx.config.visibility,
            ctx.static,
            until,
            ctx.config.oracle.collision_substeps,
            flaws,
            retimed,
        )
    return cache[key]


def _observed(
    setup: _SceneSetup, run: ego_driver.Drive, hit: bool, meet_s: float, cutoff_s: float
) -> dict:
    """Return the simulated span, the progress ratio and whether the run is incomplete.

    Incomplete means no hit, but the contact plus the oracle's cover (or the test's end) lies
    past the simulated span.
    """
    anchor, dt = setup.ctx.anchor, setup.ctx.dt
    observed_s = round((run.valid_until - 1 - anchor) * dt, 2)
    cover_end = min(meet_s + (setup.ctx.config.oracle.cover_contact_s or 0.0), cutoff_s)
    progress = None
    if setup.reference_4s > MIN_REFERENCE_4S_M:
        progress = round(run.distance_at(PROGRESS_S, dt) / setup.reference_4s, 3)
    return {
        "observed_s": observed_s,
        "progress_4s": progress,
        "incomplete": bool(not hit and cover_end > observed_s + ROUNDED_TIME_TOL_S),
    }


def _lateness(
    setup: _SceneSetup, row: TestRow, run: ego_driver.Drive, masked: bool, hit: bool
) -> dict:
    """Return a fixed-timing run's lag behind the reference at the reveal.

    ``late`` applies in masked mode only: seeing through, the ego may rightly slow early.
    """
    if row.get("t_c") is None or row.get("W") is None:
        return {}
    ctx = setup.ctx
    reveal_frame = ctx.anchor + frame_of(row["t_c"] - row["W"], ctx.dt)
    lag = lag_at_reveal(ctx.path, setup.reference, run.states, reveal_frame, ctx.scene.ego.size)
    return {"lag_at_reveal_s": lag, "late": bool(masked and not hit and lag > LATE_TOL_S)}


def _start(trigger: TriggeredReplay, target_states: np.ndarray, valid: np.ndarray) -> dict:
    """Return whether and when the run started the target, and whether that shifted it."""
    record = {
        "stalled": trigger.started_s is None,
        "stand_s": round(trigger.stand_s, 2),
        "start_s": trigger.started_s,
    }
    if trigger.started_s is not None:
        shifted_xy = target_states[valid, :2]
        record["shifted"] = not np.allclose(shifted_xy, trigger.states[valid, :2], atol=SHIFT_TOL_M)
    return record
