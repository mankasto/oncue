"""Test labels of a sample, the one labelling path shared by every target type.

The oracle judges the ego responding from the reveal and from the first frame the target would
be visible without its occluders; ``classify`` turns the two results into a label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.cutoff import Cutoff, cutoff_frame, route_band
from ..core.types import STATE_SIZE, Scene
from ..core.visibility import RevealEvent, target_visibility, visibility_from
from .evaluate import BEFORE_START, Judgement, Oracle, background_safe, counts, status_of
from .responses import prefix

if TYPE_CHECKING:  # for annotations only: no new import when running
    from collections.abc import Sequence

    import shapely

    from ..core.config import OracleConfig, VisibilityConfig
    from .responses import Candidate

# Float slack when turning ``reaction_delay_s`` into frames.
DELAY_FRAMES_SLACK = 1e-9
MIN_WINDOW_FRAMES = 3  # anything shorter than this is too short to judge
SHORT_WINDOW = "short_window"  # ``not_judged`` reason when a condition's window is too short


def label_sample(
    with_target: Scene,
    target_id: str,
    event: RevealEvent,
    contact: int,
    *,
    vis: VisibilityConfig,
    static: shapely.Geometry | None,
    oracle: Oracle,
    hits: np.ndarray | None = None,
    budget: int | None = None,
    cutoff: int | None = None,
) -> dict:
    """Label one sample through the budgeted oracle.

    ``with_target`` is the reference scene with the target inserted; the target is revealed at
    ``event.reveal`` and touches the reference ego at frame ``contact``.

    Args:
        hits: precomputed blocking array for the target, if available.
        budget: candidates per condition. None runs the configured budget, then the whole
            list if that gives no reaction test.
        cutoff: the sample's cutoff frame; None uses ``reference_cutoff``.
    """
    cfg, dt = oracle.config, with_target.dt
    if hits is None:
        hits, _ = target_visibility(with_target, target_id, vis, static)
    reveal = event.reveal
    early_frame = early_visible(with_target, target_id, event, hits, vis)
    delay = reaction_delay_frames(cfg, dt)
    out = {
        "W0": round((contact - early_frame) * dt, 2),
        "budget": cfg.budget if budget is None else budget,
        "candidate_list": oracle.digest,
        "reveal_frame": reveal,
        "early_visible_frame": early_frame,
    }
    end = evaluation_end(with_target, target_id, early_frame + delay, contact, cutoff=cutoff)
    if not isinstance(end, str) and short_window(reveal + delay, end):
        end = SHORT_WINDOW  # the revealed condition starts last: its window too must be judgeable
    if isinstance(end, str):
        return {**out, "test": "not_judged", "judge_missing": end}

    def judged(k: int | None, done: tuple = (None, None)) -> tuple:
        """Run one pass for both conditions, extending the judgements in ``done``."""
        revealed = oracle.judge(with_target, target_id, reveal + delay, end, k, done[0])
        early = (
            revealed
            if early_frame == reveal
            else oracle.judge(with_target, target_id, early_frame + delay, end, k, done[1])
        )
        stored = summary(revealed, cfg.vehicle.substeps), summary(early, cfg.vehicle.substeps)
        return (revealed, early), stored, *label_of(*stored, revealed.tried, oracle.core_size)

    runs, stored, test, missing = judged(budget)
    if budget is None and test != "reaction":
        out["budget"] = len(oracle.candidates)
        runs, stored, test, missing = judged(len(oracle.candidates), runs)
    revealed = runs[0]
    out.update(
        evaluation_end_frame=end - 1,
        # ``classify`` reports ``core_incomplete`` before any other reason
        core_coverage_complete=missing != "core_incomplete",
        oracle_revealed=stored[0],
        oracle_early=stored[1],
        test=test,
        braking_suffices=braking_suffices(revealed),
    )
    if missing is not None:
        out["judge_missing"] = missing
    return out


def braking_suffices(judgement: Judgement) -> bool:
    """Return whether a response without steering (``keep`` lateral family) is safe."""
    return any(o.result == "safe" and o.candidate.lat.name == "keep" for o in judgement.outcomes)


def early_visible(
    with_target: Scene, target_id: str, event: RevealEvent, hits: np.ndarray, vis: VisibilityConfig
) -> int:
    """Return the first frame the target is visible with its occluders made transparent.

    Returns ``event.reveal`` when no earlier frame qualifies.
    """
    responsible = hits[slice(event.occlusion_start, event.reveal)].any(axis=(0, 2))
    cleared = hits.copy()
    cleared[:, responsible, :] = False
    values = visibility_from(cleared, with_target.ego, with_target.tracks[target_id], vis)
    early_frame = first_visible(values, with_target.anchor, event.reveal, vis)
    return event.reveal if early_frame is None else early_frame


def first_visible(values: np.ndarray, start: int, stop: int, vis: VisibilityConfig) -> int | None:
    """Return the first frame in ``[start, stop)`` opening ``min_visible_frames`` visible frames."""
    ok = np.isfinite(values) & (values >= vis.threshold)
    need = vis.min_visible_frames
    for g in range(start, stop):
        if g + need <= len(ok) and ok[g : g + need].all():
            return g
    return None


def reaction_delay_frames(cfg: OracleConfig, dt: float) -> int:
    """Return the frames after a takeover before the ego may respond, rounded up."""
    return int(np.ceil(cfg.reaction_delay_s / dt - DELAY_FRAMES_SLACK))


def evaluation_end(
    scene: Scene,
    target_id: str,
    start: int,
    contact: int | None,
    *,
    cutoff: int | None = None,
) -> int | str:
    """Return the exclusive end of the window both conditions are judged over.

    The test's ``cutoff`` (None: ``reference_cutoff``), clipped to where the ego and target stay
    observed; a reason string when the window is too short or ends by the contact.
    """
    n = len(scene.timestamps)
    if cutoff is None:
        cutoff = n if contact is None else reference_cutoff(scene, target_id, contact).frame
    observed = min(scene.ego.valid_until(start, n), scene.tracks[target_id].valid_until(start, n))
    end = min(cutoff, observed)
    if contact is not None and end <= contact:
        return "contact_near_log_end" if observed <= contact else "cutoff_at_contact"
    if short_window(start, end):
        return SHORT_WINDOW
    return end


def reference_cutoff(scene: Scene, target_id: str, contact: int) -> Cutoff:
    """Return the cutoff on the reference ego's route band, for a sample without one."""
    ego, target = scene.ego, scene.tracks[target_id]
    frames = np.flatnonzero(ego.valid)
    route = ego.states[frames[frames >= scene.anchor], :2]
    band = route_band(route, ego.size[1])
    n = len(scene.timestamps)
    return cutoff_frame(band, target.states, target.valid, target.size, contact, n)


def short_window(start: int, end: int) -> bool:
    """Return whether ``[start, end)`` is shorter than ``MIN_WINDOW_FRAMES`` and so unjudgeable."""
    return end - start < MIN_WINDOW_FRAMES


def summary(j: Judgement, substeps: int) -> dict:
    """Return the stored summary of one condition's judgement.

    Witness states keep full precision: rounding can turn a near miss into a touch.
    """
    out = {
        "status": j.status,
        "start_frame": j.start,
        **counts(j.codes),
        "codes": j.codes,
        "unlawful_only": j.unlawful_only(),
        "elapsed_s": round(j.elapsed_s, 3),
    }
    if j.before_start is not None:
        out["before_start"] = list(j.before_start)
    w = j.witness()
    if w is not None:
        frames = w.executed[::substeps]
        out["witness"] = {
            "id": w.candidate.id,
            "rank": w.candidate.rank,
            "stop_accel": w.stop_accel,
            "states": frames[:, :STATE_SIZE].tolist(),
            "target_clearance_m": w.target_clearance_m,
            "background_clearance_m": w.background_clearance_m,
            "opposing": w.opposing,
            "speeding": w.speeding,
            "excused_by": list(w.excused_by),
        }
    return out


def label_of(revealed: dict, early: dict, k: int, core_size: int) -> tuple[str, str | None]:
    """Return ``(test, why_not_judged)`` from the stored summaries' first ``k`` candidates."""

    def status(c: dict) -> str:
        return c["status"] if c["status"] in BEFORE_START else status_of(c["codes"][:k])

    revealed_background_safe = background_safe(revealed["codes"][:k])
    return classify(status(revealed), revealed_background_safe, status(early), k >= core_size)


def classify(
    revealed: str, revealed_background_safe: bool, early: str, core_complete: bool
) -> tuple[str, str | None]:
    """Return ``(test, why_not_judged)`` from the two conditions' statuses.

    ``anticipation`` means no listed candidate within the budget is safe from the reveal on,
    not that no response could be.
    """
    if not core_complete:
        return "not_judged", "core_incomplete"
    if revealed == "found":
        return "reaction", None
    if revealed != "not_found":
        return "not_judged", f"revealed_{revealed}"
    if not revealed_background_safe:
        return "not_judged", "background_constrained"
    if early == "found":
        return "anticipation", None
    if early != "not_found":
        return "not_judged", f"early_{early}"
    return "unresolved", None


def label_prefix(
    revealed: dict, early: dict, k: int, candidates: Sequence[Candidate]
) -> tuple[str, str | None]:
    """Return ``label_of`` for a stored run cut to the first ``k`` candidates.

    Raises:
        ValueError: if ``k`` does not end a whole pack or exceeds the stored results.
    """
    if len(prefix(candidates, k)) != k:
        raise ValueError(f"budget {k} is not a whole number of packs of the list")
    for c in (revealed, early):
        if c["status"] not in BEFORE_START and len(c["codes"]) < k:
            raise ValueError(f"budget {k}, but only {len(c['codes'])} results stored")
    return label_of(revealed, early, k, sum(c.core for c in candidates))
