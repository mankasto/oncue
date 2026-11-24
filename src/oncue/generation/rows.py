"""The rows `oncue generate` writes: one per request and generator, plus the scene's funnel."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..core.io import digest
from ..core.runs import error_fields
from .checks.realism import FLAW_GROUPS, flaw_group, found_flaws
from .context import SceneContext
from .motion.generators import (
    Candidate,
    start_anchor_state,
)
from .motion.starts import atypical_speed
from .requests.build import Request, Start, route_limit
from .requests.sites import Route
from .targets import TargetSpec

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Sequence

    from ..core.record_types import TestRow
    from .search import SearchResult


START_STATE_DECIMALS = 4  # decimals kept for a start's anchor state in the output rows

REQUEST_ID_HEX_DIGITS = 16  # a request id is the first this many hex digits of its digest

# how a sample starts, along what route and from what kind of site
SAMPLE_KINDS = ("start_kind", "route_kind", "site_kind")

UNKNOWN_KIND = "unknown"  # used when a kind was not recorded


def is_test(row: TestRow) -> bool:
    """Return whether a row is a found sample, that is, a test."""
    return bool(row.get("found"))


def zero_flaw(row: TestRow) -> bool:
    """Return whether a row is a found sample with zero flaws (assumed flaws included)."""
    return is_test(row) and row.get("flaw_count") == 0


def request_row(ctx: SceneContext, spec: TargetSpec, req: Request) -> dict:
    """Build the request's row, with its first start in full so every method can get it."""
    r = req.route
    first = None
    if req.starts:
        start = req.starts[0]
        first = _start_fields(ctx, req, start, start_anchor_state(start, ctx.anchor))
    row = {
        "scene_id": ctx.scene.scene_id,
        "kind": spec.kind,
        **route_fields(r),
        "contact_key": req.contact_key,
        "W_req": req.W,
        "profile": req.starts[0].profile if req.starts else None,
        "realizable": req.realizable,
        # mean speed required after the reveal, and whether it is uncommon
        "mean_speed": None if req.mean_speed is None else round(req.mean_speed, 2),
        "atypical_speed": req.mean_speed is not None and atypical_speed(req.mean_speed, spec),
        "starts": len(req.starts),
        "start_drops": dict(sorted(req.start_drops.items())),
        "first_start": first,
        "limit": round(route_limit(ctx, r), 2),
        "limit_source": "route" if r.meta.get("limit") else "ego",
    }
    row["request_id"] = request_id(ctx.scene.scene_id, spec, req)
    return row


def _start_fields(
    ctx: SceneContext, req: Request, start: Start, anchor_state: Sequence[float]
) -> dict:
    """Return a start's output fields; waypoints are [arc, frame]."""
    anchor, dt = ctx.anchor, ctx.dt
    rank = next((k for k, s in enumerate(req.starts, 1) if s is start), None)
    out = {
        "start_rank": rank,
        "start_s": round(float(start.s), 3),
        "start_speed": round(float(start.speed), 3),
        "f_go": None if start.f_go is None else int(start.f_go),
        "start_back_m": start.back_m,
        "band": None,  # no longer set; the key stays so rows keep their shape
        "start_accel": None if start.accel is None else round(float(start.accel), 3),
        "start_state": [round(float(v), START_STATE_DECIMALS) for v in anchor_state],
        "wp_contact": [round(float(req.s_contact), 3), int(req.f_contact)],
        "wp_reveal": [
            None if start.s_reveal is None else round(float(start.s_reveal), 3),
            int(req.f_reveal),
        ],
    }
    if start.f_go is not None:
        out["start_lead_s"] = start.lead_s
        out["wait_s"] = round((start.f_go - anchor) * dt, 2)
    return out


def route_fields(route: Route) -> dict:
    """Return the route's identity as the output rows record it."""
    kinds = sample_kinds(route)
    return {
        "route_kind": kinds["route_kind"],
        "site_kind": kinds["site_kind"],
        "site": route.site_id,
        "lane": route.meta.get("lane"),
        "route": route.meta.get("chain"),
        "u_conflict": round(route.conflict_station, 2),
        "angle": round(route.angle_deg),
    }


def sample_kinds(
    route: Route, start: Start | None = None, candidate: Candidate | None = None
) -> dict:
    """Return the SAMPLE_KINDS of a sample, or of a route alone when start and candidate are None.

    Each comes from the first of the candidate's, start's and route's records that has it.
    """
    records = [route.meta]
    if start is not None:
        records.insert(0, start.extra)
    if candidate is not None:
        records.insert(0, candidate.extra)
    fallback = {"site_kind": route.kind, "route_kind": route.route_kind}
    return {
        key: next(
            (r[key] for r in records if r.get(key) is not None),
            fallback.get(key, UNKNOWN_KIND),
        )
        for key in SAMPLE_KINDS
    }


def request_id(scene_id: str, spec: TargetSpec, req: Request) -> str:
    """Return the request's id, a digest of its route, contact and W."""
    r = req.route
    return digest(
        [
            scene_id,
            spec.kind,
            r.kind,
            r.route_kind,
            r.site_id,
            r.meta.get("lane"),
            r.meta.get("chain"),
            round(r.angle_deg, 1),
            round(r.conflict_station, 2),
            req.contact_key,
            req.W,
        ]
    )[:REQUEST_ID_HEX_DIGITS]


def generator_row(
    ctx: SceneContext, req: Request, base: dict, kind: str, search: SearchResult
) -> dict:
    """Build one generator's row: the request's fields, the search's cost and any sample."""
    row = {
        **base,
        "generator": kind,
        "sample_id": f"{base['request_id']}/{kind}",
        "found": False,
        "rejects": search.rejects,
        "starts_tried": search.starts_tried,
        "generated": search.generated,
        "tried": search.tried,
        "first_pass_tried": search.first_pass_tried,
    }
    if search.candidate is not None:
        row.update(_sample_fields(ctx, req, search))
    return row


def _sample_fields(ctx: SceneContext, req: Request, search: SearchResult) -> dict:
    """Build the sample's output fields, with the full-precision states the label step reads."""
    anchor, dt = ctx.anchor, ctx.dt
    start, cand, res = search.start, search.candidate, search.checked
    states, f_c, reveal = cand.states, res.f_contact, res.event.reveal
    out = {
        "found": True,
        "mode": cand.mode,
        "score": None if cand.score is None else round(cand.score, 3),
        **cand.extra,
        "t_r": round((reveal - anchor) * dt, 2),
        "t_c": round((f_c - anchor) * dt, 2),
        "W": round((f_c - reveal) * dt, 2),
        "f_reveal": int(reveal),
        "f_contact": int(f_c),
        "cutoff_frame": int(res.cutoff.frame),
        "cutoff_reason": res.cutoff.reason,  # left_route, target_unobserved or run_end
        **(res.info or {}),
        "flaw_count": res.flaw_count,
        # 1 when the first candidate passing the hard checks became the sample
        "pick_rank": search.pick_rank,
    }
    xy = states[anchor:, :2][cand.valid[anchor:]]
    if len(xy) > 1:
        out["peak_speed"] = round(
            float((np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt).max()), 2
        )
    out.update(_start_fields(ctx, req, start, states[anchor]))
    out.update(sample_kinds(req.route, start, cand))
    if start.extra.get("grid"):  # a rolling target's grid start
        out["grid_start"] = {"s": round(start.s, 2), "speed": round(start.speed, 2)}
    out["_states"] = np.asarray(states, float).tolist()
    out["_valid"] = cand.valid.astype(int).tolist()
    out["_station"] = float(req.route.conflict_station)
    out["_f_hold"] = None if start.f_go is None else int(start.f_go)
    return out


def void_route_row(ctx: SceneContext, spec: TargetSpec, route: Route, error: Exception) -> dict:
    """Return the error row for a failed route."""
    return {
        "scene_id": ctx.scene.scene_id,
        "kind": spec.kind,
        **sample_kinds(route),
        "lane": route.meta.get("lane"),
        "u_conflict": round(route.conflict_station, 2),
        **error_fields(error),
    }


def scene_funnel(
    ctx: SceneContext, spec: TargetSpec, route_list: list[Route], rows: list[dict], errors: int
) -> dict:
    """Build the scene's funnel: per stage what was considered and dropped, and per generator
    what its searches gave."""
    return {
        "kind": spec.kind,
        "scene": "ok",
        "considered": dict(sorted(ctx.considered.items())),
        "dropped": dict(sorted(ctx.drops.items())),
        "atypical_speed": len({r["request_id"] for r in rows if r.get("atypical_speed")}),
        "routes": len(route_list) - errors,
        "route_errors": errors,
        "windows": len(spec.windows_s),
        **_generator_counts(rows),
    }


def _generator_counts(rows: list[TestRow]) -> dict:
    """Count, per generator, the samples, rejects, flaws, flaw groups and sample kinds."""
    searched = [r for r in rows if "generator" in r]
    samples = [r for r in searched if is_test(r)]
    rejects, flaws, groups, kinds = {}, {}, {}, {}
    for r in searched:
        rejects.setdefault(r["generator"], Counter()).update(r["rejects"])
    for r in samples:
        flaws.setdefault(r["generator"], Counter()).update(found_flaws(r["flaws"]))
        by_group = groups.setdefault(r["generator"], Counter(dict.fromkeys(FLAW_GROUPS, 0)))
        by_group[flaw_group(r["flaws"])] += 1
        by_kind = kinds.setdefault(r["generator"], {key: Counter() for key in SAMPLE_KINDS})
        for key in SAMPLE_KINDS:
            by_kind[key][r[key]] += 1
    return {
        "found": _sorted(Counter(r["generator"] for r in samples)),
        "zero_flaw": _sorted(Counter(r["generator"] for r in samples if zero_flaw(r))),
        "rejects": {c: _sorted(counts) for c, counts in sorted(rejects.items())},
        "flaws": {c: _sorted(counts) for c, counts in sorted(flaws.items())},
        "flaw_groups": {c: dict(counts) for c, counts in sorted(groups.items())},
        "kinds": {
            c: {key: _sorted(counts) for key, counts in by_kind.items()}
            for c, by_kind in sorted(kinds.items())
        },
    }


def _sorted(counts: Counter) -> dict:
    return dict(sorted(counts.items()))
