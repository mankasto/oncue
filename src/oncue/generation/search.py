"""Search a request's starts and candidates, in order, for the passing candidate with the fewest
flaws (`fewest_flaws`)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, TypeVar

from .checks.validity import (
    Checked,
    approx_timing_error,
    check_candidate,
    prescreen_passes,
    tolerance_frames,
)
from .context import SceneContext
from .motion.generators import (
    Candidate,
    model_candidates,
    rule_candidates,
)
from .motion.starts import grid_starts
from .requests.build import Request, Start
from .targets import TargetSpec

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Callable, Iterable, Iterator

    from ..models.modeling.model import RouteModel
    from .checks.diagnostics import RawCandidates


# waypoints each learned generator gives the model
WAYPOINTS = {
    "model": ("contact", "reveal"),
    "contact_only": ("contact",),
    "free": (),
    "grid": ("contact", "reveal"),
}
Item = TypeVar("Item")


@dataclass(frozen=True, eq=False)
class SceneGeneration:
    """What the routes of one scene share."""

    ctx: SceneContext
    spec: TargetSpec
    model: RouteModel | None
    generators: tuple[str, ...]
    raw: RawCandidates | None


def search_generator(
    generation: SceneGeneration, req: Request, kind: str, req_id: str
) -> SearchResult:
    """Search one generator `kind` over the request's starts; a learned one runs one forward
    pass per start."""
    ctx, spec, model, raw = generation.ctx, generation.spec, generation.model, generation.raw
    if kind == "rule":
        starts = req.starts
        return search_starts(
            ctx, req, spec, starts, lambda start: screened_rule_candidates(ctx, req, start, spec)
        )
    if kind not in WAYPOINTS:  # should be unreachable, checked by generator_kinds
        raise ValueError(f"unknown generator {kind!r}")
    starts = grid_starts(ctx, req, spec) if kind == "grid" else req.starts
    modes = model.cfg.intention_points  # candidates per forward pass
    # the search and the diagnostic share the first start's candidates
    first = None
    if raw is not None and raw.includes(req_id) and starts:
        first = list(model_candidates(model, ctx, req, starts[0], spec, WAYPOINTS[kind]))

    def candidates(start: Start) -> tuple[Iterable[Candidate], int]:
        if first is not None and start is starts[0]:
            return first, modes
        return model_candidates(model, ctx, req, start, spec, WAYPOINTS[kind]), modes

    found = search_starts(ctx, req, spec, starts, candidates)
    if raw is not None:
        raw.record(ctx, req, spec, req_id, kind, starts, first)
    return found


def screened_rule_candidates(ctx: SceneContext, req: Request, start: Start, spec: TargetSpec):
    """Return (rule candidates from `start` in profile order, how many were generated).

    With a prescreen, only the `rule_exact_checks` with the closest approximate timing are kept.
    """
    cands = list(rule_candidates(req, start))
    if not spec.prescreen:
        return cands, len(cands)
    tol = tolerance_frames(spec, ctx.dt)
    scored = []
    for order, c in enumerate(cands):
        err = approx_timing_error(ctx, req, spec, c.states, c.valid, req.table)
        if prescreen_passes(err, tol):
            scored.append((err, order))
    closest = sorted(order for _, order in sorted(scored)[: spec.rule_exact_checks])
    return [cands[order] for order in closest], len(cands)


class SearchResult(NamedTuple):
    """What a search over a request's starts found, and what it cost."""

    candidate: Candidate | None  # the sample (None when no start produced one)
    checked: Checked | None  # its checks
    rejects: dict  # rejected candidates by reason
    start: Start | None  # the start it came from
    pick_rank: int | None  # rank among the candidates passing the hard checks (1 = first)
    starts_tried: int  # starts tried (for a learned generator, equals forward passes)
    generated: int  # candidates generated
    tried: int  # candidates checked
    first_pass_tried: int | None  # count of candidates checked up to the first that passed


def search_starts(
    ctx: SceneContext,
    req: Request,
    spec: TargetSpec,
    starts: Iterable[Start],
    candidates: Callable[[Start], tuple[Iterable[Candidate], int]],
) -> SearchResult:
    """Return the sample of the request via `fewest_flaws`.

    `candidates(start)` returns (candidates for the start, how many the generator generated).
    """
    cost = _SearchCost()
    best = fewest_flaws(_passing(ctx, req, spec, starts, candidates, cost))
    rejects = dict(sorted(cost.rejects.items()))
    spent = (cost.starts, cost.generated, cost.tried, cost.first_pass_tried)
    if best is None:
        return SearchResult(None, None, rejects, None, None, *spent)
    cand, res, start, rank = best[1]
    return SearchResult(cand, res, rejects, start, rank, *spent)


def fewest_flaws(passing: Iterable[tuple[int, Item]]) -> tuple[int, Item] | None:
    """Return the earliest (flaw count, item) with the fewest flaws, stopping at zero; None if
    empty."""
    best = None
    for flaws, item in passing:
        if best is None or flaws < best[0]:
            best = (flaws, item)
        if flaws == 0:
            break
    return best


@dataclass
class _SearchCost:
    """Running cost of a search; rejects counted by their first failed hard check."""

    starts: int = 0
    generated: int = 0
    tried: int = 0
    rejects: Counter = field(default_factory=Counter)
    passing: int = 0  # candidates that passed the hard checks
    first_pass_tried: int | None = None  # `tried` when the first passing one arrived


def _passing(
    ctx: SceneContext,
    req: Request,
    spec: TargetSpec,
    starts: Iterable[Start],
    candidates: Callable[[Start], tuple[Iterable[Candidate], int]],
    cost: _SearchCost,
) -> Iterator[tuple[int, tuple[Candidate, Checked, Start, int]]]:
    """Yield (flaw count, (candidate, Checked, start, rank among passing)) in search order."""
    for start in starts:
        cands, made = candidates(start)
        cost.starts += 1
        cost.generated += made
        for cand in cands:
            cost.tried += 1
            res = check_candidate(ctx, req, spec, cand.states, cand.valid, req.table)
            if res.reject is not None:
                cost.rejects[res.reject] += 1
                continue
            cost.passing += 1
            if cost.first_pass_tried is None:
                cost.first_pass_tried = cost.tried
            yield res.flaw_count, (cand, res, start, cost.passing)
