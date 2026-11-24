"""`oncue generate`: hidden-target tests for every target type. `generate` runs `generate_scene`
over scene shards: routes, requests, starts, then a search per generator."""

from __future__ import annotations

import json
import multiprocessing
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from ..core.config import Config, DriverConfig
from ..core.io import iter_scenes, jsonable, load_config, scene_inputs
from ..core.runs import (
    PARTS,
    SceneWriter,
    done_in_both,
    error_fields,
    resume_done,
    run_parts,
    write_or_check,
)
from ..core.types import Scene
from ..utils.device import device_record, is_cuda, make_reproducible, resolve_device
from ..utils.parallel import run_jobs
from .checks.diagnostics import RawCandidates, raw_path
from .checks.validity import check_candidate
from .context import SceneContext, scene_context
from .motion.generators import (
    Candidate,
)
from .motion.starts import request_starts
from .requests.build import Request, Start, build_requests
from .requests.sites import Route, has_sites, routes
from .rows import generator_row, request_id, request_row, scene_funnel, void_route_row
from .search import (
    WAYPOINTS,
    SceneGeneration,
    screened_rule_candidates,
    search_generator,
    search_starts,
)
from .settings import run_settings
from .targets import SPECS, TargetSpec

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterable

    from ..models.modeling.model import RouteModel

GENERATORS = ("model", "rule", "contact_only", "free", "grid")


def generate_scene(
    scene: Scene,
    spec: TargetSpec,
    config: Config,
    driver: DriverConfig,
    model: RouteModel | None = None,
    generators: Iterable[str] = ("model", "rule"),
    funnel: dict | None = None,
    raw: RawCandidates | None = None,
) -> list[dict]:
    """Return the output rows of one scene, one per request and generator.

    `funnel` and `raw`, when given, collect the scene's funnel and raw-candidate rows.
    """
    generators = generator_kinds(generators, model is not None)
    if not has_sites(scene, spec.sites):
        if funnel is not None:
            funnel.update(kind=spec.kind, scene="no_site")
        return []  # no route at all: skip building the scene context (the costly part)
    ctx = scene_context(scene, config, driver)
    if ctx.reference_contact is not None:  # reference ego already hits the background: drop it
        if funnel is not None:
            funnel.update(kind=spec.kind, scene="reference_contact")
        return []
    generation = SceneGeneration(ctx, spec, model, generators, raw)
    route_list = routes(ctx, spec)
    rows, errors = [], 0
    for route in route_list:
        counts = Counter(ctx.considered), Counter(ctx.drops)
        raw_rows = None if raw is None else len(raw.rows)
        try:
            rows += _route_rows(generation, route)
        except Exception as e:
            # undo what the failed route added to the funnel and raw rows
            ctx.considered, ctx.drops = counts
            if raw is not None:
                del raw.rows[raw_rows:]
            ctx.drop("route", "error")
            rows.append(void_route_row(ctx, spec, route, e))
            errors += 1
    if funnel is not None:
        funnel.update(scene_funnel(ctx, spec, route_list, rows, errors))
    return rows


def _route_rows(generation: SceneGeneration, route: Route) -> list[dict]:
    """Return the rows of one route: one per unrealizable request, else one per generator."""
    ctx, spec = generation.ctx, generation.spec
    rows = []
    for req in build_requests(ctx, route, spec):
        _choose_starts(ctx, req, spec)
        base = request_row(ctx, spec, req)
        if req.realizable != "ok":
            ctx.drop("request", req.realizable)
            rows.append(base)
            continue
        for kind in generation.generators:
            found = search_generator(generation, req, kind, base["request_id"])
            rows.append(generator_row(ctx, req, base, kind, found))
    return rows


def _choose_starts(ctx: SceneContext, req: Request, spec: TargetSpec) -> None:
    """Count the request and give a realizable one its starts, most common first."""
    ctx.consider("request")
    if req.realizable == "ok":
        req.starts = request_starts(ctx, req, spec)
        if not req.starts and req.realizable == "ok":  # request_starts may set its own reason
            req.realizable = "no_start"


# ---------------------------------------------------------------- a test's rule motion
def rebuilt_request(ctx: SceneContext, route: Route, spec: TargetSpec, wanted_id: str) -> Request:
    """Rebuild the request of `route` whose id is `wanted_id`, with its starts as generated."""
    scene_id = ctx.scene.scene_id
    matches = [
        req
        for req in build_requests(ctx, route, spec)
        if request_id(scene_id, spec, req) == wanted_id
    ]
    if len(matches) != 1:
        raise ValueError(f"request {wanted_id}: {len(matches)} requests of its route have its id")
    req = matches[0]
    _choose_starts(ctx, req, spec)
    return req


class RuleMotion(NamedTuple):
    """The rule-based generator's motion from one start."""

    candidate: Candidate
    fallback: str | None  # None: the search's sample; otherwise the first check the fallback fails


def rule_motion(ctx: SceneContext, req: Request, spec: TargetSpec, start: Start) -> RuleMotion:
    """Return the rule-based generator's motion from one start of `req`.

    When no candidate passes, falls back to the first one checked and the hard check it failed,
    so the rule baseline still has a row for every test.
    """
    found = search_starts(
        ctx, req, spec, [start], lambda each: screened_rule_candidates(ctx, req, each, spec)
    )
    if found.candidate is not None:
        return RuleMotion(found.candidate, None)
    checked, _ = screened_rule_candidates(ctx, req, start, spec)
    if not checked:
        raise ValueError(f"request on {req.route.site_id}: no rule candidate from start {start.s}")
    first = checked[0]
    failed = check_candidate(ctx, req, spec, first.states, first.valid, req.table).reject
    return RuleMotion(first, failed)


# ---------------------------------------------------------------- runs
_MODEL: dict = {}  # (resolved checkpoint path, size, mtime, device) -> model; the last one only


def generate(
    inputs: Iterable[str | Path],
    out_path: str | Path,
    spec: str,
    model: str | Path | None = None,
    workers: int = 1,
    parts: int = PARTS,
    run_json: str | Path | None = None,
    generators: str | Iterable[str] = ("model", "rule"),
    raw_every: int | None = None,
    device: str = "cpu",
    spec_overrides: dict | None = None,
) -> int:
    """Append one line per scene to `out_path`, skipping ones already there; return lines written.

    Args:
        raw_every: record raw candidates for one request in this many, to a companion file.
        device: the trajectory generator's torch device; a resumed run on another one is refused.
        spec_overrides: TargetSpec fields replaced for a sensitivity run.
    """
    generators = generator_kinds(generators, model is not None)
    target = target_spec(spec, spec_overrides)
    files = scene_inputs(inputs)
    out_path = Path(out_path)
    raw_file, extra = None, {}
    if raw_every is not None:
        RawCandidates(raw_every)  # fail fast on a bad value, not per scene
        raw_file = raw_path(out_path)
        extra["raw_candidates"] = {"every": raw_every, "file": raw_file.name}
    if model is not None:
        device = resolve_device(device)
        make_reproducible(device)  # before the first CUDA call, which device_record makes
        extra["device"] = device_record(device)
    settings = run_settings(
        "generate",
        spec,
        target,
        load_config(run_json),
        files,
        model,
        generators=list(generators),
        run_json=run_json,
        **extra,
    )
    write_or_check(out_path, settings)
    skip = resume_done(out_path)
    if raw_file is not None:  # a scene is done only when both files have its line
        skip = done_in_both(out_path, skip, raw_file, resume_done(raw_file))
    opts = {
        "spec": spec,
        "spec_overrides": spec_overrides,
        "model": None if model is None else str(model),
        "run_json": run_json,
        "generators": tuple(generators),
        "skip": skip,
        "raw_every": raw_every,
        "device": device,
    }
    items = [(*part, opts) for part in run_parts(files, parts)]
    with ExitStack() as opened:
        writer = SceneWriter(opened.enter_context(out_path.open("a")), skip)
        raw_writer = None
        if raw_file is not None:
            raw_writer = SceneWriter(opened.enter_context(raw_file.open("a")), skip)
        # a forked process cannot use CUDA once the parent has
        context = multiprocessing.get_context("spawn") if is_cuda(device) else None
        for lines, raw_lines in run_jobs(_job, items, workers, context=context):
            if raw_writer is not None:
                raw_writer.write(raw_lines)
            writer.write(lines)
    writer.report(out_path)
    return writer.written


def generator_kinds(generators: str | Iterable[str], has_model: bool) -> tuple[str, ...]:
    """Validate and return the requested generators; the learned ones need a checkpoint."""
    if isinstance(generators, str):
        generators = generators.split(",")
    kinds = tuple(str(c).strip() for c in generators)
    unknown = [k for k in kinds if k not in GENERATORS]
    if unknown or not kinds:
        raise SystemExit(f"generators {list(kinds)}: each one of {list(GENERATORS)}")
    if len(set(kinds)) != len(kinds):
        raise SystemExit(f"generators {list(kinds)}: each at most once")
    if not has_model and set(kinds) & set(WAYPOINTS):
        raise SystemExit(
            "the model generator needs a checkpoint (--model); without one, use --generators rule"
        )
    return kinds


# TargetSpec fields a sensitivity run may override
SPEC_OVERRIDES = ("starts_per_request", "history_visibility", "start_rule", "model_history")


def target_spec(name: str, overrides: dict | None = None) -> TargetSpec:
    """Return `SPECS[name]` with the fields in `overrides` (from SPEC_OVERRIDES) replaced."""
    unknown = sorted(set(overrides or {}) - set(SPEC_OVERRIDES))
    if unknown:
        raise ValueError(f"spec overrides {unknown}: only {list(SPEC_OVERRIDES)}")
    return replace(SPECS[name], **(overrides or {}))


def _job(item: tuple):
    """Return (output lines, raw-candidate lines) for one part of one input, one line per scene."""
    path, part, parts, opts = item
    spec = target_spec(opts["spec"], opts.get("spec_overrides"))
    config = load_config(opts.get("run_json"))
    model = _load_model(opts.get("model"), opts["device"])
    out, raw_out = [], []
    for scene in iter_scenes(path, part, parts, config.replay):
        if scene.scene_id in opts["skip"]:
            continue
        funnel: dict = {}
        raw = None if opts.get("raw_every") is None else RawCandidates(opts["raw_every"])
        try:
            rows = generate_scene(
                scene,
                spec,
                config,
                DriverConfig(),
                model,
                generators=opts["generators"],
                funnel=funnel,
                raw=raw,
            )
        except Exception as e:  # the scene as a whole failed (e.g. its context)
            rows = [error_fields(e)]
            funnel = {"kind": spec.kind, "scene": "error"}
        line = {"scene_id": scene.scene_id, "rows": rows, "funnel": funnel}
        out.append(json.dumps(jsonable(line)))
        if raw is not None:
            raw_line = {"scene_id": scene.scene_id, "kind": spec.kind, "rows": raw.rows}
            if any("error" in row for row in rows):  # a resumed run will redo this scene
                raw_line["error"] = "the scene's generation failed"
            raw_out.append(json.dumps(jsonable(raw_line)))
    return out, raw_out


def _load_model(path: str | Path | None, device: str):
    """Return the trajectory generator at `path` on `device`, cached per process until the file
    changes."""
    if not path:
        return None
    p = Path(path).resolve()
    key = (str(p), p.stat().st_size, p.stat().st_mtime_ns, str(device))
    if key not in _MODEL:
        import torch

        from ..models.modeling.model import load_route_model

        torch.set_num_threads(1)
        make_reproducible(device)  # in a spawned process too, before its first CUDA use
        _MODEL.clear()
        _MODEL[key] = load_route_model(path, device)
    return _MODEL[key]
