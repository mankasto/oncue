"""``oncue closedloop``: run the tests of an ``oncue label`` output against planners.

Writes one line per scene; running the same command again resumes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.config import DriverConfig
from ..core.io import iter_scenes, jsonable, load_config, scene_inputs
from ..core.runs import PARTS, SceneWriter, error_fields, resume_done, run_parts, write_or_check
from ..generation.label import usable
from ..generation.settings import check_upstream, run_settings
from ..generation.targets import SPECS
from ..utils.parallel import run_jobs
from .registry import DEFAULT_PLANNERS, PLANNERS, import_planner_modules, planner_weights
from .scene import (
    MODES,
    ROW_FIELDS,
    RunOptions,
    checked_modes,
    closed_loop_scene,
    is_triggered,
    run_name,
    world_class,
)

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence

    from ..core.record_types import TestRow


# Labels run by default; ``tests=None`` runs every usable sample.
TESTS = ("reaction", "anticipation", "unresolved")
_ROWS_BY_SCENE: dict = {}  # scene id -> its test rows; set once per worker process


def run(
    labels: str | Path,
    scenes: Sequence[str | Path],
    out_path: str | Path,
    spec: str,
    planners: list[str] | None = None,
    fixed: bool = False,
    workers: int = 1,
    parts: int = PARTS,
    run_json: str | None = None,
    world: str = "replay",
    modes: Sequence[str] = tuple(MODES),
    tests: tuple[str, ...] | None = TESTS,
    jerk_max: float | None = None,
    planner_modules: Sequence[str] = (),
) -> int:
    """Run the closed loop on every usable sample of a label output, one line per scene.

    Args:
        labels: an ``oncue label`` output.
        out_path: output file; scenes already in it are skipped.
        planners: planner names; None uses ``DEFAULT_PLANNERS``.
        fixed: fixed timing for every target (no triggered start).
        parts: jobs per scene shard.
        run_json: optional run configuration overlay.
        world: ``"replay"`` or ``"metadrive"``.
        tests: labels to run; None runs every usable sample.
        jerk_max: the planners' jerk limit; the reference ego keeps the ``DriverConfig`` one.
        planner_modules: modules to import first, here and in every worker; each registers its
            planners (``registry.register``), which ``planners`` may then name.

    Returns:
        The number of scene lines written.
    """
    world_class(world)  # an unknown world or mode fails now, not inside every scene
    import_planner_modules(planner_modules)
    unknown = [name for name in planners or DEFAULT_PLANNERS if name not in PLANNERS]
    if unknown:
        raise SystemExit(f"unknown planners {unknown}; an outside planner needs --planner-module")
    modes = checked_modes(modes)
    check_upstream(labels, spec, SPECS[spec], load_config(run_json))
    text = Path(labels).read_text()
    tests_by_scene, labelled = _read_tests(text, tests)
    if not labelled and text.strip():  # nothing labelled, so nothing to run
        raise SystemExit(f"{labels}: no labelled samples; label it first (oncue label)")
    files = scene_inputs(scenes)
    out_path = Path(out_path)
    options = RunOptions(tuple(planners or DEFAULT_PLANNERS), fixed, world, tuple(modes), jerk_max)
    write_or_check(out_path, _settings(labels, text, files, spec, run_json, options, tests))
    skip = resume_done(out_path)
    items = [(*job, spec, run_json, options) for job in run_parts(files, parts)]
    pending = {scene_id: rows for scene_id, rows in tests_by_scene.items() if scene_id not in skip}
    with out_path.open("a") as fh:
        writer = SceneWriter(fh, skip)
        # each worker gets the tests once, not with every job
        init = (pending, tuple(planner_modules))
        for lines in run_jobs(_job, items, workers, initializer=_init, initargs=init):
            writer.write(lines)
    writer.report(out_path)
    # Refuse scenes with tests that are missing from the scene inputs.
    missing = sorted(set(pending) - writer.ids)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(pending)} scenes with tests not in the scene inputs, "
            f"e.g. {missing[0]}; {writer.written} written"
        )
    return writer.written


def _read_tests(text: str, tests: tuple[str, ...] | None = TESTS):
    """Return ``(tests_by_scene, labelled)`` from an ``oncue label`` output.

    ``tests``: the labels kept (None: any); ``labelled`` says whether any row has a label at all.
    """
    rows_by_scene: dict[str, list] = {}
    labelled = False
    # An empty line raises here (only a corrupt file has one).
    for line in text.splitlines():
        entry = json.loads(line)
        stored = [r for r in entry.get("rows", []) if isinstance(r, dict) and "_states" in r]
        labelled = labelled or any("test" in r for r in stored)
        rows = [r for r in stored if usable(r) and (tests is None or r["test"] in tests)]
        if rows:
            rows_by_scene[entry["scene_id"]] = rows
    return rows_by_scene, labelled


def _settings(
    labels: str | Path,
    text: str,
    files: list[Path],
    spec: str,
    run_json: str | None,
    options: RunOptions,
    tests: tuple | None,
) -> dict:
    """Return the run's recorded settings."""
    planners = {
        name: {
            "drive": f"{PLANNERS[name][0].__module__}.{PLANNERS[name][0].__name__}",
            "options": PLANNERS[name][1],
        }
        for name in options.planners
    }
    return run_settings(
        "closedloop",
        spec,
        SPECS[spec],
        load_config(run_json),
        files,
        generated=str(labels),  # the settings' key
        # The digest stops a resumed run from keeping results of an older file at this path.
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        planners=planners,
        weights=planner_weights(list(options.planners)),
        fixed=options.fixed,
        world=options.world,
        modes=list(options.modes),
        tests="all" if tests is None else list(tests),
        run_json=run_json,
        jerk_max=options.jerk_max,
    )


def _init(tests: dict, planner_modules: tuple[str, ...] = ()) -> None:
    global _ROWS_BY_SCENE
    _ROWS_BY_SCENE = tests
    import_planner_modules(planner_modules)  # a fresh worker knows only the built-in planners


def _job(item: tuple):
    path, part, parts, spec_name, run_json, options = item
    spec = SPECS[spec_name]
    config = load_config(run_json)
    out = []
    for scene in iter_scenes(path, part, parts, config.replay):
        rows = _ROWS_BY_SCENE.get(scene.scene_id)
        if not rows:
            continue
        try:
            recs = closed_loop_scene(scene, rows, spec, config, DriverConfig(), options)
        except Exception as e:  # the whole scene failed: carry the error into every run
            error = error_fields(e)
            recs = failed_scene_records(rows, scene.anchor, error, options)
        out.append(json.dumps(jsonable({"scene_id": scene.scene_id, "rows": recs})))
    return out


def failed_scene_records(
    rows: list[TestRow],
    anchor: int,
    error: dict,
    options: RunOptions,
) -> list[TestRow]:
    """Return the rows of a scene that failed as a whole, each run carrying the scene's error.

    The rows stay in the rates' denominators.
    """
    planners = options.planners or DEFAULT_PLANNERS
    run_names = [run_name(name, mode) for name in planners for mode in options.modes]
    return [
        {
            **{k: row.get(k) for k in ROW_FIELDS},
            "closed_loop": {name: {**error, "seconds": 0.0} for name in run_names},
            "start": "triggered" if is_triggered(row, options.fixed, anchor) else "fixed",
        }
        for row in rows
    ]
