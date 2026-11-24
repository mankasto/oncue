"""``oncue closedloop``: replay labelled tests against the planners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .options import WORKERS, chosen, comma_list, given

if TYPE_CHECKING:  # annotations only
    import argparse


def add_parsers(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("closedloop", help="replay labelled tests against planners")
    parser.add_argument("labels", type=Path, help="an `oncue label` output")
    parser.add_argument("scenes", nargs="+", type=Path, help="the scene shards it was generated on")
    parser.add_argument(
        "--out", type=Path, required=True, help="JSONL, a line per scene (resumable)"
    )
    parser.add_argument("--spec", required=True, help="the target type it was generated with")
    parser.add_argument(
        "--planners",
        help="comma list (closedloop.registry.PLANNERS; default: "
        "closedloop.registry.DEFAULT_PLANNERS)",
    )
    parser.add_argument(
        "--fixed", action="store_true", help="fixed timing for standing targets too"
    )
    parser.add_argument(
        "--world",
        default="replay",
        help="where the planners drive (closedloop.scene.WORLDS): replay (default; the "
        "experiments' world) or metadrive (the same runs mirrored in MetaDrive for drawing)",
    )
    parser.add_argument(
        "--modes",
        default="masked,see",
        help="observation modes run, comma list (closedloop.scene.MODES; experiment 4's screening: "
        "masked on the pool, then see on the candidates)",
    )
    parser.add_argument(
        "--tests",
        default="reaction,anticipation,unresolved",
        help="labels run, comma list; all: every usable sample, not_judged included (experiment 4)",
    )
    parser.add_argument(
        "--jerk-max",
        type=float,
        help="the planners' jerk limit, m/s^3 (brake-onset sensitivity; default DriverConfig's)",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--parts", type=int, help="jobs per shard (default: core.runs.PARTS)")
    parser.set_defaults(handler=run_closedloop)


def run_closedloop(args: argparse.Namespace) -> None:
    from ..closedloop.run import run

    written = run(
        args.labels,
        args.scenes,
        args.out,
        args.spec,
        planners=comma_list(args.planners),
        fixed=args.fixed,
        workers=args.workers,
        run_json=args.config,
        world=args.world,
        modes=args.modes.split(","),
        tests=chosen(args.tests),
        **given(parts=args.parts, jerk_max=args.jerk_max),
    )
    print(f"{written} scenes written to {args.out}")
