"""``oncue generate`` and ``oncue label``: the hidden-target tests and their labels."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import VEHICLE_LIMITS
from ..generation.targets import START_RULES
from .options import WORKERS, given


def add_parsers(subcommands: argparse._SubParsersAction) -> None:
    add_generate(subcommands)
    add_label(subcommands)


def add_generate(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("generate", help="generate hidden-target tests on scene shards")
    parser.add_argument("scenes", nargs="+", type=Path, help="scene shards, files or directories")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output JSONL, one line per scene; rerunning resumes it",
    )
    parser.add_argument("--spec", required=True, help="target type: pedestrian, cyclist or vehicle")
    parser.add_argument("--model", type=Path, help="trajectory generator checkpoint")
    parser.add_argument(
        "--generators",
        dest="generators",
        metavar="GENERATORS",
        default="model,rule",
        help="generators to run, comma separated: model (the learned trajectory generator) and "
        "rule (rule-based generation). The ablation also uses "
        "contact_only (the model given only the contact waypoint), free (no waypoints) and grid "
        "(grid starts)",
    )
    parser.add_argument(
        "--raw-candidates",
        type=int,
        metavar="EVERY",
        help="diagnostic: for one request in EVERY, measure every candidate of the first start and "
        "write them to <out>.raw_candidates.jsonl; the output itself does not change",
    )
    parser.add_argument(
        "--device",
        help="torch device for the trajectory generator: cpu (default), cuda, cuda:N or mps; a run "
        "resumes only on the same device",
    )
    parser.add_argument(
        "--starts-per-request",
        type=starts_budget,
        metavar="K",
        help="sensitivity runs: keep at most K starts per request, a number or all (default: the "
        "spec's)",
    )
    parser.add_argument(
        "--no-history-check",
        action="store_true",
        help="sensitivity runs: treat the assumed history before the anchor as hidden in the "
        "exposure checks",
    )
    parser.add_argument(
        "--start-rule",
        choices=START_RULES,
        help="how a moving start is placed (default: transition); mean_speed keeps the request's "
        "mean speed from the assumed history on",
    )
    parser.add_argument(
        "--no-model-history",
        action="store_true",
        help="give the trajectory generator only a moving start's state at the anchor, not its "
        "assumed history",
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--parts", type=int, help="jobs per shard (default: core.runs.PARTS)")
    parser.set_defaults(handler=run_generate)


EVERY_START = "all"  # --starts-per-request value meaning "no budget"


def starts_budget(text: str) -> int | str:
    """argparse type for ``--starts-per-request``: a positive integer or ``EVERY_START``."""
    if text != EVERY_START and not (text.isdigit() and int(text) > 0):
        raise argparse.ArgumentTypeError(f"{text!r}: a positive number or {EVERY_START}")
    return text if text == EVERY_START else int(text)


def spec_overrides(args: argparse.Namespace) -> dict | None:
    """Return target-spec overrides from the sensitivity options, or None if none were given."""
    overrides = {}
    if args.starts_per_request is not None:
        every = args.starts_per_request == EVERY_START
        overrides["starts_per_request"] = None if every else args.starts_per_request
    if args.no_history_check:
        overrides["history_visibility"] = False
    if args.start_rule is not None:
        overrides["start_rule"] = args.start_rule
    if args.no_model_history:
        overrides["model_history"] = False
    return overrides or None


def run_generate(args: argparse.Namespace) -> None:
    from ..generation.pipeline import generate

    written = generate(
        args.scenes,
        args.out,
        args.spec,
        model=args.model,
        workers=args.workers,
        run_json=args.config,
        generators=args.generators,  # validated inside generate (generator_kinds)
        raw_every=args.raw_candidates,
        spec_overrides=spec_overrides(args),
        **given(parts=args.parts, device=args.device),
    )
    print(f"{written} scenes written to {args.out}")


def add_label(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("label", help="label generated tests with the response oracle")
    parser.add_argument("generated", type=Path, help="an `oncue generate` output")
    parser.add_argument("scenes", nargs="+", type=Path, help="the scene shards it was generated on")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output JSONL, one line per scene; rerunning resumes it",
    )
    parser.add_argument("--spec", required=True, help="the target type it was generated with")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--parts", type=int, help="jobs per shard (default: core.runs.PARTS)")
    parser.add_argument(
        "--budget", type=int, help="oracle candidates per condition (default: the config's)"
    )
    parser.add_argument(
        "--relabel",
        help="relabel only samples with these labels (comma separated); the input is a label "
        "output",
    )
    parser.add_argument(
        "--brake-jerk",
        type=float,
        help="emergency brake jerk of the oracle in m/s^3, for the brake-onset sensitivity "
        f"(default {VEHICLE_LIMITS.brake_jerk:g})",
    )
    parser.set_defaults(handler=run_label)


def run_label(args: argparse.Namespace) -> None:
    from ..generation.label import label

    written = label(
        args.generated,
        args.scenes,
        args.out,
        args.spec,
        workers=args.workers,
        run_json=args.config,
        budget=args.budget,
        relabel=tuple(args.relabel.split(",")) if args.relabel else None,
        **given(parts=args.parts, brake_jerk=args.brake_jerk),
    )
    print(f"{written} scenes written to {args.out}")
