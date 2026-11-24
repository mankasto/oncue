"""``oncue export`` and ``oncue stats``: raw WOMD shards to scene files and logged statistics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .options import given

if TYPE_CHECKING:  # annotations only
    import argparse


def add_parsers(subcommands: argparse._SubParsersAction) -> None:
    add_export(subcommands)
    add_stats(subcommands)


def add_export(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("export", help="raw TFRecord shards to scene files")
    parser.add_argument("shards", nargs="+", type=Path, help="raw WOMD Scenario TFRecord shards")
    parser.add_argument("--out", type=Path, required=True, help="directory of *.scenes.jsonl.gz")
    parser.add_argument(
        "--workers",
        type=int,
        help="processes at once, each holding a raw shard (default: data.waymo.export."
        "EXPORT_WORKERS)",
    )
    parser.add_argument("--split", default="all", choices=("train", "internal_validation", "all"))
    parser.add_argument("--dataset-version", default="v1.3.1", help="WOMD release of the shards")
    parser.set_defaults(handler=run_export)


def run_export(args: argparse.Namespace) -> None:
    from ..data.waymo.export import export

    export(
        args.shards,
        args.out,
        split=args.split,
        dataset_version=args.dataset_version,
        **given(workers=args.workers),
    )


def add_stats(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("stats", help="logged statistics behind the target parameters")
    parser.add_argument(
        "scenes",
        nargs="+",
        type=Path,
        help="scene shards or directories of them, raw shards, or row files (*.rows.npz) to merge",
    )
    parser.add_argument("--out", type=Path, help="JSON output (cited by targets.py)")
    parser.add_argument(
        "--rows-out", type=Path, help="keep the road users' rows here (*.rows.npz), to merge later"
    )
    parser.add_argument(
        "--workers", type=int, help="processes at once (default: data.stats.WORKERS)"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        help="shortest future window (default: data.stats.MIN_FUTURE_FRAMES)",
    )
    parser.set_defaults(handler=run_stats)


def run_stats(args: argparse.Namespace) -> None:
    from ..data.stats import PERCENTILES, ROWS_SUFFIX, run

    if args.out is None and args.rows_out is None:
        raise SystemExit("oncue stats: give --out, --rows-out or both")
    if args.rows_out is not None and not args.rows_out.name.endswith(ROWS_SUFFIX):
        raise SystemExit(f"oncue stats: --rows-out must end with {ROWS_SUFFIX}")
    result = run(
        args.scenes,
        args.out,
        rows_out=args.rows_out,
        **given(workers=args.workers, min_frames=args.min_frames),
    )
    if result is None:
        print(f"rows written to {args.rows_out}")
        return
    print(f"{len(result['files'])} scene files; percentiles {list(PERCENTILES)}")
    for kind, statistics in result["types"].items():
        rest = {name: value for name, value in statistics.items() if name != "n"}
        print(kind, statistics["n"], rest)
