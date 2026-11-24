"""``oncue cases`` and ``oncue site``: the offline case viewer and the site around it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .options import given

if TYPE_CHECKING:  # annotations only
    import argparse


def add_parsers(subcommands: argparse._SubParsersAction) -> None:
    add_cases(subcommands)
    add_site(subcommands)


def add_cases(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "cases", help="case viewer: OnCue tests replayed with the closed loop"
    )
    parser.add_argument("labels", nargs="+", type=Path, help="`oncue label` outputs")
    parser.add_argument("--scenes", nargs="+", type=Path, required=True, help="their scene shards")
    parser.add_argument("--spec", required=True, help="the target type they were generated with")
    parser.add_argument(
        "--closed", nargs="+", type=Path, help="`oncue closedloop` outputs (ranking)"
    )
    parser.add_argument("--out", type=Path, required=True, help="page directory")
    parser.add_argument(
        "--per-kind",
        type=int,
        help="cases per site kind and test type (default: render.cases.CASES_PER_KIND)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="processes recomputing the cases (default: render.cases.WORKERS)",
    )
    parser.set_defaults(handler=run_cases)


def run_cases(args: argparse.Namespace) -> None:
    from ..render.cases import render_cases

    page = render_cases(
        args.labels,
        args.scenes,
        args.out,
        args.spec,
        closed=args.closed or (),
        run_json=args.config,
        **given(per_kind=args.per_kind, workers=args.workers),
    )
    print(page)


def add_site(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("site", help="one offline site: home, method and case viewer")
    parser.add_argument("out", type=Path, help="site directory")
    parser.add_argument("--cases", type=Path, help="a built case viewer directory (oncue cases)")
    parser.add_argument("--explained", type=Path, help="the method explainer page (HTML)")
    parser.add_argument(
        "--results",
        type=Path,
        help="a run folder: the home page's numbers come from its analyze tables "
        "(render.site.RESULTS); without it the home page shows none",
    )
    parser.set_defaults(handler=run_site)


def run_site(args: argparse.Namespace) -> None:
    from ..render.site import build_site

    page = build_site(args.out, cases=args.cases, explained=args.explained, results=args.results)
    print(page)
