"""Command-line entry point for ``oncue <command> --help``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from . import closedloop, data, generation, model, web

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence
    from types import ModuleType

# `oncue --help` lists commands in this order; experiment commands live in `python -m experiments`
COMMAND_MODULES = (
    model,
    web,
    data,
    generation,
    closedloop,
)


def main(
    argv: list[str] | None = None,
    command_modules: Sequence[ModuleType] = COMMAND_MODULES,
    prog: str = "oncue",
) -> None:
    """Parse ``argv`` and run the named command. Reused by ``experiments/cli`` with its own set."""
    args = build_parser(command_modules, prog).parse_args(argv)
    args.handler(args)


def build_parser(
    command_modules: Sequence[ModuleType] = COMMAND_MODULES, prog: str = "oncue"
) -> argparse.ArgumentParser:
    """Build the argument parser; each parsed command stores its callable in ``args.handler``."""
    description = f"Command line entry point: `{prog} <command> --help`."
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("--config", type=Path, help="JSON config overriding the defaults")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for module in command_modules:
        module.add_parsers(subcommands)
    return parser
