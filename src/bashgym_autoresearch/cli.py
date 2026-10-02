"""Command-line entry point."""

from __future__ import annotations

import argparse

from bashgym_autoresearch import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bashgym-ar", description=__doc__)
    parser.add_argument("--version", action="version", version=f"bashgym-ar {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0
