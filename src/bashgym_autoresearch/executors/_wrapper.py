"""Run one stage command and record its exit code atomically.

Usage: python -m bashgym_autoresearch.executors._wrapper <run_dir> -- <argv...>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_exit_code(run_dir: Path, code: int) -> None:
    temporary = run_dir / "exit_code.tmp"
    temporary.write_text(str(code), encoding="ascii")
    os.replace(temporary, run_dir / "exit_code")


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] != "--":
        print("usage: _wrapper <run_dir> -- <argv...>", file=sys.stderr)
        return 2
    run_dir, command = Path(argv[0]), argv[2:]
    with (
        (run_dir / "stdout.log").open("wb") as stdout,
        (run_dir / "stderr.log").open("wb") as stderr,
    ):
        try:
            code = subprocess.call(command, stdout=stdout, stderr=stderr, cwd=run_dir)
        except OSError as exc:
            stderr.write(f"failed to start stage: {exc}\n".encode())
            code = 127
    _write_exit_code(run_dir, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
