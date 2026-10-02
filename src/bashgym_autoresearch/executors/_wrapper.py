"""Run one stage command and record its exit code atomically.

Usage: python -m bashgym_autoresearch.executors._wrapper <control_dir> <run_dir> -- <argv...>

The exit code goes to ``control_dir``, which is outside the stage's working
directory; stdout and stderr go to ``run_dir``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _write_exit_code(control_dir: Path, code: int) -> None:
    temporary = control_dir / "exit_code.tmp"
    temporary.write_text(str(code), encoding="ascii")
    os.replace(temporary, control_dir / "exit_code")


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print("usage: _wrapper <control_dir> <run_dir> -- <argv...>", file=sys.stderr)
        return 2
    control_dir, run_dir, command = Path(argv[0]), Path(argv[1]), argv[3:]
    (run_dir / "outputs").mkdir(exist_ok=True)
    with (
        (run_dir / "stdout.log").open("wb") as stdout,
        (run_dir / "stderr.log").open("wb") as stderr,
    ):
        try:
            code = subprocess.call(command, stdout=stdout, stderr=stderr, cwd=run_dir)
        except OSError as exc:
            stderr.write(f"failed to start stage: {exc}\n".encode())
            code = 127
    _write_exit_code(control_dir, code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
