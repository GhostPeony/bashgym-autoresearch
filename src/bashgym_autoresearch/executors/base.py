"""Executor protocol: launch a stage, observe it across restarts, and stop it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from bashgym_autoresearch.contracts import FrozenModel

RunState = Literal["running", "exited", "lost"]


class Observation(FrozenModel):
    state: RunState
    exit_code: int | None = None


class Executor(Protocol):
    def launch(
        self,
        run_id: str,
        run_dir: Path,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
    ) -> None:
        """Start the stage. ``run_dir`` must not exist yet; it is created exclusively."""

    def observe(self, run_id: str, run_dir: Path) -> Observation:
        """Report the run's state using only what is recorded in ``run_dir``."""

    def kill(self, run_id: str, run_dir: Path) -> None:
        """Stop the run and everything it started."""
