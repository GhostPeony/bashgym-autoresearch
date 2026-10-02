"""Local subprocess executor whose runs survive worker restarts.

Each run gets its own directory. ``launch.json`` records the wrapper's pid and
process start time, so a restarted worker can tell its own live run from an
unrelated process that reused the pid. The wrapper writes ``exit_code`` last.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import psutil

from bashgym_autoresearch.executors.base import Observation

_CREATE_TIME_TOLERANCE = 0.01


def _detached_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class LocalExecutor:
    def launch(
        self,
        run_id: str,
        run_dir: Path,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
    ) -> None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
        process = subprocess.Popen(
            [sys.executable, "-m", "bashgym_autoresearch.executors._wrapper", str(run_dir), "--"]
            + list(argv),
            cwd=run_dir,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detached_kwargs(),
        )
        create_time = psutil.Process(process.pid).create_time()
        temporary = run_dir / "launch.tmp"
        temporary.write_text(
            json.dumps({"run_id": run_id, "pid": process.pid, "create_time": create_time}),
            encoding="utf-8",
        )
        os.replace(temporary, run_dir / "launch.json")

    def _process(self, run_dir: Path) -> psutil.Process | None:
        try:
            launch = json.loads((Path(run_dir) / "launch.json").read_text(encoding="utf-8"))
            process = psutil.Process(launch["pid"])
            if abs(process.create_time() - launch["create_time"]) > _CREATE_TIME_TOLERANCE:
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except (OSError, ValueError, KeyError, psutil.Error):
            return None

    def observe(self, run_id: str, run_dir: Path) -> Observation:
        exit_path = Path(run_dir) / "exit_code"
        if exit_path.is_file():
            return Observation(state="exited", exit_code=int(exit_path.read_text().strip()))
        if self._process(run_dir) is not None:
            return Observation(state="running")
        if exit_path.is_file():
            return Observation(state="exited", exit_code=int(exit_path.read_text().strip()))
        return Observation(state="lost")

    def kill(self, run_id: str, run_dir: Path) -> None:
        process = self._process(run_dir)
        if process is None:
            return
        try:
            targets = [*process.children(recursive=True), process]
        except psutil.Error:
            targets = [process]
        for target in targets:
            try:
                target.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(targets, timeout=10)
