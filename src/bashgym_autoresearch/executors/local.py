"""Local subprocess executor whose runs survive worker restarts.

Each run has a working directory (given to the stage) and a separate control
directory under ``control_root`` (not given to the stage). ``launch.json``
records the wrapper's pid, process-group id, and start time, so a restarted
worker can tell its own live run from an unrelated process that reused the pid.
The wrapper writes ``exit_code`` last. On POSIX the stage runs in its own
process group, and ``kill`` signals the whole group.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import psutil

from bashgym_autoresearch.executors.base import Observation

_CREATE_TIME_TOLERANCE = 0.01
_MALFORMED_EXIT = 255


def _detached_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class LocalExecutor:
    def __init__(self, control_root: Path):
        self.control_root = Path(control_root)

    def _control(self, run_id: str) -> Path:
        return self.control_root / run_id

    def launch(
        self,
        run_id: str,
        run_dir: Path,
        argv: Sequence[str],
        env: Mapping[str, str] | None,
    ) -> None:
        run_dir, control = Path(run_dir), self._control(run_id)
        control.mkdir(parents=True, exist_ok=False)
        run_dir.mkdir(parents=True, exist_ok=False)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bashgym_autoresearch.executors._wrapper",
                str(control),
                str(run_dir),
                "--",
                *argv,
            ],
            cwd=run_dir,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_detached_kwargs(),
        )
        record = {
            "run_id": run_id,
            "pid": process.pid,
            "pgid": os.getpgid(process.pid) if os.name != "nt" else None,
            "create_time": psutil.Process(process.pid).create_time(),
        }
        temporary = control / "launch.tmp"
        temporary.write_text(json.dumps(record), encoding="utf-8")
        os.replace(temporary, control / "launch.json")

    def _launch_record(self, run_id: str) -> dict | None:
        try:
            return json.loads((self._control(run_id) / "launch.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _process(self, run_id: str) -> psutil.Process | None:
        record = self._launch_record(run_id)
        if record is None:
            return None
        try:
            process = psutil.Process(record["pid"])
            if abs(process.create_time() - record["create_time"]) > _CREATE_TIME_TOLERANCE:
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except (KeyError, psutil.Error):
            return None

    def _exit_code(self, run_id: str) -> int | None:
        path = self._control(run_id) / "exit_code"
        if not path.is_file():
            return None
        try:
            return int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError, UnicodeDecodeError):
            return _MALFORMED_EXIT

    def launched(self, run_id: str) -> bool:
        return self._launch_record(run_id) is not None

    def observe(self, run_id: str, run_dir: Path) -> Observation:
        code = self._exit_code(run_id)
        if code is not None:
            return Observation(state="exited", exit_code=code)
        if self._process(run_id) is not None:
            return Observation(state="running")
        code = self._exit_code(run_id)
        if code is not None:
            return Observation(state="exited", exit_code=code)
        return Observation(state="lost")

    def kill(self, run_id: str, run_dir: Path) -> None:
        """Stop the run and anything it started, including after a normal exit."""
        record = self._launch_record(run_id) or {}
        if os.name != "nt" and record.get("pgid"):
            try:
                os.killpg(record["pgid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process = self._process(run_id)
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
