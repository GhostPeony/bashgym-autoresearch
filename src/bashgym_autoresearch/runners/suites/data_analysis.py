"""SmolDataEnvs data-analysis tasks.

The model writes one Python program. It runs in the sandbox with the task's
data folder mounted read-only at /home/user/input, and the last printed line is
graded on the host with the vendored grader, so gold answers never enter the
sandbox. Code extraction, the command-shaped-answer rule, and last-line
selection follow FineEnvs ``04-smoldataenvs/scripts/rollout.py`` (Apache-2.0).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from bashgym_autoresearch.grading.short_answer import grade as grade_answer
from bashgym_autoresearch.runners.suites.base import Graded
from bashgym_autoresearch.sandbox import run_program

INPUT_MOUNT = "/home/user/input"
_CODE = re.compile(r"```(?:python3?|py)?\s*\n(.*?)```", re.S | re.I)
_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|\|{1,2}\s*\S+|\S\s*>{1,2}\s*\S+|\$\(|`",
)
_PREFIX = re.compile(r"[A-Za-z0-9._-]{1,512}")


def extract_code(completion: str) -> str:
    """The last fenced block wins; unfenced text is treated as the program."""
    blocks = _CODE.findall(completion or "")
    return blocks[-1].strip() if blocks else (completion or "").strip()


def looks_like_a_command(answer: str) -> bool:
    return bool(answer) and bool(_COMMAND_SHAPED.search(answer.strip()))


def last_line(stdout: str) -> str:
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataAnalysisSuite:
    name = "data_analysis"

    def load(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            for key in ("task_id", "answer", "difficulty_tier"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"data-analysis task is missing {key!r}")
            if not isinstance(row.get("messages"), list) or not row["messages"]:
                raise ValueError("data-analysis task is missing chat messages")
            data = row.get("data") or {}
            if not _PREFIX.fullmatch(str(data.get("prefix", ""))) or not data.get("files"):
                raise ValueError("data-analysis task needs a data prefix and file manifest")
        return rows

    def prepare(self, tasks: list[dict[str, Any]], config: Any) -> None:
        """Each data folder must hold exactly the pinned files with matching digests."""
        root = Path(config.data_analysis.data_root)
        checked: dict[str, Any] = {}
        for task in tasks:
            prefix, files = task["data"]["prefix"], task["data"]["files"]
            if prefix in checked:
                if checked[prefix] != files:
                    raise ValueError(f"conflicting file manifests for {prefix}")
                continue
            directory = root / prefix
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"data folder {prefix} is missing")
            present = {
                item.relative_to(directory).as_posix(): item
                for item in directory.rglob("*")
                if not item.is_dir() or item.is_symlink()
            }
            if set(present) != {item["path"] for item in files}:
                raise ValueError(f"data folder {prefix} does not match its manifest")
            for item in files:
                path = present[item["path"]]
                if path.is_symlink() or path.stat().st_size != item["size"]:
                    raise ValueError(f"data file {prefix}/{item['path']} changed")
                if _sha256(path) != item["sha256"]:
                    raise ValueError(f"data file {prefix}/{item['path']} changed")
            checked[prefix] = files

    def prompt(self, task: dict[str, Any]) -> list[dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in task["messages"]]

    def grade(self, task: dict[str, Any], completion: str, config: Any) -> Graded:
        code = extract_code(completion)
        try:
            compile(code, "<solve>", "exec")
        except (SyntaxError, ValueError):
            return Graded("failed", method="syntax_error")
        result = run_program(
            config.sandbox.image,
            {"solve.py": code},
            ["python", "/workspace/solve.py"],
            timeout=config.sandbox.program_timeout_seconds,
            workdir=INPUT_MOUNT,
            read_only_mounts={
                INPUT_MOUNT: Path(config.data_analysis.data_root) / task["data"]["prefix"]
            },
            memory=config.sandbox.memory,
            cpus=config.sandbox.cpus,
        )
        if result.error is not None:
            return Graded("infrastructure_error", method="sandbox", detail=result.error)
        prediction = last_line(result.stdout)
        if not prediction:
            return Graded("failed", method="no_output", detail=result.stderr[-500:])
        if looks_like_a_command(prediction):
            return Graded("failed", prediction=prediction, method="command_answer")
        graded = grade_answer(
            task["answer"],
            prediction,
            reward_mode=task.get("reward_mode") or "",
            abs_tol=float(task.get("atol", 1e-3)),
            rel_tol=float(task.get("rtol", 1e-3)),
            math_verify=config.data_analysis.math_verify,
        )
        status = "passed" if graded.reward == 1.0 else "failed"
        return Graded(status, prediction=prediction[:4096], method=graded.method)

    def cluster(self, task: dict[str, Any]) -> str:
        return task["data"]["prefix"]

    def tier(self, task: dict[str, Any]) -> str | None:
        return task["difficulty_tier"]
