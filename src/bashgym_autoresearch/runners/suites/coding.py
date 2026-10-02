"""HumanEval-style function completion graded by executing the task's tests in the sandbox."""

from __future__ import annotations

import re
from typing import Any

from bashgym_autoresearch.runners.suites.base import Graded
from bashgym_autoresearch.sandbox import run_program

_ENTRY_POINT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_BODY_BOUNDARY = re.compile(
    r"^(?:def[ \t]|class[ \t]|if[ \t]|import[ \t]|from[ \t]|print(?:[ \t]|\()|#|```)",
    re.MULTILINE,
)


def process_completion(completion: str, protocol: str) -> str:
    """Apply the declared stop convention without repairing generated code.

    ``humaneval_body_v1`` ends a function-body continuation at the first
    column-zero declaration, import, print, comment, conditional, or Markdown
    fence; indented nested code is kept.
    """
    if protocol == "raw":
        return completion
    if protocol != "humaneval_body_v1":
        raise ValueError(f"unknown completion protocol {protocol!r}")
    boundary = _BODY_BOUNDARY.search(completion)
    return completion[: boundary.start()] if boundary else completion


def program(task: dict[str, Any], completion: str) -> str:
    return f"{task['prompt']}{completion}\n\n{task['test']}\n\ncheck({task['entry_point']})\n"


class CodingSuite:
    name = "coding"

    def load(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for row in rows:
            for key in ("task_id", "prompt", "test", "entry_point"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"coding task is missing {key!r}")
            if not _ENTRY_POINT.fullmatch(row["entry_point"]):
                raise ValueError("coding entry_point must be an identifier")
            if "canonical_solution" in row:
                raise ValueError("evaluation rows must not contain reference solutions")
        return rows

    def prepare(self, tasks: list[dict[str, Any]], config: Any) -> None:
        return None

    def prompt(self, task: dict[str, Any]) -> str:
        return task["prompt"]

    def grade(self, task: dict[str, Any], completion: str, config: Any) -> Graded:
        body = process_completion(completion, config.coding.completion_protocol)
        result = run_program(
            config.sandbox.image,
            {"solution.py": program(task, body)},
            ["python", "-I", "/workspace/solution.py"],
            timeout=config.sandbox.program_timeout_seconds,
            memory=config.sandbox.memory,
            cpus=config.sandbox.cpus,
        )
        if result.error is not None:
            return Graded("infrastructure_error", method="sandbox", detail=result.error)
        if result.timed_out:
            return Graded("failed", method="test_timeout")
        if result.exit_code == 0:
            return Graded("passed", method="tests")
        return Graded("failed", method="tests", detail=result.stderr[-500:])

    def cluster(self, task: dict[str, Any]) -> str:
        return task["task_id"]

    def tier(self, task: dict[str, Any]) -> str | None:
        return None
