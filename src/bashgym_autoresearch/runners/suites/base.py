"""Task-suite interface used by the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

GradeStatus = Literal["passed", "failed", "infrastructure_error"]


@dataclass(frozen=True)
class Graded:
    status: GradeStatus
    prediction: str = ""
    method: str = ""
    detail: str = ""


class Suite(Protocol):
    name: str

    def load(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate dataset rows and return tasks (each with a unique ``task_id``)."""

    def prepare(self, tasks: list[dict[str, Any]], config: Any) -> None:
        """Verify task resources before any generation; raise ValueError on mismatch."""

    def prompt(self, task: dict[str, Any]) -> str | list[dict[str, str]]:
        """A raw continuation prompt or chat messages."""

    def grade(self, task: dict[str, Any], completion: str, config: Any) -> Graded:
        """Grade one completion. Infrastructure problems return ``infrastructure_error``."""

    def cluster(self, task: dict[str, Any]) -> str:
        """Correlation unit for the bootstrap (tasks in one cluster are resampled together)."""

    def tier(self, task: dict[str, Any]) -> str | None:
        """Optional difficulty tier for per-tier metrics."""
