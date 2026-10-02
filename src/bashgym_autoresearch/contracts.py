"""Frozen domain models and canonical hashing."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

Sha256 = str
Direction = Literal["maximize", "minimize"]
CampaignStatus = Literal[
    "awaiting_start", "running", "paused", "completed", "cancelled", "exhausted"
]
TERMINAL_STATUSES = frozenset({"completed", "cancelled", "exhausted"})
ExperimentRole = Literal["baseline", "candidate"]
ExperimentStatus = Literal["queued", "running", "evaluated", "crashed", "incomplete"]
Decision = Literal["baseline", "keep", "discard", "inconclusive", "crash", "incomplete"]
ApprovalKind = Literal["start", "budget", "promote", "publish"]
StageKind = Literal["train", "evaluate"]
Scope = Literal["smoke", "development"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


def canonical_json(value: Any) -> bytes:
    """Sorted-key, compact UTF-8 JSON used for every hash and signature."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(value: Any) -> Sha256:
    return hashlib.sha256(canonical_json(value)).hexdigest()


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricSpec(FrozenModel):
    name: str = Field(pattern=_NAME_PATTERN)
    direction: Direction


class ProtectedGate(FrozenModel):
    metric: str = Field(pattern=_NAME_PATTERN)
    direction: Direction
    max_regression: float = Field(ge=0, allow_inf_nan=False)


class StopRules(FrozenModel):
    max_experiments: int = Field(ge=1, le=10_000)
    max_cost: float = Field(ge=0, allow_inf_nan=False)
    deadline: datetime | None = None
    target: float | None = Field(default=None, allow_inf_nan=False)


class StageProfile(FrozenModel):
    """A registered stage program. ``argv`` may use {script}, {run_dir}, {inputs}, {outputs}."""

    name: str = Field(pattern=_NAME_PATTERN)
    kind: StageKind
    argv: tuple[str, ...] = Field(min_length=1, max_length=64)
    script: str = Field(min_length=1, max_length=4096)
    script_sha256: Sha256 = Field(pattern=_SHA256_PATTERN)
    timeout_seconds: float = Field(gt=0, le=7 * 24 * 3600, allow_inf_nan=False)


class EvaluationBinding(FrozenModel):
    suite_id: str = Field(pattern=_NAME_PATTERN)
    dataset_sha256: Sha256 = Field(pattern=_SHA256_PATTERN)
    profile: str = Field(pattern=_NAME_PATTERN)


class CampaignSpec(FrozenModel):
    name: str = Field(pattern=_NAME_PATTERN)
    objective: str = Field(min_length=1, max_length=4000)
    primary: MetricSpec
    minimum_improvement: float = Field(ge=0, allow_inf_nan=False)
    protected: tuple[ProtectedGate, ...] = ()
    stop: StopRules
    evaluation: EvaluationBinding
    train_profile: str | None = Field(default=None, pattern=_NAME_PATTERN)
    alpha: float = Field(default=0.05, gt=0, lt=0.5)
    n_resamples: int = Field(default=2000, ge=100, le=100_000)


class Change(FrozenModel):
    variable: str = Field(min_length=1, max_length=200)
    before: JsonValue
    after: JsonValue

    @model_validator(mode="after")
    def _differs(self) -> Change:
        if canonical_hash(self.before) == canonical_hash(self.after):
            raise ValueError("a change must alter the variable's value")
        return self


class TaskOutcome(FrozenModel):
    task_id: str = Field(min_length=1, max_length=240)
    cluster: str = Field(min_length=1, max_length=240)
    value: float = Field(allow_inf_nan=False)


class EvalEvidence(FrozenModel):
    """Written by an evaluation stage as outputs/evaluation.json."""

    context_sha256: Sha256 = Field(pattern=_SHA256_PATTERN)
    scope: Scope
    metrics: dict[str, float] = Field(min_length=1, max_length=256)
    tasks: tuple[TaskOutcome, ...] = Field(max_length=100_000)
    complete: bool
    provenance: dict[str, str] = Field(default_factory=dict, max_length=32)

    @field_validator("metrics")
    @classmethod
    def _finite(cls, metrics: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError("metrics must be finite")
        return metrics

    @model_validator(mode="after")
    def _unique_tasks(self) -> EvalEvidence:
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task_id in evidence")
        return self
