import math

import pytest
from pydantic import ValidationError

from bashgym_autoresearch.contracts import (
    Change,
    EvalEvidence,
    MetricSpec,
    StageProfile,
    StopRules,
    TaskOutcome,
    canonical_hash,
)


def test_canonical_hash_ignores_key_order():
    assert canonical_hash({"a": 1, "b": [1, 2]}) == canonical_hash({"b": [1, 2], "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_models_are_frozen_and_reject_unknown_fields():
    spec = MetricSpec(name="pass_rate", direction="maximize")
    with pytest.raises(ValidationError):
        spec.name = "other"
    with pytest.raises(ValidationError):
        MetricSpec(name="pass_rate", direction="maximize", extra=True)


def test_stop_rules_require_at_least_one_experiment():
    with pytest.raises(ValidationError):
        StopRules(max_experiments=0, max_cost=1)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_evidence_rejects_non_finite_metrics(value):
    with pytest.raises(ValidationError):
        EvalEvidence(
            context_sha256="a" * 64,
            scope="smoke",
            metrics={"pass_rate": value},
            tasks=(),
            complete=True,
        )


def test_evidence_rejects_duplicate_task_ids():
    task = TaskOutcome(task_id="t1", cluster="t1", value=1.0)
    with pytest.raises(ValidationError, match="duplicate"):
        EvalEvidence(
            context_sha256="a" * 64,
            scope="smoke",
            metrics={"pass_rate": 1.0},
            tasks=(task, task),
            complete=True,
        )


def test_change_requires_a_real_difference_and_profile_argv_is_bounded():
    with pytest.raises(ValidationError):
        Change(variable="lr", before=1, after=1)
    with pytest.raises(ValidationError):
        StageProfile(
            name="eval",
            kind="evaluate",
            argv=(),
            script="evaluate.py",
            script_sha256="a" * 64,
            timeout_seconds=10,
        )
