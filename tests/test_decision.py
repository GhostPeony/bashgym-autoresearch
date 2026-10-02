from datetime import UTC, datetime, timedelta

import pytest

from bashgym_autoresearch.contracts import (
    CampaignSpec,
    Change,
    EvalEvidence,
    EvaluationBinding,
    MetricSpec,
    ProtectedGate,
    StopRules,
    TaskOutcome,
)
from bashgym_autoresearch.decision import (
    EvaluationMismatch,
    ProposalError,
    decide,
    next_action,
    validate_change,
)

NOW = datetime(2026, 10, 1, tzinfo=UTC)


def spec(**overrides):
    values = dict(
        name="demo",
        objective="improve pass rate",
        primary=MetricSpec(name="pass_rate", direction="maximize"),
        minimum_improvement=0.0,
        stop=StopRules(max_experiments=10, max_cost=100),
        evaluation=EvaluationBinding(suite_id="suite", dataset_sha256="a" * 64, profile="eval"),
        n_resamples=500,
    )
    values.update(overrides)
    return CampaignSpec(**values)


def evidence(values, *, metrics=None, complete=True):
    tasks = tuple(
        TaskOutcome(task_id=f"t{i}", cluster=f"t{i}", value=v) for i, v in enumerate(values)
    )
    mean = sum(values) / len(values)
    return EvalEvidence(
        context_sha256="b" * 64,
        scope="smoke",
        metrics=metrics or {"pass_rate": mean},
        tasks=tasks,
        complete=complete,
    )


BASE = evidence([0, 1] * 20)


def test_baseline_crash_and_incomplete_branches():
    assert decide(spec(), role="baseline", incumbent=None, result=BASE).decision == "baseline"
    assert decide(spec(), role="candidate", incumbent=BASE, result=None, crashed=True).decision == (
        "crash"
    )
    partial = evidence([1] * 40, complete=False)
    assert decide(spec(), role="candidate", incumbent=BASE, result=partial).decision == (
        "incomplete"
    )


def test_clear_gain_keeps_with_interval():
    better = evidence([1] * 40)
    comparison = decide(spec(), role="candidate", incumbent=BASE, result=better)
    assert comparison.decision == "keep"
    assert comparison.ci_low > 0 and comparison.improvement == pytest.approx(0.5)


def test_noise_level_change_is_inconclusive():
    flipped = evidence([1, 0] * 20)
    assert decide(spec(), role="candidate", incumbent=BASE, result=flipped).decision == (
        "inconclusive"
    )


def test_confident_shortfall_discards():
    worse = evidence([0] * 40)
    assert decide(spec(), role="candidate", incumbent=BASE, result=worse).decision == "discard"


def test_gain_below_minimum_improvement_is_not_kept():
    slightly = evidence([1] * 22 + [0] * 18)
    comparison = decide(
        spec(minimum_improvement=0.4), role="candidate", incumbent=BASE, result=slightly
    )
    assert comparison.decision in {"discard", "inconclusive"}


def test_minimize_direction_inverts_deltas():
    minimize = spec(primary=MetricSpec(name="pass_rate", direction="minimize"))
    lower = evidence([0] * 40)
    assert decide(minimize, role="candidate", incumbent=BASE, result=lower).decision == "keep"


def test_protected_gate_breach_overrides_gain():
    gated = spec(
        protected=(ProtectedGate(metric="safety", direction="maximize", max_regression=0.05),)
    )
    incumbent = evidence([0, 1] * 20, metrics={"pass_rate": 0.5, "safety": 0.9})
    candidate = evidence([1] * 40, metrics={"pass_rate": 1.0, "safety": 0.7})
    comparison = decide(gated, role="candidate", incumbent=incumbent, result=candidate)
    assert comparison.decision == "discard" and comparison.breached_gate == "safety"


def test_missing_protected_metric_counts_as_breach():
    gated = spec(
        protected=(ProtectedGate(metric="safety", direction="maximize", max_regression=0.5),)
    )
    incumbent = evidence([0, 1] * 20, metrics={"pass_rate": 0.5, "safety": 0.9})
    candidate = evidence([1] * 40)
    assert decide(gated, role="candidate", incumbent=incumbent, result=candidate).breached_gate == (
        "safety"
    )


def test_different_task_sets_cannot_be_compared():
    other = EvalEvidence(
        context_sha256="b" * 64,
        scope="smoke",
        metrics={"pass_rate": 1.0},
        tasks=(TaskOutcome(task_id="other", cluster="other", value=1.0),),
        complete=True,
    )
    with pytest.raises(EvaluationMismatch):
        decide(spec(), role="candidate", incumbent=BASE, result=other)


def test_candidate_without_incumbent_is_rejected():
    with pytest.raises(EvaluationMismatch):
        decide(spec(), role="candidate", incumbent=None, result=BASE)


def _next(**kwargs):
    values = dict(
        status="running",
        has_baseline=True,
        running=False,
        experiments_counted=0,
        cost_used=0.0,
        best_value=None,
        now=NOW,
    )
    values.update(kwargs)
    return next_action(spec(**values.pop("spec_overrides", {})), **values).kind


def test_next_action_follows_status_then_stop_rules_then_proposals():
    assert _next(status="awaiting_start") == "await_start"
    assert _next(status="paused") == "wait"
    assert _next(status="cancelled") == "stop"
    assert _next(running=True) == "wait"
    deadline = StopRules(max_experiments=10, max_cost=100, deadline=NOW - timedelta(seconds=1))
    assert _next(spec_overrides={"stop": deadline}) == "stop"
    assert _next(experiments_counted=10) == "stop"
    assert _next(cost_used=100.0) == "stop"
    target = StopRules(max_experiments=10, max_cost=100, target=0.9)
    assert _next(spec_overrides={"stop": target}, best_value=0.95) == "stop"
    assert _next(has_baseline=False) == "propose_baseline"
    assert _next() == "propose_candidate"


def test_change_rules_by_role():
    change = Change(variable="lr", before=1e-4, after=2e-4)
    validate_change("baseline", None)
    validate_change("candidate", change)
    with pytest.raises(ProposalError):
        validate_change("baseline", change)
    with pytest.raises(ProposalError):
        validate_change("candidate", None)
