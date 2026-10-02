"""Keep/discard/inconclusive decisions and next-action selection.

The decision compares per-task outcomes against the incumbent with a cluster
paired bootstrap. A candidate is kept only when the whole confidence interval
clears zero and the declared minimum improvement and every protected gate holds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from bashgym_autoresearch.contracts import (
    TERMINAL_STATUSES,
    CampaignSpec,
    CampaignStatus,
    Change,
    Decision,
    Direction,
    EvalEvidence,
    ExperimentRole,
    FrozenModel,
)
from bashgym_autoresearch.stats import paired_bootstrap


class EvaluationMismatch(ValueError):
    """The result cannot be compared with the incumbent."""


class ProposalError(ValueError):
    """The proposal violates the controlled-change rules."""


class Comparison(FrozenModel):
    decision: Decision
    improvement: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    breached_gate: str | None = None
    reason: str


class NextAction(FrozenModel):
    kind: Literal["await_start", "wait", "propose_baseline", "propose_candidate", "stop"]
    reason: str


def _oriented(direction: Direction, before: float, after: float) -> float:
    return after - before if direction == "maximize" else before - after


def _breached_gate(spec: CampaignSpec, incumbent: EvalEvidence, result: EvalEvidence) -> str | None:
    for gate in spec.protected:
        previous = incumbent.metrics.get(gate.metric)
        current = result.metrics.get(gate.metric)
        if previous is None or current is None:
            return gate.metric
        if -_oriented(gate.direction, previous, current) > gate.max_regression:
            return gate.metric
    return None


def decide(
    spec: CampaignSpec,
    *,
    role: ExperimentRole,
    incumbent: EvalEvidence | None,
    result: EvalEvidence | None,
    crashed: bool = False,
    seed: int = 0,
) -> Comparison:
    if crashed:
        return Comparison(decision="crash", reason="a stage exited unsuccessfully")
    if result is None or not result.complete:
        return Comparison(decision="incomplete", reason="evaluation did not complete")
    if spec.primary.name not in result.metrics:
        raise EvaluationMismatch(f"primary metric {spec.primary.name!r} missing from evidence")
    if role == "baseline":
        return Comparison(decision="baseline", reason="baseline established")
    if incumbent is None:
        raise EvaluationMismatch("a candidate needs an incumbent to compare against")

    before = {task.task_id: task for task in incumbent.tasks}
    after = {task.task_id: task for task in result.tasks}
    if not after or set(before) != set(after):
        raise EvaluationMismatch("candidate and incumbent were evaluated on different task sets")
    order = sorted(after)
    deltas = [_oriented(spec.primary.direction, before[t].value, after[t].value) for t in order]
    clusters = [after[t].cluster for t in order]
    stats = paired_bootstrap(
        deltas, clusters, n_resamples=spec.n_resamples, alpha=spec.alpha, seed=seed
    )
    interval = dict(improvement=stats.mean, ci_low=stats.ci_low, ci_high=stats.ci_high)

    gate = _breached_gate(spec, incumbent, result)
    if gate is not None:
        return Comparison(
            decision="discard",
            breached_gate=gate,
            reason=f"protected metric {gate!r} regressed",
            **interval,
        )
    if stats.ci_low > 0 and stats.ci_low >= spec.minimum_improvement:
        return Comparison(
            decision="keep", reason="improvement interval clears the minimum", **interval
        )
    if stats.ci_high < spec.minimum_improvement:
        return Comparison(
            decision="discard",
            reason="interval lies below the minimum improvement",
            **interval,
        )
    return Comparison(
        decision="inconclusive",
        reason="interval overlaps the minimum improvement; add tasks or repeats",
        **interval,
    )


def next_action(
    spec: CampaignSpec,
    status: CampaignStatus,
    *,
    has_baseline: bool,
    running: bool,
    experiments_counted: int,
    cost_used: float,
    best_value: float | None,
    now: datetime,
) -> NextAction:
    if status == "awaiting_start":
        return NextAction(kind="await_start", reason="a human must approve the campaign start")
    if status in TERMINAL_STATUSES:
        return NextAction(kind="stop", reason=f"campaign is {status}")
    if status == "paused":
        return NextAction(kind="wait", reason="campaign is paused")
    if running:
        return NextAction(kind="wait", reason="an experiment is in progress")
    stop = spec.stop
    if stop.deadline is not None and now >= stop.deadline:
        return NextAction(kind="stop", reason="deadline reached")
    if experiments_counted >= stop.max_experiments:
        return NextAction(kind="stop", reason="maximum experiments reached")
    if cost_used >= stop.max_cost:
        return NextAction(kind="stop", reason="budget exhausted")
    if stop.target is not None and best_value is not None:
        reached = (
            best_value >= stop.target
            if spec.primary.direction == "maximize"
            else best_value <= stop.target
        )
        if reached:
            return NextAction(kind="stop", reason="target metric reached")
    if not has_baseline:
        return NextAction(kind="propose_baseline", reason="no baseline yet")
    return NextAction(kind="propose_candidate", reason="propose one controlled change")


def validate_change(role: ExperimentRole, change: Change | None) -> None:
    if role == "baseline" and change is not None:
        raise ProposalError("a baseline must not declare a change")
    if role == "candidate" and change is None:
        raise ProposalError("a candidate must declare exactly one change")
