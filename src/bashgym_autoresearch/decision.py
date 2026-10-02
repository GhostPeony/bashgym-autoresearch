"""Keep/discard/inconclusive decisions and next-action selection.

The decision compares per-task outcomes against the incumbent with a cluster
paired bootstrap. A candidate is kept only when the whole confidence interval
clears zero and the declared minimum improvement and every protected gate holds.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

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
    canonical_hash,
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


def _breached_gate(
    spec: CampaignSpec, references: list[EvalEvidence], result: EvalEvidence
) -> str | None:
    """A gate is breached if the result regresses beyond the limit from any reference.

    Checking against the baseline as well as the incumbent stops a protected
    metric from sliding by ``max_regression`` with every kept candidate.
    """
    for gate in spec.protected:
        current = result.metrics.get(gate.metric)
        for reference in references:
            previous = reference.metrics.get(gate.metric)
            if previous is None or current is None:
                return gate.metric
            if -_oriented(gate.direction, previous, current) > gate.max_regression:
                return gate.metric
    return None


_PINNED_PROVENANCE = ("suite", "harness_sha256")


def decide(
    spec: CampaignSpec,
    *,
    role: ExperimentRole,
    incumbent: EvalEvidence | None,
    result: EvalEvidence | None,
    crashed: bool = False,
    seed: int = 0,
    baseline: EvalEvidence | None = None,
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

    for key in _PINNED_PROVENANCE:
        if incumbent.provenance.get(key) != result.provenance.get(key):
            raise EvaluationMismatch(f"evaluator {key} differs from the incumbent's")
    before = {task.task_id: task for task in incumbent.tasks}
    after = {task.task_id: task for task in result.tasks}
    if not after or set(before) != set(after):
        raise EvaluationMismatch("candidate and incumbent were evaluated on different task sets")
    if any(before[t].cluster != after[t].cluster for t in after):
        raise EvaluationMismatch("task cluster labels differ from the incumbent's")
    order = sorted(after)
    deltas = [_oriented(spec.primary.direction, before[t].value, after[t].value) for t in order]
    clusters = [after[t].cluster for t in order]
    stats = paired_bootstrap(
        deltas, clusters, n_resamples=spec.n_resamples, alpha=spec.alpha, seed=seed
    )
    interval = dict(improvement=stats.mean, ci_low=stats.ci_low, ci_high=stats.ci_high)

    references = [incumbent] + (
        [baseline] if baseline is not None and baseline != incumbent else []
    )
    gate = _breached_gate(spec, references, result)
    if gate is not None:
        return Comparison(
            decision="discard",
            breached_gate=gate,
            reason=f"protected metric {gate!r} regressed",
            **interval,
        )
    if stats.n_clusters < spec.min_clusters:
        return Comparison(
            decision="inconclusive",
            reason=(
                f"only {stats.n_clusters} independent clusters; at least {spec.min_clusters}"
                " are needed for a decision"
            ),
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
    incomplete_count: int = 0,
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
    if incomplete_count >= stop.max_incomplete:
        return NextAction(kind="stop", reason="too many incomplete experiments")
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


_MISSING = object()


def _get_path(recipe: dict, path: str) -> Any:
    node: Any = recipe
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        if not value:
            return {prefix: value} if prefix else {}
        leaves: dict[str, Any] = {}
        for key, item in value.items():
            leaves.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    return {prefix: value}


def check_single_change(parent: dict, recipe: dict, change: Change) -> None:
    """Require ``recipe`` to differ from ``parent`` only at ``change.variable``.

    ``variable`` is a dotted path into the recipe. Every differing leaf must lie
    at or under that path, and the old and new values at the path must equal
    the declared ``before`` and ``after`` (a missing key reads as null).
    """
    before_leaves, after_leaves = _leaves(parent), _leaves(recipe)
    differing = {
        key
        for key in before_leaves.keys() | after_leaves.keys()
        if canonical_hash(before_leaves.get(key)) != canonical_hash(after_leaves.get(key))
        or (key in before_leaves) != (key in after_leaves)
    }
    variable = change.variable
    outside = sorted(k for k in differing if k != variable and not k.startswith(variable + "."))
    if outside or not differing:
        raise ProposalError(
            f"a candidate may change only {variable!r}; the recipe also differs at {outside}"
            if outside
            else f"the recipe does not change {variable!r}"
        )
    old, new = _get_path(parent, variable), _get_path(recipe, variable)
    old = None if old is _MISSING else old
    new = None if new is _MISSING else new
    if canonical_hash(old) != canonical_hash(change.before):
        raise ProposalError(f"{variable!r} is {old!r} in the incumbent, not {change.before!r}")
    if canonical_hash(new) != canonical_hash(change.after):
        raise ProposalError(f"{variable!r} is {new!r} in the recipe, not {change.after!r}")
