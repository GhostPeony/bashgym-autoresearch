import json

import pytest
from conftest import AGENT, HUMAN

from bashgym_autoresearch.auth import Forbidden
from bashgym_autoresearch.contracts import Change, StageProfile
from bashgym_autoresearch.decision import ProposalError
from bashgym_autoresearch.service import RuleError

CHANGE = Change(variable="boost", before=0.0, after=0.3)


def propose(world, campaign_id, role="baseline", key="p1", **overrides):
    values = dict(
        role=role,
        change=CHANGE if role == "candidate" else None,
        recipe={"boost": 0.0},
        hypothesis="establish the starting point",
        estimated_cost=1.0,
        idempotency_key=key,
    )
    values.update(overrides)
    return world.service.propose(AGENT, campaign_id, **values)


def campaign_rows(world):
    return world.service.store.read("SELECT id, status, guidance_version FROM campaigns")


def test_agents_cannot_perform_human_only_actions(world):
    with pytest.raises(Forbidden):
        world.service.create_campaign(AGENT, world.spec(), "x")
    campaign_id = world.service.create_campaign(HUMAN, world.spec(), "c")["campaign_id"]
    approval = world.service.request_approval(AGENT, campaign_id, "start", {}, "s")
    with pytest.raises(Forbidden):
        world.service.decide_approval(AGENT, approval["approval_id"], True)
    with pytest.raises(Forbidden):
        world.service.set_guidance(AGENT, campaign_id, "be bold")
    with pytest.raises(Forbidden):
        world.service.register_profile(
            AGENT,
            StageProfile(
                name="x",
                kind="train",
                argv=("x",),
                script=str(world.train_script),
                script_sha256="0" * 64,
                timeout_seconds=1,
            ),
        )
    assert [tuple(row) for row in campaign_rows(world)] == [(campaign_id, "awaiting_start", 0)]
    assert world.service.brief(AGENT, campaign_id)["pending_approvals"][0]["kind"] == "start"


def test_profile_registration_checks_the_script_digest(world):
    with pytest.raises(RuleError, match="sha256"):
        world.service.register_profile(
            HUMAN,
            StageProfile(
                name="bad",
                kind="train",
                argv=("x",),
                script=str(world.train_script),
                script_sha256="0" * 64,
                timeout_seconds=1,
            ),
        )


def test_proposals_wait_for_the_start_approval(world):
    campaign_id = world.service.create_campaign(HUMAN, world.spec(), "c")["campaign_id"]
    with pytest.raises(RuleError, match="start"):
        propose(world, campaign_id)
    brief = world.service.brief(AGENT, campaign_id)
    assert brief["next_action"]["kind"] == "await_start"


def test_repeated_idempotency_key_creates_one_experiment(world):
    campaign_id = world.started_campaign()
    first = propose(world, campaign_id, key="same")
    second = propose(world, campaign_id, key="same")
    assert first == second
    assert len(world.service.state(campaign_id).experiments) == 1


def test_only_one_active_experiment_at_a_time(world):
    campaign_id = world.started_campaign()
    propose(world, campaign_id, key="a")
    with pytest.raises(RuleError, match="in progress"):
        propose(world, campaign_id, key="b")


def test_role_must_match_next_action_and_change_rules(world):
    campaign_id = world.started_campaign()
    with pytest.raises(RuleError, match="candidate"):
        propose(world, campaign_id, role="candidate")
    with pytest.raises(ProposalError):
        propose(world, campaign_id, change=CHANGE)


def test_budget_is_enforced_and_a_granted_budget_raises_it(world):
    campaign_id = world.started_campaign()
    with pytest.raises(RuleError, match="budget"):
        propose(world, campaign_id, estimated_cost=11.0)
    approval = world.service.request_approval(AGENT, campaign_id, "budget", {"amount": 5}, "b1")
    world.service.decide_approval(HUMAN, approval["approval_id"], True)
    assert world.service.brief(AGENT, campaign_id)["budget"]["max"] == 15
    assert propose(world, campaign_id, estimated_cost=11.0)["status"] == "queued"


def test_guidance_is_versioned_and_visible_in_the_brief(world):
    campaign_id = world.started_campaign()
    world.service.set_guidance(HUMAN, campaign_id, "focus on hard tasks")
    guidance = world.service.brief(AGENT, campaign_id)["guidance"]
    assert guidance == {"text": "focus on hard tasks", "version": 1}


def test_a_human_pause_cannot_be_undone_by_an_agent(world):
    campaign_id = world.started_campaign()
    world.service.pause(HUMAN, campaign_id)
    with pytest.raises(RuleError, match="human"):
        world.service.resume(AGENT, campaign_id)
    world.service.resume(HUMAN, campaign_id)
    world.service.pause(AGENT, campaign_id)
    assert world.service.resume(AGENT, campaign_id)["status"] == "running"


def test_promotion_requires_a_kept_development_result(world):
    campaign_id = world.started_campaign()
    experiment = propose(world, campaign_id)
    with pytest.raises(RuleError):
        world.service.request_approval(
            AGENT, campaign_id, "promote", {"experiment_id": experiment["experiment_id"]}, "pr"
        )


def test_cancel_is_terminal_and_reported(world):
    campaign_id = world.started_campaign()
    world.service.cancel(AGENT, campaign_id)
    with pytest.raises(RuleError):
        world.service.resume(HUMAN, campaign_id)
    report = world.service.report(HUMAN, campaign_id)
    assert report["status"] == "cancelled"
    assert json.dumps(report)


@pytest.mark.parametrize(
    "payload", [{"amount": float("nan")}, {"amount": True}, {"amount": -1}, {"x": "y" * 5000}]
)
def test_invalid_approval_payloads_are_rejected(world, payload):
    campaign_id = world.started_campaign()
    with pytest.raises(RuleError):
        world.service.request_approval(AGENT, campaign_id, "budget", payload, "k")


def test_reusing_an_idempotency_key_for_another_request_conflicts(world):
    from bashgym_autoresearch.store import IdempotencyMismatch

    campaign_id = world.started_campaign()
    world.service.request_approval(AGENT, campaign_id, "budget", {"amount": 1}, "same")
    with pytest.raises(IdempotencyMismatch):
        world.service.request_approval(AGENT, campaign_id, "budget", {"amount": 2}, "same")


def test_only_humans_mint_tokens(world):
    with pytest.raises(Forbidden):
        world.service.create_token(AGENT, "human", "escalate")
    token = world.service.create_token(HUMAN, "agent", "codex")
    assert token["token"].startswith("bgar_") and token["role"] == "agent"


def test_budget_grant_revives_a_campaign_exhausted_by_budget(world):
    from bashgym_autoresearch.contracts import StopRules

    campaign_id = world.started_campaign(stop=StopRules(max_experiments=5, max_cost=1))
    propose(world, campaign_id)
    with world.service.store.transaction() as db:
        db.execute("UPDATE experiments SET status = 'evaluated'")
        db.execute(
            "INSERT INTO results(experiment_id, decision, comparison_json, created_at)"
            ' SELECT id, \'incomplete\', \'{"decision": "incomplete", "reason": "x"}\', \'now\''
            " FROM experiments"
        )
        db.execute("UPDATE campaigns SET status = 'exhausted', version = version + 1")
    approval = world.service.request_approval(AGENT, campaign_id, "budget", {"amount": 5}, "b")
    world.service.decide_approval(HUMAN, approval["approval_id"], True)
    assert world.service.brief(AGENT, campaign_id)["status"] == "running"
