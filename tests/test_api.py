import asyncio
import time

import pytest
from conftest import World
from fastapi.testclient import TestClient

from bashgym_autoresearch.api import create_app
from bashgym_autoresearch.auth import create_token
from bashgym_autoresearch.client import Client, ClientError
from bashgym_autoresearch.executors import LocalExecutor
from bashgym_autoresearch.mcp_server import AGENT_TOOLS, build_server
from bashgym_autoresearch.worker import Worker


@pytest.fixture
def clients(world: World):
    app = create_app(world.service)
    store = world.service.store

    def make(role):
        token = create_token(store, role, role)
        return Client(
            "http://testserver", token, http=TestClient(app, base_url="http://testserver")
        )

    return make("human"), make("agent"), TestClient(app)


def spec_json(world):
    return world.spec().model_dump(mode="json")


def test_missing_or_bad_token_is_401(world, clients):
    _, _, app = clients
    app.headers.pop("Authorization", None)
    assert app.get("/v1/campaigns").status_code == 401
    assert (
        app.get("/v1/campaigns", headers={"Authorization": "Bearer bgar_nope"}).status_code == 401
    )


def test_agent_cannot_grant_or_steer_and_nothing_changes(world, clients):
    human, agent, _ = clients
    campaign_id = human.create_campaign(spec_json(world))["campaign_id"]
    approval = agent.request_approval(campaign_id, "start")
    for call in (
        lambda: agent.decide_approval(approval["approval_id"], True),
        lambda: agent.set_guidance(campaign_id, "x"),
        lambda: agent.create_campaign(spec_json(world)),
    ):
        with pytest.raises(ClientError) as error:
            call()
        assert error.value.status == 403
    brief = agent.brief(campaign_id)
    assert brief["status"] == "awaiting_start" and brief["guidance"]["version"] == 0
    assert len(human.list_campaigns()) == 1


def test_post_without_idempotency_key_is_rejected(world, clients):
    human, _, app = clients
    response = app.post(
        "/v1/campaigns",
        json=spec_json(world),
        headers={"Authorization": human._http.headers["Authorization"]},
    )
    assert response.status_code == 400


def test_idempotent_propose_replays(world, clients):
    human, agent, _ = clients
    campaign_id = human.create_campaign(spec_json(world))["campaign_id"]
    human.decide_approval(agent.request_approval(campaign_id, "start")["approval_id"], True)
    body = {"role": "baseline", "hypothesis": "start", "estimated_cost": 1, "recipe": {}}
    first = agent.propose(campaign_id, body, key="k")
    assert agent.propose(campaign_id, body, key="k") == first
    with pytest.raises(ClientError) as error:
        agent.propose(campaign_id, body, key="other")
    assert error.value.status == 409


def test_full_loop_over_http(world, clients):
    human, agent, _ = clients
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = human.create_campaign(spec_json(world))["campaign_id"]
    human.decide_approval(agent.request_approval(campaign_id, "start")["approval_id"], True)
    assert agent.brief(campaign_id)["next_action"]["kind"] == "propose_baseline"
    agent.propose(
        campaign_id,
        {"role": "baseline", "hypothesis": "start", "estimated_cost": 1, "recipe": {"boost": 0.0}},
        key="b",
    )
    settle(worker)
    agent.propose(
        campaign_id,
        {
            "role": "candidate",
            "hypothesis": "more boost passes more tasks",
            "estimated_cost": 1,
            "recipe": {"boost": 0.3},
            "change": {"variable": "boost", "before": 0.0, "after": 0.3},
        },
        key="c",
    )
    settle(worker)
    decisions = [result["decision"] for result in agent.results(campaign_id)]
    assert decisions == ["baseline", "keep"]
    events = agent.wait(campaign_id, after=0, timeout=0)["events"]
    assert any(event["type"] == "experiment_decided" for event in events)
    assert agent.report(campaign_id)["smoke_only"] is True


def settle(worker, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = worker.tick()
        if outcome == "idle":
            return
        if outcome.startswith("waiting"):
            time.sleep(0.05)
    raise AssertionError("worker did not settle")


def test_mcp_server_exposes_exactly_the_ten_agent_tools():
    server = build_server(Client("http://unused", "bgar_unused"))
    tools = server.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    assert sorted(tool.name for tool in tools) == sorted(AGENT_TOOLS)
