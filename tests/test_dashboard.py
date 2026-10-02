import json
import time

import pytest
from conftest import AGENT, HUMAN
from fastapi.testclient import TestClient

from bashgym_autoresearch.api import create_app
from bashgym_autoresearch.auth import Forbidden, create_token
from bashgym_autoresearch.contracts import Change
from bashgym_autoresearch.executors import LocalExecutor
from bashgym_autoresearch.service import DashboardService
from bashgym_autoresearch.worker import Worker


def settle(worker, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = worker.tick()
        if outcome == "idle":
            return
        if outcome.startswith("waiting"):
            time.sleep(0.05)
    raise AssertionError("worker did not settle")


@pytest.fixture
def campaign(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    common = dict(hypothesis="h", estimated_cost=1.0)
    world.service.propose(
        AGENT,
        campaign_id,
        role="baseline",
        change=None,
        recipe={"boost": 0.0},
        idempotency_key="b",
        **common,
    )
    settle(worker)
    candidate = world.service.propose(
        AGENT,
        campaign_id,
        role="candidate",
        change=Change(variable="boost", before=0.0, after=0.3),
        recipe={"boost": 0.3},
        idempotency_key="c",
        **common,
    )["experiment_id"]
    settle(worker)
    return campaign_id, candidate


def test_campaign_dashboard_includes_timeline_stages_and_events(world, campaign):
    campaign_id, candidate = campaign
    view = DashboardService(world.service).campaign(HUMAN, campaign_id)
    assert [e["decision"] for e in view["experiments"]] == ["baseline", "keep"]
    kinds = [stage["kind"] for stage in view["experiments"][1]["stages"]]
    assert kinds == ["train", "evaluate"]
    assert view["experiments"][1]["ci_low"] > 0
    assert any(event["type"] == "experiment_decided" for event in view["events"])
    assert view["approvals"][0]["kind"] == "start"
    json.dumps(view)


def test_training_metrics_are_paged(world, campaign):
    _, candidate = campaign
    service = DashboardService(world.service)
    first = service.training_metrics(AGENT, candidate)
    assert [point["loss"] for point in first["points"]] == [1.0, 0.5, 1.0 / 3]
    assert service.training_metrics(AGENT, candidate, after=first["next"])["points"] == []


def test_per_task_results_are_human_only(world, campaign):
    _, candidate = campaign
    with pytest.raises(Forbidden):
        DashboardService(world.service).task_results(AGENT, candidate)
    assert DashboardService(world.service).task_results(HUMAN, candidate)["tasks"] == []


def test_pending_approvals_inbox_and_http_routes(world, campaign):
    campaign_id, candidate = campaign
    world.service.request_approval(AGENT, campaign_id, "budget", {"amount": 2}, "more")
    client = TestClient(create_app(world.service))
    human = {"Authorization": f"Bearer {create_token(world.service.store, 'human', 'h')}"}
    agent = {"Authorization": f"Bearer {create_token(world.service.store, 'agent', 'a')}"}
    inbox = client.get("/v1/approvals", headers=human).json()
    assert [item["kind"] for item in inbox] == ["budget"] and inbox[0]["campaign_name"] == "demo"
    assert client.get(f"/v1/campaigns/{campaign_id}/dashboard", headers=agent).status_code == 200
    assert client.get(f"/v1/experiments/{candidate}/task-results", headers=agent).status_code == 403
    metrics = client.get(f"/v1/experiments/{candidate}/training-metrics", headers=agent).json()
    assert len(metrics["points"]) == 3


def test_event_stream_delivers_existing_events(world, campaign):
    campaign_id, _ = campaign
    client = TestClient(create_app(world.service))
    headers = {"Authorization": f"Bearer {create_token(world.service.store, 'human', 'h')}"}
    with client.stream(
        "GET", f"/v1/campaigns/{campaign_id}/stream?seconds=1", headers=headers
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                assert event["type"] == "campaign_created"
                break


def test_static_web_app_is_served_when_built(world, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>dashboard</title>")
    client = TestClient(create_app(world.service, static_dir=web))
    assert "dashboard" in client.get("/").text
    assert client.get("/v1/health").json() == {"ok": True}
