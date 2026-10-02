"""End to end through a real `bashgym-ar serve` process, the HTTP API, and the local executor."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from bashgym_autoresearch import cli
from bashgym_autoresearch.client import Client, ClientError

SMOKE = Path(__file__).resolve().parents[1] / "examples" / "smoke"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    home = tmp_path / "home"
    cli.main(["--home", str(home), "init"])
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; from bashgym_autoresearch.cli import main; sys.exit(main(sys.argv[1:]))",
            "--home",
            str(home),
            "serve",
            "--port",
            str(port),
            "--interval",
            "0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/v1/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        process.kill()
        raise AssertionError("server did not start")
    human = Client(url, (home / "human.token").read_text())
    agent = Client(url, human.create_token("agent", "e2e-agent")["token"])
    yield human, agent
    process.terminate()
    process.wait(timeout=30)


def register(human: Client, name: str, kind: str, script: Path) -> None:
    human.register_profile(
        {
            "name": name,
            "kind": kind,
            "argv": [sys.executable, "{script}", "{run_dir}", "{inputs}"],
            "script": str(script),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "timeout_seconds": 120,
        }
    )


def wait_for_decision(agent: Client, campaign_id: str, experiment_id: str) -> dict:
    after, deadline = 0, time.monotonic() + 120
    while time.monotonic() < deadline:
        update = agent.wait(campaign_id, after=after, timeout=10)
        for event in update["events"]:
            after = event["seq"]
            payload = event["payload"]
            if event["type"] == "experiment_decided" and payload["experiment_id"] == experiment_id:
                return payload
    raise AssertionError("experiment was not decided")


def test_smoke_campaign_end_to_end(server):
    human, agent = server
    register(human, "smoke-eval", "evaluate", SMOKE / "evaluate.py")
    register(human, "smoke-train", "train", SMOKE / "train.py")
    spec = {
        "name": "smoke",
        "objective": "Raise the smoke pass rate with one controlled change at a time.",
        "primary": {"name": "pass_rate", "direction": "maximize"},
        "minimum_improvement": 0.05,
        "stop": {"max_experiments": 4, "max_cost": 4},
        "evaluation": {
            "suite_id": "smoke-tasks",
            "dataset_sha256": hashlib.sha256((SMOKE / "tasks.json").read_bytes()).hexdigest(),
            "profile": "smoke-eval",
        },
        "train_profile": "smoke-train",
        "n_resamples": 1000,
    }
    campaign_id = human.create_campaign(spec)["campaign_id"]

    with pytest.raises(ClientError) as early:
        agent.propose(campaign_id, {"role": "baseline", "hypothesis": "x", "estimated_cost": 1})
    assert early.value.status == 409
    approval = agent.request_approval(campaign_id, "start")
    human.decide_approval(approval["approval_id"], True)

    baseline = agent.propose(
        campaign_id,
        {
            "role": "baseline",
            "hypothesis": "measure the starting point",
            "estimated_cost": 1,
            "recipe": {"boost": 0.0},
        },
    )["experiment_id"]
    assert wait_for_decision(agent, campaign_id, baseline)["decision"] == "baseline"

    human.set_guidance(campaign_id, "Prefer small boosts; report the interval.")
    brief = agent.brief(campaign_id)
    assert brief["next_action"]["kind"] == "propose_candidate"
    assert brief["guidance"]["version"] == 1

    candidate = agent.propose(
        campaign_id,
        {
            "role": "candidate",
            "hypothesis": "a 0.3 boost lets the model solve harder tasks",
            "estimated_cost": 1,
            "recipe": {"boost": 0.3},
            "change": {"variable": "boost", "before": 0.0, "after": 0.3},
        },
    )["experiment_id"]
    assert wait_for_decision(agent, campaign_id, candidate)["decision"] == "keep"
    kept = agent.results(campaign_id)[-1]
    assert kept["ci_low"] >= 0.05 and kept["scope"] == "smoke"

    with pytest.raises(ClientError) as promote:
        agent.request_approval(campaign_id, "promote", {"experiment_id": candidate})
    assert promote.value.status == 409

    report = human.report(campaign_id)
    assert [item["decision"] for item in report["experiments"]] == ["baseline", "keep"]
    assert report["smoke_only"] is True
