"""Run a demo campaign through the real API so the dashboard has data to show.

Usage: uv run python examples/smoke/demo.py <empty state directory>
Then open http://127.0.0.1:8770 and paste the printed human token. The server keeps
running in the background; its pid is written to <state directory>/server.pid.
"""

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from bashgym_autoresearch import cli
from bashgym_autoresearch.client import Client

repo = Path(__file__).resolve().parents[2]
home = Path(sys.argv[1])
shutil.rmtree(home, ignore_errors=True)
cli.main(["--home", str(home), "init"])
server = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import sys; from bashgym_autoresearch.cli import main; sys.exit(main(sys.argv[1:]))",
        "--home",
        str(home),
        "serve",
        "--port",
        "8770",
        "--interval",
        "0.2",
    ],
    stdout=open(home / "server.log", "w"),
    stderr=subprocess.STDOUT,
    creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
)
(home / "server.pid").write_text(str(server.pid))
url = "http://127.0.0.1:8770"
for _ in range(100):
    try:
        if httpx.get(url + "/v1/health", timeout=1).status_code == 200:
            break
    except httpx.HTTPError:
        time.sleep(0.2)
human = Client(url, (home / "human.token").read_text())
agent = Client(url, human.create_token("agent", "claude-code")["token"])
smoke = repo / "examples" / "smoke"
for name, kind, script in (
    ("smoke-eval", "evaluate", "evaluate.py"),
    ("smoke-train", "train", "train.py"),
):
    path = smoke / script
    human.register_profile(
        {
            "name": name,
            "kind": kind,
            "argv": [sys.executable, "{script}", "{run_dir}", "{inputs}"],
            "script": str(path),
            "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "timeout_seconds": 120,
            "cost_per_hour": 20,
        }
    )
spec = {
    "name": "smoke-pass-rate",
    "objective": "Raise the smoke pass rate with one controlled change at a time",
    "primary": {"name": "pass_rate", "direction": "maximize"},
    "minimum_improvement": 0.05,
    "stop": {"max_experiments": 8, "max_cost": 6},
    "evaluation": {
        "suite_id": "smoke-tasks",
        "dataset_sha256": hashlib.sha256((smoke / "tasks.json").read_bytes()).hexdigest(),
        "profile": "smoke-eval",
    },
    "train_profile": "smoke-train",
    "n_resamples": 1000,
}
cid = human.create_campaign(spec)["campaign_id"]
human.decide_approval(agent.request_approval(cid, "start")["approval_id"], True)


def run(body):
    eid = agent.propose(cid, body)["experiment_id"]
    after = 0
    while True:
        update = agent.wait(cid, after=after, timeout=10)
        for e in update["events"]:
            after = e["seq"]
            if e["type"] == "experiment_decided" and e["payload"]["experiment_id"] == eid:
                print(eid, e["payload"]["decision"])
                return


base = {"boost": 0.0, "seed": 1}
run(
    {
        "role": "baseline",
        "hypothesis": "Measure the base model before changing anything.",
        "estimated_cost": 1,
        "recipe": base,
    }
)
run(
    {
        "role": "candidate",
        "hypothesis": "A 0.3 boost should let the model solve the harder tasks.",
        "estimated_cost": 1,
        "recipe": {**base, "boost": 0.3},
        "change": {"variable": "boost", "before": 0.0, "after": 0.3},
    }
)
kept = {**base, "boost": 0.3}
run(
    {
        "role": "candidate",
        "hypothesis": "A different seed might change which tasks pass.",
        "estimated_cost": 1,
        "recipe": {**kept, "seed": 2},
        "change": {"variable": "seed", "before": 1, "after": 2},
    }
)
run(
    {
        "role": "candidate",
        "hypothesis": "A smaller boost may generalize better.",
        "estimated_cost": 1,
        "recipe": {**kept, "boost": 0.12},
        "change": {"variable": "boost", "before": 0.3, "after": 0.12},
    }
)
human.set_guidance(cid, "Keep the boost between 0.2 and 0.5. Try one larger step before stopping.")
agent.request_approval(cid, "budget", {"amount": 4})
print("URL", url)
print("TOKEN", (home / "human.token").read_text())
