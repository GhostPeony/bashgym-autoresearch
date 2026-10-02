"""Shared fixtures: a service with registered smoke stage profiles."""

from __future__ import annotations

import hashlib
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from bashgym_autoresearch.auth import Principal
from bashgym_autoresearch.contracts import (
    CampaignSpec,
    EvaluationBinding,
    MetricSpec,
    StageProfile,
    StopRules,
)
from bashgym_autoresearch.seal import Sealer
from bashgym_autoresearch.service import Service
from bashgym_autoresearch.store import Store

HUMAN = Principal(role="human", label="owner")
AGENT = Principal(role="agent", label="agent")

# Evaluates 40 tasks. Task i passes when its score ``(i % 10) / 10`` is below
# ``0.5 + boost``, where ``boost`` comes from the trained model (0 for the base model).
EVALUATE_SCRIPT = textwrap.dedent("""
    import json, pathlib, sys
    run_dir, inputs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    context = json.loads((inputs / "context.json").read_text())
    model = json.loads((inputs / "model.json").read_text())["path"]
    boost = 0.0
    if model:
        boost = json.loads((pathlib.Path(model) / "weights.json").read_text())["boost"]
    import hashlib
    digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    tasks = [
        {"task_id": f"t{i}", "cluster": f"t{i}", "value": float((i % 10) / 10 < 0.5 + boost)}
        for i in range(40)
    ]
    value = sum(t["value"] for t in tasks) / len(tasks)
    out = run_dir / "outputs"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps({
        "context_sha256": digest, "scope": "smoke", "metrics": {"pass_rate": value},
        "tasks": tasks, "complete": True,
    }))
    """)

TRAIN_SCRIPT = textwrap.dedent("""
    import json, pathlib, sys, time
    run_dir, inputs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    recipe = json.loads((inputs / "recipe.json").read_text())
    if recipe.get("train_crash"):
        print("training diverged", file=sys.stderr)
        sys.exit(1)
    time.sleep(recipe.get("train_seconds", 0))
    model = run_dir / "outputs" / "model"
    model.mkdir(parents=True, exist_ok=True)
    (model / "weights.json").write_text(json.dumps({"boost": recipe.get("boost", 0.0)}))
    """)


def write_script(directory: Path, name: str, body: str) -> tuple[Path, str]:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class World:
    service: Service
    home: Path
    eval_script: Path
    train_script: Path

    def spec(self, **overrides) -> CampaignSpec:
        values = dict(
            name="demo",
            objective="raise pass rate",
            primary=MetricSpec(name="pass_rate", direction="maximize"),
            minimum_improvement=0.0,
            stop=StopRules(max_experiments=5, max_cost=10),
            evaluation=EvaluationBinding(
                suite_id="smoke-suite", dataset_sha256="a" * 64, profile="eval"
            ),
            train_profile="train",
            n_resamples=500,
        )
        values.update(overrides)
        return CampaignSpec(**values)

    def started_campaign(self, **overrides) -> str:
        created = self.service.create_campaign(HUMAN, self.spec(**overrides), "create-1")
        approval = self.service.request_approval(
            AGENT, created["campaign_id"], "start", {}, "start-1"
        )
        self.service.decide_approval(HUMAN, approval["approval_id"], True)
        return created["campaign_id"]


@pytest.fixture
def world(tmp_path) -> World:
    home = tmp_path / "home"
    home.mkdir()
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    service = Service(Store(home / "state.db"), Sealer(b"s" * 32), home)
    eval_script, eval_sha = write_script(scripts, "evaluate.py", EVALUATE_SCRIPT)
    train_script, train_sha = write_script(scripts, "train.py", TRAIN_SCRIPT)
    argv = (sys.executable, "{script}", "{run_dir}", "{inputs}")
    service.register_profile(
        HUMAN,
        StageProfile(
            name="eval",
            kind="evaluate",
            argv=argv,
            script=str(eval_script),
            script_sha256=eval_sha,
            timeout_seconds=60,
        ),
    )
    service.register_profile(
        HUMAN,
        StageProfile(
            name="train",
            kind="train",
            argv=argv,
            script=str(train_script),
            script_sha256=train_sha,
            timeout_seconds=60,
        ),
    )
    return World(service, home, eval_script, train_script)
