"""Smoke evaluation stage over tasks.json.

A task passes when its difficulty is below 0.5 plus the model's boost. The
evidence echoes the platform-issued context digest and is labelled "smoke".

Usage: evaluate.py <run_dir> <inputs>
"""

import hashlib
import json
import sys
from pathlib import Path

run_dir, inputs = Path(sys.argv[1]), Path(sys.argv[2])
context_bytes = (inputs / "context.json").read_bytes()
tasks_bytes = (Path(__file__).parent / "tasks.json").read_bytes()
if hashlib.sha256(tasks_bytes).hexdigest() != json.loads(context_bytes)["dataset_sha256"]:
    sys.exit("tasks.json does not match the campaign's dataset_sha256")
tasks = json.loads(tasks_bytes)
model = json.loads((inputs / "model.json").read_text())["path"]
boost = 0.0  # the base model; evaluation stages never see the agent's recipe
if model:
    boost = json.loads((Path(model) / "weights.json").read_text())["boost"]

outcomes = [
    {
        "task_id": t["task_id"],
        "cluster": t["task_id"],
        "value": float(t["difficulty"] < 0.5 + boost),
    }
    for t in tasks
]
outputs = run_dir / "outputs"
outputs.mkdir(parents=True, exist_ok=True)
(outputs / "evaluation.json").write_text(
    json.dumps(
        {
            "context_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "scope": "smoke",
            "metrics": {"pass_rate": sum(o["value"] for o in outcomes) / len(outcomes)},
            "tasks": outcomes,
            "complete": True,
        }
    )
)
