"""Reference-answer canaries through the real harness, suites, and Docker.

Run with ``BGAR_TEST_SANDBOX_IMAGE`` set to a locally built image digest from
``sandbox-images/python-analysis``. Gold programs must pass, wrong programs must
fail, and the data mount must be read-only.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from bashgym_autoresearch.contracts import EvalEvidence
from bashgym_autoresearch.evidence import EvalContext, write_context
from bashgym_autoresearch.runners import eval_harness

IMAGE = os.environ.get("BGAR_TEST_SANDBOX_IMAGE")
pytestmark = pytest.mark.skipif(not IMAGE, reason="set BGAR_TEST_SANDBOX_IMAGE")

CODING = [
    {
        "task_id": "add",
        "prompt": 'def add(a, b):\n    """Return a + b."""\n',
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "entry_point": "add",
    },
    {
        "task_id": "mean",
        "prompt": "import numpy as np\n\ndef mean(xs):\n",
        "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 2\n",
        "entry_point": "mean",
    },
]
CODING_GOLD = {"add": "    return a + b\n", "mean": "    return float(np.mean(xs))\n"}


def run_suite(tmp_path, suite, rows, completions, **config):
    tmp_path.mkdir(parents=True)
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": "bashgym_autoresearch.eval_config.v1",
                "suite": suite,
                "scope": "smoke",
                "dataset": str(dataset),
                "dataset_sha256": digest,
                "expected_task_count": len(rows),
                "base_model": str(tmp_path),
                "sandbox": {"image": IMAGE, "program_timeout_seconds": 60},
                **config,
            }
        )
    )
    inputs, run_dir = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    write_context(
        inputs / "context.json",
        EvalContext(
            campaign_id="c",
            experiment_id="e",
            suite_id="s",
            dataset_sha256=digest,
            evaluator_sha256="b" * 64,
        ),
    )
    prompts = iter(completions)
    eval_harness.run(config_path, run_dir, inputs, complete=lambda prompt: next(prompts))
    evidence = EvalEvidence.model_validate_json(
        (run_dir / "outputs" / "evaluation.json").read_bytes()
    )
    results = json.loads((run_dir / "outputs" / "task_results.json").read_text())["tasks"]
    return evidence, results


def test_coding_gold_passes_and_wrong_fails(tmp_path):
    gold, _ = run_suite(
        tmp_path / "gold", "coding", CODING, [CODING_GOLD[t["task_id"]] for t in CODING]
    )
    assert gold.complete and gold.metrics["pass_rate"] == 1.0
    wrong, results = run_suite(tmp_path / "wrong", "coding", CODING, ["    return 0\n"] * 2)
    assert wrong.complete and wrong.metrics["pass_rate"] == 0.0
    assert {r["method"] for r in results} == {"tests"}


def test_data_analysis_gold_passes_wrong_fails_and_mount_is_read_only(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "shop__sales").mkdir(parents=True)
    table = data_root / "shop__sales" / "sales.csv"
    table.write_text("region,amount\nnorth,10\nsouth,32\n")
    task = {
        "task_id": "total",
        "messages": [{"role": "user", "content": "What is the total amount?"}],
        "answer": "42",
        "reward_mode": "numeric",
        "atol": 0.0,
        "rtol": 0.0,
        "difficulty_tier": "easy",
        "data": {
            "prefix": "shop__sales",
            "files": [
                {
                    "path": "sales.csv",
                    "size": table.stat().st_size,
                    "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    config = {"data_analysis": {"data_root": str(data_root), "math_verify": False}}
    gold_program = (
        "```python\nimport pandas as pd\n"
        "df = pd.read_csv('sales.csv')\nprint(df['amount'].sum())\n```"
    )
    gold, _ = run_suite(tmp_path / "gold", "data_analysis", [task], [gold_program], **config)
    assert gold.complete and gold.metrics["pass_rate"] == 1.0
    tamper = (
        "```python\ntry:\n    open('sales.csv', 'w').write('x')\n    print('wrote')\n"
        "except OSError:\n    print('read-only')\n```"
    )
    blocked, results = run_suite(tmp_path / "tamper", "data_analysis", [task], [tamper], **config)
    assert blocked.metrics["pass_rate"] == 0.0 and results[0]["prediction"] == "read-only"
    assert table.read_text().startswith("region")
