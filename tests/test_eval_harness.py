import hashlib
import json
from pathlib import Path

import pytest

from bashgym_autoresearch.contracts import EvalEvidence
from bashgym_autoresearch.evidence import EvalContext, write_context
from bashgym_autoresearch.runners import eval_harness
from bashgym_autoresearch.runners.suites import CodingSuite, DataAnalysisSuite, Graded
from bashgym_autoresearch.runners.suites.coding import process_completion, program
from bashgym_autoresearch.runners.suites.data_analysis import extract_code, looks_like_a_command
from bashgym_autoresearch.sandbox import SandboxResult

IMAGE = "python@sha256:" + "a" * 64
CODING_TASKS = [
    {
        "task_id": f"t{i}",
        "prompt": "def add(a, b):\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
        "entry_point": "add",
    }
    for i in range(4)
]


def setup(tmp_path, rows=CODING_TASKS, suite="coding", **config_overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    config = {
        "schema": "bashgym_autoresearch.eval_config.v1",
        "suite": suite,
        "scope": "development",
        "dataset": str(dataset),
        "dataset_sha256": digest,
        "expected_task_count": len(rows),
        "base_model": str(tmp_path / "base-model"),
        "sandbox": {"image": IMAGE},
    }
    config.update(config_overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    inputs, run_dir = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    context = EvalContext(
        campaign_id="c",
        experiment_id="e",
        suite_id="s",
        dataset_sha256=digest,
        evaluator_sha256="b" * 64,
    )
    write_context(inputs / "context.json", context)
    (inputs / "model.json").write_text(json.dumps({"path": None}))
    return config_path, run_dir, inputs, context


def read_outputs(run_dir: Path):
    evidence = EvalEvidence.model_validate_json(
        (run_dir / "outputs" / "evaluation.json").read_bytes()
    )
    results = json.loads((run_dir / "outputs" / "task_results.json").read_text())["tasks"]
    return evidence, results


def fake_grades(monkeypatch, statuses):
    queue = iter(statuses)
    monkeypatch.setattr(CodingSuite, "grade", lambda self, task, completion, config: next(queue))


def test_complete_run_echoes_context_and_reports_pass_rate(tmp_path, monkeypatch):
    config, run_dir, inputs, context = setup(tmp_path)
    fake_grades(
        monkeypatch, [Graded("passed"), Graded("failed"), Graded("passed"), Graded("passed")]
    )
    eval_harness.run(config, run_dir, inputs, complete=lambda prompt: "    return a + b\n")
    evidence, results = read_outputs(run_dir)
    assert evidence.complete and evidence.context_sha256 == context.digest()
    assert evidence.metrics == {"pass_rate": 0.75}
    assert [t.cluster for t in evidence.tasks] == ["t0", "t1", "t2", "t3"]
    assert (
        evidence.provenance["suite"] == "coding"
        and len(evidence.provenance["harness_sha256"]) == 64
    )
    assert [r["status"] for r in results] == ["passed", "failed", "passed", "passed"]


def test_infrastructure_error_stops_and_marks_incomplete(tmp_path, monkeypatch):
    config, run_dir, inputs, _ = setup(tmp_path)
    fake_grades(
        monkeypatch, [Graded("passed"), Graded("infrastructure_error", detail="docker gone")]
    )
    eval_harness.run(config, run_dir, inputs, complete=lambda prompt: "x")
    evidence, results = read_outputs(run_dir)
    assert not evidence.complete and len(results) == 2 and len(evidence.tasks) == 1
    assert evidence.metrics["pass_rate"] == 0.25


def test_generation_timeout_stops_and_marks_incomplete(tmp_path, monkeypatch):
    config, run_dir, inputs, _ = setup(tmp_path)

    def timeout(prompt):
        raise TimeoutError

    eval_harness.run(config, run_dir, inputs, complete=timeout)
    evidence, results = read_outputs(run_dir)
    assert not evidence.complete and results[0]["status"] == "generation_timeout"


@pytest.mark.parametrize("problem", ["dataset_changed", "config_digest", "count", "solution_leak"])
def test_input_problems_produce_incomplete_evidence_before_generation(tmp_path, problem):
    rows = CODING_TASKS
    overrides = {}
    if problem == "count":
        overrides["expected_task_count"] = 99
    if problem == "solution_leak":
        rows = [{**CODING_TASKS[0], "canonical_solution": "    return a + b\n"}]
    config, run_dir, inputs, _ = setup(tmp_path, rows=rows, **overrides)
    if problem == "dataset_changed":
        Path(json.loads(config.read_text())["dataset"]).write_text("{}\n")
    if problem == "config_digest":
        data = json.loads(config.read_text())
        data["dataset_sha256"] = "c" * 64
        config.write_text(json.dumps(data))
    eval_harness.run(config, run_dir, inputs, complete=lambda prompt: pytest.fail("generated"))
    evidence, _ = read_outputs(run_dir)
    assert not evidence.complete and evidence.provenance["error"]


def test_adapter_only_model_directory_is_refused(tmp_path):
    model = tmp_path / "adapter"
    model.mkdir()
    (model / "adapter_config.json").write_text("{}")
    with pytest.raises(eval_harness.HarnessError, match="adapter"):
        eval_harness.validate_model_directory(model)


def test_unpinned_sandbox_image_is_rejected(tmp_path):
    config, run_dir, inputs, _ = setup(tmp_path, sandbox={"image": "python:3.12"})
    eval_harness.run(config, run_dir, inputs, complete=lambda prompt: "x")
    evidence, _ = read_outputs(run_dir)
    assert not evidence.complete and "invalid harness config" in evidence.provenance["error"]


def test_coding_program_and_completion_protocol():
    body = "    return a + b\n\ndef unrelated():\n    pass\n"
    assert process_completion(body, "humaneval_body_v1") == "    return a + b\n\n"
    assert process_completion(body, "raw") == body
    namespace: dict = {}
    exec(program(CODING_TASKS[0], "    return a + b\n"), namespace)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SandboxResult(0, "", "", False), ("passed", "tests")),
        (SandboxResult(1, "", "AssertionError", False), ("failed", "tests")),
        (SandboxResult(None, "", "", True), ("failed", "test_timeout")),
        (SandboxResult(None, "", "", False, "no docker"), ("infrastructure_error", "sandbox")),
    ],
)
def test_coding_grade_maps_sandbox_results(tmp_path, monkeypatch, result, expected):
    from bashgym_autoresearch.runners.suites import coding

    seen = {}
    monkeypatch.setattr(
        coding, "run_program", lambda image, files, argv, **kw: seen.update(files) or result
    )
    config_path, *_ = setup(tmp_path)
    config = eval_harness.HarnessConfig.model_validate_json(config_path.read_bytes())
    graded = CodingSuite().grade(CODING_TASKS[0], "    return a + b\n", config)
    assert (graded.status, graded.method) == expected
    assert "check(add)" in seen["solution.py"]


def data_task(tmp_path, answer="42"):
    data_root = tmp_path / "data"
    (data_root / "owner__data").mkdir(parents=True)
    table = data_root / "owner__data" / "table.csv"
    table.write_text("a\n42\n")
    task = {
        "task_id": "d1",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}],
        "answer": answer,
        "reward_mode": "numeric",
        "atol": 0.0,
        "rtol": 0.0,
        "difficulty_tier": "easy",
        "data": {
            "prefix": "owner__data",
            "files": [
                {
                    "path": "table.csv",
                    "size": table.stat().st_size,
                    "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
                }
            ],
        },
    }
    return task, data_root


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("loading\n42\n", ("passed", "42")),
        ("41\n", ("failed", "41")),
        ('echo "42" > answer.txt\n', ("failed", 'echo "42" > answer.txt')),
        ("", ("failed", "")),
    ],
)
def test_data_analysis_grades_on_the_host(tmp_path, monkeypatch, stdout, expected):
    from bashgym_autoresearch.runners.suites import data_analysis

    task, data_root = data_task(tmp_path)
    seen = {}

    def fake_run(image, files, argv, **kwargs):
        seen.update(files=files, **kwargs)
        return SandboxResult(0, stdout, "", False)

    monkeypatch.setattr(data_analysis, "run_program", fake_run)
    config_path, *_ = setup(
        tmp_path / "cfg",
        rows=[task],
        suite="data_analysis",
        data_analysis={"data_root": str(data_root), "math_verify": False},
    )
    config = eval_harness.HarnessConfig.model_validate_json(config_path.read_bytes())
    suite = DataAnalysisSuite()
    suite.prepare([task], config)
    graded = suite.grade(task, "```python\nprint(42)\n```", config)
    assert (graded.status, graded.prediction) == expected
    assert task["answer"] not in json.dumps(seen["files"]).replace("print(42)", "")
    assert seen["workdir"] == "/home/user/input"
    assert set(seen["read_only_mounts"]) == {"/home/user/input"}


def test_data_analysis_prepare_rejects_changed_data(tmp_path):
    task, data_root = data_task(tmp_path)
    (data_root / "owner__data" / "table.csv").write_text("a\n43\n")
    config_path, *_ = setup(
        tmp_path / "cfg",
        rows=[task],
        suite="data_analysis",
        data_analysis={"data_root": str(data_root), "math_verify": False},
    )
    config = eval_harness.HarnessConfig.model_validate_json(config_path.read_bytes())
    with pytest.raises(ValueError, match="changed"):
        DataAnalysisSuite().prepare([task], config)


def test_data_analysis_text_helpers():
    assert extract_code("```python\nprint(1)\n```\n```py\nprint(2)\n```") == "print(2)"
    assert looks_like_a_command("cat x | head") and not looks_like_a_command(">50K")
