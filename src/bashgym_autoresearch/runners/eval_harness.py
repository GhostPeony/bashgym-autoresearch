"""Generic evaluation stage.

Profile argv: ``{python} -m bashgym_autoresearch.runners.eval_harness {script} {run_dir} {inputs}``
where ``{script}`` is the suite config JSON that the profile pins by SHA-256.

The harness checks the platform-issued context and the dataset digest, verifies
task resources, generates greedily with a local model, grades every task through
the suite, and writes ``outputs/evaluation.json`` and ``outputs/task_results.json``.
The first infrastructure failure stops evaluation and marks the evidence
incomplete; tasks are never dropped from the denominator. Configuration and
input problems also produce incomplete evidence so they do not count as an
experiment.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from bashgym_autoresearch.contracts import (
    EvalEvidence,
    FrozenModel,
    Scope,
    TaskOutcome,
    canonical_hash,
)
from bashgym_autoresearch.evidence import EVIDENCE_FILENAME, EvalContext
from bashgym_autoresearch.runners.suites import SUITES
from bashgym_autoresearch.sandbox import validate_image

CONFIG_SCHEMA = "bashgym_autoresearch.eval_config.v1"
TASK_RESULTS_FILENAME = "task_results.json"
_TIER = re.compile(r"[A-Za-z0-9_-]{1,32}")


class GenerationConfig(FrozenModel):
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    device: str = Field(default="cuda", pattern=r"^(cpu|cuda(:[0-9]+)?)$")
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    max_input_tokens: int = Field(default=8192, ge=1, le=262_144)
    max_new_tokens: int = Field(default=1024, ge=1, le=32_768)
    timeout_seconds: float = Field(default=300, gt=0, le=3600)
    chat_template_kwargs: dict[str, bool] = Field(default_factory=dict, max_length=8)


class SandboxConfig(FrozenModel):
    image: str
    memory: str = Field(default="4g", pattern=r"^[1-9][0-9]*[mg]$")
    cpus: float = Field(default=2.0, gt=0, le=64)
    program_timeout_seconds: float = Field(default=90, gt=0, le=3600)

    @field_validator("image")
    @classmethod
    def _pinned(cls, value: str) -> str:
        return validate_image(value)


class CodingConfig(FrozenModel):
    completion_protocol: Literal["raw", "humaneval_body_v1"] = "humaneval_body_v1"


class DataAnalysisConfig(FrozenModel):
    data_root: str
    math_verify: bool = True


class HarnessConfig(FrozenModel):
    schema_: Literal["bashgym_autoresearch.eval_config.v1"] = Field(alias="schema")
    suite: Literal["coding", "data_analysis"]
    scope: Scope
    dataset: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_task_count: int = Field(ge=1, le=100_000)
    primary_metric: str = Field(default="pass_rate", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    base_model: str
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sandbox: SandboxConfig
    coding: CodingConfig = Field(default_factory=CodingConfig)
    data_analysis: DataAnalysisConfig | None = None


class HarnessError(ValueError):
    """Configuration or input problem detected before generation."""


def harness_digest() -> str:
    """Digest of the harness and suite sources that produced the evidence."""
    root = Path(__file__).parent
    files = [Path(__file__), *sorted((root / "suites").glob("*.py"))]
    files.append(root.parent / "sandbox" / "docker.py")
    files.append(root.parent / "grading" / "short_answer.py")
    return canonical_hash([[f.name, hashlib.sha256(f.read_bytes()).hexdigest()] for f in files])


def validate_model_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise HarnessError("model directory is missing")
    if (path / "adapter_config.json").exists():
        raise HarnessError("model directory is an adapter; evaluation needs a merged checkpoint")
    if not (path / "config.json").is_file():
        raise HarnessError("model directory has no config.json")
    if not any(path.glob("*.safetensors")) and not any(path.glob("pytorch_model*.bin")):
        raise HarnessError("model directory has no weights")
    return path.resolve()


class LocalModel:
    """Greedy, time-bounded local generation; transformers is imported lazily."""

    def __init__(self, directory: Path, config: GenerationConfig):
        self.directory, self.config = directory, config
        self.model = self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.directory, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.directory,
                local_files_only=True,
                trust_remote_code=False,
                dtype=getattr(torch, self.config.dtype),
            )
            .to(self.config.device)
            .eval()
        )

    def __call__(self, prompt: str | list[dict[str, str]]) -> str:
        import torch
        from transformers import set_seed

        self._load()
        if isinstance(prompt, str):
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)
        else:
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **self.config.chat_template_kwargs,
            )
        length = inputs["input_ids"].shape[-1]
        if length > self.config.max_input_tokens:
            raise HarnessError("prompt exceeds max_input_tokens")
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        set_seed(self.config.seed)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config.max_new_tokens,
                max_time=self.config.timeout_seconds,
                return_dict_in_generate=True,
            )
        if time.monotonic() - started >= self.config.timeout_seconds:
            raise TimeoutError("generation timed out")
        return self.tokenizer.decode(output.sequences[0, length:], skip_special_tokens=True)


def _write(outputs: Path, evidence: EvalEvidence, task_results: list[dict]) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / TASK_RESULTS_FILENAME).write_text(
        json.dumps({"tasks": task_results}, sort_keys=True), encoding="utf-8"
    )
    (outputs / EVIDENCE_FILENAME).write_text(evidence.model_dump_json(), encoding="utf-8")


def run(
    config_path: Path,
    run_dir: Path,
    inputs: Path,
    *,
    complete: Callable[[Any], str] | None = None,
) -> EvalEvidence:
    context_bytes = (inputs / "context.json").read_bytes()
    context = EvalContext.model_validate_json(context_bytes)
    outputs = run_dir / "outputs"
    provenance = {"harness_sha256": harness_digest()}

    def incomplete(reason: str, scope: Scope = "smoke") -> EvalEvidence:
        evidence = EvalEvidence(
            context_sha256=context.digest(),
            scope=scope,
            metrics={"pass_rate": 0.0},
            tasks=(),
            complete=False,
            provenance={**provenance, "error": reason[:500]},
        )
        _write(outputs, evidence, [])
        return evidence

    try:
        config = HarnessConfig.model_validate_json(Path(config_path).read_bytes())
    except (OSError, ValueError) as exc:
        return incomplete(f"invalid harness config: {exc}")
    provenance.update(suite=config.suite, sandbox_image=config.sandbox.image)
    suite = SUITES[config.suite]()
    try:
        if config.dataset_sha256 != context.dataset_sha256:
            raise HarnessError("config dataset_sha256 differs from the campaign's dataset")
        data = Path(config.dataset).read_bytes()
        if hashlib.sha256(data).hexdigest() != config.dataset_sha256:
            raise HarnessError("dataset file does not match dataset_sha256")
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        tasks = suite.load(rows)
        if len(tasks) != config.expected_task_count:
            raise HarnessError("dataset task count differs from expected_task_count")
        if len({task["task_id"] for task in tasks}) != len(tasks):
            raise HarnessError("dataset has duplicate task ids")
        if config.suite == "data_analysis" and config.data_analysis is None:
            raise HarnessError("data_analysis suite needs a data_analysis section")
        suite.prepare(tasks, config)
        if complete is None:
            model_path = json.loads((inputs / "model.json").read_text())["path"]
            directory = validate_model_directory(Path(model_path or config.base_model))
            complete = LocalModel(directory, config.generation)
    except (OSError, ValueError, KeyError) as exc:
        return incomplete(str(exc), config.scope)

    outcomes, results, finished = [], [], True
    tiers: dict[str, list[float]] = {}
    for task in tasks:
        try:
            graded = suite.grade(task, complete(suite.prompt(task)), config)
        except TimeoutError:
            results.append({"task_id": task["task_id"], "status": "generation_timeout"})
            finished = False
            break
        except Exception as exc:  # model or grading failure is infrastructure here
            results.append(
                {"task_id": task["task_id"], "status": "error", "detail": str(exc)[:500]}
            )
            finished = False
            break
        results.append(
            {
                "task_id": task["task_id"],
                "status": graded.status,
                "prediction": graded.prediction,
                "method": graded.method,
            }
        )
        if graded.status == "infrastructure_error":
            finished = False
            break
        value = 1.0 if graded.status == "passed" else 0.0
        outcomes.append(
            TaskOutcome(task_id=task["task_id"], cluster=suite.cluster(task), value=value)
        )
        tier = suite.tier(task)
        if tier and _TIER.fullmatch(tier):
            tiers.setdefault(tier, []).append(value)

    metrics = {config.primary_metric: sum(o.value for o in outcomes) / len(tasks)}
    for tier, values in sorted(tiers.items()):
        metrics[f"{config.primary_metric}.{tier}"] = sum(values) / len(values)
    evidence = EvalEvidence(
        context_sha256=context.digest(),
        scope=config.scope,
        metrics=metrics,
        tasks=tuple(outcomes),
        complete=finished,
        provenance=provenance,
    )
    _write(outputs, evidence, results)
    return evidence


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: eval_harness <config.json> <run_dir> <inputs>", file=sys.stderr)
        return 2
    run(Path(args[0]), Path(args[1]), Path(args[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
