"""Supervised fine-tuning stage with Hugging Face TRL and a LoRA adapter.

Profile argv: ``{python} -m bashgym_autoresearch.runners.train_trl {script} {run_dir} {inputs}``
where ``{script}`` is the train config JSON that the profile pins by SHA-256.

The config (human-owned) names the base checkpoint, the datasets the agent may
choose from (each pinned by SHA-256), recipe defaults, and hard caps. The
recipe (agent-owned, ``inputs/recipe.json``) sets hyperparameters within those
caps; unknown keys are rejected. The adapter is merged into a full checkpoint
at ``outputs/model`` because evaluation requires one. Per-step metrics go to
``outputs/training_metrics.jsonl`` and a manifest to
``outputs/training_manifest.json``. A non-finite loss fails the stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from bashgym_autoresearch.contracts import FrozenModel

CONFIG_SCHEMA = "bashgym_autoresearch.train_config.v1"
METRICS_FILENAME = "training_metrics.jsonl"
MANIFEST_FILENAME = "training_manifest.json"


class TrainingFailed(RuntimeError):
    """Training ran but its result must not be used."""


class DatasetRef(FrozenModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Recipe(FrozenModel):
    """Hyperparameters the agent may set. Every field has a config default."""

    dataset: str
    learning_rate: float = Field(gt=0, le=1e-2, allow_inf_nan=False)
    num_train_epochs: float = Field(gt=0, le=100, allow_inf_nan=False)
    max_steps: int = Field(ge=-1, le=1_000_000)
    per_device_train_batch_size: int = Field(ge=1, le=1024)
    gradient_accumulation_steps: int = Field(ge=1, le=1024)
    max_length: int = Field(ge=16, le=262_144)
    lora_r: int = Field(ge=1, le=1024)
    lora_alpha: int = Field(ge=1, le=4096)
    lora_dropout: float = Field(ge=0, lt=1, allow_inf_nan=False)
    warmup_ratio: float = Field(ge=0, lt=1, allow_inf_nan=False)
    weight_decay: float = Field(ge=0, le=1, allow_inf_nan=False)
    lr_scheduler_type: Literal["linear", "cosine", "constant", "constant_with_warmup"]
    packing: bool
    seed: int = Field(ge=0, le=2**32 - 1)


class Caps(FrozenModel):
    max_steps: int = Field(ge=1, le=1_000_000)
    max_examples: int = Field(ge=1, le=10_000_000)
    max_length: int = Field(ge=16, le=262_144)
    max_lora_r: int = Field(ge=1, le=1024)


class TrainConfig(FrozenModel):
    schema_: Literal["bashgym_autoresearch.train_config.v1"] = Field(alias="schema")
    base_model: str
    datasets: dict[str, DatasetRef] = Field(min_length=1, max_length=64)
    defaults: dict[str, Any]
    caps: Caps
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    target_modules: str | list[str] = "all-linear"
    assistant_only_loss: bool = False

    @model_validator(mode="after")
    def _defaults_form_a_recipe(self) -> TrainConfig:
        Recipe.model_validate(self.defaults)
        return self


def resolve_recipe(config: TrainConfig, recipe: dict[str, Any]) -> Recipe:
    """Merge the agent's recipe over the config defaults and enforce the caps."""
    resolved = Recipe.model_validate({**config.defaults, **recipe})
    caps = config.caps
    if resolved.dataset not in config.datasets:
        raise ValueError(f"dataset {resolved.dataset!r} is not allowed by the train config")
    if resolved.max_steps == -1:
        # Epoch-based training is still bounded by the step cap.
        resolved = resolved.model_copy(update={"max_steps": caps.max_steps})
    elif resolved.max_steps > caps.max_steps:
        raise ValueError(f"max_steps exceeds the cap of {caps.max_steps}")
    if resolved.max_length > caps.max_length:
        raise ValueError(f"max_length exceeds the cap of {caps.max_length}")
    if resolved.lora_r > caps.max_lora_r:
        raise ValueError(f"lora_r exceeds the cap of {caps.max_lora_r}")
    return resolved


def load_examples(ref: DatasetRef, max_examples: int) -> list[dict[str, Any]]:
    data = Path(ref.path).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError("training dataset does not match its pinned sha256")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not rows:
        raise ValueError("training dataset is empty")
    if len(rows) > max_examples:
        raise ValueError(f"training dataset has more than {max_examples} examples")
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("every training row needs a non-empty 'messages' list")
        if any(
            not isinstance(m, dict)
            or m.get("role") not in ("system", "user", "assistant", "tool")
            or not isinstance(m.get("content"), str)
            for m in messages
        ):
            raise ValueError("training messages need a role and string content")
    return [{"messages": row["messages"]} for row in rows]


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for package in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


def train_with_trl(
    config: TrainConfig, recipe: Recipe, examples: list[dict[str, Any]], outputs: Path
) -> dict[str, Any]:
    """Run LoRA SFT, merge the adapter, and save a full checkpoint to ``outputs/model``."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    metrics_path = outputs / METRICS_FILENAME
    losses: list[float] = []

    class RecordMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            record = {
                "step": state.global_step,
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
            }
            if "loss" in record:
                losses.append(float(record["loss"]))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    dtype = getattr(torch, config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, local_files_only=True, trust_remote_code=False, dtype=dtype
    )
    arguments = SFTConfig(
        output_dir=str(outputs.parent / "checkpoints"),
        learning_rate=recipe.learning_rate,
        num_train_epochs=recipe.num_train_epochs,
        max_steps=recipe.max_steps,
        per_device_train_batch_size=recipe.per_device_train_batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        max_length=recipe.max_length,
        warmup_ratio=recipe.warmup_ratio,
        weight_decay=recipe.weight_decay,
        lr_scheduler_type=recipe.lr_scheduler_type,
        packing=recipe.packing,
        assistant_only_loss=config.assistant_only_loss,
        seed=recipe.seed,
        bf16=config.dtype == "bfloat16",
        fp16=config.dtype == "float16",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_list(examples),
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            lora_dropout=recipe.lora_dropout,
            target_modules=config.target_modules,
            task_type="CAUSAL_LM",
        ),
        callbacks=[RecordMetrics()],
    )
    result = trainer.train()
    if not losses or not all(math.isfinite(loss) for loss in losses):
        raise TrainingFailed("training loss was missing or not finite")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(outputs / "model", safe_serialization=True)
    tokenizer.save_pretrained(outputs / "model")
    return {"global_step": result.global_step, "final_loss": losses[-1]}


def run(
    config_path: Path,
    run_dir: Path,
    inputs: Path,
    *,
    train: Callable[[TrainConfig, Recipe, list, Path], dict[str, Any]] = train_with_trl,
) -> dict[str, Any]:
    config = TrainConfig.model_validate_json(Path(config_path).read_bytes())
    recipe = resolve_recipe(config, json.loads((inputs / "recipe.json").read_text()))
    examples = load_examples(config.datasets[recipe.dataset], config.caps.max_examples)
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    summary = train(config, recipe, examples, outputs)
    if not (outputs / "model").is_dir():
        raise TrainingFailed("training did not produce outputs/model")
    manifest = {
        "schema": "bashgym_autoresearch.training_manifest.v1",
        "base_model": Path(config.base_model).name,
        "dataset": recipe.dataset,
        "dataset_sha256": config.datasets[recipe.dataset].sha256,
        "examples": len(examples),
        "recipe": recipe.model_dump(),
        "versions": _versions(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **summary,
    }
    (outputs / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: train_trl <config.json> <run_dir> <inputs>", file=sys.stderr)
        return 2
    try:
        run(Path(args[0]), Path(args[1]), Path(args[2]))
    except (ValueError, TrainingFailed) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
