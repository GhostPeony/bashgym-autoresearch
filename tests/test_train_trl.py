import hashlib
import json

import pytest

from bashgym_autoresearch.runners import train_trl

DEFAULTS = {
    "dataset": "sft-v1",
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "max_steps": -1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler_type": "cosine",
    "packing": False,
    "seed": 0,
}
ROWS = [
    {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
] * 3


def setup(tmp_path, rows=ROWS, recipe=None, **caps):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "sft.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = {
        "schema": "bashgym_autoresearch.train_config.v1",
        "base_model": str(tmp_path / "base"),
        "datasets": {
            "sft-v1": {"path": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}
        },
        "defaults": DEFAULTS,
        "caps": {
            "max_steps": 500,
            "max_examples": 100,
            "max_length": 4096,
            "max_lora_r": 64,
            **caps,
        },
    }
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    inputs, run_dir = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir()
    run_dir.mkdir()
    (inputs / "recipe.json").write_text(json.dumps(recipe or {}))
    return config_path, run_dir, inputs


def fake_train(config, recipe, examples, outputs):
    (outputs / "model").mkdir()
    (outputs / "model" / "config.json").write_text("{}")
    return {"global_step": 3, "final_loss": 0.5}


def test_recipe_overrides_defaults_and_manifest_is_written(tmp_path):
    seen = {}

    def train(config, recipe, examples, outputs):
        seen.update(recipe=recipe, examples=examples)
        return fake_train(config, recipe, examples, outputs)

    config, run_dir, inputs = setup(tmp_path, recipe={"learning_rate": 1e-4})
    manifest = train_trl.run(config, run_dir, inputs, train=train)
    assert seen["recipe"].learning_rate == 1e-4 and seen["recipe"].max_steps == 500
    assert len(seen["examples"]) == 3
    written = json.loads((run_dir / "outputs" / "training_manifest.json").read_text())
    assert written == manifest and manifest["dataset"] == "sft-v1" and manifest["global_step"] == 3
    assert str(tmp_path) not in json.dumps(manifest)


@pytest.mark.parametrize(
    ("recipe", "message"),
    [
        ({"surprise": 1}, "Extra inputs"),
        ({"dataset": "secret-test-set"}, "not allowed"),
        ({"max_steps": 10_000}, "max_steps"),
        ({"max_length": 100_000}, "max_length"),
        ({"lora_r": 128}, "lora_r"),
        ({"learning_rate": 1.0}, "learning_rate"),
    ],
)
def test_recipes_outside_the_config_are_rejected(tmp_path, recipe, message):
    config, run_dir, inputs = setup(tmp_path, recipe=recipe)
    with pytest.raises(ValueError, match=message):
        train_trl.run(config, run_dir, inputs, train=fake_train)


def test_changed_or_malformed_datasets_are_rejected(tmp_path):
    config, run_dir, inputs = setup(tmp_path)
    (tmp_path / "sft.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="sha256"):
        train_trl.run(config, run_dir, inputs, train=fake_train)
    config, run_dir, inputs = setup(tmp_path / "bad", rows=[{"text": "no messages"}])
    with pytest.raises(ValueError, match="messages"):
        train_trl.run(config, run_dir, inputs, train=fake_train)


def test_training_that_produces_no_model_fails(tmp_path):
    config, run_dir, inputs = setup(tmp_path)
    with pytest.raises(train_trl.TrainingFailed):
        train_trl.run(config, run_dir, inputs, train=lambda *args: {})


def test_main_exit_codes(tmp_path, monkeypatch):
    config, run_dir, inputs = setup(tmp_path, recipe={"surprise": True})
    assert train_trl.main([str(config), str(run_dir), str(inputs)]) == 1
    assert train_trl.main([]) == 2
