import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from conftest import AGENT, HUMAN

from bashgym_autoresearch.contracts import Change, StopRules
from bashgym_autoresearch.executors import LocalExecutor
from bashgym_autoresearch.worker import Worker


class CountingExecutor(LocalExecutor):
    def __init__(self, control_root):
        super().__init__(control_root)
        self.launches = 0

    def launch(self, *args, **kwargs):
        self.launches += 1
        return super().launch(*args, **kwargs)


def run_until_settled(worker, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = worker.tick()
        if outcome == "idle":
            return
        if outcome.startswith("waiting"):
            time.sleep(0.05)
    raise AssertionError("worker did not settle")


def propose(world, campaign_id, role, key, recipe, change=None):
    return world.service.propose(
        AGENT,
        campaign_id,
        role=role,
        change=change,
        recipe=recipe,
        hypothesis="test",
        estimated_cost=1.0,
        idempotency_key=key,
    )["experiment_id"]


def decision(world, campaign_id, experiment_id):
    results = world.service.results(AGENT, campaign_id)
    return next(r for r in results if r["experiment_id"] == experiment_id)


CHANGE = Change(variable="boost", before=0.0, after=0.3)


def baseline(world, worker, campaign_id, **extra):
    experiment = propose(world, campaign_id, "baseline", "b", {"boost": 0.0, **extra})
    run_until_settled(worker)
    return experiment


def test_baseline_then_kept_candidate(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    base = baseline(world, worker, campaign_id)
    assert decision(world, campaign_id, base)["decision"] == "baseline"
    candidate = propose(world, campaign_id, "candidate", "c", {"boost": 0.3}, CHANGE)
    run_until_settled(worker)
    result = decision(world, campaign_id, candidate)
    assert result["decision"] == "keep" and result["ci_low"] > 0
    brief = world.service.brief(AGENT, campaign_id)
    assert brief["incumbent"]["experiment_id"] == candidate
    assert brief["smoke_only"] is True


def test_equal_candidate_is_inconclusive(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
    same = Change(variable="seed", before=None, after=2)
    candidate = propose(world, campaign_id, "candidate", "c", {"boost": 0.0, "seed": 2}, same)
    run_until_settled(worker)
    assert decision(world, campaign_id, candidate)["decision"] == "inconclusive"
    assert world.service.brief(AGENT, campaign_id)["incumbent"]["experiment_id"] != candidate


def test_training_crash_is_recorded_with_stderr(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id, train_crash=True)
    candidate = propose(
        world, campaign_id, "candidate", "c", {"boost": 0.3, "train_crash": True}, CHANGE
    )
    run_until_settled(worker)
    assert decision(world, campaign_id, candidate)["decision"] == "crash"
    failures = world.service.failures(AGENT, campaign_id)
    assert failures["experiment_id"] == candidate
    assert "training diverged" in failures["stages"][0]["stderr_tail"]


def test_stage_timeout_marks_experiment_incomplete(world):
    clock = {"now": datetime(2026, 10, 1, tzinfo=UTC)}
    world.service.clock = lambda: clock["now"]
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id, train_seconds=30)
    candidate = propose(
        world, campaign_id, "candidate", "c", {"boost": 0.3, "train_seconds": 30}, CHANGE
    )
    assert worker.tick().startswith("launched train")
    clock["now"] += timedelta(seconds=61)
    run_until_settled(worker)
    result = decision(world, campaign_id, candidate)
    assert result["decision"] == "incomplete" and "time limit" in result["reason"]


def test_edited_evaluator_script_is_refused(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    world.eval_script.write_text(world.eval_script.read_text() + "\n# edited\n")
    experiment = propose(world, campaign_id, "baseline", "b", {"boost": 0.0})
    run_until_settled(worker)
    result = decision(world, campaign_id, experiment)
    assert result["decision"] == "incomplete" and "script_changed" in result["reason"]


def test_restarted_worker_adopts_a_running_stage(world):
    executor = CountingExecutor(world.home / "control")
    campaign_id = world.started_campaign()
    baseline(world, Worker(world.service, executor, owner="w"), campaign_id, train_seconds=1)
    candidate = propose(
        world, campaign_id, "candidate", "c", {"boost": 0.3, "train_seconds": 1}, CHANGE
    )
    first = Worker(world.service, executor, owner="w")
    assert first.tick().startswith("launched train")
    launches = executor.launches
    restarted = Worker(world.service, executor, owner="w")
    run_until_settled(restarted)
    assert executor.launches == launches + 1  # only the evaluate stage was launched afterwards
    assert decision(world, campaign_id, candidate)["decision"] == "keep"


def test_second_worker_stands_by_while_lease_is_held(world):
    Worker(world.service, LocalExecutor(world.home / "control"), owner="a").tick()
    assert (
        Worker(world.service, LocalExecutor(world.home / "control"), owner="b")
        .tick()
        .startswith("standby")
    )


def test_campaign_exhausts_after_max_experiments(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign(stop=StopRules(max_experiments=1, max_cost=10))
    baseline(world, worker, campaign_id)
    run_until_settled(worker)
    brief = world.service.brief(HUMAN, campaign_id)
    assert brief["status"] == "exhausted"
    assert brief["next_action"]["kind"] == "stop"


def test_model_changed_after_training_seal_is_not_graded(world):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
    candidate = propose(world, campaign_id, "candidate", "c", {"boost": 0.3}, CHANGE)
    assert worker.tick().startswith("launched train")
    while not worker.tick().startswith("trained"):
        time.sleep(0.05)
    weights = world.home / "runs" / candidate / "train" / "outputs" / "model" / "weights.json"
    weights.write_text('{"boost": 0.5}')
    run_until_settled(worker)
    result = decision(world, campaign_id, candidate)
    assert result["decision"] == "incomplete" and "changed after sealing" in result["reason"]


def test_stages_get_an_allow_listed_environment_and_no_recipe_for_evaluation(world, monkeypatch):
    monkeypatch.setenv("BGAR_TOKEN", "bgar_secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    executor = CountingExecutor(world.home / "control")
    seen = []
    original = executor.launch

    def launch(run_id, run_dir, argv, env):
        seen.append((Path(run_dir).name, env))
        return original(run_id, run_dir, argv, env)

    executor.launch = launch
    worker = Worker(world.service, executor)
    campaign_id = world.started_campaign()
    experiment = baseline(world, worker, campaign_id)
    assert seen and all("BGAR_TOKEN" not in env and "HF_TOKEN" not in env for _, env in seen)
    assert "PATH" in seen[0][1]
    assert not (world.home / "runs" / experiment / "evaluate-inputs" / "recipe.json").exists()


def test_internal_error_marks_experiment_incomplete_without_stopping_the_worker(world, monkeypatch):
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    experiment = propose(world, campaign_id, "baseline", "b", {"boost": 0.0})

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(worker, "_launch", boom)
    outcome = worker.tick()
    assert outcome.endswith("incomplete")
    assert "disk full" in decision(world, campaign_id, experiment)["reason"]
    monkeypatch.undo()
    assert worker.tick() == "idle"


def test_profile_reregistration_does_not_change_a_running_campaign(world):
    from conftest import HUMAN, write_script

    from bashgym_autoresearch.contracts import StageProfile

    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    other, other_sha = write_script(world.eval_script.parent, "other.py", "raise SystemExit(9)\n")
    world.service.register_profile(
        HUMAN,
        StageProfile(
            name="eval",
            kind="evaluate",
            argv=(sys.executable, "{script}"),
            script=str(other),
            script_sha256=other_sha,
            timeout_seconds=60,
        ),
    )
    experiment = baseline(world, worker, campaign_id)
    assert decision(world, campaign_id, experiment)["decision"] == "baseline"


def test_measured_cost_is_charged_when_it_exceeds_the_estimate(world):
    from conftest import HUMAN

    from bashgym_autoresearch.contracts import StageProfile

    profile = world.service.profile("eval")
    world.service.register_profile(
        HUMAN, StageProfile(**{**profile.model_dump(), "cost_per_hour": 360_000.0})
    )
    worker = Worker(world.service, LocalExecutor(world.home / "control"))
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
    assert world.service.brief(AGENT, campaign_id)["budget"]["used"] > 1.0
