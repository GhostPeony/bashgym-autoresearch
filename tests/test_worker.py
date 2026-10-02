import time
from datetime import UTC, datetime, timedelta

from conftest import AGENT, HUMAN

from bashgym_autoresearch.contracts import Change, StopRules
from bashgym_autoresearch.executors import LocalExecutor
from bashgym_autoresearch.worker import Worker


class CountingExecutor(LocalExecutor):
    def __init__(self):
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


def baseline(world, worker, campaign_id):
    experiment = propose(world, campaign_id, "baseline", "b", {"boost": 0.0})
    run_until_settled(worker)
    return experiment


def test_baseline_then_kept_candidate(world):
    worker = Worker(world.service, LocalExecutor())
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
    worker = Worker(world.service, LocalExecutor())
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
    same = Change(variable="seed", before=1, after=2)
    candidate = propose(world, campaign_id, "candidate", "c", {"boost": 0.0}, same)
    run_until_settled(worker)
    assert decision(world, campaign_id, candidate)["decision"] == "inconclusive"
    assert world.service.brief(AGENT, campaign_id)["incumbent"]["experiment_id"] != candidate


def test_training_crash_is_recorded_with_stderr(world):
    worker = Worker(world.service, LocalExecutor())
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
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
    worker = Worker(world.service, LocalExecutor())
    campaign_id = world.started_campaign()
    baseline(world, worker, campaign_id)
    candidate = propose(
        world, campaign_id, "candidate", "c", {"boost": 0.3, "train_seconds": 30}, CHANGE
    )
    assert worker.tick().startswith("launched train")
    clock["now"] += timedelta(seconds=61)
    run_until_settled(worker)
    result = decision(world, campaign_id, candidate)
    assert result["decision"] == "incomplete" and "time limit" in result["reason"]


def test_edited_evaluator_script_is_refused(world):
    worker = Worker(world.service, LocalExecutor())
    campaign_id = world.started_campaign()
    world.eval_script.write_text(world.eval_script.read_text() + "\n# edited\n")
    experiment = propose(world, campaign_id, "baseline", "b", {"boost": 0.0})
    run_until_settled(worker)
    result = decision(world, campaign_id, experiment)
    assert result["decision"] == "incomplete" and "script_changed" in result["reason"]


def test_restarted_worker_adopts_a_running_stage(world):
    executor = CountingExecutor()
    campaign_id = world.started_campaign()
    baseline(world, Worker(world.service, executor, owner="w"), campaign_id)
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
    Worker(world.service, LocalExecutor(), owner="a").tick()
    assert Worker(world.service, LocalExecutor(), owner="b").tick().startswith("standby")


def test_campaign_exhausts_after_max_experiments(world):
    worker = Worker(world.service, LocalExecutor())
    campaign_id = world.started_campaign(stop=StopRules(max_experiments=1, max_cost=10))
    baseline(world, worker, campaign_id)
    run_until_settled(worker)
    brief = world.service.brief(HUMAN, campaign_id)
    assert brief["status"] == "exhausted"
    assert brief["next_action"]["kind"] == "stop"
