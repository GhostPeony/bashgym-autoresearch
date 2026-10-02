import sys
import time

import pytest

from bashgym_autoresearch.executors import LocalExecutor


def wait_for_exit(executor, run_id, run_dir, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observation = executor.observe(run_id, run_dir)
        if observation.state != "running":
            return observation
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_successful_command_reports_exit_code_and_output(tmp_path):
    executor, run_dir = LocalExecutor(), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "print('hello')"], env=None)
    observation = wait_for_exit(executor, "r1", run_dir)
    assert (observation.state, observation.exit_code) == ("exited", 0)
    assert (run_dir / "stdout.log").read_text().strip() == "hello"


def test_failing_command_reports_its_exit_code(tmp_path):
    executor, run_dir = LocalExecutor(), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "import sys; sys.exit(7)"], env=None)
    assert wait_for_exit(executor, "r1", run_dir).exit_code == 7


def test_missing_program_is_reported_as_failure(tmp_path):
    executor, run_dir = LocalExecutor(), tmp_path / "run"
    executor.launch("r1", run_dir, [str(tmp_path / "does-not-exist")], env=None)
    observation = wait_for_exit(executor, "r1", run_dir)
    assert observation.state == "exited" and observation.exit_code != 0


def test_running_command_can_be_killed_and_adopted_by_a_new_executor(tmp_path):
    run_dir = tmp_path / "run"
    LocalExecutor().launch(
        "r1", run_dir, [sys.executable, "-c", "import time; time.sleep(60)"], env=None
    )
    restarted = LocalExecutor()
    assert restarted.observe("r1", run_dir).state == "running"
    restarted.kill("r1", run_dir)
    assert wait_for_exit(restarted, "r1", run_dir).state in {"exited", "lost"}


def test_relaunch_into_existing_run_directory_is_refused(tmp_path):
    executor, run_dir = LocalExecutor(), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)
    with pytest.raises(FileExistsError):
        executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)


def test_unknown_run_is_lost(tmp_path):
    (tmp_path / "run").mkdir()
    assert LocalExecutor().observe("r1", tmp_path / "run").state == "lost"
