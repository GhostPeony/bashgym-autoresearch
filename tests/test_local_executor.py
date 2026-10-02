import os
import sys
import time

import psutil
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
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "print('hello')"], env=None)
    observation = wait_for_exit(executor, "r1", run_dir)
    assert (observation.state, observation.exit_code) == ("exited", 0)
    assert (run_dir / "stdout.log").read_text().strip() == "hello"


def test_failing_command_reports_its_exit_code(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "import sys; sys.exit(7)"], env=None)
    assert wait_for_exit(executor, "r1", run_dir).exit_code == 7


def test_missing_program_is_reported_as_failure(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [str(tmp_path / "does-not-exist")], env=None)
    observation = wait_for_exit(executor, "r1", run_dir)
    assert observation.state == "exited" and observation.exit_code != 0


def test_running_command_can_be_killed_and_adopted_by_a_new_executor(tmp_path):
    run_dir = tmp_path / "run"
    LocalExecutor(tmp_path / "control").launch(
        "r1", run_dir, [sys.executable, "-c", "import time; time.sleep(60)"], env=None
    )
    restarted = LocalExecutor(tmp_path / "control")
    assert restarted.observe("r1", run_dir).state == "running"
    restarted.kill("r1", run_dir)
    assert wait_for_exit(restarted, "r1", run_dir).state in {"exited", "lost"}


def test_relaunch_into_existing_run_directory_is_refused(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)
    with pytest.raises(FileExistsError):
        executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)


def test_unknown_run_is_lost(tmp_path):
    (tmp_path / "run").mkdir()
    assert LocalExecutor(tmp_path / "control").observe("r1", tmp_path / "run").state == "lost"


def test_control_files_live_outside_the_stage_directory(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)
    wait_for_exit(executor, "r1", run_dir)
    assert not (run_dir / "exit_code").exists() and not (run_dir / "launch.json").exists()
    assert (tmp_path / "control" / "r1" / "exit_code").is_file()
    assert (run_dir / "outputs").is_dir()


def test_a_stage_forging_an_exit_code_in_its_directory_is_ignored(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    script = "import pathlib, sys; pathlib.Path('exit_code').write_text('0'); sys.exit(4)"
    executor.launch("r1", run_dir, [sys.executable, "-c", script], env=None)
    assert wait_for_exit(executor, "r1", run_dir).exit_code == 4


def test_malformed_exit_code_counts_as_failure(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    executor.launch("r1", run_dir, [sys.executable, "-c", "pass"], env=None)
    wait_for_exit(executor, "r1", run_dir)
    (tmp_path / "control" / "r1" / "exit_code").write_text("zero")
    assert executor.observe("r1", run_dir).exit_code == 255


@pytest.mark.skipif(os.name == "nt", reason="process-group kill is POSIX only")
def test_kill_reaches_grandchildren(tmp_path):
    executor, run_dir = LocalExecutor(tmp_path / "control"), tmp_path / "run"
    script = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open('child.pid', 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    executor.launch("r1", run_dir, [sys.executable, "-c", script], env=None)
    deadline = time.monotonic() + 10
    while not (run_dir / "child.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child = int((run_dir / "child.pid").read_text())
    executor.kill("r1", run_dir)
    time.sleep(0.5)
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE
