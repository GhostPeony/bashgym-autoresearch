"""Mechanical campaign progression: one durable effect per tick.

Order of work in ``tick``:
1. observe the oldest stage that is launching or running (deadline, exit, loss);
2. launch the next stage of an active experiment in a running campaign;
3. stop a running campaign whose stop rules are met;
4. otherwise idle.

Every effect is recorded before or after it happens in a way that a restarted
worker can resume: a run directory is created once, and an existing run is
adopted instead of relaunched.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sqlite3
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from bashgym_autoresearch.contracts import CampaignSpec, EvalEvidence, StageProfile
from bashgym_autoresearch.decision import Comparison, EvaluationMismatch, decide
from bashgym_autoresearch.evidence import (
    EvalContext,
    EvidenceError,
    directory_digest,
    read_evidence,
    write_context,
)
from bashgym_autoresearch.executors import Executor
from bashgym_autoresearch.seal import SEAL_FILENAME, SealError, hash_file
from bashgym_autoresearch.service import CampaignState, Service
from bashgym_autoresearch.store import utc_now

LEASE_SECONDS = 30


def _stages(spec: CampaignSpec, role: str) -> list[tuple[str, str]]:
    stages = []
    if role == "candidate" and spec.train_profile is not None:
        stages.append(("train", spec.train_profile))
    stages.append(("evaluate", spec.evaluation.profile))
    return stages


def _seed(experiment_id: str) -> int:
    return int(hashlib.sha256(experiment_id.encode()).hexdigest()[:8], 16)


class Worker:
    def __init__(self, service: Service, executor: Executor, *, owner: str | None = None):
        self.service = service
        self.store = service.store
        self.executor = executor
        self.owner = owner or f"{socket.gethostname()}:{os.getpid()}:{id(self)}"

    # ----- public loop ---------------------------------------------------------

    def tick(self) -> str:
        if not self._hold_lease():
            return "standby: another worker holds the lease"
        stage = self.store.read_one(
            "SELECT * FROM stage_runs WHERE status IN ('launching', 'running')"
            " ORDER BY created_at LIMIT 1"
        )
        if stage is not None:
            return self._observe(stage)
        experiment = self.store.read_one(
            "SELECT e.* FROM experiments e JOIN campaigns c ON c.id = e.campaign_id"
            " WHERE e.status IN ('queued', 'running') AND c.status IN ('running', 'cancelled')"
            " ORDER BY e.created_at LIMIT 1"
        )
        if experiment is not None:
            return self._advance(experiment)
        for campaign in self.store.read("SELECT id FROM campaigns WHERE status = 'running'"):
            stopped = self._maybe_stop(campaign["id"])
            if stopped:
                return stopped
        return "idle"

    def run(self, stop: threading.Event, interval: float = 1.0) -> None:
        while not stop.is_set():
            outcome = self.tick()
            if outcome == "idle" or outcome.startswith(("waiting", "standby")):
                stop.wait(interval)

    # ----- lease ---------------------------------------------------------------

    def _hold_lease(self) -> bool:
        now = self.service.clock()
        with self.store.transaction() as db:
            row = db.execute("SELECT value FROM meta WHERE key = 'worker_lease'").fetchone()
            if row is not None:
                lease = json.loads(row["value"])
                expires = datetime.fromisoformat(lease["expires_at"])
                if lease["owner"] != self.owner and expires > now:
                    return False
            value = json.dumps(
                {
                    "owner": self.owner,
                    "expires_at": (now + timedelta(seconds=LEASE_SECONDS)).isoformat(),
                }
            )
            db.execute(
                "INSERT INTO meta(key, value) VALUES ('worker_lease', ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (value,),
            )
        return True

    # ----- launching -----------------------------------------------------------

    def _advance(self, experiment: sqlite3.Row) -> str:
        with self.store.transaction() as db:
            campaign = self.service.campaign_row(db, experiment["campaign_id"])
        if campaign["status"] == "cancelled":
            return self._finish(experiment["id"], "incomplete", "campaign cancelled")
        spec = CampaignSpec.model_validate_json(campaign["spec_json"])
        existing = {
            row["kind"]: row
            for row in self.store.read(
                "SELECT * FROM stage_runs WHERE experiment_id = ?", (experiment["id"],)
            )
        }
        pending = [
            (kind, name) for kind, name in _stages(spec, experiment["role"]) if kind not in existing
        ]
        if not pending:
            return self._finish(experiment["id"], "incomplete", "no stage left to run")
        kind, profile_name = pending[0]
        profile = self.service.profile(profile_name)
        script = Path(profile.script)
        if not script.is_file() or hash_file(script)[0] != profile.script_sha256:
            return self._finish(
                experiment["id"],
                "incomplete",
                f"script_changed: {profile.name} no longer matches its registered sha256",
            )

        base = self.service.home / "runs" / experiment["id"]
        inputs = base / f"{kind}-inputs"
        run_dir = base / kind
        inputs.mkdir(parents=True, exist_ok=True)
        (inputs / "recipe.json").write_text(experiment["recipe_json"], encoding="utf-8")
        context = None
        if kind == "evaluate":
            train = existing.get("train")
            model_path = str(Path(train["run_dir"]) / "outputs" / "model") if train else None
            (inputs / "model.json").write_text(json.dumps({"path": model_path}), encoding="utf-8")
            context = EvalContext(
                campaign_id=campaign["id"],
                experiment_id=experiment["id"],
                suite_id=spec.evaluation.suite_id,
                dataset_sha256=spec.evaluation.dataset_sha256,
                evaluator_sha256=profile.script_sha256,
                model_digest=train["output_digest"] if train else None,
            )
            write_context(inputs / "context.json", context)

        stage_id = hashlib.sha256(f"{experiment['id']}:{kind}".encode()).hexdigest()[:24]
        deadline = self.service.clock() + timedelta(seconds=profile.timeout_seconds)
        with self.store.transaction() as db:
            now = utc_now()
            db.execute(
                "INSERT INTO stage_runs(id, experiment_id, profile, kind, status, run_dir,"
                " deadline_at, script_sha256, context_json, version, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, 'launching', ?, ?, ?, ?, 1, ?, ?)",
                (
                    stage_id,
                    experiment["id"],
                    profile.name,
                    kind,
                    str(run_dir),
                    deadline.isoformat(),
                    profile.script_sha256,
                    context.model_dump_json() if context else None,
                    now,
                    now,
                ),
            )
            if experiment["status"] == "queued":
                self.store.cas_update(
                    db, "experiments", experiment["id"], experiment["version"], status="running"
                )
        stage = self.store.read_one("SELECT * FROM stage_runs WHERE id = ?", (stage_id,))
        return self._launch(stage, profile, inputs)

    def _launch(self, stage: sqlite3.Row, profile: StageProfile, inputs: Path) -> str:
        run_dir = Path(stage["run_dir"])
        if not run_dir.exists():
            values = {
                "python": sys.executable,
                "script": profile.script,
                "run_dir": str(run_dir),
                "inputs": str(inputs),
                "outputs": str(run_dir / "outputs"),
            }
            argv = [part.format(**values) for part in profile.argv]
            self.executor.launch(stage["id"], run_dir, argv, env=dict(os.environ))
        experiment_id = stage["experiment_id"]
        with self.store.transaction() as db:
            self.store.cas_update(db, "stage_runs", stage["id"], stage["version"], status="running")
            campaign_id = db.execute(
                "SELECT campaign_id FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()["campaign_id"]
            self.store.append_event(
                db,
                campaign_id,
                "stage_launched",
                {"experiment_id": experiment_id, "kind": stage["kind"]},
            )
        return f"launched {stage['kind']} for {experiment_id}"

    # ----- observing -----------------------------------------------------------

    def _observe(self, stage: sqlite3.Row) -> str:
        experiment_id, run_dir = stage["experiment_id"], Path(stage["run_dir"])
        campaign = self.store.read_one(
            "SELECT c.status FROM campaigns c JOIN experiments e ON e.campaign_id = c.id"
            " WHERE e.id = ?",
            (experiment_id,),
        )
        if stage["status"] == "launching":
            if not run_dir.exists():
                profile = self.service.profile(stage["profile"])
                inputs = run_dir.parent / f"{stage['kind']}-inputs"
                return self._launch(stage, profile, inputs)
            if not (run_dir / "launch.json").exists():
                return self._close_stage(stage, "lost", None, "stage launch was interrupted")
            with self.store.transaction() as db:
                self.store.cas_update(
                    db, "stage_runs", stage["id"], stage["version"], status="running"
                )
            return f"adopted {stage['kind']} for {experiment_id}"
        if campaign["status"] == "cancelled":
            self.executor.kill(stage["id"], run_dir)
            return self._close_stage(stage, "killed", None, "campaign cancelled")
        if self.service.clock() > datetime.fromisoformat(stage["deadline_at"]):
            self.executor.kill(stage["id"], run_dir)
            return self._close_stage(stage, "timeout", None, "stage exceeded its time limit")
        observation = self.executor.observe(stage["id"], run_dir)
        if observation.state == "running":
            return f"waiting on {stage['kind']} for {experiment_id}"
        if observation.state == "lost":
            return self._close_stage(stage, "lost", None, "stage process was lost")
        if observation.exit_code != 0:
            return self._close_stage(
                stage, "failed", observation.exit_code, f"stage exited with {observation.exit_code}"
            )
        return self._complete(stage)

    def _identity(self, stage: sqlite3.Row) -> dict[str, str]:
        return {
            "experiment_id": stage["experiment_id"],
            "stage_id": stage["id"],
            "kind": stage["kind"],
        }

    def _complete(self, stage: sqlite3.Row) -> str:
        run_dir = Path(stage["run_dir"])
        try:
            if not (run_dir / SEAL_FILENAME).exists():
                self.service.sealer.seal(run_dir, identity=self._identity(stage))
            self.service.sealer.verify(run_dir, identity=self._identity(stage))
        except SealError as exc:
            return self._close_stage(stage, "invalid", 0, f"outputs could not be sealed: {exc}")
        if stage["kind"] == "train":
            model = run_dir / "outputs" / "model"
            if not model.is_dir():
                return self._close_stage(stage, "invalid", 0, "training produced no outputs/model")
            with self.store.transaction() as db:
                self.store.cas_update(
                    db,
                    "stage_runs",
                    stage["id"],
                    stage["version"],
                    status="succeeded",
                    exit_code=0,
                    output_digest=directory_digest(model),
                )
            return f"trained {stage['experiment_id']}"
        return self._evaluate(stage)

    def _evaluate(self, stage: sqlite3.Row) -> str:
        experiment_id = stage["experiment_id"]
        context = EvalContext.model_validate_json(stage["context_json"])
        try:
            evidence = read_evidence(Path(stage["run_dir"]) / "outputs", context)
        except EvidenceError as exc:
            return self._close_stage(stage, "invalid", 0, str(exc))
        experiment = self.store.read_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        state = self.service.state(experiment["campaign_id"])
        try:
            comparison = decide(
                state.spec,
                role=experiment["role"],
                incumbent=state.incumbent_evidence,
                result=evidence,
                seed=_seed(experiment_id),
            )
        except EvaluationMismatch as exc:
            return self._close_stage(stage, "invalid", 0, str(exc))
        with self.store.transaction() as db:
            self.store.cas_update(
                db, "stage_runs", stage["id"], stage["version"], status="succeeded", exit_code=0
            )
        return self._record(experiment_id, comparison, evidence)

    def _close_stage(
        self, stage: sqlite3.Row, status: str, exit_code: int | None, reason: str
    ) -> str:
        with self.store.transaction() as db:
            self.store.cas_update(
                db,
                "stage_runs",
                stage["id"],
                stage["version"],
                status=status,
                exit_code=exit_code,
                reason=reason,
            )
        decision = "crash" if status == "failed" else "incomplete"
        return self._finish(stage["experiment_id"], decision, reason)

    # ----- recording -----------------------------------------------------------

    def _finish(self, experiment_id: str, decision: str, reason: str) -> str:
        comparison = Comparison(decision=decision, reason=reason)
        return self._record(experiment_id, comparison, None)

    def _record(
        self, experiment_id: str, comparison: Comparison, evidence: EvalEvidence | None
    ) -> str:
        status = {"crash": "crashed", "incomplete": "incomplete"}.get(
            comparison.decision, "evaluated"
        )
        with self.store.transaction() as db:
            experiment = db.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            db.execute(
                "INSERT INTO results(experiment_id, decision, comparison_json, evidence_json,"
                " created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    experiment_id,
                    comparison.decision,
                    comparison.model_dump_json(),
                    evidence.model_dump_json() if evidence else None,
                    utc_now(),
                ),
            )
            self.store.cas_update(
                db, "experiments", experiment_id, experiment["version"], status=status
            )
            self.store.append_event(
                db,
                experiment["campaign_id"],
                "experiment_decided",
                {
                    "experiment_id": experiment_id,
                    "decision": comparison.decision,
                    "reason": comparison.reason,
                },
            )
        return f"{experiment_id}: {comparison.decision}"

    def _maybe_stop(self, campaign_id: str) -> str | None:
        with self.store.transaction() as db:
            campaign = self.service.campaign_row(db, campaign_id)
            state = CampaignState(db, campaign)
            action = state.next_action(self.service.clock())
            if action.kind != "stop":
                return None
            target = "completed" if action.reason == "target metric reached" else "exhausted"
            self.store.cas_update(db, "campaigns", campaign_id, campaign["version"], status=target)
            self.store.append_event(db, campaign_id, "campaign_stopped", {"reason": action.reason})
        return f"{campaign_id}: {target} ({action.reason})"
