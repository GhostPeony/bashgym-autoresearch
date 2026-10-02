"""Mechanical campaign progression: one durable effect per tick.

Order of work in ``tick``:
1. observe the oldest stage that is launching or running (cancel, deadline, exit, loss);
2. launch the next stage of an active experiment in a running campaign;
3. stop a running campaign whose stop rules are met;
4. otherwise idle.

Each effect is recorded so a restarted worker can resume: a run directory is
created once, an existing run is adopted instead of relaunched, and a stage's
closure and its experiment's result are written in one transaction. Stages use
the profiles snapshotted when the campaign was created, receive only an
allow-listed environment, and evaluation stages never receive the agent's recipe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import socket
import sqlite3
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

from bashgym_autoresearch.contracts import EvalEvidence, StageProfile
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

log = logging.getLogger(__name__)
LEASE_SECONDS = 120
BASE_ENVIRONMENT = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "TMPDIR",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LC_ALL",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "DOCKER_HOST",
)


def stage_environment(profile: StageProfile) -> dict[str, str]:
    names = set(BASE_ENVIRONMENT) | set(profile.env_passthrough)
    return {name: os.environ[name] for name in names if name in os.environ}


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
            return self._guard(stage["experiment_id"], lambda: self._observe(stage))
        experiment = self.store.read_one(
            "SELECT e.* FROM experiments e JOIN campaigns c ON c.id = e.campaign_id"
            " WHERE e.status IN ('queued', 'running') AND c.status IN ('running', 'cancelled')"
            " ORDER BY e.created_at LIMIT 1"
        )
        if experiment is not None:
            return self._guard(experiment["id"], lambda: self._advance(experiment))
        for campaign in self.store.read("SELECT id FROM campaigns WHERE status = 'running'"):
            stopped = self._maybe_stop(campaign["id"])
            if stopped:
                return stopped
        return "idle"

    def run(self, stop: threading.Event, interval: float = 1.0) -> None:
        while not stop.is_set():
            try:
                outcome = self.tick()
            except Exception:  # keep the worker alive; the failure is logged
                log.exception("worker tick failed")
                outcome = "waiting after error"
            if outcome == "idle" or outcome.startswith(("waiting", "standby")):
                stop.wait(interval)

    def _guard(self, experiment_id: str, effect) -> str:
        """Run one effect; an unexpected error ends that experiment as incomplete."""
        try:
            return effect()
        except Exception as exc:
            log.exception("stage handling failed for %s", experiment_id)
            return self._abort(experiment_id, f"internal error: {type(exc).__name__}: {exc}")

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

    def _campaign(self, experiment_id: str) -> tuple[sqlite3.Row, CampaignState]:
        with self.store.transaction() as db:
            campaign_id = db.execute(
                "SELECT campaign_id FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()["campaign_id"]
            campaign = self.service.campaign_row(db, campaign_id)
            return campaign, CampaignState(db, campaign)

    @staticmethod
    def _plan(state: CampaignState, role: str) -> list[tuple[str, StageProfile]]:
        spec, stages = state.spec, []
        if role == "candidate" and spec.train_profile is not None:
            stages.append(("train", state.profiles[spec.train_profile]))
        stages.append(("evaluate", state.profiles[spec.evaluation.profile]))
        return stages

    def _advance(self, experiment: sqlite3.Row) -> str:
        campaign, state = self._campaign(experiment["id"])
        if campaign["status"] == "cancelled":
            return self._abort(experiment["id"], "campaign cancelled")
        existing = {
            row["kind"]: row
            for row in self.store.read(
                "SELECT * FROM stage_runs WHERE experiment_id = ?", (experiment["id"],)
            )
        }
        for row in existing.values():
            if row["status"] != "succeeded":
                decision = "crash" if row["status"] == "failed" else "incomplete"
                return self._finish(
                    experiment["id"], decision, row["reason"] or "stage did not succeed"
                )
        pending = [(k, p) for k, p in self._plan(state, experiment["role"]) if k not in existing]
        if not pending:
            return self._abort(experiment["id"], "no stage left to run")
        kind, profile = pending[0]
        script = Path(profile.script)
        if not script.is_file() or hash_file(script)[0] != profile.script_sha256:
            return self._abort(
                experiment["id"],
                f"script_changed: {profile.name} no longer matches its registered sha256",
            )

        base = self.service.home / "runs" / experiment["id"]
        inputs = base / f"{kind}-inputs"
        run_dir = base / kind
        inputs.mkdir(parents=True, exist_ok=True)
        context = None
        if kind == "train":
            (inputs / "recipe.json").write_text(experiment["recipe_json"], encoding="utf-8")
        else:
            train = existing.get("train")
            if train is not None:
                reason = self._verify_training(train)
                if reason:
                    return self._abort(experiment["id"], reason)
            model_path = str(Path(train["run_dir"]) / "outputs" / "model") if train else None
            (inputs / "model.json").write_text(json.dumps({"path": model_path}), encoding="utf-8")
            spec = state.spec
            context = EvalContext(
                campaign_id=campaign["id"],
                experiment_id=experiment["id"],
                suite_id=spec.evaluation.suite_id,
                dataset_sha256=spec.evaluation.dataset_sha256,
                evaluator_sha256=profile.script_sha256,
                model_digest=train["output_digest"] if train else None,
                nonce=secrets.token_hex(16),
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
        return self._launch(stage, profile)

    def _launch(self, stage: sqlite3.Row, profile: StageProfile) -> str:
        run_dir = Path(stage["run_dir"])
        inputs = run_dir.parent / f"{stage['kind']}-inputs"
        if not self.executor.launched(stage["id"]):
            values = {
                "python": sys.executable,
                "script": profile.script,
                "run_dir": str(run_dir),
                "inputs": str(inputs),
                "outputs": str(run_dir / "outputs"),
            }
            argv = [part.format(**values) for part in profile.argv]
            self.executor.launch(stage["id"], run_dir, argv, env=stage_environment(profile))
        with self.store.transaction() as db:
            self.store.cas_update(db, "stage_runs", stage["id"], stage["version"], status="running")
            campaign_id = db.execute(
                "SELECT campaign_id FROM experiments WHERE id = ?", (stage["experiment_id"],)
            ).fetchone()["campaign_id"]
            self.store.append_event(
                db,
                campaign_id,
                "stage_launched",
                {"experiment_id": stage["experiment_id"], "kind": stage["kind"]},
            )
        return f"launched {stage['kind']} for {stage['experiment_id']}"

    # ----- observing -----------------------------------------------------------

    def _observe(self, stage: sqlite3.Row) -> str:
        experiment_id, run_dir = stage["experiment_id"], Path(stage["run_dir"])
        campaign, state = self._campaign(experiment_id)
        if campaign["status"] == "cancelled":
            self.executor.kill(stage["id"], run_dir)
            return self._close(stage, state, "killed", None, "campaign cancelled")
        if stage["status"] == "launching":
            if not self.executor.launched(stage["id"]):
                if run_dir.exists():
                    return self._close(stage, state, "lost", None, "stage launch was interrupted")
                return self._launch(stage, state.profiles[stage["profile"]])
            with self.store.transaction() as db:
                self.store.cas_update(
                    db, "stage_runs", stage["id"], stage["version"], status="running"
                )
            return f"adopted {stage['kind']} for {experiment_id}"
        if self.service.clock() > datetime.fromisoformat(stage["deadline_at"]):
            self.executor.kill(stage["id"], run_dir)
            return self._close(stage, state, "timeout", None, "stage exceeded its time limit")
        observation = self.executor.observe(stage["id"], run_dir)
        if observation.state == "running":
            return f"waiting on {stage['kind']} for {experiment_id}"
        self.executor.kill(stage["id"], run_dir)  # reap anything the stage left behind
        if observation.state == "lost":
            return self._close(stage, state, "lost", None, "stage process was lost")
        if observation.exit_code != 0:
            return self._close(
                stage,
                state,
                "failed",
                observation.exit_code,
                f"stage exited with {observation.exit_code}",
            )
        return self._complete(stage, state)

    @staticmethod
    def _identity(stage: sqlite3.Row) -> dict[str, str]:
        return {
            "experiment_id": stage["experiment_id"],
            "stage_id": stage["id"],
            "kind": stage["kind"],
        }

    def _verify_training(self, train: sqlite3.Row) -> str | None:
        """The trained model must still match its seal and recorded digest."""
        run_dir = Path(train["run_dir"])
        try:
            self.service.sealer.verify(run_dir, identity=self._identity(train))
        except SealError as exc:
            return f"trained model changed after sealing: {exc}"
        if directory_digest(run_dir / "outputs" / "model") != train["output_digest"]:
            return "trained model changed after sealing"
        return None

    def _cost(self, stage: sqlite3.Row, state: CampaignState) -> float:
        profile = state.profiles.get(stage["profile"])
        started = datetime.fromisoformat(stage["created_at"])
        hours = max(0.0, (self.service.clock() - started).total_seconds() / 3600)
        return hours * profile.cost_per_hour if profile else 0.0

    def _complete(self, stage: sqlite3.Row, state: CampaignState) -> str:
        run_dir = Path(stage["run_dir"])
        try:
            if not (run_dir / SEAL_FILENAME).exists():
                self.service.sealer.seal(run_dir, identity=self._identity(stage))
            self.service.sealer.verify(run_dir, identity=self._identity(stage))
        except SealError as exc:
            return self._close(stage, state, "invalid", 0, f"outputs could not be sealed: {exc}")
        if stage["kind"] == "train":
            model = run_dir / "outputs" / "model"
            if not model.is_dir():
                return self._close(stage, state, "invalid", 0, "training produced no outputs/model")
            with self.store.transaction() as db:
                self.store.cas_update(
                    db,
                    "stage_runs",
                    stage["id"],
                    stage["version"],
                    status="succeeded",
                    exit_code=0,
                    output_digest=directory_digest(model),
                    cost=self._cost(stage, state),
                )
            return f"trained {stage['experiment_id']}"
        return self._evaluate(stage, state)

    def _evaluate(self, stage: sqlite3.Row, state: CampaignState) -> str:
        experiment_id = stage["experiment_id"]
        context = EvalContext.model_validate_json(stage["context_json"])
        train = self.store.read_one(
            "SELECT * FROM stage_runs WHERE experiment_id = ? AND kind = 'train'", (experiment_id,)
        )
        if train is not None:
            reason = self._verify_training(train)
            if reason:
                return self._close(stage, state, "invalid", 0, reason)
        try:
            evidence = read_evidence(Path(stage["run_dir"]) / "outputs", context)
        except EvidenceError as exc:
            return self._close(stage, state, "invalid", 0, str(exc))
        experiment = self.store.read_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        try:
            comparison = decide(
                state.spec,
                role=experiment["role"],
                incumbent=state.incumbent_evidence,
                result=evidence,
                seed=_seed(experiment_id),
                baseline=state.baseline_evidence,
            )
        except EvaluationMismatch as exc:
            return self._close(stage, state, "invalid", 0, str(exc))
        return self._close(stage, state, "succeeded", 0, None, comparison, evidence)

    # ----- recording -----------------------------------------------------------

    def _close(
        self,
        stage: sqlite3.Row,
        state: CampaignState,
        status: str,
        exit_code: int | None,
        reason: str | None,
        comparison: Comparison | None = None,
        evidence: EvalEvidence | None = None,
    ) -> str:
        """Close a stage and record its experiment's result in one transaction."""
        if comparison is None:
            decision = "crash" if status == "failed" else "incomplete"
            comparison = Comparison(decision=decision, reason=reason or status)
        with self.store.transaction() as db:
            self.store.cas_update(
                db,
                "stage_runs",
                stage["id"],
                stage["version"],
                status=status,
                exit_code=exit_code,
                reason=reason,
                cost=self._cost(stage, state),
            )
            self._record_in(db, stage["experiment_id"], comparison, evidence)
        return f"{stage['experiment_id']}: {comparison.decision}"

    def _finish(self, experiment_id: str, decision: str, reason: str) -> str:
        with self.store.transaction() as db:
            self._record_in(db, experiment_id, Comparison(decision=decision, reason=reason), None)
        return f"{experiment_id}: {decision}"

    def _abort(self, experiment_id: str, reason: str) -> str:
        """End an experiment as incomplete, closing any stage that is still open."""
        with self.store.transaction() as db:
            if db.execute(
                "SELECT 1 FROM results WHERE experiment_id = ?", (experiment_id,)
            ).fetchone():
                return f"{experiment_id}: already decided"
            for stage in db.execute(
                "SELECT * FROM stage_runs WHERE experiment_id = ? AND status IN"
                " ('launching', 'running')",
                (experiment_id,),
            ).fetchall():
                self.executor.kill(stage["id"], Path(stage["run_dir"]))
                self.store.cas_update(
                    db, "stage_runs", stage["id"], stage["version"], status="aborted", reason=reason
                )
            self._record_in(
                db, experiment_id, Comparison(decision="incomplete", reason=reason), None
            )
        return f"{experiment_id}: incomplete"

    def _record_in(
        self,
        db: sqlite3.Connection,
        experiment_id: str,
        comparison: Comparison,
        evidence: EvalEvidence | None,
    ) -> None:
        status = {"crash": "crashed", "incomplete": "incomplete"}.get(
            comparison.decision, "evaluated"
        )
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
