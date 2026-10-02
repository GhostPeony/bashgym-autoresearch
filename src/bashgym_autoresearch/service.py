"""Campaign rules. Every agent and human operation goes through this layer."""

from __future__ import annotations

import json
import math
import secrets
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bashgym_autoresearch.auth import Principal, Role, create_token, require_human
from bashgym_autoresearch.contracts import (
    TERMINAL_STATUSES,
    ApprovalKind,
    CampaignSpec,
    Change,
    EvalEvidence,
    ExperimentRole,
    StageProfile,
    canonical_hash,
    canonical_json,
)
from bashgym_autoresearch.decision import check_single_change, next_action, validate_change
from bashgym_autoresearch.seal import Sealer, hash_file
from bashgym_autoresearch.store import Store, utc_now

MAX_RECIPE_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 4 * 1024
MAX_HYPOTHESIS_CHARS = 4000
STDERR_TAIL_BYTES = 4096
ACTIVE_EXPERIMENT_STATUSES = ("queued", "running")


class NotFound(LookupError):
    """The referenced object does not exist."""


class RuleError(ValueError):
    """The request is valid but not allowed in the current state."""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class CampaignState:
    """Derived state used by the next-action rule and the brief."""

    def __init__(self, db: sqlite3.Connection, campaign: sqlite3.Row):
        self.campaign = campaign
        self.spec = CampaignSpec.model_validate_json(campaign["spec_json"])
        rows = db.execute(
            "SELECT e.*, r.decision, r.comparison_json, r.evidence_json FROM experiments e"
            " LEFT JOIN results r ON r.experiment_id = e.id WHERE e.campaign_id = ? ORDER BY e.seq",
            (campaign["id"],),
        ).fetchall()
        self.experiments = rows
        self.active = next(
            (row for row in rows if row["status"] in ACTIVE_EXPERIMENT_STATUSES), None
        )
        measured = {
            row["experiment_id"]: row["cost"]
            for row in db.execute(
                "SELECT s.experiment_id, SUM(s.cost) AS cost FROM stage_runs s"
                " JOIN experiments e ON e.id = s.experiment_id WHERE e.campaign_id = ?"
                " GROUP BY s.experiment_id",
                (campaign["id"],),
            ).fetchall()
        }
        # The platform charges whichever is larger: the agent's estimate or measured cost.
        self.cost_used = sum(
            max(row["estimated_cost"], measured.get(row["id"], 0.0)) for row in rows
        )
        self.counted = sum(1 for row in rows if row["decision"] not in (None, "incomplete"))
        self.incomplete = sum(1 for row in rows if row["decision"] == "incomplete")
        self.has_baseline = any(row["decision"] == "baseline" for row in rows)
        self.incumbent = next(
            (row for row in reversed(rows) if row["decision"] in ("baseline", "keep")), None
        )
        self.incumbent_evidence = (
            EvalEvidence.model_validate_json(self.incumbent["evidence_json"])
            if self.incumbent is not None
            else None
        )
        baseline = next((row for row in rows if row["decision"] == "baseline"), None)
        self.baseline_evidence = (
            EvalEvidence.model_validate_json(baseline["evidence_json"]) if baseline else None
        )
        self.profiles = {
            name: StageProfile.model_validate(profile)
            for name, profile in json.loads(campaign["profiles_json"]).items()
        }

    @property
    def best_value(self) -> float | None:
        if self.incumbent_evidence is None:
            return None
        return self.incumbent_evidence.metrics.get(self.spec.primary.name)

    def next_action(self, now: datetime):
        return next_action(
            self.spec,
            self.campaign["status"],
            has_baseline=self.has_baseline,
            running=self.active is not None,
            experiments_counted=self.counted,
            cost_used=self.cost_used,
            best_value=self.best_value,
            now=now,
            incomplete_count=self.incomplete,
        )


def _result_summary(row: sqlite3.Row) -> dict[str, Any]:
    comparison = json.loads(row["comparison_json"]) if row["comparison_json"] else {}
    evidence = json.loads(row["evidence_json"]) if row["evidence_json"] else None
    return {
        "experiment_id": row["id"],
        "seq": row["seq"],
        "role": row["role"],
        "status": row["status"],
        "change": json.loads(row["change_json"]) if row["change_json"] else None,
        "hypothesis": row["hypothesis"],
        "decision": row["decision"],
        "improvement": comparison.get("improvement"),
        "ci_low": comparison.get("ci_low"),
        "ci_high": comparison.get("ci_high"),
        "breached_gate": comparison.get("breached_gate"),
        "reason": comparison.get("reason"),
        "metrics": evidence["metrics"] if evidence else None,
        "scope": evidence["scope"] if evidence else None,
    }


class Service:
    def __init__(
        self,
        store: Store,
        sealer: Sealer,
        home: Path,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self.store = store
        self.sealer = sealer
        self.home = Path(home)
        self.clock = clock

    # ----- lookups -------------------------------------------------------------

    def campaign_row(self, db: sqlite3.Connection, campaign_id: str) -> sqlite3.Row:
        row = db.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        if row is None:
            raise NotFound(f"campaign {campaign_id} not found")
        return row

    def profile(self, name: str) -> StageProfile:
        row = self.store.read_one("SELECT json FROM profiles WHERE name = ?", (name,))
        if row is None:
            raise NotFound(f"stage profile {name} not found")
        return StageProfile.model_validate_json(row["json"])

    def state(self, campaign_id: str) -> CampaignState:
        with self.store.transaction() as db:
            return CampaignState(db, self.campaign_row(db, campaign_id))

    # ----- human setup ---------------------------------------------------------

    def create_token(self, principal: Principal, role: Role, label: str) -> dict:
        require_human(principal)
        if not label or len(label) > 100:
            raise RuleError("token label must be 1-100 characters")
        return {"token": create_token(self.store, role, label), "role": role, "label": label}

    def register_profile(self, principal: Principal, profile: StageProfile) -> dict:
        require_human(principal)
        script = Path(profile.script)
        if not script.is_absolute() or not script.is_file():
            raise RuleError("profile script must be an existing absolute file path")
        digest, _ = hash_file(script)
        if digest != profile.script_sha256:
            raise RuleError("profile script sha256 does not match the file")
        with self.store.transaction() as db:
            db.execute(
                "INSERT INTO profiles(name, json) VALUES (?, ?)"
                " ON CONFLICT(name) DO UPDATE SET json = excluded.json",
                (profile.name, profile.model_dump_json()),
            )
        return profile.model_dump(mode="json")

    def create_campaign(
        self, principal: Principal, spec: CampaignSpec, idempotency_key: str
    ) -> dict:
        require_human(principal)
        evaluation = self.profile(spec.evaluation.profile)
        if evaluation.kind != "evaluate":
            raise RuleError("evaluation profile must have kind 'evaluate'")
        profiles = {evaluation.name: evaluation.model_dump(mode="json")}
        if spec.train_profile is not None:
            train = self.profile(spec.train_profile)
            if train.kind != "train":
                raise RuleError("train profile must have kind 'train'")
            profiles[train.name] = train.model_dump(mode="json")

        def compute(db: sqlite3.Connection) -> dict:
            campaign_id = _new_id("cmp")
            now = utc_now()
            # Profiles are snapshotted so later re-registration cannot change how this
            # campaign's baseline and candidates are trained or graded.
            db.execute(
                "INSERT INTO campaigns(id, status, version, spec_json, profiles_json, guidance,"
                " guidance_version, created_at, updated_at)"
                " VALUES (?, 'awaiting_start', 1, ?, ?, '', 0, ?, ?)",
                (campaign_id, spec.model_dump_json(), json.dumps(profiles), now, now),
            )
            self.store.append_event(db, campaign_id, "campaign_created", {"by": principal.label})
            return {"campaign_id": campaign_id, "status": "awaiting_start"}

        return self.store.remember(
            f"create_campaign:{principal.label}:{idempotency_key}",
            compute,
            fingerprint=canonical_hash(spec),
        )

    def list_campaigns(self, principal: Principal) -> list[dict]:
        rows = self.store.read(
            "SELECT id, status, spec_json, created_at FROM campaigns ORDER BY created_at"
        )
        return [
            {
                "campaign_id": row["id"],
                "name": json.loads(row["spec_json"])["name"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def set_guidance(self, principal: Principal, campaign_id: str, text: str) -> dict:
        require_human(principal)
        if len(text) > 20_000:
            raise RuleError("guidance is limited to 20000 characters")
        with self.store.transaction() as db:
            row = self.campaign_row(db, campaign_id)
            db.execute(
                "UPDATE campaigns SET guidance = ?, guidance_version = guidance_version + 1,"
                " version = version + 1, updated_at = ? WHERE id = ?",
                (text, utc_now(), campaign_id),
            )
            version = row["guidance_version"] + 1
            self.store.append_event(db, campaign_id, "guidance_updated", {"version": version})
        return {"guidance_version": version}

    # ----- approvals -----------------------------------------------------------

    def request_approval(
        self,
        principal: Principal,
        campaign_id: str,
        kind: ApprovalKind,
        payload: dict,
        idempotency_key: str,
    ) -> dict:
        if kind not in ("start", "budget", "promote", "publish"):
            raise RuleError(f"unknown approval kind {kind!r}")
        try:
            encoded_payload = canonical_json(payload)
        except ValueError as exc:
            raise RuleError("approval payload must be finite JSON") from exc
        if len(encoded_payload) > MAX_PAYLOAD_BYTES:
            raise RuleError("approval payload is limited to 4 KiB of JSON")

        def compute(db: sqlite3.Connection) -> dict:
            campaign = self.campaign_row(db, campaign_id)
            if kind == "start" and campaign["status"] != "awaiting_start":
                raise RuleError("campaign is not awaiting a start approval")
            if kind == "budget":
                amount = payload.get("amount")
                if (
                    isinstance(amount, bool)
                    or not isinstance(amount, (int, float))
                    or not math.isfinite(amount)
                    or amount <= 0
                ):
                    raise RuleError("budget approvals need a positive, finite 'amount'")
            if kind in ("promote", "publish"):
                self._eligible_for_release(db, campaign_id, payload.get("experiment_id"))
            approval_id = _new_id("apr")
            now = utc_now()
            db.execute(
                "INSERT INTO approvals(id, campaign_id, kind, status, payload_json, requested_by,"
                " version, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?, 1, ?, ?)",
                (approval_id, campaign_id, kind, json.dumps(payload), principal.label, now, now),
            )
            self.store.append_event(
                db, campaign_id, "approval_requested", {"approval_id": approval_id, "kind": kind}
            )
            return {"approval_id": approval_id, "kind": kind, "status": "pending"}

        return self.store.remember(
            f"approval:{campaign_id}:{principal.label}:{idempotency_key}",
            compute,
            fingerprint=canonical_hash({"kind": kind, "payload": payload}),
        )

    def _eligible_for_release(
        self, db: sqlite3.Connection, campaign_id: str, experiment_id: Any
    ) -> None:
        row = db.execute(
            "SELECT r.decision, r.evidence_json FROM experiments e JOIN results r"
            " ON r.experiment_id = e.id WHERE e.id = ? AND e.campaign_id = ?",
            (experiment_id, campaign_id),
        ).fetchone()
        if row is None or row["decision"] != "keep":
            raise RuleError("only a kept result can be promoted or published")
        if json.loads(row["evidence_json"])["scope"] != "development":
            raise RuleError("smoke-scope results cannot be promoted or published")

    def decide_approval(self, principal: Principal, approval_id: str, grant: bool) -> dict:
        require_human(principal)
        with self.store.transaction() as db:
            approval = db.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
            if approval is None:
                raise NotFound(f"approval {approval_id} not found")
            if approval["status"] != "pending":
                raise RuleError(f"approval is already {approval['status']}")
            campaign = self.campaign_row(db, approval["campaign_id"])
            payload = json.loads(approval["payload_json"])
            if grant:
                self._apply_grant(db, campaign, approval["kind"], payload)
            self.store.cas_update(
                db,
                "approvals",
                approval_id,
                approval["version"],
                status="granted" if grant else "denied",
                decided_by=principal.label,
            )
            self.store.append_event(
                db,
                campaign["id"],
                "approval_decided",
                {"approval_id": approval_id, "kind": approval["kind"], "granted": grant},
            )
        return {"approval_id": approval_id, "status": "granted" if grant else "denied"}

    def _apply_grant(
        self, db: sqlite3.Connection, campaign: sqlite3.Row, kind: str, payload: dict
    ) -> None:
        if kind == "start":
            if campaign["status"] != "awaiting_start":
                raise RuleError("campaign is not awaiting a start approval")
            self.store.cas_update(
                db, "campaigns", campaign["id"], campaign["version"], status="running"
            )
        elif kind == "budget":
            spec = CampaignSpec.model_validate_json(campaign["spec_json"])
            stop = spec.stop.model_copy(update={"max_cost": spec.stop.max_cost + payload["amount"]})
            updated = CampaignSpec.model_validate({**spec.model_dump(), "stop": stop.model_dump()})
            # A campaign that stopped on its budget resumes once more budget is granted;
            # any other stop rule that still holds stops it again on the next worker tick.
            status = "running" if campaign["status"] == "exhausted" else campaign["status"]
            self.store.cas_update(
                db,
                "campaigns",
                campaign["id"],
                campaign["version"],
                spec_json=updated.model_dump_json(),
                status=status,
            )
        else:
            self._eligible_for_release(db, campaign["id"], payload.get("experiment_id"))

    # ----- agent operations ----------------------------------------------------

    def propose(
        self,
        principal: Principal,
        campaign_id: str,
        *,
        role: ExperimentRole,
        change: Change | None,
        recipe: dict,
        hypothesis: str,
        estimated_cost: float,
        idempotency_key: str,
    ) -> dict:
        validate_change(role, change)
        try:
            encoded_recipe = canonical_json(recipe)
        except ValueError as exc:
            raise RuleError("recipe must be finite JSON") from exc
        if len(encoded_recipe) > MAX_RECIPE_BYTES:
            raise RuleError("recipe is limited to 64 KiB of JSON")
        if not hypothesis.strip() or len(hypothesis) > MAX_HYPOTHESIS_CHARS:
            raise RuleError("a hypothesis of at most 4000 characters is required")
        if not (math.isfinite(estimated_cost) and estimated_cost >= 0):
            raise RuleError("estimated_cost must be a non-negative number")

        def compute(db: sqlite3.Connection) -> dict:
            state = CampaignState(db, self.campaign_row(db, campaign_id))
            action = state.next_action(self.clock())
            expected = "propose_baseline" if role == "baseline" else "propose_candidate"
            if action.kind != expected:
                raise RuleError(f"cannot propose a {role} now: {action.reason}")
            if role == "candidate":
                parent_recipe = json.loads(state.incumbent["recipe_json"])
                check_single_change(parent_recipe, recipe, change)
            if state.cost_used + estimated_cost > state.spec.stop.max_cost:
                raise RuleError(
                    "proposal would exceed the campaign budget; request a budget approval"
                )
            experiment_id = _new_id("exp")
            now = utc_now()
            db.execute(
                "INSERT INTO experiments(id, campaign_id, seq, role, parent_id, status,"
                " change_json, recipe_json, hypothesis, estimated_cost, version, created_at,"
                " updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, 1, ?, ?)",
                (
                    experiment_id,
                    campaign_id,
                    len(state.experiments) + 1,
                    role,
                    state.incumbent["id"] if role == "candidate" else None,
                    change.model_dump_json() if change else None,
                    json.dumps(recipe, sort_keys=True),
                    hypothesis,
                    estimated_cost,
                    now,
                    now,
                ),
            )
            self.store.append_event(
                db,
                campaign_id,
                "experiment_proposed",
                {"experiment_id": experiment_id, "role": role},
            )
            return {"experiment_id": experiment_id, "status": "queued"}

        return self.store.remember(
            f"propose:{campaign_id}:{principal.label}:{idempotency_key}",
            compute,
            fingerprint=canonical_hash(
                {
                    "role": role,
                    "change": change.model_dump(mode="json") if change else None,
                    "recipe": recipe,
                    "hypothesis": hypothesis,
                    "estimated_cost": estimated_cost,
                }
            ),
        )

    def brief(self, principal: Principal, campaign_id: str) -> dict:
        with self.store.transaction() as db:
            state = CampaignState(db, self.campaign_row(db, campaign_id))
            approvals = db.execute(
                "SELECT id, kind, payload_json, created_at FROM approvals"
                " WHERE campaign_id = ? AND status = 'pending' ORDER BY created_at",
                (campaign_id,),
            ).fetchall()
            last_seq = db.execute(
                "SELECT COALESCE(MAX(seq), 0) AS seq FROM events WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()["seq"]
        action = state.next_action(self.clock())
        spec, campaign = state.spec, state.campaign
        evaluated = [row for row in state.experiments if row["decision"] is not None]
        scopes = {
            json.loads(row["evidence_json"])["scope"] for row in evaluated if row["evidence_json"]
        }
        return {
            "campaign_id": campaign_id,
            "name": spec.name,
            "objective": spec.objective,
            "status": campaign["status"],
            "next_action": action.model_dump(),
            "primary_metric": spec.primary.model_dump(),
            "minimum_improvement": spec.minimum_improvement,
            "incumbent": (
                {
                    "experiment_id": state.incumbent["id"],
                    "value": state.best_value,
                    "scope": state.incumbent_evidence.scope,
                }
                if state.incumbent is not None
                else None
            ),
            "active_experiment": (
                {"experiment_id": state.active["id"], "status": state.active["status"]}
                if state.active is not None
                else None
            ),
            "recent_results": [_result_summary(row) for row in evaluated[-5:]],
            "budget": {"used": state.cost_used, "max": spec.stop.max_cost},
            "experiments": {"counted": state.counted, "max": spec.stop.max_experiments},
            "guidance": {"text": campaign["guidance"], "version": campaign["guidance_version"]},
            "pending_approvals": [
                {
                    "approval_id": row["id"],
                    "kind": row["kind"],
                    "payload": json.loads(row["payload_json"]),
                }
                for row in approvals
            ],
            "smoke_only": bool(scopes) and scopes == {"smoke"},
            "event_seq": last_seq,
        }

    def results(self, principal: Principal, campaign_id: str) -> list[dict]:
        state = self.state(campaign_id)
        return [_result_summary(row) for row in state.experiments if row["decision"] is not None]

    def failures(self, principal: Principal, campaign_id: str) -> dict:
        state = self.state(campaign_id)
        failed = next(
            (
                row
                for row in reversed(state.experiments)
                if row["decision"] in ("crash", "incomplete")
            ),
            None,
        )
        if failed is None:
            return {"experiment_id": None, "stages": []}
        stages = self.store.read(
            "SELECT kind, profile, status, exit_code, reason, run_dir FROM stage_runs"
            " WHERE experiment_id = ? ORDER BY created_at",
            (failed["id"],),
        )
        return {
            "experiment_id": failed["id"],
            "decision": failed["decision"],
            "reason": json.loads(failed["comparison_json"]).get("reason"),
            "stages": [
                {
                    "kind": stage["kind"],
                    "profile": stage["profile"],
                    "status": stage["status"],
                    "exit_code": stage["exit_code"],
                    "reason": stage["reason"],
                    # Evaluation stderr can reveal task contents or answers; only
                    # training output is shown to the agent.
                    "stderr_tail": (
                        _tail(Path(stage["run_dir"]) / "stderr.log")
                        if stage["kind"] == "train"
                        else "(withheld for evaluation stages)"
                    ),
                }
                for stage in stages
            ],
        }

    def _transition(
        self, principal: Principal, campaign_id: str, allowed: set[str], target: str
    ) -> dict:
        with self.store.transaction() as db:
            campaign = self.campaign_row(db, campaign_id)
            if campaign["status"] not in allowed:
                raise RuleError(f"cannot move a {campaign['status']} campaign to {target}")
            if target == "running" and principal.role != "human":
                paused = db.execute(
                    "SELECT payload_json FROM events WHERE campaign_id = ? AND type = 'paused'"
                    " ORDER BY seq DESC LIMIT 1",
                    (campaign_id,),
                ).fetchone()
                if paused and json.loads(paused["payload_json"]).get("role") == "human":
                    raise RuleError("a campaign paused by a human can only be resumed by a human")
            self.store.cas_update(db, "campaigns", campaign_id, campaign["version"], status=target)
            event = {"running": "resumed", "paused": "paused", "cancelled": "cancelled"}[target]
            self.store.append_event(
                db, campaign_id, event, {"by": principal.label, "role": principal.role}
            )
        return {"campaign_id": campaign_id, "status": target}

    def pause(self, principal: Principal, campaign_id: str) -> dict:
        return self._transition(principal, campaign_id, {"running"}, "paused")

    def resume(self, principal: Principal, campaign_id: str) -> dict:
        return self._transition(principal, campaign_id, {"paused"}, "running")

    def cancel(self, principal: Principal, campaign_id: str) -> dict:
        allowed = {"awaiting_start", "running", "paused"}
        return self._transition(principal, campaign_id, allowed, "cancelled")

    def report(self, principal: Principal, campaign_id: str) -> dict:
        state = self.state(campaign_id)
        approvals = self.store.read(
            "SELECT id, kind, status, requested_by, decided_by FROM approvals WHERE campaign_id = ?"
            " ORDER BY created_at",
            (campaign_id,),
        )
        brief = self.brief(principal, campaign_id)
        return {
            "campaign_id": campaign_id,
            "name": state.spec.name,
            "objective": state.spec.objective,
            "status": state.campaign["status"],
            "spec": json.loads(state.campaign["spec_json"]),
            "incumbent": brief["incumbent"],
            "smoke_only": brief["smoke_only"],
            "experiments": [_result_summary(row) for row in state.experiments],
            "approvals": [dict(row) for row in approvals],
        }

    def wait(
        self, principal: Principal, campaign_id: str, after_seq: int, timeout_s: float
    ) -> dict:
        self.state(campaign_id)
        deadline = time.monotonic() + max(0.0, min(timeout_s, 300.0))
        while True:
            events = self.store.events_after(campaign_id, after_seq, limit=100)
            if events or time.monotonic() >= deadline:
                return {"events": events, "brief": self.brief(principal, campaign_id)}
            time.sleep(0.25)


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - STDERR_TAIL_BYTES))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = ["CampaignState", "NotFound", "RuleError", "Service", "TERMINAL_STATUSES"]


MAX_METRIC_LINES = 2000


class DashboardService:
    """Read-only projections for the web dashboard."""

    def __init__(self, service: Service):
        self.service = service
        self.store = service.store

    def _experiment(self, experiment_id: str) -> sqlite3.Row:
        row = self.store.read_one("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        if row is None:
            raise NotFound(f"experiment {experiment_id} not found")
        return row

    def campaign(self, principal: Principal, campaign_id: str) -> dict:
        brief = self.service.brief(principal, campaign_id)
        state = self.service.state(campaign_id)
        stages: dict[str, list[dict]] = {}
        for row in self.store.read(
            "SELECT s.experiment_id, s.kind, s.profile, s.status, s.exit_code, s.reason, s.cost,"
            " s.created_at, s.updated_at FROM stage_runs s JOIN experiments e"
            " ON e.id = s.experiment_id WHERE e.campaign_id = ? ORDER BY s.created_at",
            (campaign_id,),
        ):
            stages.setdefault(row["experiment_id"], []).append(
                {key: row[key] for key in row.keys() if key != "experiment_id"}
            )
        approvals = self.store.read(
            "SELECT id, kind, status, payload_json, requested_by, decided_by, created_at"
            " FROM approvals WHERE campaign_id = ? ORDER BY created_at",
            (campaign_id,),
        )
        experiments = []
        for row in state.experiments:
            summary = _result_summary(row)
            summary.update(
                created_at=row["created_at"],
                estimated_cost=row["estimated_cost"],
                stages=stages.get(row["id"], []),
            )
            experiments.append(summary)
        return {
            **brief,
            "spec": json.loads(state.campaign["spec_json"]),
            "experiments": experiments,
            "approvals": [
                {**dict(row), "payload": json.loads(row["payload_json"])} for row in approvals
            ],
            "events": self.store.events_after(campaign_id, max(0, brief["event_seq"] - 50)),
        }

    def training_metrics(self, principal: Principal, experiment_id: str, after: int = 0) -> dict:
        """Per-step training metrics written by the training stage, from line ``after`` on."""
        self._experiment(experiment_id)
        stage = self.store.read_one(
            "SELECT run_dir FROM stage_runs WHERE experiment_id = ? AND kind = 'train'",
            (experiment_id,),
        )
        points: list[dict] = []
        line_count = 0
        if stage is not None:
            path = Path(stage["run_dir"]) / "outputs" / "training_metrics.jsonl"
            try:
                with path.open(encoding="utf-8") as handle:
                    for line_count, line in enumerate(handle, start=1):
                        if line_count <= after or len(points) >= MAX_METRIC_LINES:
                            continue
                        try:
                            record = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(record, dict):
                            points.append(
                                {
                                    k: v
                                    for k, v in record.items()
                                    if isinstance(v, (int, float)) and math.isfinite(v)
                                }
                            )
            except OSError:
                pass
        return {"experiment_id": experiment_id, "points": points, "next": after + len(points)}

    def task_results(self, principal: Principal, experiment_id: str) -> dict:
        """Per-task evaluation outcomes. Human-only: they reveal which tasks fail."""
        require_human(principal)
        self._experiment(experiment_id)
        stage = self.store.read_one(
            "SELECT run_dir FROM stage_runs WHERE experiment_id = ? AND kind = 'evaluate'",
            (experiment_id,),
        )
        tasks: list = []
        if stage is not None:
            path = Path(stage["run_dir"]) / "outputs" / "task_results.json"
            try:
                if path.stat().st_size <= 16 * 1024 * 1024:
                    tasks = json.loads(path.read_text(encoding="utf-8")).get("tasks", [])
            except (OSError, ValueError, AttributeError):
                tasks = []
        return {"experiment_id": experiment_id, "tasks": tasks}

    def pending_approvals(self, principal: Principal) -> list[dict]:
        rows = self.store.read(
            "SELECT a.id, a.campaign_id, a.kind, a.payload_json, a.requested_by, a.created_at,"
            " c.spec_json FROM approvals a JOIN campaigns c ON c.id = a.campaign_id"
            " WHERE a.status = 'pending' ORDER BY a.created_at"
        )
        return [
            {
                "approval_id": row["id"],
                "campaign_id": row["campaign_id"],
                "campaign_name": json.loads(row["spec_json"])["name"],
                "kind": row["kind"],
                "payload": json.loads(row["payload_json"]),
                "requested_by": row["requested_by"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
