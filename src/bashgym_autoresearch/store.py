"""SQLite persistence: versioned rows, idempotent writes, and a per-campaign event log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS campaigns (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    version INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    profiles_json TEXT NOT NULL DEFAULT '{}',
    guidance TEXT NOT NULL,
    guidance_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    parent_id TEXT,
    status TEXT NOT NULL,
    change_json TEXT,
    recipe_json TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    estimated_cost REAL NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (campaign_id, seq)
);
CREATE TABLE IF NOT EXISTS stage_runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    profile TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    run_dir TEXT NOT NULL,
    deadline_at TEXT NOT NULL,
    exit_code INTEGER,
    reason TEXT,
    output_digest TEXT,
    script_sha256 TEXT NOT NULL,
    context_json TEXT,
    cost REAL NOT NULL DEFAULT 0,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (experiment_id, kind)
);
CREATE TABLE IF NOT EXISTS results (
    experiment_id TEXT PRIMARY KEY REFERENCES experiments(id),
    decision TEXT NOT NULL,
    comparison_json TEXT NOT NULL,
    evidence_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(id),
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    decided_by TEXT,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    hash TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (name TEXT PRIMARY KEY, json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id TEXT NOT NULL,
    type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_campaign ON events(campaign_id, seq);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    response_json TEXT NOT NULL
);
"""

_VERSIONED_TABLES = frozenset({"campaigns", "experiments", "stage_runs", "approvals"})


class ConflictError(RuntimeError):
    """A versioned row changed since it was read."""


class IdempotencyMismatch(ConflictError):
    """An idempotency key was reused for a different request."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)
            db.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One serialized write transaction; rolls back on any exception."""
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def read(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        db = self._connect()
        try:
            return db.execute(sql, params).fetchall()
        finally:
            db.close()

    def read_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self.read(sql, params)
        return rows[0] if rows else None

    @staticmethod
    def cas_update(
        db: sqlite3.Connection, table: str, row_id: str, expected_version: int, **fields: Any
    ) -> None:
        """Update a versioned row only if it still has ``expected_version``."""
        if table not in _VERSIONED_TABLES:
            raise ValueError(f"{table} is not a versioned table")
        assignments = "".join(f"{name} = ?, " for name in fields)
        cursor = db.execute(
            f"UPDATE {table} SET {assignments}version = version + 1, updated_at = ?"
            " WHERE id = ? AND version = ?",
            (*fields.values(), utc_now(), row_id, expected_version),
        )
        if cursor.rowcount != 1:
            raise ConflictError(f"{table} {row_id} changed or does not exist")

    @staticmethod
    def append_event(
        db: sqlite3.Connection, campaign_id: str, event_type: str, payload: dict
    ) -> int:
        cursor = db.execute(
            "INSERT INTO events(campaign_id, type, payload_json, at) VALUES (?, ?, ?, ?)",
            (campaign_id, event_type, json.dumps(payload, sort_keys=True), utc_now()),
        )
        return int(cursor.lastrowid)

    def events_after(self, campaign_id: str, seq: int, *, limit: int = 100) -> list[dict]:
        rows = self.read(
            "SELECT seq, type, payload_json, at FROM events WHERE campaign_id = ? AND seq > ?"
            " ORDER BY seq LIMIT ?",
            (campaign_id, seq, limit),
        )
        return [
            {
                "seq": row["seq"],
                "type": row["type"],
                "payload": json.loads(row["payload_json"]),
                "at": row["at"],
            }
            for row in rows
        ]

    def remember(
        self,
        key: str,
        compute: Callable[[sqlite3.Connection], dict],
        *,
        fingerprint: str = "",
    ) -> dict:
        """Run ``compute`` once per key inside one transaction and replay its response.

        ``fingerprint`` identifies the request; reusing a key for a different
        request raises ``IdempotencyMismatch`` instead of replaying.
        """
        with self.transaction() as db:
            row = db.execute(
                "SELECT fingerprint, response_json FROM idempotency WHERE key = ?", (key,)
            ).fetchone()
            if row is not None:
                if row["fingerprint"] != fingerprint:
                    raise IdempotencyMismatch(
                        "idempotency key was already used for another request"
                    )
                return json.loads(row["response_json"])
            response = compute(db)
            db.execute(
                "INSERT INTO idempotency(key, fingerprint, response_json) VALUES (?, ?, ?)",
                (key, fingerprint, json.dumps(response, sort_keys=True)),
            )
            return response
