import threading

import pytest

from bashgym_autoresearch.store import ConflictError, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "state.db")


def _insert_campaign(store, campaign_id="c1"):
    with store.transaction() as db:
        db.execute(
            "INSERT INTO campaigns(id, status, version, spec_json, guidance, guidance_version,"
            " created_at, updated_at) VALUES (?, 'awaiting_start', 1, '{}', '', 0, 'now', 'now')",
            (campaign_id,),
        )


def test_schema_creation_is_idempotent(tmp_path):
    Store(tmp_path / "state.db")
    store = Store(tmp_path / "state.db")
    tables = {
        row["name"] for row in store.read("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"campaigns", "experiments", "stage_runs", "results", "approvals", "tokens"} <= tables


def test_cas_update_rejects_stale_version(store):
    _insert_campaign(store)
    with store.transaction() as db:
        store.cas_update(db, "campaigns", "c1", 1, status="running")
    with pytest.raises(ConflictError), store.transaction() as db:
        store.cas_update(db, "campaigns", "c1", 1, status="paused")
    assert store.read_one("SELECT status, version FROM campaigns WHERE id='c1'")["version"] == 2


def test_remember_computes_once_per_key(store):
    calls = []

    def compute(db):
        calls.append(1)
        return {"value": len(calls)}

    first = store.remember("key-1", compute)
    second = store.remember("key-1", compute)
    assert first == second == {"value": 1} and len(calls) == 1


def test_failed_compute_does_not_remember(store):
    def boom(db):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError):
        store.remember("key-2", boom)
    assert store.remember("key-2", lambda db: {"ok": True}) == {"ok": True}


def test_events_are_ordered_and_paged(store):
    _insert_campaign(store)
    with store.transaction() as db:
        for index in range(5):
            store.append_event(db, "c1", "tick", {"i": index})
    events = store.events_after("c1", 0, limit=3)
    assert [event["payload"]["i"] for event in events] == [0, 1, 2]
    rest = store.events_after("c1", events[-1]["seq"], limit=10)
    assert [event["payload"]["i"] for event in rest] == [3, 4]


def test_concurrent_writers_serialize_without_lost_updates(tmp_path):
    path = tmp_path / "state.db"
    Store(path)
    _insert_campaign(Store(path))

    def bump():
        local = Store(path)
        for _ in range(20):
            while True:
                try:
                    with local.transaction() as db:
                        row = db.execute("SELECT version FROM campaigns WHERE id='c1'").fetchone()
                        local.cas_update(db, "campaigns", "c1", row["version"])
                    break
                except ConflictError:
                    continue

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert Store(path).read_one("SELECT version FROM campaigns WHERE id='c1'")["version"] == 81
