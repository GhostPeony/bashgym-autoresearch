"""SmolDataEnvs preparation and bucket staging; no network access."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from bashgym_autoresearch.datasets import smoldataenvs as dab


def _row(task_id="t1", prefix="owner__data", files=("data.csv",), **overrides):
    row = {
        "task_id": task_id,
        "question": "How many rows?",
        "answer": "42",
        "reward_mode": "numeric",
        "atol": 0.0,
        "rtol": 0.0,
        "difficulty_tier": "easy",
        "hf_bucket": dab.SMOLDATAENVS_BUCKET,
        "bucket_prefix": prefix,
        "files": list(files),
    }
    row.update(overrides)
    return row


def _staged(prefix="owner__data", names=("data.csv",)):
    return {
        prefix: [
            {"path": name, "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
            for name in names
        ]
    }


def test_messages_match_upstream_prompt_shape():
    messages = dab.smoldataenvs_messages(_row(files=("a.csv", "b.csv")))
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == dab.SMOLDATAENVS_SYSTEM
    user = messages[1]["content"]
    assert user.startswith("How many rows?\n\nThe files are in /home/user/input")
    assert "- a.csv\n- b.csv" in user
    assert "Write one Python program in a ```python block, then stop." in user


def test_messages_without_file_names_tell_model_to_list_inputs():
    user = dab.smoldataenvs_messages(_row(files=()))[1]["content"]
    assert "os.listdir('/home/user/input')" in user


def test_prepare_projects_selected_rows_with_pinned_data_and_separate_canaries():
    rows = [_row("t1"), _row("t2", answer="Male", reward_mode="exact_short")]
    prepared = dab.prepare_smoldataenvs(rows, task_ids=["t2"], split="test", staged_files=_staged())
    assert [task["task_id"] for task in prepared.tasks] == ["t2"]
    task = prepared.tasks[0]
    assert task["answer"] == "Male"
    assert task["reward_mode"] == "exact_short"
    assert task["data"]["prefix"] == "owner__data"
    assert task["data"]["files"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert task["provenance"] == {
        "source": dab.SMOLDATAENVS_SOURCE,
        "revision": dab.SMOLDATAENVS_REVISION,
        "split": "test",
    }
    assert "Male" not in str(task["messages"])
    assert prepared.canaries == [{"task_id": "t2", "program": "print('Male')"}]


@pytest.mark.parametrize(
    ("row", "staged", "error"),
    [
        (_row(files=("missing.csv",)), _staged(), "data_analysis_listed_file_not_staged"),
        (_row(prefix="other"), _staged(), "data_analysis_prefix_not_staged"),
        (_row(hf_bucket="someone/else"), _staged(), "data_analysis_bucket_mismatch"),
        (_row(atol=-1.0), _staged(), "data_analysis_tolerance_invalid"),
        (_row(answer=""), _staged(), "coding_field_invalid:answer"),
    ],
)
def test_prepare_rejects_inconsistent_rows(row, staged, error):
    with pytest.raises(ValueError, match=error):
        dab.prepare_smoldataenvs([row], task_ids=["t1"], split="test", staged_files=staged)


def test_prepare_rejects_unknown_split():
    with pytest.raises(ValueError, match="data_analysis_split_invalid"):
        dab.prepare_smoldataenvs([_row()], task_ids=["t1"], split="dev", staged_files=_staged())


def test_stage_downloads_only_exact_prefix_and_records_digests(tmp_path):
    listed = [
        SimpleNamespace(path="owner__data/data.csv", size=3),
        SimpleNamespace(path="owner__data/sub/notes.txt", size=2),
        SimpleNamespace(path="owner__database/other.csv", size=5),
    ]
    downloads = []

    def list_tree(bucket, prefix, recursive):
        assert (bucket, prefix, recursive) == (dab.SMOLDATAENVS_BUCKET, "owner__data/", True)
        return listed

    def download(bucket, files):
        for remote, local in files:
            downloads.append(remote.path)
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"abc" if remote.path.endswith(".csv") else b"hi")

    manifest = dab.stage_bucket_prefixes(
        ["owner__data"], tmp_path, list_tree=list_tree, download=download, max_bytes=100
    )
    assert downloads == ["owner__data/data.csv", "owner__data/sub/notes.txt"]
    assert manifest == {
        "owner__data": [
            {"path": "data.csv", "size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()},
            {"path": "sub/notes.txt", "size": 2, "sha256": hashlib.sha256(b"hi").hexdigest()},
        ]
    }
    assert (tmp_path / "owner__data" / "sub" / "notes.txt").read_bytes() == b"hi"


def test_stage_rejects_size_mismatch_and_byte_budget(tmp_path):
    listed = [SimpleNamespace(path="p/a.csv", size=3)]

    def bad_download(bucket, files):
        for _, local in files:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"abcd")

    with pytest.raises(ValueError, match="data_analysis_staged_size_mismatch"):
        dab.stage_bucket_prefixes(
            ["p"], tmp_path / "a", list_tree=lambda *a: listed, download=bad_download, max_bytes=100
        )
    with pytest.raises(ValueError, match="data_analysis_stage_byte_limit"):
        dab.stage_bucket_prefixes(
            ["p"], tmp_path / "b", list_tree=lambda *a: listed, download=bad_download, max_bytes=2
        )


@pytest.mark.parametrize("path", ["p/../escape.csv", "p//a.csv", "p/./a.csv"])
def test_stage_rejects_unsafe_bucket_paths(tmp_path, path):
    listed = [SimpleNamespace(path=path, size=1)]
    with pytest.raises(ValueError, match="data_analysis_bucket_path_unsafe"):
        dab.stage_bucket_prefixes(
            ["p"], tmp_path, list_tree=lambda *a: listed, download=lambda *a: None, max_bytes=10
        )
