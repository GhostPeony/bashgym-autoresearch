import json

import pytest

from bashgym_autoresearch.contracts import EvalEvidence, TaskOutcome
from bashgym_autoresearch.evidence import (
    EvalContext,
    EvidenceError,
    directory_digest,
    read_evidence,
    write_context,
)
from bashgym_autoresearch.seal import Sealer, SealError

KEY = b"k" * 32
IDENTITY = {"campaign_id": "c1", "experiment_id": "e1", "stage": "evaluate"}


def make_run(tmp_path, files=None):
    run_dir = tmp_path / "run"
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True)
    for name, data in (files or {"evaluation.json": b"{}", "nested/a.txt": b"a"}).items():
        path = outputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return run_dir


def test_seal_round_trip(tmp_path):
    run_dir = make_run(tmp_path)
    sealed = Sealer(KEY).seal(run_dir, identity=IDENTITY)
    verified = Sealer(KEY).verify(run_dir, identity=IDENTITY)
    assert verified == sealed
    assert {item.path for item in sealed.outputs} == {"evaluation.json", "nested/a.txt"}


@pytest.mark.parametrize("tamper", ["modify", "add", "remove", "key", "identity"])
def test_seal_detects_tampering(tmp_path, tamper):
    run_dir = make_run(tmp_path)
    Sealer(KEY).seal(run_dir, identity=IDENTITY)
    sealer, identity = Sealer(KEY), IDENTITY
    if tamper == "modify":
        (run_dir / "outputs" / "nested" / "a.txt").write_bytes(b"changed")
    elif tamper == "add":
        (run_dir / "outputs" / "extra.txt").write_bytes(b"x")
    elif tamper == "remove":
        (run_dir / "outputs" / "nested" / "a.txt").unlink()
    elif tamper == "key":
        sealer = Sealer(b"z" * 32)
    else:
        identity = {**IDENTITY, "experiment_id": "other"}
    with pytest.raises(SealError):
        sealer.verify(run_dir, identity=identity)


def test_sealer_rejects_short_keys_and_resealing(tmp_path):
    with pytest.raises(ValueError):
        Sealer(b"short")
    run_dir = make_run(tmp_path)
    Sealer(KEY).seal(run_dir, identity=IDENTITY)
    with pytest.raises(SealError):
        Sealer(KEY).seal(run_dir, identity=IDENTITY)


def context():
    return EvalContext(
        campaign_id="c1",
        experiment_id="e1",
        suite_id="suite",
        dataset_sha256="a" * 64,
        evaluator_sha256="b" * 64,
        model_digest=None,
    )


def write_evidence(outputs, ctx, **overrides):
    evidence = EvalEvidence(
        context_sha256=ctx.digest(),
        scope="smoke",
        metrics={"pass_rate": 1.0},
        tasks=(TaskOutcome(task_id="t1", cluster="t1", value=1.0),),
        complete=True,
    ).model_dump(mode="json")
    evidence.update(overrides)
    (outputs / "evaluation.json").write_text(json.dumps(evidence))


def test_evidence_must_echo_the_issued_context(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    ctx = context()
    write_context(tmp_path / "context.json", ctx)
    assert EvalContext.model_validate_json((tmp_path / "context.json").read_bytes()) == ctx
    write_evidence(outputs, ctx)
    assert read_evidence(outputs, ctx).metrics == {"pass_rate": 1.0}
    write_evidence(outputs, ctx, context_sha256="f" * 64)
    with pytest.raises(EvidenceError, match="context"):
        read_evidence(outputs, ctx)


def test_evidence_rejects_missing_invalid_or_oversized_files(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    with pytest.raises(EvidenceError):
        read_evidence(outputs, context())
    (outputs / "evaluation.json").write_text("not json")
    with pytest.raises(EvidenceError):
        read_evidence(outputs, context())
    (outputs / "evaluation.json").write_bytes(b" " * (4 * 1024 * 1024 + 1))
    with pytest.raises(EvidenceError, match="large"):
        read_evidence(outputs, context())


def test_directory_digest_is_order_independent(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for root, names in ((first, ["x", "y/z"]), (second, ["y/z", "x"])):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
    assert directory_digest(first) == directory_digest(second)
    (second / "x").write_bytes(b"different")
    assert directory_digest(first) != directory_digest(second)
