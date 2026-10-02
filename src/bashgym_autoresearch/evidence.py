"""Evaluation context issued by the platform and evidence returned by an evaluation stage."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, ValidationError

from bashgym_autoresearch.contracts import (
    EvalEvidence,
    FrozenModel,
    Sha256,
    canonical_hash,
    canonical_json,
)
from bashgym_autoresearch.seal import hash_file

EVIDENCE_FILENAME = "evaluation.json"
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024


class EvidenceError(ValueError):
    """Evaluation evidence is missing, malformed, or bound to another context."""


class EvalContext(FrozenModel):
    """What the evaluator was asked to evaluate; its digest must be echoed in the evidence."""

    campaign_id: str
    experiment_id: str
    suite_id: str
    dataset_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    model_digest: Sha256 | None = None
    nonce: str | None = Field(default=None, max_length=64)

    def digest(self) -> Sha256:
        return canonical_hash(self)


def write_context(path: Path, context: EvalContext) -> None:
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_bytes(canonical_json(context))
    os.replace(temporary, path)


def read_evidence(outputs_dir: Path, context: EvalContext) -> EvalEvidence:
    path = Path(outputs_dir) / EVIDENCE_FILENAME
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"{EVIDENCE_FILENAME} is missing")
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise EvidenceError(f"{EVIDENCE_FILENAME} is too large")
    try:
        evidence = EvalEvidence.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise EvidenceError(f"invalid evidence: {exc.error_count()} error(s)") from exc
    if evidence.context_sha256 != context.digest():
        raise EvidenceError("evidence was produced for a different evaluation context")
    return evidence


def directory_digest(root: Path) -> Sha256:
    """Content digest of a directory tree: sorted (relative path, sha256, size)."""
    items = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest, size = hash_file(path)
            items.append([path.relative_to(root).as_posix(), digest, size])
    return canonical_hash(items)
