"""HMAC sealing of stage outputs, simplified from BashGym ``campaigns/artifacts.py``.

A stage writes files under ``<run_dir>/outputs``. Sealing hashes every file,
binds the manifest to the stage identity, and signs it, so any later change to
the outputs or a replay under another identity fails verification.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

from pydantic import Field

from bashgym_autoresearch.contracts import FrozenModel, canonical_json

SEAL_FILENAME = "seal.json"
SEAL_SCHEMA = "bashgym_autoresearch.seal.v1"


class SealError(RuntimeError):
    """Outputs are missing, changed, or not authenticated."""


class SealedFile(FrozenModel):
    path: str
    sha256: str
    size: int = Field(ge=0)


class SealManifest(FrozenModel):
    identity: dict[str, str]
    outputs: tuple[SealedFile, ...]


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _describe(outputs_dir: Path) -> tuple[SealedFile, ...]:
    if not outputs_dir.is_dir() or outputs_dir.is_symlink():
        raise SealError("outputs directory is missing")
    files = []
    for path in sorted(outputs_dir.rglob("*")):
        if path.is_symlink():
            raise SealError(f"symbolic link in outputs: {path.name}")
        if path.is_file():
            digest, size = hash_file(path)
            files.append(
                SealedFile(path=path.relative_to(outputs_dir).as_posix(), sha256=digest, size=size)
            )
    return tuple(files)


class Sealer:
    def __init__(self, key: bytes):
        if len(key) < 32:
            raise ValueError("seal key must contain at least 32 bytes")
        self._key = key

    def _signature(self, manifest: SealManifest) -> str:
        message = SEAL_SCHEMA.encode() + b"\0" + canonical_json(manifest)
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def seal(self, run_dir: Path, *, identity: dict[str, str]) -> SealManifest:
        seal_path = Path(run_dir) / SEAL_FILENAME
        if seal_path.exists():
            raise SealError("run is already sealed")
        manifest = SealManifest(identity=identity, outputs=_describe(Path(run_dir) / "outputs"))
        envelope = {
            "schema": SEAL_SCHEMA,
            "manifest": manifest.model_dump(mode="json"),
            "signature": self._signature(manifest),
        }
        temporary = seal_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            handle.write(canonical_json(envelope))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, seal_path)
        return manifest

    def verify(self, run_dir: Path, *, identity: dict[str, str]) -> SealManifest:
        try:
            envelope = json.loads((Path(run_dir) / SEAL_FILENAME).read_bytes())
            if envelope.get("schema") != SEAL_SCHEMA:
                raise ValueError("wrong seal schema")
            manifest = SealManifest.model_validate(envelope["manifest"])
            if not hmac.compare_digest(envelope["signature"], self._signature(manifest)):
                raise ValueError("signature mismatch")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SealError(f"invalid seal: {exc}") from exc
        if manifest.identity != identity:
            raise SealError("seal identity mismatch")
        if _describe(Path(run_dir) / "outputs") != manifest.outputs:
            raise SealError("sealed outputs changed")
        return manifest
