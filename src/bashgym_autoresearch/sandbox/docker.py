"""Fail-closed Docker execution for untrusted model programs.

Simplified from BashGym ``environments/docker_coding.py``. A program runs once in
a fresh container with no network, a read-only root filesystem, an unprivileged
user, no capabilities, and bounded memory, CPU, processes, time, and output.
Images are never pulled; they must already exist locally and be referenced by
digest. Infrastructure problems are reported in ``SandboxResult.error`` rather
than raised, so callers can mark an evaluation incomplete.
"""

from __future__ import annotations

import re
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

WORKSPACE = "/workspace"
_IMAGE_PATTERN = re.compile(
    r"(?:sha256:[0-9a-f]{64}|[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64})"
)
_MEMORY_PATTERN = re.compile(r"[1-9][0-9]*[mg]")


@dataclass(frozen=True)
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    error: str | None = None


def validate_image(image: str) -> str:
    if not isinstance(image, str) or not _IMAGE_PATTERN.fullmatch(image):
        raise ValueError("sandbox image must be pinned by digest (name@sha256:... or sha256:...)")
    return image


def _validate_mounts(mounts: Mapping[str, Path]) -> dict[str, Path]:
    validated = {}
    for target, source in mounts.items():
        path = PurePosixPath(target)
        if (
            not path.is_absolute()
            or path == PurePosixPath("/")
            or ".." in path.parts
            or str(path) != target
            or path == PurePosixPath(WORKSPACE)
            or PurePosixPath(WORKSPACE) in path.parents
        ):
            raise ValueError("read-only mount target must be an absolute path outside /workspace")
        source = Path(source)
        if source.is_symlink() or not source.is_dir():
            raise ValueError("read-only mount source must be an existing, non-linked directory")
        validated[target] = source.resolve()
    return validated


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError(f"unsafe sandbox file name: {name!r}")
    return path


class _Tail:
    """Keep only the last ``limit`` bytes of a stream."""

    def __init__(self, limit: int):
        self.limit = limit
        self.data = bytearray()

    def add(self, chunk: bytes) -> None:
        self.data.extend(chunk)
        if len(self.data) > self.limit:
            del self.data[: len(self.data) - self.limit]

    def text(self) -> str:
        return bytes(self.data).decode("utf-8", errors="replace")


def run_program(
    image: str,
    files: Mapping[str, str],
    argv: Sequence[str],
    *,
    timeout: float,
    workdir: str = WORKSPACE,
    read_only_mounts: Mapping[str, Path] | None = None,
    memory: str = "2g",
    cpus: float = 1.0,
    pids_limit: int = 128,
    max_output: int = 64 * 1024,
    client=None,
) -> SandboxResult:
    validate_image(image)
    mounts = _validate_mounts(read_only_mounts or {})
    if not _MEMORY_PATTERN.fullmatch(memory):
        raise ValueError("memory must look like '512m' or '8g'")
    if not 0 < cpus <= 64 or not 0 < timeout <= 24 * 3600 or not argv:
        raise ValueError("invalid sandbox limits or empty argv")
    relative = {_safe_relative(name): content for name, content in files.items()}

    owns_client = client is None
    try:
        if client is None:
            import docker

            client = docker.from_env(timeout=10)
        client.ping()
        client.images.get(image)
    except Exception as exc:  # docker SDK raises several unrelated types here
        if owns_client and client is not None:
            client.close()
        return SandboxResult(None, "", "", False, f"docker unavailable or image missing: {exc}")

    container = None
    try:
        with tempfile.TemporaryDirectory(prefix="bgar-sandbox-") as workspace:
            root = Path(workspace)
            for path, content in relative.items():
                target = root.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                target.chmod(0o444)
            for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
                directory.chmod(0o755)
            from docker.types import Mount

            volume_mounts = [Mount(target=WORKSPACE, source=str(root), type="bind", read_only=True)]
            volume_mounts += [
                Mount(target=target, source=str(source), type="bind", read_only=True)
                for target, source in sorted(mounts.items())
            ]
            container = client.containers.run(
                image=image,
                command=list(argv),
                name="bgar-sandbox-" + uuid.uuid4().hex,
                working_dir=workdir,
                mounts=volume_mounts,
                user="65534:65534",
                network_mode="none",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                pids_limit=pids_limit,
                mem_limit=memory,
                nano_cpus=int(cpus * 1e9),
                tmpfs={"/tmp": "rw,nosuid,size=256m,mode=1777"},
                environment={
                    "HOME": "/tmp",
                    "MPLCONFIGDIR": "/tmp/matplotlib",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                detach=True,
                stdin_open=False,
                tty=False,
            )
            state: dict = {}

            def wait() -> None:
                try:
                    state["result"] = container.wait()
                except Exception as exc:  # connection errors from the daemon
                    state["error"] = str(exc)

            waiter = threading.Thread(target=wait, daemon=True)
            waiter.start()
            waiter.join(timeout)
            timed_out = waiter.is_alive()
            if timed_out:
                container.kill()
                waiter.join(10)
            stdout, stderr = _Tail(max_output), _Tail(max_output)
            for chunk in container.logs(stdout=True, stderr=False, stream=True):
                stdout.add(chunk)
            for chunk in container.logs(stdout=False, stderr=True, stream=True):
                stderr.add(chunk)
            if "error" in state and not timed_out:
                return SandboxResult(None, stdout.text(), stderr.text(), False, state["error"])
            exit_code = None if timed_out else int(state["result"]["StatusCode"])
            return SandboxResult(exit_code, stdout.text(), stderr.text(), timed_out)
    except Exception as exc:  # container creation or log retrieval failed
        return SandboxResult(None, "", "", False, f"sandbox failure: {exc}")
    finally:
        if container is not None:
            try:
                container.remove(force=True, v=True)
            except Exception:
                pass
        if owns_client:
            client.close()
