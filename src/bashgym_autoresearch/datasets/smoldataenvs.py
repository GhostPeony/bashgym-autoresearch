"""Deterministic preparation of pinned SmolDataEnvs data-analysis tasks.

SmolDataEnvs (https://huggingface.co/datasets/FineEnvs/SmolDataEnvs, MIT) asks a
question about real Kaggle data; the model writes one Python program and the last
printed line is graded against a gold answer. The prompt text below is copied
from FineEnvs ``04-smoldataenvs/scripts/rollout.py`` (Apache-2.0) so scores stay
comparable with the published training and evaluation runs.

Unlike HumanEval tasks, the gold answer is the grading input itself: evaluation
rows carry it for the host-side grader, and it never enters the model prompt or
the sandbox. Canaries are gold-printing programs for validating the sandbox and
grader before any model is scored.

``prepare_smoldataenvs`` is pure. ``stage_bucket_prefixes`` downloads task data
from the Hub bucket, which has no revisions, so every staged file is pinned by
its size and SHA-256 in the prepared rows instead.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from bashgym_autoresearch.datasets.coding import PreparedBenchmark, _select, _text

SMOLDATAENVS_SOURCE = "FineEnvs/SmolDataEnvs"
SMOLDATAENVS_REVISION = "6c439f075cf550793b1bfd7b887a41e51f3b23ab"
SMOLDATAENVS_BUCKET = "AdithyaSK/jupyter-agent-kaggle-all"
SMOLDATAENVS_SPLITS = frozenset({"train", "test", "eval"})

SMOLDATAENVS_SYSTEM = (
    "You are a data analyst. You answer questions about CSV files by writing a short "
    "Python program and reading what it prints."
)

SMOLDATAENVS_PROMPT = """{question}

The files are in /home/user/input and your program runs in that directory:
{files}

Write one Python program in a ```python block, then stop.

- Look at the data if you need to, then compute the answer.
- The LAST thing the program prints must be the answer on its own: a bare number
  (no commas or units), a short label, yes/no, or a comma-separated list.
- Keep it under 40 lines. pandas, numpy, scipy, sklearn and statsmodels are installed."""

_NO_FILES = (
    "- No file names were provided by the dataset. The data is still staged in "
    "/home/user/input; list /home/user/input first, for example with "
    "os.listdir('/home/user/input'), then read the discovered files."
)

ListTree = Callable[[str, str, bool], Iterable[Any]]
Download = Callable[[str, list[tuple[Any, Path]]], None]


def smoldataenvs_messages(row: Mapping) -> list[dict[str, str]]:
    """Build the upstream chat prompt; training and evaluation must share it."""
    names = row.get("files") or []
    files = "\n".join(f"- {name}" for name in names) if names else _NO_FILES
    return [
        {"role": "system", "content": SMOLDATAENVS_SYSTEM},
        {
            "role": "user",
            "content": SMOLDATAENVS_PROMPT.format(question=_text(row, "question"), files=files),
        },
    ]


def _tolerance(row: Mapping, key: str) -> float:
    value = row.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("data_analysis_tolerance_invalid")
    return float(value)


def prepare_smoldataenvs(
    rows: Iterable[Mapping],
    *,
    task_ids: Sequence[str],
    split: str,
    staged_files: Mapping[str, Sequence[Mapping]],
) -> PreparedBenchmark:
    """Project selected rows with their staged data manifest and grading fields."""
    if split not in SMOLDATAENVS_SPLITS:
        raise ValueError("data_analysis_split_invalid")
    tasks, canaries = [], []
    for row in _select(rows, task_ids, key="task_id", id_type=str):
        answer = _text(row, "answer")
        if row.get("hf_bucket") != SMOLDATAENVS_BUCKET:
            raise ValueError("data_analysis_bucket_mismatch")
        prefix = _text(row, "bucket_prefix")
        if prefix not in staged_files:
            raise ValueError("data_analysis_prefix_not_staged")
        files = [dict(item) for item in staged_files[prefix]]
        staged_names = {item["path"] for item in files}
        if any(name not in staged_names for name in row.get("files") or []):
            raise ValueError("data_analysis_listed_file_not_staged")
        tasks.append(
            {
                "task_id": row["task_id"],
                "messages": smoldataenvs_messages(row),
                "answer": answer,
                "reward_mode": row.get("reward_mode") or "",
                "atol": _tolerance(row, "atol"),
                "rtol": _tolerance(row, "rtol"),
                "difficulty_tier": _text(row, "difficulty_tier"),
                "data": {"prefix": prefix, "files": files},
                "provenance": {
                    "source": SMOLDATAENVS_SOURCE,
                    "revision": SMOLDATAENVS_REVISION,
                    "split": split,
                },
            }
        )
        canaries.append({"task_id": row["task_id"], "program": f"print({answer!r})"})
    return PreparedBenchmark(tasks, canaries)


def _relative_bucket_path(path: str, prefix: str) -> str:
    relative = path[len(prefix) + 1 :]
    parts = relative.split("/")
    if not relative or any(part in ("", ".", "..") for part in parts) or "\\" in relative:
        raise ValueError("data_analysis_bucket_path_unsafe")
    return str(PurePosixPath(*parts))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_bucket_prefixes(
    prefixes: Iterable[str],
    destination: Path,
    *,
    bucket: str = SMOLDATAENVS_BUCKET,
    max_bytes: int,
    list_tree: ListTree | None = None,
    download: Download | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Download whole task-data prefixes and return a size/SHA-256 manifest.

    The bucket listing is prefix-based, so ``owner__data`` also matches
    ``owner__database``; only paths under ``<prefix>/`` are staged. Existing
    files with the listed size are reused and re-hashed rather than downloaded.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("data_analysis_stage_byte_limit")
    if list_tree is None or download is None:
        from huggingface_hub import download_bucket_files, list_bucket_tree

        list_tree = list_tree or (lambda b, p, r: list_bucket_tree(b, p, recursive=r))
        download = download or (
            lambda b, files: download_bucket_files(b, files, raise_on_missing_files=True)
        )
    plan: dict[str, list[tuple[Any, str]]] = {}
    total = 0
    for prefix in sorted(set(prefixes)):
        if not prefix or "/" in prefix or prefix in (".", ".."):
            raise ValueError("data_analysis_prefix_invalid")
        entries = []
        for item in list_tree(bucket, prefix + "/", True):
            path = getattr(item, "path", "")
            if not path.startswith(prefix + "/") or not hasattr(item, "size"):
                continue
            entries.append((item, _relative_bucket_path(path, prefix)))
            total += int(item.size)
        if not entries:
            raise ValueError("data_analysis_prefix_empty")
        plan[prefix] = sorted(entries, key=lambda entry: entry[1])
    if total > max_bytes:
        raise ValueError("data_analysis_stage_byte_limit")

    root = Path(destination)
    manifest: dict[str, list[dict[str, Any]]] = {}
    for prefix, entries in plan.items():
        pending = [
            (item, root / prefix / relative)
            for item, relative in entries
            if not (
                (root / prefix / relative).is_file()
                and (root / prefix / relative).stat().st_size == int(item.size)
            )
        ]
        if pending:
            download(bucket, pending)
        files = []
        for item, relative in entries:
            local = root / prefix / relative
            if local.is_symlink() or not local.is_file() or local.stat().st_size != int(item.size):
                raise ValueError("data_analysis_staged_size_mismatch")
            files.append({"path": relative, "size": int(item.size), "sha256": _sha256(local)})
        manifest[prefix] = files
    return manifest
