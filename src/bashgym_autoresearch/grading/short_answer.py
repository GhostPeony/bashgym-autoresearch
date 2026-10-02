"""Deterministic short-answer grader for SmolDataEnvs-style tasks.

Vendored from the ``grader.py`` file of the FineEnvs/SmolDataEnvs dataset
(https://huggingface.co/datasets/FineEnvs/SmolDataEnvs, MIT license, revision
6c439f075cf550793b1bfd7b887a41e51f3b23ab). The tier order and tolerance rules
are unchanged so scores stay comparable with published FineEnvs results.

Differences from upstream:
- the unused ``question``, ``judge`` and ``judge_model`` parameters are removed;
- the math-verify tier raises when ``math-verify`` is not installed instead of
  silently scoring every symbolic answer as a miss. Callers that intentionally
  grade without it pass ``math_verify=False``.

Tiers, applied in order:
1. exact match after lowercasing and collapsing whitespace;
2. numeric match within tolerance, with a percent/fraction bridge;
3. order-insensitive comma-separated list match, numeric-tolerant per element;
4. math-verify symbolic equivalence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMERIC_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
MATH_VERIFY_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class GradeResult:
    reward: float
    method: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _to_float(text: str) -> float | None:
    if not text:
        return None
    match = _NUMERIC_RE.search(str(text).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _num_close(gold: float, candidate: float, abs_tol: float, rel_tol: float) -> bool:
    difference = abs(gold - candidate)
    return difference <= abs_tol or difference / max(abs(gold), 1e-9) <= rel_tol


def _is_clean_number(text: str) -> bool:
    stripped = (text or "").strip().strip("%$").strip().replace(",", "")
    return bool(_NUMERIC_RE.fullmatch(stripped))


def _element_match(gold: str, candidate: str, abs_tol: float, rel_tol: float) -> bool:
    if _normalize(gold) == _normalize(candidate):
        return True
    gold_value, candidate_value = _to_float(gold), _to_float(candidate)
    if gold_value is not None and candidate_value is not None:
        return _num_close(gold_value, candidate_value, abs_tol, rel_tol)
    return False


def _list_match(gold: str, candidate: str, abs_tol: float, rel_tol: float) -> bool:
    gold_items = [item.strip() for item in gold.split(",") if item.strip()]
    candidate_items = [item.strip() for item in candidate.split(",") if item.strip()]
    if len(gold_items) < 2 or len(gold_items) != len(candidate_items):
        return False
    orderings = (
        (gold_items, candidate_items),
        (sorted(gold_items, key=str.lower), sorted(candidate_items, key=str.lower)),
    )
    return any(
        all(_element_match(g, c, abs_tol, rel_tol) for g, c in zip(golds, candidates, strict=True))
        for golds, candidates in orderings
    )


def _math_verify_match(gold: str, candidate: str) -> bool:
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise RuntimeError(
            "The math-verify grading tier requires the 'math-verify' package; "
            "install it or grade with math_verify=False."
        ) from exc
    try:
        return bool(
            verify(parse(gold), parse(candidate), timeout_seconds=MATH_VERIFY_TIMEOUT_SECONDS)
        )
    except Exception:
        return False


def grade(
    gold: str,
    candidate: str | None,
    *,
    reward_mode: str = "",
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
    math_verify: bool = True,
) -> GradeResult:
    """Grade ``candidate`` against ``gold`` and return a 0/1 reward with its tier."""
    if not gold or candidate is None:
        return GradeResult(0.0, "miss")

    if _normalize(gold) == _normalize(candidate):
        return GradeResult(1.0, "exact")

    if reward_mode in ("numeric", "flexible") or _is_clean_number(gold):
        gold_value, candidate_value = _to_float(gold), _to_float(candidate)
        if gold_value is not None and candidate_value is not None:
            if _num_close(gold_value, candidate_value, abs_tol, rel_tol):
                return GradeResult(1.0, "numeric")
            one_side_fraction = (0 < abs(candidate_value) < 1 <= abs(gold_value)) or (
                0 < abs(gold_value) < 1 <= abs(candidate_value)
            )
            if one_side_fraction and (
                _num_close(gold_value, candidate_value * 100, abs_tol, rel_tol)
                or _num_close(gold_value, candidate_value / 100, abs_tol, rel_tol)
            ):
                return GradeResult(1.0, "numeric_scaled")

    if reward_mode in ("list", "list_csv") or ("," in gold and "," in candidate):
        if _list_match(gold, candidate, abs_tol, rel_tol):
            return GradeResult(1.0, "list")

    if math_verify and _math_verify_match(gold, candidate):
        return GradeResult(1.0, "math_verify")

    return GradeResult(0.0, "miss")
