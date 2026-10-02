"""Gold cases for the vendored SmolDataEnvs short-answer grader."""

from __future__ import annotations

import builtins

import pytest

from bashgym_autoresearch.grading.short_answer import GradeResult, grade


def _grade(gold: str, candidate: str | None, **kwargs) -> GradeResult:
    return grade(gold, candidate, math_verify=False, **kwargs)


@pytest.mark.parametrize(
    ("gold", "candidate", "method"),
    [
        ("Paris", "  paris ", "exact"),
        ("New   York", "new york", "exact"),
        ("42", "42.0004", "numeric"),
        ("1,234", "1234", "numeric"),
        ("96.00", "0.96", "numeric_scaled"),
        ("0.25", "25", "numeric_scaled"),
        ("apple, banana, cherry", "cherry,apple,banana", "list"),
        ("1.5, 2.5", "1.5001, 2.4999", "list"),
    ],
)
def test_accepts_equivalent_answers(gold: str, candidate: str, method: str) -> None:
    assert _grade(gold, candidate) == GradeResult(1.0, method)


@pytest.mark.parametrize(
    ("gold", "candidate"),
    [
        ("42", "43"),
        ("Paris", "London"),
        ("apple, banana", "apple"),
        ("apple, banana", "apple, banana, cherry"),
        ("", "anything"),
        ("42", None),
    ],
)
def test_rejects_wrong_or_missing_answers(gold: str, candidate: str | None) -> None:
    assert _grade(gold, candidate) == GradeResult(0.0, "miss")


def test_numeric_tier_respects_explicit_tolerances() -> None:
    assert _grade("100", "100.5", abs_tol=1.0, rel_tol=0.0).reward == 1.0
    assert _grade("100", "100.5", abs_tol=0.1, rel_tol=0.0).reward == 0.0


def test_reward_mode_enables_numeric_tier_for_unclean_gold() -> None:
    assert _grade("about 12 items", "12", reward_mode="numeric").method == "numeric"
    assert _grade("about 12 items", "12").method == "miss"


def test_math_verify_tier_fails_closed_when_dependency_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def _no_math_verify(name, *args, **kwargs):
        if name == "math_verify":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_math_verify)
    with pytest.raises(RuntimeError, match="math-verify"):
        grade("x^2", "x**2")
