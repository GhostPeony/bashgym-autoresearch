"""Evaluation task suites."""

from bashgym_autoresearch.runners.suites.base import Graded, Suite
from bashgym_autoresearch.runners.suites.coding import CodingSuite
from bashgym_autoresearch.runners.suites.data_analysis import DataAnalysisSuite

SUITES: dict[str, type] = {"coding": CodingSuite, "data_analysis": DataAnalysisSuite}

__all__ = ["SUITES", "CodingSuite", "DataAnalysisSuite", "Graded", "Suite"]
