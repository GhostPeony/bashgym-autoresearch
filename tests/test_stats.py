"""Tests for the cluster-paired bootstrap (ported from BashGym)."""

import pytest

from bashgym_autoresearch.stats import paired_bootstrap


class TestPairedBootstrap:
    def test_clear_improvement_is_better(self):
        deltas = [0.2] * 30
        clusters = list(range(30))  # 30 distinct sessions
        r = paired_bootstrap(deltas, clusters, seed=1)
        assert r.better and r.significant
        assert r.ci_low > 0
        assert r.mean == pytest.approx(0.2)

    def test_no_difference_is_not_significant(self):
        deltas = [0.0] * 30
        r = paired_bootstrap(deltas, list(range(30)), seed=1)
        assert not r.significant and not r.better

    def test_regression_is_significant_but_not_better(self):
        deltas = [-0.3] * 30
        r = paired_bootstrap(deltas, list(range(30)), seed=1)
        assert r.significant and not r.better
        assert r.ci_high < 0

    def test_deterministic_with_seed(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.15] * 4
        clusters = list(range(20))
        a = paired_bootstrap(deltas, clusters, seed=7)
        b = paired_bootstrap(deltas, clusters, seed=7)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_single_cluster_has_zero_width_ci(self):
        # One session => no resampling variance => CI collapses to the mean.
        deltas = [0.2, 0.2, 0.2]
        r = paired_bootstrap(deltas, ["s1", "s1", "s1"], seed=1)
        assert r.ci_low == r.ci_high == pytest.approx(0.2)
        assert r.n_clusters == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap([0.1, 0.2], ["s1"], seed=1)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            paired_bootstrap([], [], seed=1)
