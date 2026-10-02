"""Cluster-paired bootstrap for comparing a candidate with its incumbent.

A candidate is better only if its per-task advantage survives resampling. Tasks
in the same cluster are not independent, so whole clusters are resampled, as in
"Adding Error Bars to Evals" (arXiv:2411.00640). Ported from BashGym
``eval/stats.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int
    n_clusters: int

    @property
    def significant(self) -> bool:
        """The confidence interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    @property
    def better(self) -> bool:
        """The whole confidence interval is above zero."""
        return self.ci_low > 0


def paired_bootstrap(
    deltas: list[float],
    clusters: list,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Cluster bootstrap of aligned paired deltas (candidate minus incumbent).

    Each resample draws ``n_clusters`` clusters with replacement and pools their
    deltas; the interval is the ``alpha/2 .. 1-alpha/2`` percentiles of the
    resampled means.
    """
    if len(deltas) != len(clusters):
        raise ValueError("deltas and clusters must be the same length")
    if not deltas:
        raise ValueError("no data to bootstrap")

    by_cluster: dict = {}
    for delta, cluster in zip(deltas, clusters, strict=True):
        by_cluster.setdefault(cluster, []).append(delta)
    cluster_keys = list(by_cluster)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        pooled: list[float] = []
        for _ in range(len(cluster_keys)):
            pooled.extend(by_cluster[rng.choice(cluster_keys)])
        means.append(sum(pooled) / len(pooled))
    means.sort()

    low = means[int((alpha / 2) * n_resamples)]
    high = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    observed = sum(deltas) / len(deltas)
    return BootstrapResult(
        mean=observed, ci_low=low, ci_high=high, n=len(deltas), n_clusters=len(cluster_keys)
    )
