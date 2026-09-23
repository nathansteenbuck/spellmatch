"""Integration tests for the Stage 6 pre-flight memory guardrail wired into
Spellmatch._compute_distance_terms (plan_spellmatch.md, matchingwizard repo).

These exercise the real code path used during matching, not just the
underlying _kron_free estimator (see test_kron_free.py for that).
"""

from __future__ import annotations

import numpy as np
import pytest

from spellmatch.matching._kron_free import CorrectionMatrixTooLargeError
from spellmatch.matching.algorithms.spellmatch import Spellmatch


def _make_symmetric_adjacency(rng, n, p_edge):
    a = (rng.random((n, n)) < p_edge).astype(bool)
    a = np.triu(a, 1)
    return a | a.T


@pytest.fixture
def dense_graph_pair():
    rng = np.random.default_rng(2)
    n1, n2 = 150, 130
    adj1 = _make_symmetric_adjacency(rng, n1, 0.3)
    adj2 = _make_symmetric_adjacency(rng, n2, 0.3)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(1, 15, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    return adj1, adj2, dists1, dists2, n1, n2


def test_guardrail_raises_when_budget_too_small(dense_graph_pair):
    adj1, adj2, dists1, dists2, n1, n2 = dense_graph_pair
    algo = Spellmatch(distance_weight=1, max_correction_gib=1e-9)
    with pytest.raises(CorrectionMatrixTooLargeError, match="max_correction_gib"):
        algo._compute_distance_terms(adj1, adj2, dists1, dists2, n1, n2)


def test_guardrail_allows_build_when_budget_is_generous(dense_graph_pair):
    adj1, adj2, dists1, dists2, n1, n2 = dense_graph_pair
    algo = Spellmatch(distance_weight=1, max_correction_gib=32.0)
    d1, d2, correction = algo._compute_distance_terms(
        adj1, adj2, dists1, dists2, n1, n2
    )
    assert d1.shape == (n1, n1)
    assert d2.shape == (n2, n2)
    assert correction.shape == (n1 * n2, n1 * n2)


def test_guardrail_disabled_when_budget_is_none(dense_graph_pair):
    adj1, adj2, dists1, dists2, n1, n2 = dense_graph_pair
    algo = Spellmatch(distance_weight=1, max_correction_gib=None)
    # Should not raise even though a tiny budget would have rejected this.
    algo._compute_distance_terms(adj1, adj2, dists1, dists2, n1, n2)
