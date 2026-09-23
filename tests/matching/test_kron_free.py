"""Numerical correctness tests for spellmatch.matching._kron_free.

Ports the verification checks from plan_spellmatch.md (matchingwizard repo)
into permanent tests: every identity the matrix-free reformulation of
Spellmatch's distance term relies on is checked against a dense,
brute-force ground truth, not just asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from spellmatch.matching._kron_free import (
    build_distance_correction,
    correction_matvec,
    distance_term_matvec,
    estimate_correction_bytes,
    estimate_correction_nnz,
    kron_matvec,
    outer_sum_matvec,
)


def _make_symmetric_adjacency(rng, n, p_edge):
    a = (rng.random((n, n)) < p_edge).astype(float)
    a = np.triu(a, 1)
    return a + a.T


@pytest.fixture
def small_graphs():
    rng = np.random.default_rng(0)
    n1, n2 = 40, 55
    a1 = _make_symmetric_adjacency(rng, n1, 0.15)
    a2 = _make_symmetric_adjacency(rng, n2, 0.15)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(1, 15, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    d1 = a1 * dists1
    d2 = a2 * dists2
    return rng, n1, n2, a1, a2, d1, d2


def test_kron_matvec_matches_dense_kronecker_product(small_graphs):
    rng, n1, n2, a1, a2, _, _ = small_graphs
    s = rng.random(n1 * n2)
    expected = sparse.kron(a1, a2).toarray() @ s
    actual = kron_matvec(a1, a2, s, n1, n2)
    assert np.allclose(expected, actual)


def test_kron_matvec_handles_rectangular_smaller_or_larger_second_graph():
    # n1 != n2 in both directions -- guards against an accidental square-only
    # reshape/transpose bug (see plan_spellmatch.md's "must A be square?").
    rng = np.random.default_rng(1)
    for n1, n2 in [(5, 30), (30, 5)]:
        a1 = _make_symmetric_adjacency(rng, n1, 0.3)
        a2 = _make_symmetric_adjacency(rng, n2, 0.3)
        s = rng.random(n1 * n2)
        expected = sparse.kron(a1, a2).toarray() @ s
        actual = kron_matvec(a1, a2, s, n1, n2)
        assert np.allclose(expected, actual)


def test_outer_sum_matvec_matches_dense_broadcast(small_graphs):
    rng, n1, n2, a1, a2, _, _ = small_graphs
    s = rng.random(n1 * n2)
    u = rng.random(n1 * n2)
    adj = sparse.kron(a1, a2).toarray()
    # This is exactly how the original code builds the degree/intensity
    # contribution to w_csr: adj * u[:,None] + adj * u[None,:].
    expected = (adj * (u[:, None] + u[None, :])) @ s
    actual = outer_sum_matvec(a1, a2, u, s, n1, n2)
    assert np.allclose(expected, actual)


def test_distance_term_matvec_matches_exact_clipped_ground_truth(small_graphs):
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    s = rng.random(n1 * n2)
    thresh = 3.0  # small threshold to exercise the clip / correction path

    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    y_true = np.clip((diff / thresh) ** 2, 0, 1)
    expected = y_true @ s

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2, chunk_size=7
    )
    actual = distance_term_matvec(a1, a2, d1, d2, thresh, correction, s, n1, n2)
    assert np.allclose(expected, actual, atol=1e-9)


def test_correction_matrix_avoids_double_counting_and_matches_upper_triangle(
    small_graphs,
):
    # This is the specific bug class caught during development: iterating
    # both directions of D1's edges against both directions of D2's edges
    # revisits every unordered candidate pair twice (once via (a,b)-(x,y),
    # once via its mirror (b,a)-(y,x)), which would double the stored
    # correction value if not handled. Verify nnz and values match a dense
    # brute-force upper-triangle computation exactly.
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    thresh = 3.0
    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    full_correction = np.maximum((diff / thresh) ** 2 - 1, 0)
    expected_upper = np.triu(full_correction, 1)

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2, chunk_size=7
    )
    assert correction.nnz == int((expected_upper > 0).sum())
    dense_actual = correction.toarray()
    assert np.allclose(dense_actual, expected_upper)


def test_distance_term_matvec_combined_with_degree_term(small_graphs):
    # The hybrid must remain exact when multiple weighted terms are summed
    # together (plan_spellmatch.md: "does degree_weight and distance_weight
    # on together require computing the clip once or twice?" -- answer:
    # once, and this test guards the combined-matvec arithmetic itself).
    rng, n1, n2, a1, a2, d1, d2 = small_graphs
    s = rng.random(n1 * n2)
    thresh = 3.0
    w_deg, w_dist = 1.0, 1.0

    deg1, deg2 = a1.sum(1), a2.sum(1)
    u_deg = np.abs(deg1[:, None] - deg2[None, :]).ravel()

    adj = sparse.kron(a1, a2).toarray()
    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    y_true = np.clip((diff / thresh) ** 2, 0, 1)
    w_full = (
        adj
        * (w_deg * (u_deg[:, None] + u_deg[None, :]) + w_dist * y_true)
        / (2 * w_deg + w_dist)
    )
    expected = w_full @ s

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2, chunk_size=7
    )
    deg_term = outer_sum_matvec(a1, a2, u_deg, s, n1, n2)
    dist_term = distance_term_matvec(a1, a2, d1, d2, thresh, correction, s, n1, n2)
    actual = (w_deg * deg_term + w_dist * dist_term) / (2 * w_deg + w_dist)
    assert np.allclose(expected, actual, atol=1e-9)


def test_float32_precision_gap_and_float64_mitigation():
    # plan_spellmatch.md §Step 4b: the d1^2+d2^2-2*d1*d2 expansion loses
    # precision relative to (d1-d2)^2 directly; verifies the required
    # float64 mitigation actually closes the gap, on a near-perfect-match
    # (worst-case cancellation) scenario.
    rng = np.random.default_rng(2)
    n1, n2 = 60, 70
    a1 = _make_symmetric_adjacency(rng, n1, 0.2)
    a2 = _make_symmetric_adjacency(rng, n2, 0.2)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    # near-perfect alignment: d2 is d1 plus tiny noise wherever shapes allow
    noise = rng.normal(0, 1e-3, (n1, n1))
    dists2_pattern = np.abs(dists1 + noise)
    dists2 = np.resize(dists2_pattern, (n2, n2))
    dists2 = np.triu(dists2, 1)
    dists2 = dists2 + dists2.T

    d1 = (a1 * dists1).astype(np.float64)
    d2 = (a2 * dists2).astype(np.float64)
    thresh = 5.0
    s = rng.random(n1 * n2)

    diff = sparse.kron(d1, a2).toarray() - sparse.kron(a1, d2).toarray()
    y_true = np.clip((diff / thresh) ** 2, 0, 1)
    expected = y_true @ s

    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2
    )
    actual = distance_term_matvec(a1, a2, d1, d2, thresh, correction, s, n1, n2)
    rel_err = np.abs(actual - expected) / (np.abs(expected).max() + 1e-12)
    # float64 throughout (as required by _kron_free.distance_term_matvec)
    # should land close to machine epsilon, not the ~1e-4 float32 would give
    # in this near-perfect-alignment / peaked scenario (see plan §Step 4b).
    assert rel_err.max() < 1e-9


def test_estimate_correction_nnz_matches_actual_build_when_unsampled(small_graphs):
    # small_graphs has far fewer than 2000 upper-triangle D1 edges, so the
    # default sample_size covers every edge: the estimate should be exact,
    # not merely close (Stage 6 pre-flight guardrail, plan_spellmatch.md).
    _, n1, n2, _, _, d1, d2 = small_graphs
    thresh = 5.0
    correction = build_distance_correction(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh, n1, n2
    )
    estimated_nnz, rel_stderr = estimate_correction_nnz(
        sparse.csr_matrix(d1), sparse.csr_matrix(d2), thresh
    )
    assert estimated_nnz == correction.nnz
    assert rel_stderr == 0.0


def test_estimate_correction_nnz_extrapolates_under_subsampling():
    # With more upper-triangle edges than the sample size, the estimate is a
    # genuine extrapolation -- check it lands within a generous tolerance of
    # the true count on a graph large enough for the law of large numbers to
    # apply, and that it reports nonzero sampling uncertainty.
    rng = np.random.default_rng(1)
    n1, n2 = 120, 90
    a1 = _make_symmetric_adjacency(rng, n1, 0.3)
    a2 = _make_symmetric_adjacency(rng, n2, 0.3)
    dists1 = np.triu(rng.uniform(1, 15, (n1, n1)), 1)
    dists1 = dists1 + dists1.T
    dists2 = np.triu(rng.uniform(1, 15, (n2, n2)), 1)
    dists2 = dists2 + dists2.T
    d1 = sparse.csr_matrix(a1 * dists1)
    d2 = sparse.csr_matrix(a2 * dists2)
    thresh = 5.0

    correction = build_distance_correction(d1, d2, thresh, n1, n2)
    estimated_nnz, rel_stderr = estimate_correction_nnz(
        d1, d2, thresh, sample_size=200
    )

    assert rel_stderr > 0.0
    rel_error = abs(estimated_nnz - correction.nnz) / correction.nnz
    assert rel_error < 0.25


def test_estimate_correction_bytes_scales_linearly_with_nnz():
    assert estimate_correction_bytes(0) == 0
    assert estimate_correction_bytes(1000) == 40 * 1000
    assert estimate_correction_bytes(2000) == 2 * estimate_correction_bytes(1000)
