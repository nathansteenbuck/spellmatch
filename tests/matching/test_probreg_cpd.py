"""Tests for the CPD wrapper's soft correspondence output.

Verifies the fix to spellmatch.matching.algorithms.probreg: RigidCoherentPointDrift
(and its Affine/NonRigid siblings, which share the same _register_points) must
return a continuous (n_source, n_target) correspondence matrix reproducing CPD's
own E-step, not the hard nearest-neighbor 0/1 matrix the wrapper used to fall
back to. See jazzy-crafting-sonnet.md ("Fix CPD wrapper...") for the derivation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import distance

from spellmatch.matching.algorithms.probreg import (
    RigidCoherentPointDrift,
    _cpd_correspondence_matrix,
)


@pytest.fixture
def synthetic_points():
    # A little Gaussian noise (not a perfectly exact rigid transform) is
    # deliberate: on perfectly clean data CPD's own sigma2 correctly converges
    # toward its numerical floor, making the Gaussian kernel a near-delta
    # function that legitimately underflows to exact 0/1 in float64 -- correct
    # CPD behavior, but useless for testing that the *soft* matrix is wired
    # through. A little noise keeps sigma2 away from that floor.
    rng = np.random.default_rng(0)
    n = 30
    source = rng.uniform(0, 100, (n, 2))
    theta = np.deg2rad(7.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = np.array([3.0, -2.0])
    target = source @ rot.T + translation + rng.normal(0, 1.5, (n, 2))
    return source, target


def _run_rigid_cpd(source, target, **kwargs):
    algo = RigidCoherentPointDrift(w=0.0, maxiter=100, tol=1e-6, **kwargs)
    source_df = pd.DataFrame(source, columns=["x", "y"])
    target_df = pd.DataFrame(target, columns=["x", "y"])
    info, scores = algo._match_points(
        "source", "target", source_df, target_df, None, None, None
    )
    return info, scores.to_numpy()


def _register(algo, source, target):
    # _register_points relies on _current_iteration being initialized, which
    # _match_points normally does; replicate that for tests calling it directly.
    algo._current_iteration = 0
    transform = algo._register_points(source, target)
    algo._current_iteration = None
    return transform


def test_returns_continuous_not_binary_scores(synthetic_points):
    source, target = synthetic_points
    _, scores = _run_rigid_cpd(source, target)
    assert scores.shape == (len(source), len(target))
    unique_vals = np.unique(scores)
    assert not np.all(np.isin(unique_vals, [0.0, 1.0])), (
        "scores are still binary -- soft correspondence was not wired through"
    )


def test_columns_sum_to_one_when_no_outlier_term(synthetic_points):
    # w=0 => CPD's own E-step normalization has no outlier absorption term,
    # so every target point's column should sum to exactly 1.
    source, target = synthetic_points
    _, scores = _run_rigid_cpd(source, target)
    col_sums = scores.sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-6)


def test_matches_independent_reimplementation_of_e_step(synthetic_points):
    # Recompute the correspondence matrix a second, independently-written way
    # (brute-force cdist + exp + normalize using the actual converged rigid
    # transform) and check it agrees with what the algorithm returns -- same
    # "port the derivation into a permanent ground-truth test" discipline used
    # for the matrix-free work (test_kron_free.py).
    source, target = synthetic_points
    algo = RigidCoherentPointDrift(w=0.0, maxiter=100, tol=1e-6)
    transform = _register(algo, source, target)
    t_source = transform.transform(source)

    d2 = distance.cdist(t_source, target, "sqeuclidean")
    brute = np.exp(-d2 / (2.0 * algo._current_correspondence_sigma2))
    brute = brute / brute.sum(axis=0, keepdims=True)

    np.testing.assert_allclose(algo._current_correspondence, brute, rtol=1e-10)


def test_row_argmax_matches_true_nearest_target_for_well_separated_points():
    # Sparse, well-separated points: the soft matrix should still be sharply
    # peaked at the geometrically correct correspondence.
    rng = np.random.default_rng(1)
    n = 12
    source = rng.uniform(0, 500, (n, 2))
    target = source + rng.normal(0, 0.05, source.shape)  # near-identity + tiny noise
    _, scores = _run_rigid_cpd(source, target)
    row_argmax = np.argmax(scores, axis=1)
    np.testing.assert_array_equal(row_argmax, np.arange(n))


def test_max_dist_masks_far_pairs_without_reintroducing_hard_assignment(
    synthetic_points,
):
    source, target = synthetic_points
    _, scores_unmasked = _run_rigid_cpd(source, target, max_dist=None)
    _, scores_masked = _run_rigid_cpd(source, target, max_dist=5.0)

    algo = RigidCoherentPointDrift(w=0.0, maxiter=100, tol=1e-6, max_dist=5.0)
    transform = _register(algo, source, target)
    t_source = transform.transform(source)
    far = distance.cdist(t_source, target) > 5.0

    assert np.all(scores_masked[far] == 0.0)
    assert np.any(scores_masked[~far] > 0.0)
    # masking must not collapse the remaining entries to hard 0/1
    remaining = scores_masked[~far]
    assert not np.all(np.isin(np.unique(remaining), [0.0, 1.0]))
    # unmasked run should have (weakly) more nonzero mass than the masked one
    assert (scores_unmasked > 0).sum() >= (scores_masked > 0).sum()


def test_intersect_max_assignment_recovers_correct_matches_on_clean_case():
    # Reproduces the same "direction=intersect, max=True" logic spellmatch
    # assign already applies to any continuous score matrix -- confirms a
    # binary assignment is still recoverable downstream, unchanged.
    rng = np.random.default_rng(2)
    n = 10
    source = rng.uniform(0, 200, (n, 2))
    theta = np.deg2rad(3.0)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = source @ rot.T + np.array([1.0, 0.5])
    _, scores = _run_rigid_cpd(source, target)

    row_argmax = np.argmax(scores, axis=1)
    col_argmax = np.argmax(scores, axis=0)
    mutual = row_argmax == np.arange(n)
    mutual &= col_argmax[row_argmax] == np.arange(n)
    assert mutual.all(), "expected every point to be its own mutual best match"
    np.testing.assert_array_equal(row_argmax, np.arange(n))
