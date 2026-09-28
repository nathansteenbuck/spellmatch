"""Regression test for the uint8 degree-product overflow bug.

`Spellmatch._match_graphs` (spellmatch.py) computes per-node degree via
`np.sum(adj, axis=1, dtype=X)`, then a full (n1, n2) outer product
`deg1[:, None] * deg2[None, :]` -- computed in dtype X too, since numpy does
not promote same-dtype arithmetic. With X=uint8 (the original and, until
this fix, the fork's dtype), individual degrees never overflow at realistic
adj_radius values (max 20-30 seen on real IMC data), but their *product*
routinely does: two real degree-17/16 nodes give 17*16=272, silently
wrapping to 16 in uint8 arithmetic (mod 256), no warning, no error.

This corrupts the D^-1/2 degree normalization for the affected candidate
pairs (an artificially small "degree" gives an artificially large
normalization weight), breaking the assumption that guarantees the
assembled operator's spectral radius stays <=1 -- confirmed as the root
cause of a real NaN-divergence failure on the 2022 Melanoma dataset (see
convergence_hypotheses.md). Fixed by widening to uint16 (65,535 max),
a >3000x margin over any degree observed in practice.
"""

from __future__ import annotations

import numpy as np
import pytest


def _degree_product(adj1: np.ndarray, adj2: np.ndarray, dtype) -> np.ndarray:
    """Mirrors spellmatch.py's exact degree + outer-product computation."""
    deg1 = np.sum(adj1, axis=1, dtype=dtype)
    deg2 = np.sum(adj2, axis=1, dtype=dtype)
    return np.asarray(deg1[:, np.newaxis] * deg2[np.newaxis, :])


def test_uint8_silently_wrapped_this_exact_real_case():
    # Real values from the diverging FOV pair (p28_r32_a32 <-> p18_r43_a43):
    # a source node with 17 neighbors, a target node with 16.
    adj1 = np.zeros((18, 18), dtype=bool)
    adj1[0, 1:18] = True  # node 0 has degree 17
    adj2 = np.zeros((17, 17), dtype=bool)
    adj2[0, 1:17] = True  # node 0 has degree 16

    prod_uint8 = _degree_product(adj1, adj2, np.uint8)
    assert prod_uint8.dtype == np.uint8
    assert prod_uint8[0, 0] == 16  # 272 mod 256 -- the historical bug

    prod_uint16 = _degree_product(adj1, adj2, np.uint16)
    assert prod_uint16.dtype == np.uint16
    assert prod_uint16[0, 0] == 272  # true value, no wraparound


def test_uint16_has_large_margin_over_realistic_degrees():
    # Max degree observed on real data at adj_radius=15 was 20; stress-test
    # with degrees an order of magnitude higher on both sides (200*200 =
    # 40,000 < 65,535) to confirm headroom, not just the one real example.
    adj1 = np.zeros((201, 201), dtype=bool)
    adj1[0, 1:201] = True  # degree 200
    adj2 = np.zeros((201, 201), dtype=bool)
    adj2[0, 1:201] = True  # degree 200

    prod = _degree_product(adj1, adj2, np.uint16)
    assert prod[0, 0] == 200 * 200 == 40000


def test_uint16_still_overflows_far_beyond_any_realistic_degree():
    # Documents where uint16's own limit sits, for completeness -- this
    # regime (degree ~256+ simultaneously on both graphs) would require
    # ~2.8 px^2/cell at adj_radius=15, far denser than any real segmented
    # cell, so this is not expected to occur in practice.
    adj1 = np.zeros((257, 257), dtype=bool)
    adj1[0, 1:257] = True  # degree 256
    adj2 = np.zeros((257, 257), dtype=bool)
    adj2[0, 1:257] = True  # degree 256

    prod = _degree_product(adj1, adj2, np.uint16)
    assert prod[0, 0] == 0  # 65536 mod 65536 -- confirms the boundary exists


def test_spellmatch_match_graphs_uses_uint16_not_uint8():
    import inspect

    from spellmatch.matching.algorithms.spellmatch import Spellmatch

    src = inspect.getsource(Spellmatch._match_graphs)
    assert "np.uint16" in src
    assert "np.uint8" not in src
