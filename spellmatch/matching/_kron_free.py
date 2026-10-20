"""Matrix-free replacements for the Kronecker-product-scale computations in
:class:`spellmatch.matching.algorithms.spellmatch.Spellmatch`.

See ``plan_spellmatch.md`` (matchingwizard) for the full derivation and the
numerical investigation behind every choice made here. In one line: the
fixed-point iteration only ever needs ``W @ s`` for a changing vector ``s``,
never the ``(n1*n2) x (n1*n2)`` matrix ``W`` itself, so every term is
rewritten as a sequence of ordinary ``n1 x n1`` / ``n2 x n2`` operations
instead of ever materializing ``A1 (x) A2``.

Precision note: ``A1, A2, D1, D2`` and the internal distance-term algebra are
required to be float64 (see plan §Step 4b) -- these matrices are always small
regardless of dtype, and float32 there produces a measurable, avoidable
catastrophic-cancellation error. The public functions below therefore work in
float64 internally and cast to the caller's target dtype only on return.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def kron_matvec(
    x: np.ndarray, y: np.ndarray, v: np.ndarray, n1: int, n2: int
) -> np.ndarray:
    """Compute ``(x kron y) @ v`` without forming ``x kron y``.

    ``x`` is ``(n1, n1)``, ``y`` is ``(n2, n2)``, ``v`` has length ``n1*n2``
    and is interpreted with the row-major convention ``v[a*n2+x] = V[a, x]``.

    Identity (derived and numerically verified in ``plan_spellmatch.md``):
    ``(x kron y) @ v == vec(x @ V @ y.T)``.
    """
    v_mat = np.asarray(v).reshape(n1, n2)
    if sparse.issparse(x):
        out = x @ v_mat
    else:
        out = x.dot(v_mat)
    if sparse.issparse(y):
        out = out @ y.T
    else:
        out = out.dot(y.T)
    return np.asarray(out).reshape(n1 * n2)


def outer_sum_matvec(
    a1: np.ndarray, a2: np.ndarray, u: np.ndarray, s: np.ndarray, n1: int, n2: int
) -> np.ndarray:
    """Compute ``[A ⊙ (u 1^T + 1 u^T)] @ s`` where ``A = a1 kron a2``.

    This is exactly how the *original* implementation already builds the
    degree/intensity contributions to ``w_csr`` (two broadcast ``+=``
    statements) -- this function is a matrix-free restatement of that
    existing broadcast, not new algebra. Requires ``u`` to index the same
    candidate space as both the rows and columns of ``A`` (only meaningful
    because ``A`` is square: see plan_spellmatch.md's answer to "does this
    require A to be square").
    """
    a_s = kron_matvec(a1, a2, s, n1, n2)
    return u * a_s + kron_matvec(a1, a2, u * s, n1, n2)


# Default number of correction entries generated per block in
# build_distance_correction. Block temporaries cost roughly
# _BLOCK_BYTES_PER_NNZ bytes per entry (measured ~45), so 1e7 keeps them below 1 GB.
DEFAULT_BLOCK_NNZ = 10_000_000
_BLOCK_BYTES_PER_NNZ = 60
_INT32_LIMIT = 2**31


def correction_index_dtype(n: int, nnz: int) -> type:
    """Index dtype of the correction matrix: int32 unless an index (up to
    ``n - 1``, with ``n = n1*n2``) or an ``indptr`` value (up to ``nnz``)
    would overflow it. scipy requires ``indices`` and ``indptr`` to share one
    dtype, so both limits apply to both arrays."""
    return np.int32 if n < _INT32_LIMIT and nnz < _INT32_LIMIT else np.int64


def _clipped(d1_val: np.ndarray, d2_val: np.ndarray, thresh: np.float64) -> np.ndarray:
    """The clip predicate ``y > 1``, computed with exactly the same arithmetic
    as the stored value ``y - 1`` so that counting and building agree."""
    return ((d1_val - d2_val) / thresh) ** 2 > 1.0


def _correction_layout(d1: sparse.spmatrix, d2: sparse.spmatrix, thresh: float):
    """Shared setup for counting and building the correction matrix.

    Returns the upper-triangle D1 edges sorted by source cell ``a``, the D2
    edges sorted by length, and per D1 edge the bounds ``lo``/``hi`` such that
    its clipped D2 partners are exactly sorted positions ``[0, lo)`` and
    ``[hi, E2)``.

    The predicate ``((d - e)/thresh)**2 > 1`` is monotone in the sorted D2
    length ``e`` on either side of ``d`` (every floating-point step is
    monotone), so the clipped partners always form a prefix and a suffix of
    the sorted order. ``searchsorted`` finds the bounds up to rounding; the
    bounds are then moved one position at a time until the exact predicate
    holds on the inside and fails on the outside.
    """
    thresh = np.float64(thresh)

    d1_coo = sparse.coo_array(d1)
    upper = d1_coo.row < d1_coo.col
    e1_a = d1_coo.row[upper]
    e1_b = d1_coo.col[upper]
    e1_val = d1_coo.data[upper].astype(np.float64)
    order = np.argsort(e1_a, kind="stable")
    e1_a, e1_b, e1_val = e1_a[order], e1_b[order], e1_val[order]

    d2_coo = sparse.coo_array(d2)
    order = np.argsort(d2_coo.data, kind="stable")
    e2_x = d2_coo.row[order]
    e2_y = d2_coo.col[order]
    e2_val = d2_coo.data[order].astype(np.float64)
    n_e2 = len(e2_val)

    if len(e1_val) == 0 or n_e2 == 0:
        empty = np.zeros(len(e1_val), dtype=np.int64)
        return e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, empty, empty + n_e2

    # Prefix [0, lo): clipped partners shorter than d.
    lo = np.searchsorted(e2_val, e1_val - thresh, side="left").astype(np.int64)
    while True:
        i = np.nonzero(lo > 0)[0]
        i = i[~_clipped(e1_val[i], e2_val[lo[i] - 1], thresh)]
        if len(i) == 0:
            break
        lo[i] -= 1
    while True:
        i = np.nonzero(lo < n_e2)[0]
        nxt = e2_val[lo[i]]
        i = i[(nxt < e1_val[i]) & _clipped(e1_val[i], nxt, thresh)]
        if len(i) == 0:
            break
        lo[i] += 1

    # Suffix [hi, E2): clipped partners longer than d.
    hi = np.searchsorted(e2_val, e1_val + thresh, side="right").astype(np.int64)
    while True:
        i = np.nonzero(hi < n_e2)[0]
        i = i[~_clipped(e1_val[i], e2_val[hi[i]], thresh)]
        if len(i) == 0:
            break
        hi[i] += 1
    while True:
        i = np.nonzero(hi > 0)[0]
        prv = e2_val[hi[i] - 1]
        i = i[(prv > e1_val[i]) & _clipped(e1_val[i], prv, thresh)]
        if len(i) == 0:
            break
        hi[i] -= 1

    return e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, lo, hi


def count_distance_correction(
    d1: sparse.spmatrix, d2: sparse.spmatrix, thresh: float
) -> np.ndarray:
    """Exact number of correction entries contributed by each upper-triangle
    D1 edge (in source-cell order); ``.sum()`` is the correction's nnz.

    Costs O(E1 log E2) and allocates nothing of edge-pair size, so it is used
    as the pre-flight check before any large allocation.
    """
    *_, e2_val, lo, hi = _correction_layout(d1, d2, thresh)
    return lo + (len(e2_val) - hi)


def build_distance_correction(
    d1: sparse.spmatrix,
    d2: sparse.spmatrix,
    thresh: float,
    n1: int,
    n2: int,
    chunk_size: int = 2000,
    block_nnz: int = DEFAULT_BLOCK_NNZ,
) -> sparse.csr_array:
    """Build the sparse correction matrix ``relu(y-1)`` for the distance term.

    ``d1 = A1 ⊙ dists1``, ``d2 = A2 ⊙ dists2`` (edge-masked distances, i.e.
    zero everywhere the corresponding adjacency is zero). Only entries where
    the clip actually fires (``y = ((d1[a,b]-d2[x,y])/thresh)**2 > 1``) are
    stored, and only the upper triangle (``v < w``) -- see plan §Step 4/§8:
    the whole formulation is exactly symmetric with a structurally zero
    diagonal, so this loses no information and the matvec later reconstructs
    the full effect via ``U@s + U.T@s``.

    Iterating the *undirected* edges of G1 once each (``a < b``) against *all*
    directed edges of G2 visits every unordered candidate pair exactly once.
    Since ``a < b``, ``v = a*n2 + x < b*n2 <= w``, so every entry already lies
    in the upper triangle.

    Memory: the exact nnz is counted first (``count_distance_correction``) and
    the CSR arrays are allocated once at their final size, with int32 indices
    whenever they fit (``correction_index_dtype``). Rows ``v = a*n2 + x`` of
    consecutive source cells ``a`` form a contiguous row range, so entries are
    generated in blocks of source cells (at most ``block_nnz`` entries each,
    unless a single cell has more), converted to CSR locally and copied into
    place. Peak memory is therefore the final matrix (12 B/nnz with int32)
    plus one block's temporaries -- not a global COO staging copy.

    ``chunk_size`` is accepted for backwards compatibility and ignored.
    """
    del chunk_size
    e1_a, e1_b, e1_val, e2_x, e2_y, e2_val, lo, hi = _correction_layout(d1, d2, thresh)
    thresh = np.float64(thresh)
    n = n1 * n2
    counts = lo + (len(e2_val) - hi)
    nnz = int(counts.sum())

    idx_dtype = correction_index_dtype(n, nnz)
    if idx_dtype is np.int64:
        logger.warning(
            "Distance-correction matrix needs int64 indices (n1*n2=%s, nnz=%s); "
            "it will use 16 instead of 12 bytes per entry",
            f"{n:,}",
            f"{nnz:,}",
        )
    if nnz == 0:
        return sparse.csr_array((n, n), dtype=np.float64)

    indptr = np.empty(n + 1, dtype=idx_dtype)
    indices = np.empty(nnz, dtype=idx_dtype)
    data = np.empty(nnz, dtype=np.float64)
    indptr[0] = 0

    # Blocks of whole source cells: edges of cell a are contiguous (sorted by
    # a). Greedily close a block once it would exceed block_nnz entries.
    cell_starts = np.flatnonzero(np.r_[True, e1_a[1:] != e1_a[:-1]])
    cell_ends = np.r_[cell_starts[1:], len(e1_a)]
    cell_cum = np.cumsum(np.add.reduceat(counts, cell_starts))

    offset = 0
    filled_rows = 0  # indptr[1 : filled_rows + 1] is final
    first_cell = 0
    while first_cell < len(cell_starts):
        before = cell_cum[first_cell - 1] if first_cell > 0 else 0
        last_cell = int(np.searchsorted(cell_cum, before + block_nnz, side="right"))
        last_cell = max(last_cell, first_cell + 1)  # at least one cell per block
        e0, e1 = cell_starts[first_cell], cell_ends[last_cell - 1]
        a0, a1 = int(e1_a[e0]), int(e1_a[e1 - 1]) + 1
        first_cell = last_cell

        block_counts = counts[e0:e1]
        block_total = int(block_counts.sum())
        if block_total == 0:
            continue

        # Expand each edge's ranges [0, lo) and [hi, E2) into D2 positions.
        edge = np.repeat(np.arange(e0, e1), block_counts)
        pos = np.arange(block_total, dtype=np.int64)
        pos -= np.repeat(np.cumsum(block_counts) - block_counts, block_counts)
        edge_lo = lo[edge]
        j = pos + (hi[edge] - edge_lo) * (pos >= edge_lo)
        del pos, edge_lo

        # Same arithmetic as the predicate in _clipped: bit-identical values.
        y = ((e1_val[edge] - e2_val[j]) / thresh) ** 2
        vals = y - 1.0
        del y
        rows = (e1_a[edge].astype(np.int64) - a0) * n2 + e2_x[j]
        cols = e1_b[edge].astype(np.int64) * n2 + e2_y[j]
        del edge, j

        block = sparse.coo_array((vals, (rows, cols)), shape=((a1 - a0) * n2, n)).tocsr()
        del vals, rows, cols

        # Rows of cells without upper edges (between blocks) are empty.
        indptr[filled_rows + 1 : a0 * n2 + 1] = offset
        indptr[a0 * n2 + 1 : a1 * n2 + 1] = block.indptr[1:] + offset
        indices[offset : offset + block.nnz] = block.indices
        data[offset : offset + block.nnz] = block.data
        offset += block.nnz
        filled_rows = a1 * n2
        del block

    indptr[filled_rows + 1 :] = offset
    assert offset == nnz, f"correction count mismatch: built {offset}, counted {nnz}"
    return sparse.csr_array((data, indices, indptr), shape=(n, n))


class CorrectionMatrixTooLargeError(MemoryError):
    """Raised when the estimated distance-correction matrix would exceed the
    configured memory budget, instead of silently attempting an allocation
    that could OOM the process (plan_spellmatch.md, Stage 6)."""


def estimate_correction_nnz(
    d1: sparse.spmatrix,
    d2: sparse.spmatrix,
    thresh: float,
    sample_size: int = 2000,
    seed: int = 0,
) -> tuple[int, float]:
    """Estimate ``build_distance_correction``'s output size by running its
    exact per-edge comparison on a random sample of D1's upper-triangle
    edges (instead of all of them) against the full D2 edge list, then
    extrapolating linearly -- the "hits per D1 edge" this measures does not
    depend on which edges are sampled, only on the (fixed, already-known) D2
    edge population, so a small sample is representative.

    Returns ``(estimated_nnz, relative_stderr)``: the point estimate and the
    sampling distribution's relative standard error, so callers can apply a
    safety margin instead of trusting a single number exactly at a budget
    boundary.
    """
    d1_coo = sparse.coo_array(d1)
    upper = d1_coo.row < d1_coo.col
    e1_val = d1_coo.data[upper].astype(np.float64)
    n_e1 = len(e1_val)
    if n_e1 == 0:
        return 0, 0.0

    d2_coo = sparse.coo_array(d2)
    e2_val = d2_coo.data.astype(np.float64)
    if len(e2_val) == 0:
        return 0, 0.0

    rng = np.random.default_rng(seed)
    exhaustive = sample_size >= n_e1
    sample_size = min(sample_size, n_e1)
    sample = e1_val[rng.choice(n_e1, size=sample_size, replace=False)]

    thresh = np.float64(thresh)
    diff = sample[:, np.newaxis] - e2_val[np.newaxis, :]
    y = (diff / thresh) ** 2
    hits_per_edge = (y > 1.0).sum(axis=1).astype(np.float64)

    mean_hits = hits_per_edge.mean()
    estimated_nnz = int(round(mean_hits * n_e1))

    if exhaustive:
        # The whole population was measured, not sampled -- the count is
        # exact, with no extrapolation uncertainty to report.
        rel_stderr = 0.0
    elif mean_hits > 0 and sample_size > 1:
        std_hits = hits_per_edge.std(ddof=1)
        rel_stderr = float((std_hits / np.sqrt(sample_size)) / mean_hits)
    else:
        rel_stderr = 0.0
    return estimated_nnz, rel_stderr


def estimate_correction_bytes(
    nnz: int, n: int = 0, block_nnz: int = DEFAULT_BLOCK_NNZ
) -> int:
    """Peak-memory estimate (bytes) for ``build_distance_correction``.

    The CSR arrays are allocated once at their final size: ``indices`` and
    ``indptr`` (4 bytes each with int32, 8 with int64, see
    ``correction_index_dtype``) plus float64 ``data``. On top of that, one
    block of at most ``block_nnz`` entries is alive while it is generated, at
    roughly ``_BLOCK_BYTES_PER_NNZ`` bytes per entry. ``n = n1*n2`` is the
    number of rows (sizes ``indptr``).
    """
    nnz = int(nnz)
    if nnz == 0:
        return 0
    index_bytes = np.dtype(correction_index_dtype(n, nnz)).itemsize
    final = nnz * (index_bytes + 8) + (n + 1) * index_bytes
    return final + min(nnz, block_nnz) * _BLOCK_BYTES_PER_NNZ


def correction_matvec(correction: sparse.csr_array, s: np.ndarray) -> np.ndarray:
    """Apply an upper-triangular-only symmetric correction matrix to ``s``.

    ``correction`` stores only ``v < w``; the true (symmetric, zero-diagonal)
    matrix's action is ``U@s + U.T@s`` (plan §8 -- pure memory win, roughly
    the same total FLOPs as one matvec on the full matrix).
    """
    return correction @ s + correction.T @ s


def distance_term_matvec(
    a1: np.ndarray,
    a2: np.ndarray,
    d1: sparse.spmatrix,
    d2: sparse.spmatrix,
    thresh: float,
    correction: sparse.csr_array,
    s: np.ndarray,
    n1: int,
    n2: int,
) -> np.ndarray:
    """Compute ``Wdistance @ s`` (i.e. ``clip(y,0,1) @ s``) without ever
    materializing an edge-pair-scale dense/kron object.

    ``y@s`` is computed via the exact three-term Kronecker decomposition of
    the squared difference (plan §Step 3), in float64 (required -- plan
    §Step 4b); the clip is applied via ``y - relu(y-1)`` (plan §Step 4), with
    ``relu(y-1)`` supplied pre-built as ``correction``.
    """
    d1_f64 = d1.astype(np.float64) if sparse.issparse(d1) else np.asarray(d1, dtype=np.float64)
    d2_f64 = d2.astype(np.float64) if sparse.issparse(d2) else np.asarray(d2, dtype=np.float64)
    a1_f64 = a1.astype(np.float64) if sparse.issparse(a1) else np.asarray(a1, dtype=np.float64)
    a2_f64 = a2.astype(np.float64) if sparse.issparse(a2) else np.asarray(a2, dtype=np.float64)
    s_f64 = np.asarray(s, dtype=np.float64)

    d1_sq = d1_f64.multiply(d1_f64) if sparse.issparse(d1_f64) else d1_f64 * d1_f64
    d2_sq = d2_f64.multiply(d2_f64) if sparse.issparse(d2_f64) else d2_f64 * d2_f64

    thresh2 = np.float64(thresh) ** 2
    algebraic = (
        kron_matvec(d1_sq, a2_f64, s_f64, n1, n2)
        - 2.0 * kron_matvec(d1_f64, d2_f64, s_f64, n1, n2)
        + kron_matvec(a1_f64, d2_sq, s_f64, n1, n2)
    ) / thresh2

    if correction.nnz == 0:
        # Nothing clipped (e.g. distance_cdiff_thres >= longest edge).
        return algebraic
    corr = correction_matvec(correction, s_f64)
    return algebraic - corr
