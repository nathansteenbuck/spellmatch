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

import numpy as np
from scipy import sparse


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


def build_distance_correction(
    d1: sparse.spmatrix,
    d2: sparse.spmatrix,
    thresh: float,
    n1: int,
    n2: int,
    chunk_size: int = 2000,
) -> sparse.csr_array:
    """Build the sparse correction matrix ``relu(y-1)`` for the distance term.

    ``d1 = A1 ⊙ dists1``, ``d2 = A2 ⊙ dists2`` (edge-masked distances, i.e.
    zero everywhere the corresponding adjacency is zero). Only entries where
    the clip actually fires (``y = ((d1[a,b]-d2[x,y])/thresh)**2 > 1``) are
    stored, and only the upper triangle (``v < w``) -- see plan §Step 4/§8:
    the whole formulation is exactly symmetric with a structurally zero
    diagonal, verified numerically, so this loses no information and the
    matvec later reconstructs the full effect via ``U@s + U.T@s``.

    To avoid double-counting: iterating the *undirected* edges of G1 once
    each (upper triangle of ``d1``) against *all* directed edges of G2 visits
    every unordered candidate pair ``{(a,x),(b,y)}`` exactly once. (Using the
    full, both-directions edge list of ``d1`` would revisit every pair twice,
    once via ``(a,b)`` and once via its mirror ``(b,a)`` together with the
    matching ``(y,x)`` entry of G2 -- verified against a dense brute-force
    ground truth while developing this.)
    """
    d1_coo = sparse.coo_array(d1)
    upper = d1_coo.row < d1_coo.col
    e1_a = d1_coo.row[upper]
    e1_b = d1_coo.col[upper]
    e1_val = d1_coo.data[upper].astype(np.float64)

    d2_coo = sparse.coo_array(d2)
    e2_x = d2_coo.row
    e2_y = d2_coo.col
    e2_val = d2_coo.data.astype(np.float64)

    thresh = np.float64(thresh)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []

    n_e1 = len(e1_val)
    for start in range(0, n_e1, chunk_size):
        stop = min(start + chunk_size, n_e1)
        a_chunk = e1_a[start:stop]
        b_chunk = e1_b[start:stop]
        d1_chunk = e1_val[start:stop]

        diff = d1_chunk[:, np.newaxis] - e2_val[np.newaxis, :]
        y = (diff / thresh) ** 2
        mask = y > 1.0
        if not mask.any():
            continue
        ci, cj = np.nonzero(mask)

        a_idx = a_chunk[ci].astype(np.int64)
        b_idx = b_chunk[ci].astype(np.int64)
        x_idx = e2_x[cj].astype(np.int64)
        y_idx = e2_y[cj].astype(np.int64)

        v = a_idx * n2 + x_idx
        w = b_idx * n2 + y_idx
        excess = y[ci, cj] - 1.0

        vv = np.minimum(v, w)
        ww = np.maximum(v, w)
        keep = vv != ww
        rows.append(vv[keep])
        cols.append(ww[keep])
        vals.append(excess[keep])

    n = n1 * n2
    if not rows:
        return sparse.csr_array((n, n), dtype=np.float64)
    rows_arr = np.concatenate(rows)
    cols_arr = np.concatenate(cols)
    vals_arr = np.concatenate(vals)
    correction = sparse.coo_array((vals_arr, (rows_arr, cols_arr)), shape=(n, n))
    return sparse.csr_array(correction)


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


def estimate_correction_bytes(estimated_nnz: int) -> int:
    """Rough peak-memory estimate (bytes) for ``build_distance_correction``.

    The final CSR array costs ~16-20 bytes/nnz (int32/int64 indices + float64
    data). During construction the COO staging arrays (int64 row + int64 col
    + float64 val = 24 bytes/nnz, per chunk, concatenated) are briefly alive
    at the same time as that CSR array, plus numpy's own transient copies in
    ``np.concatenate``/``coo_array``/``tocsr``. 40 bytes/nnz is a conservative
    round number covering that overlap, chosen to be an upper bound rather
    than a tight prediction.
    """
    return int(estimated_nnz) * 40


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

    corr = correction_matvec(correction, s_f64)
    return algebraic - corr
