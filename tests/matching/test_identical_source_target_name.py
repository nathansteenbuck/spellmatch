"""Dimension names on the final scores DataArray must be the real
source/target identifiers, not generic placeholders -- and identical
source/target names are prevented earlier (in cli_match), not something
these internals handle.

icp.py, probreg.py and spellmatch.py build their final scores xr.DataArray
as `coords={source_name: ..., target_name: ...}` (source_adj.name/
target_adj.name for spellmatch.py), using the mask's own identifier (its
filename stem, via `source_mask.name or "source"` in _algorithms.py). This
is deliberate: downstream outputs -- in particular the assignment CSV's
column headers, written via assignment.py's `scores.dims[0]: ...,
scores.dims[1]: ...` -- show the real file names this way, not a generic
"source"/"target" label.

Whenever source and target happen to share that identifier (e.g. matching
two panels of the same physical tissue core, both labeled "A3"), the dict
literal collapses to a single key, leaving xr.DataArray unable to infer two
dimensions from one coordinate:

    ValueError: coordinate A3 has dimensions ('A3',), but these are not a
    subset of the DataArray dimensions ('dim_0', 'dim_1')

Observed in a real pipeline run where every source/target pair shared the
same identifier: every pair failed this way, across all three algorithms
(cpd, icp, spellmatch).

This is prevented one layer up instead of papered over here: `cli_match`
refuses identical source_mask/target_mask filenames before any matching
runs (see tests/test_cli_match_validation.py). These internals are
intentionally left able to fail on identical names -- that failure should
never be reachable in practice, and a regression here is exactly what
should be caught if the cli_match guard is ever removed or bypassed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spellmatch.matching.algorithms.icp import IterativeClosestPoints
from spellmatch.matching.algorithms.probreg import RigidCoherentPointDrift
from spellmatch.matching.algorithms.spellmatch import Spellmatch


@pytest.fixture
def synthetic_points():
    rng = np.random.default_rng(0)
    n = 20
    source = pd.DataFrame(rng.uniform(0, 50, (n, 2)), columns=["x", "y"])
    target = pd.DataFrame(
        source.to_numpy() + rng.normal(0, 0.5, (n, 2)), columns=["x", "y"]
    )
    return source, target


def test_icp_match_points_uses_real_names_as_dims(synthetic_points):
    source, target = synthetic_points
    algo = IterativeClosestPoints(max_dist=None)
    info, scores = algo._match_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source_mask", "target_mask")
    assert scores.shape == (len(source), len(target))


def test_rigid_cpd_match_points_uses_real_names_as_dims(synthetic_points):
    source, target = synthetic_points
    algo = RigidCoherentPointDrift()
    info, scores = algo._match_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source_mask", "target_mask")
    assert scores.shape == (len(source), len(target))


def test_spellmatch_match_graphs_uses_real_names_as_dims(synthetic_points):
    source, target = synthetic_points
    algo = Spellmatch(adj_radius=15, max_iter=1, opt_max_iter=20)
    info, scores = algo._match_graphs_from_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source_mask", "target_mask")
    assert scores.shape == (len(source), len(target))


def test_icp_still_fails_on_identical_names_at_this_level(synthetic_points):
    # Intentionally NOT handled here -- see module docstring. If this starts
    # passing without the cli_match guard also covering the case, that's a
    # gap, not an improvement.
    source, target = synthetic_points
    algo = IterativeClosestPoints(max_dist=None)
    with pytest.raises(ValueError, match="coordinate"):
        algo._match_points("A3", "A3", source, target, None, None, {})


def test_rigid_cpd_still_fails_on_identical_names_at_this_level(synthetic_points):
    source, target = synthetic_points
    algo = RigidCoherentPointDrift()
    with pytest.raises(ValueError, match="coordinate"):
        algo._match_points("A3", "A3", source, target, None, None, {})


def test_spellmatch_still_fails_on_identical_names_at_this_level(synthetic_points):
    source, target = synthetic_points
    algo = Spellmatch(adj_radius=15, max_iter=1, opt_max_iter=20)
    with pytest.raises(ValueError, match="coordinate"):
        algo._match_graphs_from_points("A3", "A3", source, target, None, None, {})
