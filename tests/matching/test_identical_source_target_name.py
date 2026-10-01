"""Regression test: matching crashes when source and target share a name.

icp.py, probreg.py and spellmatch.py all used to build their final scores
xr.DataArray as:

    coords={source_name: ..., target_name: ...}

source_name/target_name come from the mask's own identifier (its filename
stem, via `source_mask.name or "source"` in _algorithms.py). Whenever source
and target happen to share that identifier -- e.g. matching two panels of
the same physical tissue core, both labeled "A3" -- the dict literal
collapses to a single key (the second entry silently overwrites the first),
leaving xr.DataArray unable to infer two dimensions from one coordinate:

    ValueError: coordinate A3 has dimensions ('A3',), but these are not a
    subset of the DataArray dimensions ('dim_0', 'dim_1')

Confirmed in a real pipeline run (matchingwizard inst/HochSchulz, 2026-09-30)
where every single registration row names source and target identically
(e.g. "A3,A3,A3,A3" in section_registrations.csv): 163/163 pairs failed,
identically, across all three algorithms (cpd, icp, spellmatch). Fixed by
using fixed "source"/"target" dimension names instead of the per-run
identifier, which nothing downstream (assignment.py, benchmark/semisynthetic.py,
utils.restore_outlier_scores) actually relies on -- they all index
scores.dims[0]/scores.dims[1] positionally, never by the mask's own name.
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


def test_icp_match_points_with_identical_source_target_name(synthetic_points):
    source, target = synthetic_points
    algo = IterativeClosestPoints(max_dist=None)
    info, scores = algo._match_points("A3", "A3", source, target, None, None, {})
    assert scores.dims == ("source", "target")
    assert scores.shape == (len(source), len(target))


def test_rigid_cpd_match_points_with_identical_source_target_name(synthetic_points):
    source, target = synthetic_points
    algo = RigidCoherentPointDrift()
    info, scores = algo._match_points("A3", "A3", source, target, None, None, {})
    assert scores.dims == ("source", "target")
    assert scores.shape == (len(source), len(target))


def test_spellmatch_match_graphs_with_identical_source_target_name(synthetic_points):
    source, target = synthetic_points
    algo = Spellmatch(adj_radius=15, max_iter=1, opt_max_iter=20)
    info, scores = algo._match_graphs_from_points(
        "A3", "A3", source, target, None, None, {}
    )
    assert scores.dims == ("source", "target")
    assert scores.shape == (len(source), len(target))


def test_source_target_dims_distinct_even_with_different_names(synthetic_points):
    # Guard against a fix that accidentally only works for the identical-name
    # case: a normal run (distinct names) must still produce a valid,
    # 2-distinct-dims DataArray.
    source, target = synthetic_points
    algo = IterativeClosestPoints(max_dist=None)
    info, scores = algo._match_points(
        "source_mask", "target_mask", source, target, None, None, {}
    )
    assert scores.dims == ("source", "target")
    assert scores.shape == (len(source), len(target))
