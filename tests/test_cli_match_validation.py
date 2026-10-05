"""Tests for cli_match's same-name-refusal check (__main__.py).

Identical source/target filenames used to crash deep inside _match_graphs
(the dict-literal coords={source_name: ..., target_name: ...} collision,
fixed in commit e0099d8 by switching to fixed "source"/"target" dimension
names). That fix makes identical names not crash -- but identical
source/target is still invalid input (matching a section against itself),
so cli_match now refuses it outright, before reading any file. See also
tests/matching/test_identical_source_target_name.py, which covers the
lower-level _match_points/_match_graphs_from_points methods directly; this
file covers the CLI-level refusal instead.
"""

from __future__ import annotations

import numpy as np
import tifffile
from click.testing import CliRunner

from spellmatch.__main__ import cli


def _write_mask(path, labels):
    tifffile.imwrite(path, np.array(labels, dtype=np.uint16))


def _simple_mask():
    # A tiny 2-cell labeled mask: label 1 in the top-left, label 2 in the
    # bottom-right, background 0 elsewhere.
    arr = np.zeros((10, 10), dtype=np.uint16)
    arr[1:3, 1:3] = 1
    arr[6:8, 6:8] = 2
    return arr


def test_refuses_identical_mask_filenames(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    source_mask = source_dir / "same.tiff"
    target_mask = target_dir / "same.tiff"
    _write_mask(source_mask, _simple_mask())
    _write_mask(target_mask, _simple_mask())

    result = CliRunner().invoke(
        cli,
        [
            "match",
            str(source_mask),
            str(target_mask),
            "--algorithm",
            "icp",
            str(tmp_path / "scores.nc"),
        ],
    )

    assert result.exit_code != 0
    assert "same filename" in result.output
    assert "same.tiff" in result.output


def test_distinctly_named_masks_are_not_rejected_by_this_check(tmp_path):
    source_mask = tmp_path / "source_mask.tiff"
    target_mask = tmp_path / "target_mask.tiff"
    _write_mask(source_mask, _simple_mask())
    _write_mask(target_mask, _simple_mask())

    result = CliRunner().invoke(
        cli,
        [
            "match",
            str(source_mask),
            str(target_mask),
            "--algorithm",
            "icp",
            str(tmp_path / "scores.nc"),
        ],
    )

    # Regardless of whether the full icp run succeeds end-to-end on this
    # minimal synthetic input, it must not fail with the same-name refusal.
    assert "same filename" not in result.output
