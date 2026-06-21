"""Unit tests for `src/data/gene_alignment.py`.

In-memory fixtures only — no real data files are read. Synthetic gene symbols
and float arrays are constructed inline.

Behaviors covered:

coverage_fraction:
  1. Exact overlap returns 1.0.
  2. No overlap returns 0.0.
  3. Partial overlap returns correct fraction.
  4. Duplicate symbols in source_genes are deduplicated before counting.
  5. Empty target_genes returns 0.0.

align_to_gene_space — input acceptance:
  6. Accepts ndarray + source_genes of matching length.
  7. Accepts dict profile (source_genes used for symbol order).

align_to_gene_space — missing-gene policy:
  8. missing='zero' fills absent target genes with 0.0.
  9. missing='nan' fills absent target genes with NaN.

align_to_gene_space — extra-gene policy:
  10. Genes present in source but absent from target are silently dropped
      (output length == len(target_genes)).

align_to_gene_space — duplicate symbol policy:
  11. duplicates='first' keeps value at first occurrence.
  12. duplicates='mean' averages all occurrences.

align_to_gene_space — target ordering:
  13. Output values appear in target_genes order, not source order.

align_to_gene_space — output dtype:
  14. Output dtype is float32 regardless of input dtype.

align_to_gene_space — validation errors:
  15. Duplicate symbols in target_genes raise ValueError.
  16. ndarray with len != len(source_genes) raises ValueError.
  17. 2-D ndarray raises ValueError.
  18. Unsupported `missing` value raises ValueError.
  19. Unsupported `duplicates` value raises ValueError.
  20. Non-array/dict profile raises TypeError.

align_to_gene_space — low-coverage warning:
  21. Coverage < 50% triggers a WARNING log via the module logger.
  22. Coverage >= 50% does not trigger the WARNING.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from src.spatial.gene_alignment import align_to_gene_space, coverage_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    source_genes: list[str],
    values: list[float],
) -> np.ndarray:
    """Return a float32 1-D array paired with source_genes."""
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests 1-5: coverage_fraction
# ---------------------------------------------------------------------------


def test_coverage_exact_overlap() -> None:
    """Test 1: complete overlap returns 1.0."""
    assert coverage_fraction(["A", "B", "C"], ["A", "B", "C"]) == pytest.approx(1.0)


def test_coverage_no_overlap() -> None:
    """Test 2: zero overlap returns 0.0."""
    assert coverage_fraction(["A", "B"], ["C", "D"]) == pytest.approx(0.0)


def test_coverage_partial_overlap() -> None:
    """Test 3: 2 of 3 target genes covered = 2/3."""
    result = coverage_fraction(["A", "B", "C"], ["B", "C", "D"])
    assert result == pytest.approx(2.0 / 3.0)


def test_coverage_deduplicates_source() -> None:
    """Test 4: duplicate symbols in source_genes are deduplicated before
    the intersection computation.

    source has ["A", "A", "B"] — effective unique set is {A, B}.
    target is ["A", "B", "C"] — 2 of 3 hit = 2/3.
    """
    result = coverage_fraction(["A", "A", "B"], ["A", "B", "C"])
    assert result == pytest.approx(2.0 / 3.0)


def test_coverage_empty_target() -> None:
    """Test 5: empty target returns 0.0 (no division by zero)."""
    assert coverage_fraction(["A", "B"], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests 6-7: input acceptance
# ---------------------------------------------------------------------------


def test_align_accepts_ndarray() -> None:
    """Test 6: ndarray + matching source_genes is accepted."""
    src = ["G1", "G2", "G3"]
    tgt = ["G1", "G2", "G3"]
    prof = _make_profile(src, [1.0, 2.0, 3.0])
    out = align_to_gene_space(prof, src, tgt)
    assert out.shape == (3,)
    np.testing.assert_array_almost_equal(out, [1.0, 2.0, 3.0], decimal=5)


def test_align_accepts_dict_profile() -> None:
    """Test 7: dict profile is accepted; source_genes defines symbol order."""
    src = ["G1", "G2", "G3"]
    tgt = ["G3", "G1"]
    prof_dict = {"G1": 10.0, "G2": 20.0, "G3": 30.0}
    out = align_to_gene_space(prof_dict, src, tgt)
    assert out.shape == (2,)
    np.testing.assert_array_almost_equal(out, [30.0, 10.0], decimal=5)


# ---------------------------------------------------------------------------
# Tests 8-9: missing-gene policy
# ---------------------------------------------------------------------------


def test_align_missing_zero_fills_zero() -> None:
    """Test 8: missing='zero' fills absent target genes with 0.0."""
    src = ["G1", "G2"]
    tgt = ["G1", "G3"]  # G3 not in src
    prof = _make_profile(src, [5.0, 6.0])
    out = align_to_gene_space(prof, src, tgt, missing="zero")
    np.testing.assert_array_almost_equal(out, [5.0, 0.0], decimal=5)


def test_align_missing_nan_fills_nan() -> None:
    """Test 9: missing='nan' fills absent target genes with NaN."""
    src = ["G1", "G2"]
    tgt = ["G1", "G3"]  # G3 not in src
    prof = _make_profile(src, [5.0, 6.0])
    out = align_to_gene_space(prof, src, tgt, missing="nan")
    assert out[0] == pytest.approx(5.0)
    assert math.isnan(out[1])


# ---------------------------------------------------------------------------
# Test 10: extra genes in source are dropped
# ---------------------------------------------------------------------------


def test_align_drops_extra_source_genes() -> None:
    """Test 10: genes present in source but not target are silently dropped."""
    src = ["G1", "G2", "G3", "G4"]
    tgt = ["G1", "G3"]
    prof = _make_profile(src, [1.0, 2.0, 3.0, 4.0])
    out = align_to_gene_space(prof, src, tgt)
    assert out.shape == (2,)  # only target-length
    np.testing.assert_array_almost_equal(out, [1.0, 3.0], decimal=5)


# ---------------------------------------------------------------------------
# Tests 11-12: duplicate symbol policy
# ---------------------------------------------------------------------------


def test_align_duplicates_first() -> None:
    """Test 11: duplicates='first' keeps value at first occurrence.

    source has G2 twice: values 2.0 and 99.0.
    First occurrence (2.0) should win.
    """
    src = ["G1", "G2", "G2", "G3"]
    tgt = ["G1", "G2", "G3"]
    prof = _make_profile(src, [1.0, 2.0, 99.0, 3.0])
    out = align_to_gene_space(prof, src, tgt, duplicates="first")
    np.testing.assert_array_almost_equal(out, [1.0, 2.0, 3.0], decimal=5)


def test_align_duplicates_mean() -> None:
    """Test 12: duplicates='mean' averages all occurrences.

    G2 appears at values 2.0 and 8.0 → mean = 5.0.
    G3 appears at values 3.0, 7.0, 10.0 → mean = 20/3 ≈ 6.666...
    """
    src = ["G1", "G2", "G2", "G3", "G3", "G3"]
    tgt = ["G1", "G2", "G3"]
    prof = _make_profile(src, [1.0, 2.0, 8.0, 3.0, 7.0, 10.0])
    out = align_to_gene_space(prof, src, tgt, duplicates="mean")
    np.testing.assert_array_almost_equal(out, [1.0, 5.0, 20.0 / 3.0], decimal=5)


# ---------------------------------------------------------------------------
# Test 13: output ordering
# ---------------------------------------------------------------------------


def test_align_target_ordering() -> None:
    """Test 13: output values follow target_genes order, not source order.

    source = [G3, G1, G2], target = [G2, G3, G1].
    Expected output: [value(G2), value(G3), value(G1)] = [3.0, 1.0, 2.0].
    """
    src = ["G3", "G1", "G2"]
    tgt = ["G2", "G3", "G1"]
    prof = _make_profile(src, [1.0, 2.0, 3.0])  # G3=1, G1=2, G2=3
    out = align_to_gene_space(prof, src, tgt)
    np.testing.assert_array_almost_equal(out, [3.0, 1.0, 2.0], decimal=5)


# ---------------------------------------------------------------------------
# Test 14: output dtype
# ---------------------------------------------------------------------------


def test_align_output_dtype_float32() -> None:
    """Test 14: output is always float32 regardless of input dtype."""
    src = ["G1", "G2"]
    tgt = ["G1", "G2"]
    # Feed float64 input — output must still be float32.
    prof_f64 = np.array([1.0, 2.0], dtype=np.float64)
    out = align_to_gene_space(prof_f64, src, tgt)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests 15-20: validation errors
# ---------------------------------------------------------------------------


def test_align_raises_on_duplicate_target_genes() -> None:
    """Test 15: duplicate symbols in target_genes raise ValueError."""
    src = ["G1", "G2"]
    tgt = ["G1", "G1"]  # duplicate
    prof = _make_profile(src, [1.0, 2.0])
    with pytest.raises(ValueError, match=r"target_genes contains duplicate"):
        align_to_gene_space(prof, src, tgt)


def test_align_raises_on_length_mismatch() -> None:
    """Test 16: ndarray with len != len(source_genes) raises ValueError."""
    src = ["G1", "G2", "G3"]
    prof = _make_profile(["G1", "G2"], [1.0, 2.0])  # len=2, but src has 3
    with pytest.raises(ValueError, match=r"len\(source_genes\)"):
        align_to_gene_space(prof, src, ["G1"])


def test_align_raises_on_2d_ndarray() -> None:
    """Test 17: 2-D ndarray raises ValueError."""
    src = ["G1", "G2"]
    tgt = ["G1"]
    prof_2d = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"1-D"):
        align_to_gene_space(prof_2d, src, tgt)


def test_align_raises_on_bad_missing_policy() -> None:
    """Test 18: unrecognized `missing` value raises ValueError."""
    src = ["G1"]
    tgt = ["G1"]
    prof = _make_profile(src, [1.0])
    with pytest.raises(ValueError, match=r"missing"):
        align_to_gene_space(prof, src, tgt, missing="interpolate")  # type: ignore[arg-type]


def test_align_raises_on_bad_duplicates_policy() -> None:
    """Test 19: unrecognized `duplicates` value raises ValueError."""
    src = ["G1"]
    tgt = ["G1"]
    prof = _make_profile(src, [1.0])
    with pytest.raises(ValueError, match=r"duplicates"):
        align_to_gene_space(prof, src, tgt, duplicates="max")  # type: ignore[arg-type]


def test_align_raises_on_non_array_non_dict_profile() -> None:
    """Test 20: a list (not ndarray or dict) raises TypeError."""
    src = ["G1", "G2"]
    tgt = ["G1"]
    prof_list = [1.0, 2.0]
    with pytest.raises(TypeError, match=r"np.ndarray or dict"):
        align_to_gene_space(prof_list, src, tgt)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests 21-22: low-coverage warning
# ---------------------------------------------------------------------------


def test_align_low_coverage_triggers_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 21: coverage < 50% triggers a WARNING from the module logger.

    source has 1 gene (G1); target has 5 genes (G1..G5).
    Coverage = 1/5 = 20% < 50% → warning expected.
    """
    src = ["G1"]
    tgt = ["G1", "G2", "G3", "G4", "G5"]
    prof = _make_profile(src, [1.0])
    with caplog.at_level(logging.WARNING, logger="src.data.gene_alignment"):
        align_to_gene_space(prof, src, tgt, missing="zero")
    assert any("low coverage" in record.message for record in caplog.records), (
        "Expected a low-coverage WARNING but none was emitted."
    )


def test_align_sufficient_coverage_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test 22: coverage >= 50% does NOT trigger the low-coverage WARNING.

    source has 3 genes (G1..G3); target has 4 genes (G1..G4).
    Coverage = 3/4 = 75% >= 50% → no warning.
    """
    src = ["G1", "G2", "G3"]
    tgt = ["G1", "G2", "G3", "G4"]
    prof = _make_profile(src, [1.0, 2.0, 3.0])
    with caplog.at_level(logging.WARNING, logger="src.data.gene_alignment"):
        align_to_gene_space(prof, src, tgt, missing="zero")
    low_cov_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "low coverage" in r.message
    ]
    assert low_cov_warnings == [], (
        f"Expected no low-coverage WARNING but got: {[r.message for r in low_cov_warnings]}"
    )
