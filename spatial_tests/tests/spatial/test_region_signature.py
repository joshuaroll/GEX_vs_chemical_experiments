"""Unit tests for ``src/spatial_signatures/region_signature.py``.

In-memory fixture tests — no real-data reads, no model calls.

All model inference is stubbed out in the library behind NotImplementedError.
These tests cover only the PURE parts:

  1. ``compute_de``: DE subtraction (predicted_treated - region_basal).
  2. ``make_cache_key``: determinism, gene-order sensitivity, namespace
     separation (different model_variant → different key).
  3. ``build_manifest``: alignment metadata construction, duplicate/empty
     validation, model_variant validation.
  4. ``assemble_cache``: end-to-end pure assembly from injected prediction
     arrays (bypassing the model stub), including shape, dtype, axis alignment,
     and error paths.
  5. ``RegionSignatureCacher`` constructor validation and NotImplementedError
     guard on ``load_model`` and ``_call_model``.

Purity gate
-----------
- No ``/raid`` paths anywhere in this file.
- No real-data filenames.
- All numpy arrays are constructed inline with small synthetic values.
- No import of torch or any model checkpoint utilities.
- N_LANDMARK=978 is used only in the gene-space contract test; the majority of
  tests use small synthetic gene counts (e.g. 5 or 8 genes) to keep fixtures
  readable.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.spatial.region_signature import (
    N_LANDMARK,
    RegionDE,
    RegionSignatureCache,
    RegionSignatureManifest,
    RegionSignatureCacher,
    assemble_cache,
    build_manifest,
    compute_de,
    make_cache_key,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
# NOTE (Phase 2 / Wave 0): the shared fixtures `three_genes`, `two_regions`,
# `two_pert_ids`, `small_manifest`, `region_basal_map` now live in
# `tests/spatial/conftest.py` and resolve via pytest fixture discovery. The new
# Phase-2 fixtures `synthetic_treated_control` and `gene_order_slice` are also
# defined there. Do NOT redefine them here.


def _make_predicted_treated_map(
    pert_ids: list[str],
    regions: list[str],
    n_genes: int,
    seed: int = 0,
) -> dict[tuple[str, str], np.ndarray]:
    """Build a synthetic predicted_treated_map for testing assemble_cache."""
    rng = np.random.default_rng(seed)
    return {
        (pid, reg): rng.standard_normal(n_genes).astype(np.float32)
        for pid in pert_ids
        for reg in regions
    }


def _make_predicted_control_map(
    pert_ids: list[str],
    regions: list[str],
    n_genes: int,
    seed: int = 1,
) -> dict[tuple[str, str], np.ndarray]:
    """Build a synthetic predicted_control_map (rule B, D-02) keyed (pid, reg).

    Rule B subtracts the model's OWN predicted control output, so the second
    operand to ``assemble_cache`` is a per-(pert_id, region) predicted-control
    map, NOT a raw region_basal map.
    """
    rng = np.random.default_rng(seed)
    return {
        (pid, reg): rng.standard_normal(n_genes).astype(np.float32)
        for pid in pert_ids
        for reg in regions
    }


# ---------------------------------------------------------------------------
# Tests: compute_de
# ---------------------------------------------------------------------------


class TestComputeDe:
    """Tests for the pure DE subtraction function."""

    def test_subtraction_correctness(self):
        """DE = predicted_treated - region_basal, element-wise."""
        pred = np.array([3.0, 1.0, 5.0], dtype=np.float32)
        basal = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        de = compute_de(pred, basal)
        expected = np.array([2.0, -1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(de, expected)

    def test_output_dtype_is_float32(self):
        """Output is float32 regardless of input dtype."""
        pred = np.array([1.0, 2.0], dtype=np.float64)
        basal = np.array([0.5, 0.5], dtype=np.float64)
        de = compute_de(pred, basal)
        assert de.dtype == np.float32

    def test_output_shape_matches_input(self):
        """Output has same shape as the 1-D inputs."""
        n = 978
        pred = np.zeros(n, dtype=np.float32)
        basal = np.ones(n, dtype=np.float32)
        de = compute_de(pred, basal)
        assert de.shape == (n,)

    def test_zero_basal_returns_predicted(self):
        """DE = predicted_treated when basal is all-zero."""
        pred = np.array([1.5, -2.3, 0.0], dtype=np.float32)
        basal = np.zeros(3, dtype=np.float32)
        de = compute_de(pred, basal)
        np.testing.assert_array_almost_equal(de, pred)

    def test_identical_arrays_returns_zero(self):
        """DE = 0 when predicted_treated == region_basal."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        de = compute_de(arr, arr.copy())
        np.testing.assert_array_almost_equal(de, np.zeros(3, dtype=np.float32))

    def test_raises_on_2d_predicted(self):
        """2-D predicted_treated raises ValueError."""
        with pytest.raises(ValueError, match="predicted_treated must be 1-D"):
            compute_de(np.ones((2, 3)), np.ones(3))

    def test_raises_on_2d_basal(self):
        """2-D predicted_control raises ValueError (rule B, D-02)."""
        with pytest.raises(ValueError, match="predicted_control must be 1-D"):
            compute_de(np.ones(3), np.ones((2, 3)))

    def test_raises_on_shape_mismatch(self):
        """Different lengths raise ValueError mentioning shape mismatch."""
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_de(np.ones(5), np.ones(3))

    def test_large_array_no_overflow(self):
        """Large float32 values should not overflow in subtraction."""
        big = np.full(978, 1e30, dtype=np.float32)
        zero = np.zeros(978, dtype=np.float32)
        de = compute_de(big, zero)
        assert np.all(np.isfinite(de))

    def test_negative_values_preserved(self):
        """Negative DE values are preserved accurately."""
        pred = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        basal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        de = compute_de(pred, basal)
        np.testing.assert_array_almost_equal(de, np.array([-1.0, -2.0, -3.0]))


# ---------------------------------------------------------------------------
# Tests: make_cache_key
# ---------------------------------------------------------------------------


class TestMakeCacheKey:
    """Tests for deterministic cache key construction."""

    def test_returns_64_char_hex_string(self, three_genes):
        """Key is a 64-character lowercase hex SHA-256 digest."""
        key = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        assert isinstance(key, str)
        assert len(key) == 64
        assert key == key.lower()
        # All hex characters
        int(key, 16)

    def test_determinism(self, three_genes):
        """Same inputs always produce the same key."""
        k1 = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        k2 = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        assert k1 == k2

    def test_different_pert_id_different_key(self, three_genes):
        """Different pert_id → different key."""
        k1 = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        k2 = make_cache_key("drug_B", "periportal", "multidcp_pdg", three_genes)
        assert k1 != k2

    def test_different_region_different_key(self, three_genes):
        """Different region → different key."""
        k1 = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        k2 = make_cache_key("drug_A", "pericentral", "multidcp_pdg", three_genes)
        assert k1 != k2

    def test_different_model_variant_different_key(self, three_genes):
        """Different model_variant → different key (namespace separation)."""
        k1 = make_cache_key("drug_A", "periportal", "multidcp_pdg", three_genes)
        k2 = make_cache_key("drug_A", "periportal", "multidcp_chemoe", three_genes)
        assert k1 != k2

    def test_gene_order_sensitive(self):
        """Different gene_ids orderings produce different keys."""
        genes_ab = ("GENE_A", "GENE_B", "GENE_C")
        genes_ba = ("GENE_B", "GENE_A", "GENE_C")
        k1 = make_cache_key("drug_X", "zone_1", "multidcp_pdg", genes_ab)
        k2 = make_cache_key("drug_X", "zone_1", "multidcp_pdg", genes_ba)
        assert k1 != k2, (
            "Gene order must be embedded in the key; reordering the same "
            "genes should produce a different hash."
        )

    def test_gene_id_set_sensitive(self):
        """Different gene_ids sets (not just ordering) produce different keys."""
        genes_3 = ("GENE_A", "GENE_B", "GENE_C")
        genes_4 = ("GENE_A", "GENE_B", "GENE_C", "GENE_D")
        k1 = make_cache_key("drug_X", "zone_1", "multidcp_pdg", genes_3)
        k2 = make_cache_key("drug_X", "zone_1", "multidcp_pdg", genes_4)
        assert k1 != k2


# ---------------------------------------------------------------------------
# Tests: build_manifest
# ---------------------------------------------------------------------------


class TestBuildManifest:
    """Tests for manifest construction and validation."""

    def test_returns_named_tuple(self, two_pert_ids, two_regions, three_genes):
        """Return type is RegionSignatureManifest NamedTuple."""
        m = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=list(three_genes),
            model_variant="multidcp_pdg",
        )
        assert isinstance(m, RegionSignatureManifest)

    def test_axes_are_frozen_tuples(self, two_pert_ids, two_regions, three_genes):
        """pert_ids, regions, gene_ids are stored as tuples."""
        m = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=list(three_genes),
            model_variant="multidcp_pdg",
        )
        assert isinstance(m.pert_ids, tuple)
        assert isinstance(m.regions, tuple)
        assert isinstance(m.gene_ids, tuple)

    def test_counts_match_axes(self, two_pert_ids, two_regions, three_genes):
        """n_pert_ids, n_regions, n_genes equal the axis lengths."""
        m = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=list(three_genes),
            model_variant="multidcp_pdg",
        )
        assert m.n_pert_ids == len(two_pert_ids)
        assert m.n_regions == len(two_regions)
        assert m.n_genes == len(three_genes)

    def test_de_convention_field(self, two_pert_ids, two_regions, three_genes):
        """de_convention is fixed to the spatial-arm string."""
        m = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=list(three_genes),
            model_variant="multidcp_pdg",
        )
        assert m.de_convention == (
            "predicted_treated(drug) - predicted_control(region_basal)"
        )

    def test_model_variant_preserved(self, two_pert_ids, two_regions, three_genes):
        """model_variant is stored unchanged."""
        m = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=list(three_genes),
            model_variant="multidcp_chemoe",
        )
        assert m.model_variant == "multidcp_chemoe"

    def test_raises_on_empty_pert_ids(self, two_regions, three_genes):
        """Empty pert_ids raises ValueError."""
        with pytest.raises(ValueError, match="pert_ids"):
            build_manifest(
                pert_ids=[],
                regions=two_regions,
                gene_ids=list(three_genes),
                model_variant="multidcp_pdg",
            )

    def test_raises_on_empty_regions(self, two_pert_ids, three_genes):
        """Empty regions raises ValueError."""
        with pytest.raises(ValueError, match="regions"):
            build_manifest(
                pert_ids=two_pert_ids,
                regions=[],
                gene_ids=list(three_genes),
                model_variant="multidcp_pdg",
            )

    def test_raises_on_empty_gene_ids(self, two_pert_ids, two_regions):
        """Empty gene_ids raises ValueError."""
        with pytest.raises(ValueError, match="gene_ids"):
            build_manifest(
                pert_ids=two_pert_ids,
                regions=two_regions,
                gene_ids=[],
                model_variant="multidcp_pdg",
            )

    def test_raises_on_duplicate_pert_ids(self, two_regions, three_genes):
        """Duplicate pert_ids raise ValueError."""
        with pytest.raises(ValueError, match="pert_ids"):
            build_manifest(
                pert_ids=["drug_A", "drug_A"],
                regions=two_regions,
                gene_ids=list(three_genes),
                model_variant="multidcp_pdg",
            )

    def test_raises_on_duplicate_regions(self, two_pert_ids, three_genes):
        """Duplicate regions raise ValueError."""
        with pytest.raises(ValueError, match="regions"):
            build_manifest(
                pert_ids=two_pert_ids,
                regions=["periportal", "periportal"],
                gene_ids=list(three_genes),
                model_variant="multidcp_pdg",
            )

    def test_raises_on_duplicate_gene_ids(self, two_pert_ids, two_regions):
        """Duplicate gene_ids raise ValueError."""
        with pytest.raises(ValueError, match="gene_ids"):
            build_manifest(
                pert_ids=two_pert_ids,
                regions=two_regions,
                gene_ids=["G1", "G1", "G2"],
                model_variant="multidcp_pdg",
            )

    def test_raises_on_invalid_model_variant(self, two_pert_ids, two_regions, three_genes):
        """Unknown model_variant raises ValueError."""
        with pytest.raises(ValueError, match="model_variant"):
            build_manifest(
                pert_ids=two_pert_ids,
                regions=two_regions,
                gene_ids=list(three_genes),
                model_variant="unknown_model_xyz",
            )

    def test_axis_order_preserved(self):
        """pert_ids, regions, gene_ids axes preserve caller-supplied order."""
        pids = ["C", "A", "B"]
        regs = ["zone_3", "zone_1"]
        genes = ["G_Z", "G_A", "G_M"]
        m = build_manifest(
            pert_ids=pids,
            regions=regs,
            gene_ids=genes,
            model_variant="multidcp_pdg",
        )
        assert list(m.pert_ids) == pids
        assert list(m.regions) == regs
        assert list(m.gene_ids) == genes


# ---------------------------------------------------------------------------
# Tests: assemble_cache
# ---------------------------------------------------------------------------


class TestAssembleCache:
    """Tests for the pure cache assembly function."""

    def test_output_type(self, small_manifest, two_pert_ids, two_regions):
        """Return type is RegionSignatureCache."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        cache = assemble_cache(pred_map, ctrl_map, small_manifest)
        assert isinstance(cache, RegionSignatureCache)

    def test_de_array_shape(self, small_manifest, two_pert_ids, two_regions):
        """de_array shape is (n_pert_ids, n_regions, n_genes)."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        cache = assemble_cache(pred_map, ctrl_map, small_manifest)
        expected = (small_manifest.n_pert_ids, small_manifest.n_regions, small_manifest.n_genes)
        assert cache.de_array.shape == expected

    def test_de_array_dtype_float32(
        self, small_manifest, two_pert_ids, two_regions
    ):
        """de_array dtype is float32."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        cache = assemble_cache(pred_map, ctrl_map, small_manifest)
        assert cache.de_array.dtype == np.float32

    def test_manifest_passthrough(
        self, small_manifest, two_pert_ids, two_regions
    ):
        """Manifest stored in cache is identical to the one passed in."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        cache = assemble_cache(pred_map, ctrl_map, small_manifest)
        assert cache.manifest is small_manifest

    def test_de_values_are_correct(self):
        """de_array[i, j, :] == treated[(pid, reg)] - control[(pid, reg)] (rule B)."""
        pert_ids = ["drug_1", "drug_2"]
        regions = ["zone_A", "zone_B"]
        gene_ids = ["G1", "G2", "G3", "G4"]

        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=gene_ids,
            model_variant="multidcp_pdg",
        )

        # Use fully deterministic synthetic values so we can compute expected.
        pred_vals = {
            ("drug_1", "zone_A"): np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            ("drug_1", "zone_B"): np.array([11.0, 21.0, 31.0, 41.0], dtype=np.float32),
            ("drug_2", "zone_A"): np.array([12.0, 22.0, 32.0, 42.0], dtype=np.float32),
            ("drug_2", "zone_B"): np.array([13.0, 23.0, 33.0, 43.0], dtype=np.float32),
        }
        # Rule B (D-02): control is the model's own predicted control output,
        # keyed per (pert_id, region). Region-determined, so both drugs share the
        # same control vector within a region.
        ctrl_vals = {
            ("drug_1", "zone_A"): np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ("drug_2", "zone_A"): np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            ("drug_1", "zone_B"): np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ("drug_2", "zone_B"): np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        }

        cache = assemble_cache(pred_vals, ctrl_vals, manifest)

        for (pid, reg, i, j) in [
            ("drug_1", "zone_A", 0, 0),
            ("drug_1", "zone_B", 0, 1),
            ("drug_2", "zone_A", 1, 0),
            ("drug_2", "zone_B", 1, 1),
        ]:
            expected = pred_vals[(pid, reg)] - ctrl_vals[(pid, reg)]
            np.testing.assert_array_equal(cache.de_array[i, j, :], expected)
            np.testing.assert_array_equal(cache.treated_array[i, j, :], pred_vals[(pid, reg)])
            np.testing.assert_array_equal(cache.control_array[i, j, :], ctrl_vals[(pid, reg)])

    def test_axis_alignment_pert_id_order(self):
        """Axis 0 follows the manifest pert_id order, not dict insertion order."""
        # Manifest has pert_ids in order [B, A]. The de_array axis 0 must follow
        # this order even though the maps may be stored in a different dict order.
        pert_ids = ["drug_B", "drug_A"]
        regions = ["zone_1"]
        gene_ids = ["G1", "G2"]

        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=gene_ids,
            model_variant="multidcp_pdg",
        )

        pred_map = {
            ("drug_B", "zone_1"): np.array([100.0, 200.0], dtype=np.float32),
            ("drug_A", "zone_1"): np.array([300.0, 400.0], dtype=np.float32),
        }
        ctrl_map = {
            ("drug_B", "zone_1"): np.array([10.0, 20.0], dtype=np.float32),
            ("drug_A", "zone_1"): np.array([10.0, 20.0], dtype=np.float32),
        }
        cache = assemble_cache(pred_map, ctrl_map, manifest)

        # Axis 0 index 0 = drug_B (the first element in pert_ids)
        expected_drug_B = pred_map[("drug_B", "zone_1")] - ctrl_map[("drug_B", "zone_1")]
        np.testing.assert_array_equal(cache.de_array[0, 0, :], expected_drug_B)

        # Axis 0 index 1 = drug_A
        expected_drug_A = pred_map[("drug_A", "zone_1")] - ctrl_map[("drug_A", "zone_1")]
        np.testing.assert_array_equal(cache.de_array[1, 0, :], expected_drug_A)

    def test_raises_on_missing_pert_id_region_pair(
        self, small_manifest, two_pert_ids, two_regions
    ):
        """KeyError if a (pert_id, region) pair is absent from treated map."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        # Remove one entry.
        missing_key = (two_pert_ids[0], two_regions[0])
        del pred_map[missing_key]
        with pytest.raises(KeyError):
            assemble_cache(pred_map, ctrl_map, small_manifest)

    def test_raises_on_missing_pair_in_control_map(
        self, small_manifest, two_pert_ids, two_regions
    ):
        """KeyError if predicted_control_map is missing a manifest pair (rule B)."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        del ctrl_map[(two_pert_ids[0], two_regions[0])]
        with pytest.raises(KeyError, match="predicted_control_map"):
            assemble_cache(pred_map, ctrl_map, small_manifest)

    def test_raises_on_gene_count_mismatch(self, two_pert_ids, two_regions):
        """ValueError if a predicted array has wrong gene count."""
        n_genes = 5
        gene_ids = [f"G{i}" for i in range(n_genes)]
        manifest = build_manifest(
            pert_ids=two_pert_ids,
            regions=two_regions,
            gene_ids=gene_ids,
            model_variant="multidcp_pdg",
        )
        # treated + control arrays have 5 genes each — this is fine.
        pred_map = _make_predicted_treated_map(two_pert_ids, two_regions, n_genes)
        ctrl_map = _make_predicted_control_map(two_pert_ids, two_regions, n_genes)

        # Now corrupt one treated array to the wrong size.
        pred_map[(two_pert_ids[0], two_regions[0])] = np.zeros(n_genes + 1, dtype=np.float32)
        with pytest.raises(ValueError):
            assemble_cache(pred_map, ctrl_map, manifest)

    def test_determinism(self, small_manifest, two_pert_ids, two_regions):
        """Same inputs → identical de_array contents."""
        pred_map = _make_predicted_treated_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        ctrl_map = _make_predicted_control_map(
            two_pert_ids, two_regions, small_manifest.n_genes
        )
        cache_a = assemble_cache(pred_map, ctrl_map, small_manifest)
        cache_b = assemble_cache(pred_map, ctrl_map, small_manifest)
        np.testing.assert_array_equal(cache_a.de_array, cache_b.de_array)

    def test_many_pert_ids_many_regions(self):
        """Functional test: 10 drugs × 5 regions × 50 genes assembles without error."""
        n_p, n_r, n_g = 10, 5, 50
        pert_ids = [f"drug_{i}" for i in range(n_p)]
        regions = [f"region_{j}" for j in range(n_r)]
        gene_ids = [f"gene_{k}" for k in range(n_g)]

        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=gene_ids,
            model_variant="multidcp_chemoe",
        )

        pred_map = _make_predicted_treated_map(pert_ids, regions, n_g, seed=7)
        ctrl_map = _make_predicted_control_map(pert_ids, regions, n_g, seed=8)

        cache = assemble_cache(pred_map, ctrl_map, manifest)
        assert cache.de_array.shape == (n_p, n_r, n_g)
        assert cache.treated_array.shape == (n_p, n_r, n_g)
        assert cache.control_array.shape == (n_p, n_r, n_g)
        assert cache.de_array.dtype == np.float32
        assert cache.manifest.n_pert_ids == n_p
        assert cache.manifest.n_regions == n_r
        assert cache.manifest.n_genes == n_g


# ---------------------------------------------------------------------------
# Tests: RegionSignatureCacher (constructor, NotImplementedError stubs)
# ---------------------------------------------------------------------------


class TestRegionSignatureCacher:
    """Tests for cacher construction and model-stub guard."""

    def test_constructor_valid_multidcp_pdg(self, three_genes):
        """Constructor accepts multidcp_pdg model variant."""
        cacher = RegionSignatureCacher(
            model_variant="multidcp_pdg",
            gene_ids=three_genes,
        )
        assert cacher.model_variant == "multidcp_pdg"
        assert cacher.gene_ids == three_genes

    def test_constructor_valid_multidcp_chemoe(self, three_genes):
        """Constructor accepts multidcp_chemoe model variant."""
        cacher = RegionSignatureCacher(
            model_variant="multidcp_chemoe",
            gene_ids=three_genes,
        )
        assert cacher.model_variant == "multidcp_chemoe"

    def test_constructor_invalid_variant_raises(self, three_genes):
        """Unknown model_variant raises ValueError."""
        with pytest.raises(ValueError, match="model_variant"):
            RegionSignatureCacher(
                model_variant="gpt4_chemoe",
                gene_ids=three_genes,
            )

    def test_gene_ids_stored_as_tuple(self):
        """gene_ids are stored as a tuple regardless of input type."""
        cacher = RegionSignatureCacher(
            model_variant="multidcp_pdg",
            gene_ids=("G1", "G2", "G3"),
        )
        assert isinstance(cacher.gene_ids, tuple)

    def test_load_model_missing_checkpoint_raises(self, three_genes):
        """load_model on a missing checkpoint path raises FileNotFoundError.

        The seam is now a REAL strict-load (02-02): there is no NotImplementedError
        stub. A nonexistent path fails fast (no GPU/checkpoint needed, so this
        stays in the pure suite). Real strict-load + forward are covered by the
        gpu-marked test_model_load.py smoke (Hard Rule 1: not mocked).
        """
        cacher = RegionSignatureCacher(
            model_variant="multidcp_chemoe",
            gene_ids=three_genes,
        )
        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            cacher.load_model("/some/path/does_not_exist_best.pt")

    def test_load_model_does_not_reference_row18_kpgt(self):
        """The wired backbone is row-17 only; row-18 chemoe_kpgt is NOT wired."""
        import src.spatial.region_signature as rs

        src = open(rs.__file__).read()
        assert "chemoe_kpgt" not in src, (
            "S-B descoped (D-04 amendment): the collapsed row-18 chemoe_kpgt "
            "checkpoint must not be referenced as a wired backbone."
        )

    def test_call_model_before_load_raises(self, three_genes):
        """_call_model before load_model raises RuntimeError (model not loaded)."""
        cacher = RegionSignatureCacher(
            model_variant="multidcp_chemoe",
            gene_ids=three_genes,
        )
        with pytest.raises(RuntimeError, match="model not loaded"):
            cacher._call_model("CCO", np.zeros(len(three_genes), dtype=np.float32))

    def test_run_before_load_raises(self, three_genes):
        """run() before load_model raises RuntimeError (model not loaded)."""
        cacher = RegionSignatureCacher(
            model_variant="multidcp_chemoe",
            gene_ids=three_genes,
        )
        with pytest.raises(RuntimeError, match="model not loaded"):
            cacher.run(
                pert_ids=["drug_A"],
                smiles_map={"drug_A": "CCO"},
                region_basal_map={
                    "periportal": np.zeros(len(three_genes), dtype=np.float32)
                },
            )


# ---------------------------------------------------------------------------
# Tests: N_LANDMARK constant (gene space contract)
# ---------------------------------------------------------------------------


class TestGeneSpaceConstant:
    """Verify the N_LANDMARK constant matches the project-wide contract."""

    def test_n_landmark_value(self):
        """N_LANDMARK is 978 (LINCS L1000 landmark gene count)."""
        assert N_LANDMARK == 978, (
            f"N_LANDMARK must be 978 to match MultiDCP training space; "
            f"got {N_LANDMARK}. If this changed, update spatial_basal.config too."
        )

    def test_compute_de_accepts_978_genes(self):
        """compute_de works correctly on N_LANDMARK-sized vectors."""
        pred = np.ones(N_LANDMARK, dtype=np.float32)
        basal = np.zeros(N_LANDMARK, dtype=np.float32)
        de = compute_de(pred, basal)
        assert de.shape == (N_LANDMARK,)
        assert np.all(de == 1.0)

    def test_build_manifest_accepts_978_genes(self):
        """build_manifest accepts a 978-element gene_ids list."""
        gene_ids = [f"GENE_{i:04d}" for i in range(N_LANDMARK)]
        m = build_manifest(
            pert_ids=["drug_A"],
            regions=["periportal"],
            gene_ids=gene_ids,
            model_variant="multidcp_pdg",
        )
        assert m.n_genes == N_LANDMARK


# ---------------------------------------------------------------------------
# Tests: RegionDE NamedTuple (result container)
# ---------------------------------------------------------------------------


class TestRegionDEContainer:
    """Verify RegionDE stores all fields correctly."""

    def test_fields_accessible(self):
        """All four fields are accessible as named attributes."""
        vec = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        rde = RegionDE(
            pert_id="drug_X",
            region="periportal",
            de_vector=vec,
            n_genes=3,
        )
        assert rde.pert_id == "drug_X"
        assert rde.region == "periportal"
        np.testing.assert_array_equal(rde.de_vector, vec)
        assert rde.n_genes == 3

    def test_is_named_tuple(self):
        """RegionDE is a NamedTuple (tuple subclass)."""
        rde = RegionDE(
            pert_id="x", region="r",
            de_vector=np.zeros(3, dtype=np.float32), n_genes=3
        )
        assert isinstance(rde, tuple)


# ---------------------------------------------------------------------------
# Phase 2 / Wave 0 — RED contracts for rule B + the 3-vector cache (D-02/D-03)
# ---------------------------------------------------------------------------
# These tests assert the NEW Phase-2 contracts that land in 02-02:
#   * compute_de subtracts a SECOND PREDICTED vector (rule B), and the manifest
#     de_convention string records "predicted_treated(drug) - predicted_control".
#   * RegionSignatureCache carries treated_array AND control_array AND de_array
#     (D-03), and assemble_cache accepts a predicted_control_map.
#   * N_PDG == 10716 is a module constant alongside N_LANDMARK (D-01).
# They are EXPECTED to FAIL (RED) now: the current rule-A convention string,
# the 1-vector cache, and the missing N_PDG constant. 02-02 flips them to GREEN
# by editing source only.


class TestRuleBAndThreeVectorCache:
    """RED until 02-02: rule-B DE convention + 3-vector cache + N_PDG."""

    def test_rule_b_de_subtracts_control(self, synthetic_treated_control):
        # RED until 02-02
        treated, control = synthetic_treated_control

        # The rule-B subtraction itself: treated - control, element-wise float32.
        de = compute_de(treated, control)
        np.testing.assert_array_almost_equal(de, treated - control)
        assert de.dtype == np.float32

        # Force RED on the convention string: the manifest must record the
        # rule-B convention. The current source returns the rule-A string.
        manifest = build_manifest(
            pert_ids=["drug_X"],
            regions=["pericentral"],
            gene_ids=["GENE_A", "GENE_B", "GENE_C"],
            model_variant="multidcp_chemoe",
        )
        assert manifest.de_convention == (
            "predicted_treated(drug) - predicted_control(region_basal)"
        ), (
            "Phase-2 rule B (D-02): de_convention must record the "
            "predicted-control subtraction, not the rule-A region_basal string."
        )

    def test_cache_three_vector_schema(self):
        # RED until 02-02
        pert_ids = ["drug_1", "drug_2"]
        regions = ["pericentral", "periportal"]
        gene_ids = ["G1", "G2", "G3", "G4"]
        n_p, n_r, n_g = len(pert_ids), len(regions), len(gene_ids)

        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=gene_ids,
            model_variant="multidcp_chemoe",
        )

        rng = np.random.default_rng(3)
        predicted_treated_map = {
            (p, r): rng.standard_normal(n_g).astype(np.float32)
            for p in pert_ids
            for r in regions
        }
        predicted_control_map = {
            (p, r): rng.standard_normal(n_g).astype(np.float32)
            for p in pert_ids
            for r in regions
        }

        # Rule-B assembly takes a predicted_control_map (NOT a region_basal_map).
        # Current assemble_cache signature differs -> TypeError RED now.
        cache = assemble_cache(
            predicted_treated_map, predicted_control_map, manifest
        )

        # 3-vector schema (D-03): all three arrays present, same shape, float32.
        for attr in ("treated_array", "control_array", "de_array"):
            assert hasattr(cache, attr), (
                f"RegionSignatureCache must expose '{attr}' (D-03 3-vector cache)."
            )
        expected_shape = (n_p, n_r, n_g)
        assert cache.treated_array.shape == expected_shape
        assert cache.control_array.shape == expected_shape
        assert cache.de_array.shape == expected_shape
        assert cache.treated_array.dtype == np.float32
        assert cache.control_array.dtype == np.float32
        assert cache.de_array.dtype == np.float32

        # de_array == treated_array - control_array (rule B at the cache layer).
        np.testing.assert_array_almost_equal(
            cache.de_array, cache.treated_array - cache.control_array
        )

    def test_n_pdg_constant(self):
        # RED until 02-02 (ImportError on the missing constant; D-01).
        from src.spatial.region_signature import N_PDG

        assert N_PDG == 10716, (
            f"N_PDG must be 10716 (MultiDCP-CheMoE PDG gene space); got {N_PDG}."
        )
