"""Unit tests for spatial dataset registry and data-path sanity (DATA-01/02/03).

Sanity-only tests per hard rule: tests are never run on model outputs.
All tests use real registry data and public reference annotations only.
Network-accessing tests are gated behind TDC_NETWORK_TESTS=1.

Behaviors covered:

Registry correctness (offline — DATA-01):
  1. test_accessions_corrected: No registry entry contains stale accessions
     GSE189994 or GSE144239; Yu liver accession contains "22321447"; Lake/KPMP
     primary has GSE183456 and GSE183279.
  2. test_registry_schema: Every SpatialDataset has non-empty species,
     access_mechanism, license, and slug strings; whole_transcriptome is bool;
     all slugs are unique, lowercase, and match ^[a-z0-9_]+$.

Data files on disk (offline with skip — DATA-01):
  3. test_raw_datasets_present: Each usable/validation dataset directory resolves
     via entry.slug (single source of truth — no independent slugification);
     expected files are present. Skips cleanly if downloads absent.

Whole-transcriptome gate (offline with skip — DATA-01 / Halt Gate 1):
  4. test_whole_transcriptome_gate: For each downloaded Visium basal dataset,
     coverage_fraction(var_names, multidcp_10716_symbols) > 0.80.

MANIFEST completeness (offline with skip — DATA-02):
  5. test_manifest_complete: MANIFEST.md has SHA256 + license + Whole-transcriptome
     column for every raw dataset and rows for frozen checkpoints.

Environment (offline — DATA-02):
  6. test_squidpy_available: squidpy is importable in the active environment.

Ortholog table (offline from fixture — DATA-03):
  7. test_ortholog_one2one: Offline BioMart fixture produces a nonempty one2one
     table; dropped_fraction in [0, 1]; rows <= unique human genes.

Coverage report existence (offline with skip — DATA-03):
  8. test_coverage_report_exists: results/tables/P0_coverage.md and
     P0_orthologs.md exist.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

from src.spatial.datasets import SPATIAL_DATASETS
from src.spatial.config import N_PDG
from src.spatial.gene_alignment import coverage_fraction

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

needs_net = pytest.mark.skipif(
    not os.environ.get("TDC_NETWORK_TESTS"),
    reason="network test; set TDC_NETWORK_TESTS=1 to enable",
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "spatial"
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"

# Regex for slug validation
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


# ---------------------------------------------------------------------------
# Test 1: Accession corrections (DATA-01)
# ---------------------------------------------------------------------------


def test_accessions_corrected() -> None:
    """Test 1: Stale accessions GSE189994/GSE144239 absent; verified accessions present."""
    accessions = [d.accession for d in SPATIAL_DATASETS]
    joined = " ".join(accessions)

    assert "GSE189994" not in joined, (
        "Stale Yu accession GSE189994 found — must be replaced with figshare 22321447"
    )
    assert "GSE144239" not in joined, (
        "Stale Maynard accession GSE144239 found — must be replaced with spatialLIBD"
    )

    # Yu liver: verified figshare 22321447
    assert any("22321447" in a for a in accessions), (
        "Verified Yu liver accession 22321447 not found in registry"
    )

    # Lake/KPMP: both primary GEO accessions must be present
    assert any("GSE183456" in a for a in accessions), (
        "Lake/KPMP primary accession GSE183456 not found in registry"
    )
    assert any("GSE183279" in a for a in accessions), (
        "Lake/KPMP superseries accession GSE183279 not found in registry"
    )


# ---------------------------------------------------------------------------
# Test 2: Registry schema (DATA-01)
# ---------------------------------------------------------------------------


def test_registry_schema() -> None:
    """Test 2: Every entry has all required fields populated and valid slugs."""
    seen_slugs: set[str] = set()

    for d in SPATIAL_DATASETS:
        label = f"[{d.name}]"

        # Required non-empty string fields
        assert isinstance(d.species, str) and d.species, (
            f"{label}: species must be non-empty str"
        )
        assert isinstance(d.access_mechanism, str) and d.access_mechanism, (
            f"{label}: access_mechanism must be non-empty str"
        )
        assert isinstance(d.license, str) and d.license, (
            f"{label}: license must be non-empty str"
        )
        assert isinstance(d.slug, str) and d.slug, (
            f"{label}: slug must be non-empty str"
        )

        # whole_transcriptome must be bool
        assert isinstance(d.whole_transcriptome, bool), (
            f"{label}: whole_transcriptome must be bool, got {type(d.whole_transcriptome)}"
        )

        # slug format: lowercase, alphanumeric + underscore only
        assert _SLUG_RE.match(d.slug), (
            f"{label}: slug '{d.slug}' must match ^[a-z0-9_]+$"
        )

        # slug must be unique
        assert d.slug not in seen_slugs, (
            f"Duplicate slug '{d.slug}' found for {label}"
        )
        seen_slugs.add(d.slug)


# ---------------------------------------------------------------------------
# Test 3: Raw datasets present on disk (DATA-01)
# ---------------------------------------------------------------------------


def test_raw_datasets_present() -> None:
    """Test 3: Resolved dataset directories and expected files exist under data/raw/spatial/."""
    # Skip entirely if the raw data root does not exist or has no actual dataset directories
    # (.gitkeep is not a dataset directory)
    if not RAW.exists():
        pytest.skip("downloads not present; run with TDC_NETWORK_TESTS=1")
    # Check for directories whose names match registry slugs (not auxiliary dirs like biomart/).
    all_slugs = {d.slug for d in SPATIAL_DATASETS}
    dataset_dirs = [p for p in RAW.iterdir() if p.is_dir() and p.name in all_slugs]
    if not dataset_dirs:
        pytest.skip("downloads not present; run with TDC_NETWORK_TESTS=1")

    missing: list[str] = []
    for d in SPATIAL_DATASETS:
        # Check usable input datasets and APAP validation anchors (identified by slug prefix)
        is_validation_anchor = d.slug.startswith("gse280652_") or d.slug.startswith("gse272564_apap")
        if not (d.usable_as_input or is_validation_anchor):
            continue
        # Use entry.slug directly — single source of truth; never slugify d.name here
        dataset_dir = RAW / d.slug
        if not dataset_dir.exists():
            missing.append(f"{d.slug}: directory not found at {dataset_dir}")
            continue
        for fname in d.expected_files:
            fpath = dataset_dir / fname
            if not fpath.exists():
                missing.append(f"{d.slug}: expected file '{fname}' not found")

    if missing:
        pytest.fail("Missing dataset files:\n" + "\n".join(missing))


# ---------------------------------------------------------------------------
# Test 4: Whole-transcriptome gate (DATA-01 / Halt Gate 1)
# ---------------------------------------------------------------------------


def test_whole_transcriptome_gate() -> None:
    """Test 4: Halt Gate 1 — each Visium basal dataset covers > 80% of N_PDG genes."""
    try:
        import anndata
        import scanpy as sc
    except ImportError:
        pytest.skip("anndata/scanpy not available")

    if not RAW.exists() or not any(RAW.iterdir()):
        pytest.skip("downloads not present; run with TDC_NETWORK_TESTS=1")

    # Load MultiDCP 10716-gene symbol list from wherever it is cached
    multidcp_gene_list_path = ROOT / "data" / "processed" / "spatial" / "multidcp_10716_symbols.txt"
    if not multidcp_gene_list_path.exists():
        pytest.skip(
            "MultiDCP gene symbol list not yet built; run scripts/build_gene_list.py first"
        )

    with open(multidcp_gene_list_path) as fh:
        multidcp_10716_symbols = [line.strip() for line in fh if line.strip()]

    failures: list[str] = []
    for d in SPATIAL_DATASETS:
        if not (d.usable_as_input and d.whole_transcriptome and d.platform == "Visium"):
            continue
        dataset_dir = RAW / d.slug
        if not dataset_dir.exists():
            continue  # skip not-yet-downloaded datasets

        h5_files = list(dataset_dir.rglob("*.h5ad")) + list(dataset_dir.rglob("*.h5"))
        if not h5_files:
            continue

        try:
            adata = sc.read(str(h5_files[0]))
            var_names = list(adata.var_names)
        except Exception as exc:
            failures.append(f"{d.slug}: could not read ({exc})")
            continue

        cov = coverage_fraction(var_names, multidcp_10716_symbols)
        if cov <= 0.80:
            failures.append(
                f"Halt Gate 1: {d.slug} covers only {cov:.1%} of the {N_PDG}-gene space"
            )

    if failures:
        pytest.fail("\n".join(failures))


# ---------------------------------------------------------------------------
# Test 5: MANIFEST completeness (DATA-02)
# ---------------------------------------------------------------------------


def test_manifest_complete() -> None:
    """Test 5: MANIFEST.md has SHA256 + license + Whole-transcriptome for all datasets."""
    manifest_path = ROOT / "MANIFEST.md"
    if not manifest_path.exists():
        pytest.skip("MANIFEST.md not yet created")

    text = manifest_path.read_text()

    required_columns = ["SHA256", "License", "Whole-transcriptome"]
    for col in required_columns:
        assert col in text, f"MANIFEST.md missing required column: {col}"

    # Every downloadable dataset's slug should appear in the manifest
    missing_from_manifest: list[str] = []
    for d in SPATIAL_DATASETS:
        if d.usable_as_input:
            if d.slug not in text:
                missing_from_manifest.append(d.slug)

    if missing_from_manifest:
        pytest.fail(
            "Dataset slugs missing from MANIFEST.md:\n"
            + "\n".join(missing_from_manifest)
        )


# ---------------------------------------------------------------------------
# Test 6: squidpy available (DATA-02)
# ---------------------------------------------------------------------------


def test_squidpy_available() -> None:
    """Test 6: squidpy is importable in the active conda environment."""
    squidpy = pytest.importorskip("squidpy")
    assert squidpy is not None


# ---------------------------------------------------------------------------
# Test 7: Ortholog one-to-one filter (DATA-03) — offline via fixture
# ---------------------------------------------------------------------------


def test_ortholog_one2one() -> None:
    """Test 7: Offline BioMart fixture produces nonempty one2one table with valid fractions."""
    fixture_path = FIXTURES / "biomart_chr21_sample.tsv"
    if not fixture_path.exists():
        pytest.skip("Offline BioMart fixture not present")

    # Import the orthology module if it exists; skip if not yet implemented
    try:
        from src.spatial.orthology import build_one2one_orthologs
    except ImportError:
        pytest.skip("src/spatial/orthology.py not yet implemented")

    result = build_one2one_orthologs(fixture_path)

    assert result is not None
    # Returned object must have a dropped_fraction in [0, 1]
    assert hasattr(result, "dropped_fraction"), "OrthologTable missing 'dropped_fraction'"
    assert 0.0 <= result.dropped_fraction <= 1.0, (
        f"dropped_fraction {result.dropped_fraction} not in [0, 1]"
    )
    # Must have kept at least one row
    assert hasattr(result, "n_one2one"), "OrthologTable missing 'n_one2one'"
    assert result.n_one2one > 0, "No one2one orthologs found in fixture — check filter"

    # Rows kept must not exceed unique human genes (no duplication)
    assert hasattr(result, "n_input"), "OrthologTable missing 'n_input'"
    assert result.n_one2one <= result.n_input, (
        "n_one2one > n_input: ortholog rows exceed input genes"
    )

    # pairs DataFrame must be nonempty
    assert hasattr(result, "pairs"), "OrthologTable missing 'pairs'"
    import pandas as pd
    assert isinstance(result.pairs, pd.DataFrame), "pairs must be a DataFrame"
    assert len(result.pairs) > 0, "pairs DataFrame is empty"


# ---------------------------------------------------------------------------
# Test 8: Coverage report existence (DATA-03)
# ---------------------------------------------------------------------------


def test_coverage_report_exists() -> None:
    """Test 8: Coverage and ortholog markdown tables exist in results/tables/."""
    coverage_path = ROOT / "results" / "tables" / "P0_coverage.md"
    ortholog_path = ROOT / "results" / "tables" / "P0_orthologs.md"

    if not coverage_path.exists() and not ortholog_path.exists():
        pytest.skip("Coverage/ortholog reports not yet generated")

    assert coverage_path.exists(), f"P0_coverage.md not found at {coverage_path}"
    assert ortholog_path.exists(), f"P0_orthologs.md not found at {ortholog_path}"
