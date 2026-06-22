"""Content/shape guard for acquired spatial datasets (DATA-01 hardening).

Motivation: existence of a named file is NOT proof of usable data. Phase 0 found
three ways a dataset can "pass" acquisition yet be unusable as input:
  - wrong study entirely (chen_brain_mtg was an ALS bulk RNA-seq series),
  - images-only archive (abedini GSE211785_RAW.tar = .tif + spatial .json, no counts),
  - metadata-only (maynard = sample-sheet CSV, no expression).
`expected_files` checks existence, never content. This module's `dataset_has_counts`
inspects what is actually on disk (loose files + one level of archive nesting) and
decides whether a gene-expression COUNT artifact is present.

These tests are sanity-only (never run on model outputs). The unit tests use
synthetic archives in tmp_path (offline, deterministic) to prove the guard has
teeth; the integration test runs the guard over the real downloaded inputs.
"""

from __future__ import annotations

import gzip
import io
import pathlib
import tarfile
import zipfile

import pytest

from src.spatial.datasets import SPATIAL_DATASETS
from src.spatial.data_validation import dataset_has_counts, validate_usable_inputs

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "spatial"


# ---------------------------------------------------------------------------
# Helpers: build synthetic dataset directories
# ---------------------------------------------------------------------------


def _touch(path: pathlib.Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _make_tar(path: pathlib.Path, members: dict[str, bytes], gz: bool = False) -> None:
    """Write a tar (optionally gzipped) with {arcname: bytes}."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if gz else "w"
    with tarfile.open(path, mode) as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _make_zip(path: pathlib.Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _entry(slug: str):
    """A minimal duck-typed registry entry (only .slug is read by the guard)."""
    class _E:
        pass

    e = _E()
    e.slug = slug
    return e


# ---------------------------------------------------------------------------
# Unit tests — the guard must ACCEPT real count artifacts
# ---------------------------------------------------------------------------


def test_accepts_loose_10x_triplet(tmp_path):
    d = tmp_path / "ds_a"
    _touch(d / "matrix.mtx.gz")
    _touch(d / "barcodes.tsv.gz")
    _touch(d / "features.tsv.gz")
    assert dataset_has_counts(_entry("ds_a"), tmp_path).has_counts


def test_accepts_loose_spaceranger_h5(tmp_path):
    d = tmp_path / "ds_b"
    _touch(d / "GSM1_filtered_feature_bc_matrix.h5")
    assert dataset_has_counts(_entry("ds_b"), tmp_path).has_counts


def test_accepts_seurat_rds(tmp_path):
    # GSE252772 mouse kidney ships per-sample Seurat objects.
    d = tmp_path / "ds_c"
    _touch(d / "GSM9_sample_obj.rds.gz")
    assert dataset_has_counts(_entry("ds_c"), tmp_path).has_counts


def test_accepts_zip_with_matrix(tmp_path):
    # yu2022 / andrews style: counts inside a per-sample zip.
    d = tmp_path / "ds_d"
    d.mkdir()
    _make_zip(d / "L5_upload.zip", {"L5_upload/filtered_feature_bc_matrix.h5": b"\x00"})
    assert dataset_has_counts(_entry("ds_d"), tmp_path).has_counts


def test_accepts_nested_tar(tmp_path):
    # lake_kpmp style: RAW.tar -> per-sample .tar.gz -> filtered_feature_bc_matrix.h5
    d = tmp_path / "ds_e"
    d.mkdir()
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w:gz") as tf:
        payload = b"\x00"
        info = tarfile.TarInfo("filtered_feature_bc_matrix.h5")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    inner_bytes = inner.getvalue()
    _make_tar(
        d / "GSE_RAW.tar",
        {
            "GSM1_sample.tar.gz": inner_bytes,
            "GSM1_sample.tif.gz": b"\x00",  # decoy image
        },
    )
    res = dataset_has_counts(_entry("ds_e"), tmp_path)
    assert res.has_counts, res.checked


# ---------------------------------------------------------------------------
# Unit tests — the guard must REJECT the Phase-0 failure modes
# ---------------------------------------------------------------------------


def test_rejects_images_only(tmp_path):
    # abedini GSE211785_RAW.tar failure mode: only images + spatial json.
    d = tmp_path / "ds_f"
    d.mkdir()
    _make_tar(
        d / "GSE211785_RAW.tar",
        {
            "GSM1_sec.tif.gz": b"\x00",
            "GSM1_sec.json.gz": b"{}",
        },
    )
    assert not dataset_has_counts(_entry("ds_f"), tmp_path).has_counts


def test_rejects_metadata_only(tmp_path):
    # maynard failure mode: only a sample-sheet CSV.
    d = tmp_path / "ds_g"
    _touch(d / "metadata_spatialLIBD.csv", b"a,b,c\n1,2,3\n")
    assert not dataset_has_counts(_entry("ds_g"), tmp_path).has_counts


def test_rejects_de_table_only(tmp_path):
    # chen failure mode (pre-fix): a DESeq2 expression text table, not a matrix.
    d = tmp_path / "ds_h"
    _touch(d / "GSE200474_Deseq2_normalized.txt.gz", gzip.compress(b"gene\tx\n"))
    assert not dataset_has_counts(_entry("ds_h"), tmp_path).has_counts


def test_rejects_longread_gtf(tmp_path):
    # canela failure mode: long-read isoform outputs, no count matrix.
    d = tmp_path / "ds_i"
    d.mkdir()
    _make_tar(
        d / "GSE202327_RAW.tar",
        {
            "GSM1_corrected.gtf.gz": b"\x00",
            "GSM1.bed12.FILTERED.bb": b"\x00",
        },
    )
    assert not dataset_has_counts(_entry("ds_i"), tmp_path).has_counts


def test_missing_directory_is_not_counts(tmp_path):
    res = dataset_has_counts(_entry("does_not_exist"), tmp_path)
    assert not res.has_counts


# ---------------------------------------------------------------------------
# Integration — every usable_as_input dataset on disk must contain counts
# ---------------------------------------------------------------------------


def test_validate_skips_empty_dir(tmp_path):
    # KPMP-consistency: an empty dir (DUA/ToS-gated skip) is not a content failure.
    (tmp_path / "ds_empty").mkdir()
    e = _entry("ds_empty")
    e.usable_as_input = True
    assert validate_usable_inputs([e], tmp_path) == {}


def test_validate_flags_present_but_countless(tmp_path):
    d = tmp_path / "ds_bad"
    _touch(d / "only_image.tif")
    e = _entry("ds_bad")
    e.usable_as_input = True
    assert "ds_bad" in validate_usable_inputs([e], tmp_path)


# ---------------------------------------------------------------------------
# Driver regression guards (CR-01 / CR-02 / CR-03)
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402

_DRIVER = ROOT / "scripts" / "download_spatial.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("download_spatial_mod", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cr01_figshare_id_parsed_from_accession():
    drv = _load_driver()
    assert drv._figshare_article_id(
        "figshare: 22321447 (DOI 10.6084/m9.figshare.22321447.v1)"
    ) == "22321447"
    assert drv._figshare_article_id("figshare.17058105") == "17058105"
    assert drv._figshare_article_id("GEO: GSE185477") == ""  # non-figshare -> no id


def test_cr01_no_hardcoded_figshare_constant():
    # The module must not reintroduce a hardcoded article-id constant.
    assert "FIGSHARE_ARTICLE_ID" not in _DRIVER.read_text()


def test_cr02_lake_kpmp_uses_nonfatal_kpmp_mechanism():
    # lake/KPMP must dispatch through the non-fatal "kpmp" path so a DUA-gated 404
    # does not abort the whole run via the hard-halting geo_supp path.
    lake = next(d for d in SPATIAL_DATASETS if d.slug == "lake_kpmp_kidney")
    assert lake.access_mechanism == "kpmp"


def test_cr03_md5_mismatch_is_fatal():
    # The figshare md5-mismatch branch must call _halt (not just log.warning).
    text = _DRIVER.read_text()
    assert "recorded for MANIFEST review" not in text  # old warn-only text removed
    idx = text.index("MD5 MISMATCH")
    assert "_halt(" in text[idx - 200 : idx + 200]


def test_usable_inputs_have_counts() -> None:
    """The guard: no usable_as_input dataset present on disk may lack counts.

    Skips datasets not yet downloaded. Fails loudly if a present input is
    counts-free (the regression this whole module exists to prevent).
    """
    if not RAW.exists() or not any(RAW.iterdir()):
        pytest.skip("downloads not present; run with TDC_NETWORK_TESTS=1")

    present = [d for d in SPATIAL_DATASETS if d.usable_as_input and (RAW / d.slug).exists()]
    if not present:
        pytest.skip("no usable_as_input datasets on disk")

    failures = validate_usable_inputs(SPATIAL_DATASETS, RAW)
    if failures:
        lines = [f"{slug}: no count artifact found (checked {len(c.checked)} files)"
                 for slug, c in failures.items()]
        pytest.fail("usable_as_input datasets missing expression counts:\n" + "\n".join(lines))
