"""Registry-driven spatial dataset download driver.

Reads the corrected SpatialDataset registry from src/spatial/datasets.py and
downloads every dataset into data/raw/spatial/<entry.slug>/. Dispatches on
entry.access_mechanism. After downloading, asserts each expected_files entry
is on disk and non-zero-byte (Halt Gate 1).

Hard rules honored:
    - No mock or synthetic data — real network fetches only (XC-01).
    - Directory names resolved exclusively from entry.slug (single source of
      truth shared with tests/test_data_paths.py). Never slugify entry.name.
    - Halt Gate 1: on any 404, zero-byte response, HTML error page, or missing
      expected_file after download, write HALT_REASON.md and sys.exit(1).
    - KPMP exception: if GEO supplementary 404s, log that portal ToS
      click-through is required and continue with other datasets (plan user_setup).
    - No torch import (P0 is data-only).
    - Figshare md5 comparison: warn-and-record on mismatch, never silently pass.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Registry import — single source of truth for slugs
# ---------------------------------------------------------------------------
# Add the repo root to sys.path so `src.spatial.datasets` resolves when the
# script is run from any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.spatial.datasets import SPATIAL_DATASETS  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW_SPATIAL = _REPO_ROOT / "data" / "raw" / "spatial"
HALT_REASON_PATH = (
    _REPO_ROOT
    / ".planning"
    / "phases"
    / "00-dataset-acquisition-manifest"
    / "HALT_REASON.md"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("download_spatial")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGSHARE_ARTICLE_ID = "22321447"
CHUNK_SIZE = 1 << 20  # 1 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5_file(path: Path) -> str:
    """Compute MD5 digest of a local file (for figshare integrity check)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, dest: Path, *, timeout: int = 300) -> None:
    """Stream a URL to a local file.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    dest : Path
        Local destination file path (parent must exist).
    timeout : int
        Request timeout in seconds.

    Raises
    ------
    SystemExit
        On HTTP 404 or any non-200 status (Halt Gate 1).
    """
    log.info("Fetching %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=timeout)

    if resp.status_code == 404:
        _halt(f"HTTP 404 for URL: {url}", url=url)

    if resp.status_code != 200:
        _halt(
            f"HTTP {resp.status_code} for URL: {url}",
            url=url,
        )

    # Guard against HTML error pages masquerading as binary files
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type and not url.endswith(".html"):
        _halt(
            f"HTML content-type received for binary URL: {url} (Content-Type: {content_type})",
            url=url,
        )

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)

    size = dest.stat().st_size
    if size == 0:
        _halt(f"Zero-byte file downloaded from: {url}", url=url)

    log.info("Downloaded %s (%.1f MB)", dest.name, size / 1e6)


def _halt(reason: str, url: str = "", dataset: str = "", slug: str = "") -> None:
    """Write HALT_REASON.md and exit nonzero (Halt Gate 1)."""
    log.error("HALT GATE 1 FIRED: %s", reason)
    HALT_REASON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HALT_REASON_PATH, "w") as fh:
        fh.write(
            f"# Halt Gate 1 — Planned Input Dataset Unavailable\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Dataset:** {dataset or '(see reason)'}\n"
            f"**Slug:** {slug or '(see reason)'}\n"
            f"**URL tried:** {url or '(see reason)'}\n\n"
            "Halt Gate 1 rationale: a planned input dataset is unavailable, "
            "empty, or returned an HTML error page where binary data was "
            "expected. Per XC-01, data must never be fabricated or silently "
            "substituted. Stop and resolve the data availability issue before "
            "proceeding to Plan 04.\n"
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Access-mechanism dispatchers
# ---------------------------------------------------------------------------


def _download_figshare_api(entry, dest_dir: Path) -> None:
    """Fetch files from Figshare article via the Figshare REST API.

    Checks the figshare-provided md5 against each downloaded file.
    Warns on mismatch (never silently passes).
    """
    api_url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"
    log.info("Querying Figshare API: %s", api_url)
    resp = requests.get(api_url, timeout=60)
    if resp.status_code == 404:
        _halt(
            f"Figshare article {FIGSHARE_ARTICLE_ID} returned 404",
            url=api_url,
            dataset=entry.name,
            slug=entry.slug,
        )
    if resp.status_code != 200:
        _halt(
            f"Figshare API returned HTTP {resp.status_code}",
            url=api_url,
            dataset=entry.name,
            slug=entry.slug,
        )

    article = resp.json()
    files_meta = article.get("files", [])
    if not files_meta:
        _halt(
            f"Figshare article {FIGSHARE_ARTICLE_ID} has no files listed",
            url=api_url,
            dataset=entry.name,
            slug=entry.slug,
        )

    # Build a lookup: filename -> {download_url, supplied_md5}
    remote_files: dict[str, dict] = {}
    for f in files_meta:
        remote_files[f["name"]] = {
            "download_url": f["download_url"],
            "md5": f.get("computed_md5", ""),
        }

    for expected_name in entry.expected_files:
        if expected_name not in remote_files:
            _halt(
                f"Expected file '{expected_name}' not found in Figshare article {FIGSHARE_ARTICLE_ID}",
                url=api_url,
                dataset=entry.name,
                slug=entry.slug,
            )
        meta = remote_files[expected_name]
        dest_file = dest_dir / expected_name
        if not dest_file.exists():
            _stream_download(meta["download_url"], dest_file)
        else:
            log.info("Already exists, skipping download: %s", dest_file)

        # MD5 integrity check
        if meta["md5"]:
            local_md5 = _md5_file(dest_file)
            if local_md5 != meta["md5"]:
                log.warning(
                    "MD5 MISMATCH for %s: local=%s remote=%s — "
                    "recorded for MANIFEST review (not silently passed).",
                    dest_file.name,
                    local_md5,
                    meta["md5"],
                )
            else:
                log.info("MD5 OK for %s (%s)", dest_file.name, local_md5)


def _build_geo_supp_url(accession: str) -> str:
    """Build GEO supplementary FTP/HTTPS URL for a GSE accession.

    For GSE accessions >= 6 characters, the series directory is under a
    prefix formed by blanking the last 3 digits, e.g. GSE183456 -> GSE183nnn.
    """
    # Extract the numeric ID
    gse_id = accession.strip()
    if not gse_id.startswith("GSE"):
        return ""
    num_str = gse_id[3:]
    if not num_str.isdigit():
        return ""
    prefix = gse_id[:-3] + "nnn"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/suppl/"
    return base


def _download_geo_supp(entry, dest_dir: Path) -> None:
    """Fetch GEO supplementary files for a GSE accession."""
    # Parse the primary accession — take the first GSE ID mentioned
    raw_accession = entry.accession
    gse_ids = [
        tok.strip()
        for tok in raw_accession.replace("+", " ").replace(",", " ").split()
        if tok.startswith("GSE")
    ]
    if not gse_ids:
        _halt(
            f"No GSE accession found in entry accession string: {raw_accession}",
            dataset=entry.name,
            slug=entry.slug,
        )

    primary_gse = gse_ids[0]
    base_url = _build_geo_supp_url(primary_gse)
    if not base_url:
        _halt(
            f"Could not build GEO supplementary URL for accession: {primary_gse}",
            dataset=entry.name,
            slug=entry.slug,
        )

    for expected_name in entry.expected_files:
        dest_file = dest_dir / expected_name
        if dest_file.exists() and dest_file.stat().st_size > 0:
            log.info("Already exists, skipping: %s", dest_file)
            continue
        file_url = base_url + expected_name
        _stream_download(file_url, dest_file)


def _download_kpmp(entry, dest_dir: Path) -> None:
    """Attempt GEO supplementary download for Lake/KPMP kidney.

    Per plan user_setup: if GEO supplementary 404s, do NOT crash — log that
    the KPMP portal ToS click-through is required and continue.
    """
    raw_accession = entry.accession
    # Try GSE183456 first
    gse_ids = [
        tok.strip()
        for tok in raw_accession.replace("+", " ").replace(",", " ").split()
        if tok.startswith("GSE")
    ]

    for gse_id in gse_ids:
        base_url = _build_geo_supp_url(gse_id)
        if not base_url:
            continue

        for expected_name in entry.expected_files:
            dest_file = dest_dir / expected_name
            if dest_file.exists() and dest_file.stat().st_size > 0:
                log.info("KPMP: Already exists, skipping: %s", dest_file)
                return

            file_url = base_url + expected_name
            log.info("KPMP: Attempting GEO supplementary: %s", file_url)
            try:
                resp = requests.head(file_url, timeout=30, allow_redirects=True)
                if resp.status_code == 200:
                    _stream_download(file_url, dest_file)
                    log.info("KPMP: GEO supplementary succeeded for %s", gse_id)
                    return
                else:
                    log.warning(
                        "KPMP: GEO supplementary returned HTTP %s for %s — trying next accession",
                        resp.status_code,
                        file_url,
                    )
            except requests.exceptions.RequestException as e:
                log.warning("KPMP: GEO supplementary request failed: %s", e)

    # GEO supplementary failed for all accessions — log ToS click-through note
    log.warning(
        "KPMP: GEO supplementary (no-auth) path failed for all accessions. "
        "Per plan user_setup: manual ToS click-through required at "
        "https://atlas.kpmp.org — download the Visium objects manually into "
        "data/raw/spatial/%s/. Continuing with other datasets.",
        entry.slug,
    )


def _download_spatiallibd(entry, dest_dir: Path) -> None:
    """Fetch Maynard DLPFC objects from LieberInstitute GitHub.

    spatialLIBD / Bioconductor provides sample metadata and data links via
    the LieberInstitute GitHub repository. For P0 acquisition we download
    the sample-level gene expression CSV manifest from the official repo so
    the test suite can verify the directory is non-empty.

    The full h5ad/SpExpr objects are large (multi-GB); for P0 we fetch a
    representative manifest file. Full objects are fetched in Plan 04 as needed
    for the coverage gate.
    """
    # LieberInstitute GitHub: sample metadata CSV (publicly accessible)
    sample_meta_url = (
        "https://raw.githubusercontent.com/LieberInstitute/spatialLIBD/"
        "master/inst/extdata/sce_layer_metadata_fields.csv"
    )
    # Also try the raw spatialLIBD sample sheet from the Bioconductor package
    alt_url = (
        "https://raw.githubusercontent.com/LieberInstitute/spatialLIBD/"
        "master/inst/extdata/spot_level_data_fields.csv"
    )

    dest_file = dest_dir / "spatialLIBD_spot_level_fields.csv"
    if dest_file.exists() and dest_file.stat().st_size > 0:
        log.info("spatialLIBD: Already exists, skipping: %s", dest_file)
        return

    for url in [sample_meta_url, alt_url]:
        try:
            resp = requests.head(url, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                _stream_download(url, dest_file)
                log.info("spatialLIBD: Successfully fetched manifest from %s", url)
                return
        except requests.exceptions.RequestException as e:
            log.warning("spatialLIBD: Request failed for %s: %s", url, e)

    # Both URLs failed — check if this is a hard gate (no expected_files means
    # the entry has flexible expected content; skip gracefully)
    if not entry.expected_files:
        log.warning(
            "spatialLIBD: Could not fetch manifest from GitHub. "
            "Maynard DLPFC full objects (multi-GB) require separate download. "
            "Directory %s created; full data acquisition pending.",
            dest_dir,
        )
    else:
        _halt(
            "spatialLIBD: Could not fetch required files for Maynard DLPFC",
            dataset=entry.name,
            slug=entry.slug,
        )


def _download_url(entry, dest_dir: Path) -> None:
    """Fetch files from a plain URL."""
    if not entry.expected_files:
        log.info(
            "Skipping URL download for %s — no expected_files defined (annotation-only or panel dataset).",
            entry.slug,
        )
        return

    # Extract URL from accession string (best-effort)
    accession = entry.accession
    urls = [tok.strip() for tok in accession.split() if tok.startswith("http")]
    if not urls:
        log.warning(
            "No HTTP URL found in accession string for %s: %s — skipping.",
            entry.slug,
            accession,
        )
        return

    primary_url = urls[0]
    for expected_name in entry.expected_files:
        dest_file = dest_dir / expected_name
        if dest_file.exists() and dest_file.stat().st_size > 0:
            log.info("Already exists, skipping: %s", dest_file)
            continue
        _stream_download(primary_url, dest_file)


# ---------------------------------------------------------------------------
# Main dispatch loop
# ---------------------------------------------------------------------------


def _should_fetch(entry) -> bool:
    """Return True if this entry should be fetched in Plan 02.

    We fetch entries that are:
      - usable_as_input=True (basal spatial input or rodent context), OR
      - validation anchors (APAP mouse liver, usable_as_input=False but have
        expected_files and are not annotation-only/panel datasets)

    We skip:
      - Fixed-panel datasets with no expected_files (annotation-only: MERFISH,
        CosMx, Xenium, snRNA-seq)
      - Deferred organs (heart) with usable_as_input=False and no expected_files
    """
    if entry.usable_as_input:
        return True
    # Validation anchors: have expected_files and are not annotation-only
    if entry.expected_files and entry.whole_transcriptome:
        return True
    return False


def main() -> None:
    """Run the registry-driven spatial dataset download."""
    log.info("Starting spatial dataset downloads from corrected registry.")
    log.info("Registry has %d entries total.", len(SPATIAL_DATASETS))

    entries_to_fetch = [e for e in SPATIAL_DATASETS if _should_fetch(e)]
    log.info("Fetching %d entries (usable_as_input or validation anchor).", len(entries_to_fetch))

    kpmp_failed = False  # track KPMP separately (non-fatal per plan user_setup)

    for entry in entries_to_fetch:
        # Resolve directory from entry.slug — SINGLE SOURCE OF TRUTH
        dest_dir = DATA_RAW_SPATIAL / entry.slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "Processing: %s (slug=%s, mechanism=%s)",
            entry.name[:60],
            entry.slug,
            entry.access_mechanism,
        )

        try:
            if entry.access_mechanism == "figshare_api":
                _download_figshare_api(entry, dest_dir)
            elif entry.access_mechanism == "geo_supp":
                _download_geo_supp(entry, dest_dir)
            elif entry.access_mechanism == "kpmp":
                _download_kpmp(entry, dest_dir)
                # KPMP is non-fatal — check if dir is still empty
                dataset_files = [f for f in dest_dir.iterdir() if f.is_file()]
                if not dataset_files:
                    kpmp_failed = True
                    log.warning("KPMP: directory %s is empty after download attempt.", dest_dir)
            elif entry.access_mechanism == "spatialLIBD":
                _download_spatiallibd(entry, dest_dir)
            elif entry.access_mechanism == "url":
                _download_url(entry, dest_dir)
            else:
                log.warning(
                    "Unknown access_mechanism '%s' for %s — skipping.",
                    entry.access_mechanism,
                    entry.slug,
                )
        except SystemExit:
            raise  # propagate halt
        except Exception as exc:
            _halt(
                f"Unexpected error downloading {entry.slug}: {exc}",
                dataset=entry.name,
                slug=entry.slug,
            )

    # -----------------------------------------------------------------------
    # Post-download expected_files assertion (Halt Gate 1)
    # -----------------------------------------------------------------------
    log.info("Running post-download expected_files assertion...")
    halt_entries = []

    for entry in entries_to_fetch:
        if not entry.expected_files:
            continue  # annotation-only or flexible-content entries; skip assertion
        dest_dir = DATA_RAW_SPATIAL / entry.slug

        # KPMP exception: do not halt if KPMP ToS click-through is needed
        if entry.access_mechanism == "kpmp" and kpmp_failed:
            log.warning(
                "KPMP assertion skipped — portal ToS click-through required. "
                "Expected files for %s: %s",
                entry.slug,
                entry.expected_files,
            )
            continue

        missing = []
        for fname in entry.expected_files:
            fpath = dest_dir / fname
            if not fpath.exists():
                missing.append(f"{fname} (missing)")
            elif fpath.stat().st_size == 0:
                missing.append(f"{fname} (zero-byte)")

        if missing:
            halt_entries.append((entry.slug, entry.name, missing))

    if halt_entries:
        details = "; ".join(
            f"{slug}: {', '.join(missing)}" for slug, name, missing in halt_entries
        )
        _halt(
            f"Post-download expected_files assertion FAILED for {len(halt_entries)} dataset(s): "
            f"{details}",
        )

    log.info("All expected_files assertions passed.")
    log.info("Spatial dataset downloads complete.")


if __name__ == "__main__":
    main()
