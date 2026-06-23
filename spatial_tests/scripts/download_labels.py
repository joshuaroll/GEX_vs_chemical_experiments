"""Per-organ toxicity label-set fetcher.

Downloads per-organ toxicity label sets into data/raw/labels/<set>/:
  - liver: DILIst + DILIrank (FDA Excel, direct URLs from sibling dili_downstream/MANIFEST.md)
  - kidney: DIRIL (Connor 2024 Drug Discov Today supplement) — resolve at download time;
    if unresolved, write a TODO note into HALT_REASON candidates (never fabricate)
  - brain: SIDER (sideeffects.embl.de) meddra_all_se.tsv.gz ONLY — nervous-system
    SOC (serious terms) filtered at use time; seizure/DNT-IVB are explicitly NOT
    fetched in P0 (DEFERRED to Phase 1 per plan DECISION below)
  - heart: DICTrank fetched from the FDA (heart IN-SCOPE as of 2026-06-21; non-fatal
    on failure, mirroring the DIRIL pattern)

DECISION (recorded per Plan 02 Task 2):
  P0 brain toxicity labels = SIDER nervous-system SOC (serious terms) ONLY.
  Lane-Ekins seizure and DNT-IVB are DEFERRED to Phase 1 because brain is the
  last organ sequenced (liver → kidney → brain). DIRIL supplement URL is
  resolved at download time; if unresolved, a TODO line is written — never
  fabricated. This converts the previously-silent brain-label scope reduction
  into a recorded decision; DATA-01 coverage stays intact (SIDER brain labels
  still acquired).

Hard rules honored:
    - Real data only; no fabricated or substituted labels (XC-01).
    - Halt Gate 1: if any required (non-deferred) label set returns 404 / empty,
      write HALT_REASON.md and sys.exit(1).
    - No torch import (P0 data-only rule honored).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_LABELS = _REPO_ROOT / "data" / "raw" / "labels"
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
log = logging.getLogger("download_labels")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1 << 20  # 1 MB

# ---------------------------------------------------------------------------
# URLs — sourced from sibling dili_downstream/MANIFEST.md (verified 2026-06-20)
# ---------------------------------------------------------------------------
DILIST_URL = "https://www.fda.gov/media/160597/download?attachment"
DILIRANK_URL = "https://www.fda.gov/media/113052/download?attachment"
DILIST_EXPECTED_SHA256 = "4331ee9d16ae7641488161e4dc2c603c29e06baa8667dd16e7f3d635366e7e5e"
DILIRANK_EXPECTED_SHA256 = "1ca1352ff727af68e68e250eae2ed775bca8492335140ac0afd2233248694993"

# Sibling project paths — the same files were downloaded 2026-05-05 with verified SHAs.
# The spatial_tests project lives inside the same GEX_vs_chemical_experiments umbrella repo
# as dili_downstream. Copying from the sibling (with SHA verification) is equivalent to
# re-downloading and avoids FDA bot-protection blocks on automated downloads.
_SIBLING_ROOT = _REPO_ROOT.parent / "dili_downstream"
DILIST_SIBLING = _SIBLING_ROOT / "data" / "raw" / "DILIst" / "dilist.xlsx"
DILIRANK_SIBLING = _SIBLING_ROOT / "data" / "raw" / "DILIrank" / "dilirank.xlsx"

# SIDER: meddra_all_se.tsv.gz (all side effects, MedDRA preferred terms + SOC)
# Verified HTTP 200 at sideeffects.embl.de 2026-06-20
SIDER_MEDDRA_ALL_SE_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
SIDER_MEDDRA_FREQ_URL = "http://sideeffects.embl.de/media/download/meddra_freq_parsed.tsv.gz"

# DIRIL (Connor et al. 2024, Drug Discov Today 29(4)) — kidney nephrotoxicity label set.
# RESOLVED 2026-06-21: the FDA hosts DIRIL directly (gold source, public, no journal
# gate). The prior Elsevier-CDN candidate had the wrong article S-number and 404'd.
# FDA "Drug-Induced Renal Injury List (DIRIL) Dataset": diril_dataset_508.xlsx,
# sheet "A. DIRIL (317)" — 317 drugs with SMILES + binary DIRI label
# ("My Findings (Toxicity)": 171 Nephrotoxic / 146 Non-Nephrotoxic, no NaN).
DIRIL_CANDIDATE_URLS = [
    "https://www.fda.gov/media/178824/download?attachment",  # FDA diril_dataset_508.xlsx (primary)
    "https://ars.els-cdn.com/content/image/1-s2.0-S1359644624000631-mmc1.xlsx",  # Elsevier suppl (fallback)
]

# Browser User-Agent — FDA /media endpoints reject the default python-requests UA.
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

# DICTrank (Qu et al. 2023, Drug Discov Today 28(11)) — heart cardiotoxicity label set.
# FDA-hosted (public): 1318 drugs ranked into most/less/no/ambiguous DICT-concern.
# Heart is IN-SCOPE (activated 2026-06-21); DICTrank is the cardiac analogue of
# DILIrank/DIRIL.
DICTRANK_URL = "https://www.fda.gov/media/178811/download?attachment"  # dictrank_dataset_508.xlsx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute streaming SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_from_sibling(src: Path, dest: Path, expected_sha256: str, label: str) -> bool:
    """Copy a file from the sibling dili_downstream project if SHA matches.

    Returns True if copied successfully, False if src does not exist.
    """
    if not src.exists():
        return False
    local_sha = _sha256_file(src)
    if local_sha != expected_sha256:
        log.warning(
            "%s: sibling file SHA256 mismatch (local=%s expected=%s) — "
            "will try network download instead.",
            label,
            local_sha,
            expected_sha256,
        )
        return False
    shutil.copy2(src, dest)
    log.info(
        "%s: Copied from sibling dili_downstream (SHA256 verified: %s -> %s)",
        label,
        src,
        dest,
    )
    return True


def _stream_download(url: str, dest: Path, *, timeout: int = 300, label_set: str = "") -> None:
    """Stream a URL to a local file.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    dest : Path
        Local destination file path (parent must exist).
    timeout : int
        Request timeout in seconds.
    label_set : str
        Label set name for Halt Gate 1 error messages.

    Raises
    ------
    SystemExit
        On HTTP 404 or any non-200 status for required (non-DIRIL) downloads.
    """
    log.info("Fetching %s -> %s", url, dest)
    resp = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)

    if resp.status_code == 404:
        _halt(f"HTTP 404 for URL: {url}", url=url, label_set=label_set)

    if resp.status_code != 200:
        _halt(f"HTTP {resp.status_code} for URL: {url}", url=url, label_set=label_set)

    # Guard against HTML error pages masquerading as binary files
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        _halt(
            f"HTML content-type received for binary URL: {url} (Content-Type: {content_type})",
            url=url,
            label_set=label_set,
        )

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)

    size = dest.stat().st_size
    if size == 0:
        _halt(f"Zero-byte file downloaded from: {url}", url=url, label_set=label_set)

    log.info("Downloaded %s (%.1f MB)", dest.name, size / 1e6)


def _halt(reason: str, url: str = "", label_set: str = "") -> None:
    """Write HALT_REASON.md and exit nonzero (Halt Gate 1)."""
    log.error("HALT GATE 1 FIRED: %s", reason)
    HALT_REASON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HALT_REASON_PATH, "w") as fh:
        fh.write(
            f"# Halt Gate 1 — Required Label Set Unavailable\n\n"
            f"**Reason:** {reason}\n\n"
            f"**Label set:** {label_set or '(see reason)'}\n"
            f"**URL tried:** {url or '(see reason)'}\n\n"
            "Halt Gate 1 rationale: a planned (non-deferred) toxicity label set is "
            "unavailable, empty, or returned HTML where binary data was expected. "
            "Per XC-01, labels must never be fabricated or silently substituted. "
            "Stop and resolve the data availability issue before proceeding to Plan 04.\n"
        )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Label-set downloaders
# ---------------------------------------------------------------------------


def download_liver_labels() -> None:
    """Download DILIst and DILIrank FDA Excel files into data/raw/labels/dilist/
    and data/raw/labels/dilirank/ respectively.

    URLs and expected SHA256s are sourced from sibling dili_downstream/MANIFEST.md
    (verified 2026-06-20 by the sibling project).
    """
    # DILIst
    dilist_dir = DATA_RAW_LABELS / "dilist"
    dilist_dir.mkdir(parents=True, exist_ok=True)
    dilist_dest = dilist_dir / "dilist.xlsx"
    if not dilist_dest.exists():
        # Try sibling project first (same umbrella repo; SHA-verified)
        if not _copy_from_sibling(DILIST_SIBLING, dilist_dest, DILIST_EXPECTED_SHA256, "DILIst"):
            # Fallback to network download (FDA direct URL)
            _stream_download(DILIST_URL, dilist_dest, label_set="DILIst")
    else:
        log.info("DILIst already on disk: %s", dilist_dest)

    # DILIrank
    dilirank_dir = DATA_RAW_LABELS / "dilirank"
    dilirank_dir.mkdir(parents=True, exist_ok=True)
    dilirank_dest = dilirank_dir / "dilirank.xlsx"
    if not dilirank_dest.exists():
        # Try sibling project first (same umbrella repo; SHA-verified)
        if not _copy_from_sibling(DILIRANK_SIBLING, dilirank_dest, DILIRANK_EXPECTED_SHA256, "DILIrank"):
            # Fallback to network download (FDA direct URL)
            _stream_download(DILIRANK_URL, dilirank_dest, label_set="DILIrank")
    else:
        log.info("DILIrank already on disk: %s", dilirank_dest)

    log.info("Liver label sets complete (DILIst + DILIrank).")


def download_kidney_labels() -> None:
    """Attempt to download DIRIL (Connor 2024) into data/raw/labels/diril/.

    DIRIL is the kidney nephrotoxicity label set. The supplement URL is
    resolved at runtime. If ALL candidate URLs fail (non-200 or HTML response),
    a TODO note is written into the halt-reason candidates file — DIRIL download
    failure is NOT fatal to the full download run per the plan DECISION (the
    exact supplement URL was uncertain at planning time).

    Never fabricates kidney labels (XC-01).
    """
    diril_dir = DATA_RAW_LABELS / "diril"
    diril_dir.mkdir(parents=True, exist_ok=True)

    todo_note_path = diril_dir / "DIRIL_TODO.txt"
    dest = diril_dir / "diril_dataset_508.xlsx"
    if dest.exists() and dest.stat().st_size > 0:
        log.info("DIRIL already on disk: %s", dest)
        todo_note_path.unlink(missing_ok=True)  # clear any stale TODO once resolved
        return

    for url in DIRIL_CANDIDATE_URLS:
        log.info("DIRIL: Attempting: %s", url)
        try:
            resp = requests.head(url, timeout=30, allow_redirects=True, headers=_HEADERS)
            if resp.status_code != 200:
                log.warning("DIRIL: %s returned HTTP %s — skipping.", url, resp.status_code)
                continue
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                log.warning("DIRIL: %s returned HTML — likely a journal gate. Skipping.", url)
                continue
            # URL looks good — fetch it
            _stream_download_non_fatal(url, dest)
            if dest.exists() and dest.stat().st_size > 0:
                log.info("DIRIL: Downloaded successfully from %s", url)
                todo_note_path.unlink(missing_ok=True)  # resolved — remove stale TODO
                return
        except requests.exceptions.RequestException as e:
            log.warning("DIRIL: Request failed for %s: %s", url, e)

    # All candidate URLs failed — write TODO note (never halt for DIRIL per DECISION)
    log.warning(
        "DIRIL: All candidate URLs failed. Writing TODO note. "
        "Kidney nephrotoxicity labels must be resolved before Phase 4 analysis."
    )
    with open(todo_note_path, "w") as fh:
        fh.write(
            "# DIRIL TODO — Kidney Nephrotoxicity Labels Not Downloaded\n\n"
            "DECISION (Plan 02 Task 2): DIRIL (Connor 2024 Drug Discov Today supplement) "
            "URL was not resolved automatically at P0 download time. "
            "Candidate URLs attempted:\n"
        )
        for u in DIRIL_CANDIDATE_URLS:
            fh.write(f"  - {u}\n")
        fh.write(
            "\nAction required before Phase 4 kidney analysis:\n"
            "  1. DIRIL is hosted by the FDA: 'Drug-Induced Renal Injury List (DIRIL) Dataset'\n"
            "     (file diril_dataset_508.xlsx). Connor S et al., Drug Discov Today 29(4), 2024.\n"
            "  2. Download the dataset and place it at:\n"
            "     data/raw/labels/diril/diril_dataset_508.xlsx\n"
            "  3. Never fabricate kidney labels (XC-01).\n"
        )


def _stream_download_non_fatal(url: str, dest: Path, *, timeout: int = 120) -> None:
    """Stream a URL to a local file without halting on failure.

    Used for DIRIL where failure is non-fatal per plan DECISION.
    """
    try:
        resp = requests.get(
            url, stream=True, timeout=timeout, allow_redirects=True, headers=_HEADERS
        )
        if resp.status_code != 200:
            return
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            return
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
    except requests.exceptions.RequestException:
        pass


def download_brain_labels() -> None:
    """Download SIDER nervous-system SOC labels into data/raw/labels/sider/.

    DECISION: P0 brain toxicity labels = SIDER meddra_all_se.tsv.gz ONLY.
    Nervous-system SOC (serious terms) filtering is applied at use-time
    (Phase 1 analysis), not here. Lane-Ekins seizure and DNT-IVB are
    explicitly DEFERRED to Phase 1 — they are NOT fetched in P0.

    Source: sideeffects.embl.de (verified HTTP 200 on 2026-06-20).
    """
    # Brain label-scope DECISION is honored: only SIDER meddra_all_se; no seizure/DNT-IVB
    sider_dir = DATA_RAW_LABELS / "sider"
    sider_dir.mkdir(parents=True, exist_ok=True)

    meddra_all_dest = sider_dir / "meddra_all_se.tsv.gz"
    if not meddra_all_dest.exists():
        _stream_download(SIDER_MEDDRA_ALL_SE_URL, meddra_all_dest, label_set="SIDER meddra_all_se")
    else:
        log.info("SIDER meddra_all_se.tsv.gz already on disk: %s", meddra_all_dest)

    # Also fetch the frequency-parsed file (contains SOC-level assignments)
    meddra_freq_dest = sider_dir / "meddra_freq_parsed.tsv.gz"
    if not meddra_freq_dest.exists():
        try:
            resp = requests.head(SIDER_MEDDRA_FREQ_URL, timeout=30, allow_redirects=True)
            if resp.status_code == 200:
                _stream_download(SIDER_MEDDRA_FREQ_URL, meddra_freq_dest, label_set="SIDER meddra_freq_parsed")
            else:
                log.info(
                    "SIDER meddra_freq_parsed.tsv.gz not available (HTTP %s) — "
                    "meddra_all_se.tsv.gz is sufficient for SOC filtering.",
                    resp.status_code,
                )
        except requests.exceptions.RequestException as e:
            log.info("SIDER freq file not available: %s — meddra_all_se.tsv.gz is sufficient.", e)
    else:
        log.info("SIDER meddra_freq_parsed.tsv.gz already on disk: %s", meddra_freq_dest)

    # Write explicit deferred-note for seizure/DNT-IVB
    deferred_note = sider_dir / "BRAIN_LABELS_DEFERRED_NOTE.txt"
    if not deferred_note.exists():
        with open(deferred_note, "w") as fh:
            fh.write(
                "# Brain Label Scope — P0 DECISION\n\n"
                "P0 brain toxicity labels = SIDER nervous-system SOC (serious terms) ONLY.\n\n"
                "DEFERRED to Phase 1 (brain is the last organ sequenced: liver → kidney → brain):\n"
                "  - Lane-Ekins seizure label set (not fetched in P0)\n"
                "  - DNT-IVB developmental neurotoxicity labels (not fetched in P0)\n\n"
                "SIDER meddra_all_se.tsv.gz provides comprehensive MedDRA side-effect data.\n"
                "Filter to nervous-system SOC (System Organ Class: 'Nervous system disorders')\n"
                "and 'serious' frequency flag at use time in Phase 1 analysis.\n"
            )

    log.info("Brain label sets complete (SIDER SOC only; seizure/DNT-IVB deferred to P1).")


def download_heart_labels() -> None:
    """Acquire DICTrank (heart cardiotoxicity labels) from the FDA.

    Heart is IN-SCOPE (activated 2026-06-21) — DICTrank is the cardiac analogue of
    DILIrank/DIRIL. Failure stays NON-FATAL (logged; mirrors the DIRIL pattern) since
    the FDA source is stable and the file is staged on disk. The FDA hosts DICTrank
    publicly (Qu et al. 2023, Drug Discov Today 28(11)): 1318 drugs ranked into
    most/less/no/ambiguous DICT-concern. Idempotent: skips if already on disk.
    Never fabricates labels (XC-01).
    """
    dict_dir = DATA_RAW_LABELS / "dictrank"
    dict_dir.mkdir(parents=True, exist_ok=True)
    dest = dict_dir / "dictrank_dataset_508.xlsx"
    if dest.exists() and dest.stat().st_size > 0:
        log.info("DICTrank already on disk: %s", dest)
        return

    log.info("DICTrank: Attempting FDA source: %s", DICTRANK_URL)
    try:
        resp = requests.head(DICTRANK_URL, timeout=30, allow_redirects=True, headers=_HEADERS)
        if resp.status_code == 200 and "text/html" not in resp.headers.get("Content-Type", ""):
            _stream_download_non_fatal(DICTRANK_URL, dest)
            if dest.exists() and dest.stat().st_size > 0:
                log.info("DICTrank: Downloaded successfully (heart in-scope).")
                return
        log.warning(
            "DICTrank: FDA source returned HTTP %s / unexpected type — skipping "
            "(heart deferred; non-fatal).",
            resp.status_code,
        )
    except requests.exceptions.RequestException as e:
        log.warning("DICTrank: request failed: %s (heart deferred; non-fatal).", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all per-organ toxicity label-set downloads."""
    log.info("Starting per-organ toxicity label-set downloads.")

    # Liver labels (required, non-deferred — DILIst + DILIrank)
    log.info("=== Liver labels (DILIst + DILIrank) ===")
    download_liver_labels()

    # Kidney labels (DIRIL — non-fatal on URL failure; log TODO if unresolved)
    log.info("=== Kidney labels (DIRIL) ===")
    download_kidney_labels()

    # Brain labels (SIDER SOC only; seizure/DNT-IVB explicitly deferred)
    log.info("=== Brain labels (SIDER SOC only; seizure/DNT-IVB deferred to P1) ===")
    download_brain_labels()

    # Heart labels (DICTrank deferred per ROADMAP)
    log.info("=== Heart labels (DICTrank — FDA; heart in-scope) ===")
    download_heart_labels()

    log.info("Per-organ toxicity label-set downloads complete.")


if __name__ == "__main__":
    main()
