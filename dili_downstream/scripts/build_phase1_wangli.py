#!/usr/bin/env python3
"""Phase 1 driver: build wangli_profiles.csv + wangli_measured_de.npy +
wangli_drug_splits.npz + P1_wangli_data_summary.md from Wang/Li 2020 sources.

Source-of-truth paths declared at top; libs in src/data/ are pure and accept
paths as args. End-to-end:
  1. Download xlsx + pickle into data/raw/wangli_2020/ (skip if already present).
     Synapse pickle requires auth: tries the synapseclient login chain
     (cached config, DILI_V05_SYNAPSE_AUTH env var) and exits 2 with clear
     manual-download instructions if neither path works.
  2. Compute SHA256 for xlsx, pickle, local h5, dili_canonical.csv;
     append v0.5 section to MANIFEST.md (does not touch v0.4 entries).
  3. load_inst_ids + load_split_pickle (Wave-1 Plan 01).
  4. lookup_inst_ids on local h5; if >1% miss, log + exit 3 with manual-escalation
     pointer (caller decides GEO fallback — out of scope for this driver).
  5. crossvalidate_pearson on 100-profile random sample (seed 42).
  6. resolve_smiles via dili_canonical.csv; drop unresolved.
  7. Build wangli_profiles.csv with 11-col schema; tuple_key = sha256(smiles|cell_id|dose_str)[:16].
  8. Stack wangli_measured_de.npy aligned to retained rows.
  9. Save wangli_drug_splits.npz with 50 boolean arrays (split_00..split_49).
 10. Render P1_wangli_data_summary.md with 9 section headers.
 11. Print pass/fail summary; exit 0 on success.

Hard rule: --no-gpu — Phase 1 doesn't touch torch/cuda.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Make repo importable from anywhere.
REPO = Path(__file__).resolve().parents[1]  # .../dili_downstream/
sys.path.insert(0, str(REPO))

from src.data.wangli_loader import load_inst_ids, load_split_pickle
from src.data.wangli_lincs_lookup import (
    crossvalidate_pearson,
    lookup_inst_ids,
)
from src.data.wangli_smiles_resolver import resolve_smiles


# ---------------------------------------------------------------------------
# Source-of-truth paths (only place these live)
# ---------------------------------------------------------------------------
LINCS_H5 = Path(
    "/raid/home/joshua/data/L1000_and_CMap/Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5"
)
CANONICAL = REPO / "data" / "processed" / "dili_canonical.csv"
RAW_DIR = REPO / "data" / "raw" / "wangli_2020"
PROCESSED = REPO / "data" / "processed"
SUMMARY_MD = REPO / "results" / "tables" / "P1_wangli_data_summary.md"
MANIFEST = REPO / "MANIFEST.md"

XLSX_NAME = "6000_transcriptomic_profiles_id.xlsx"
PICKLE_NAME = "drug_split_index.pickle"
XLSX_URL = "https://github.com/TingLi2016/L1000_DILI/raw/master/6000_transcriptomic_profiles_id.xlsx"
PICKLE_SYN_ID = "syn22910821"  # File entity inside project syn22910750
PICKLE_SYN_PROJECT = "syn22910750"

SEED = 42
SAMPLE_N_FOR_PEARSON = 100
PEARSON_THRESHOLD = 0.99
INST_ID_MISS_FRACTION_LIMIT = 0.01  # >1% miss → halt with manual-escalation message
RETAINED_ROW_LO = 5700
RETAINED_ROW_HI = 6000
PN_RATIO_LO = 1.25
PN_RATIO_HI = 1.69
PAPER_PN_RATIO = 1.47
PAPER_POS = 3568
PAPER_NEG = 2432

EXIT_OK = 0
EXIT_SYNAPSE_AUTH_REQUIRED = 2
EXIT_INST_ID_MISS_OVER_LIMIT = 3
EXIT_PEARSON_FAIL = 4
EXIT_RETAINED_OUT_OF_RANGE = 5

log = logging.getLogger("build_phase1_wangli")


# ---------------------------------------------------------------------------
# Helpers: SHA256, downloads
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_retry(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            log.info("download[%d/%d]: %s -> %s", attempt, retries, url, dest)
            urllib.request.urlretrieve(url, str(dest))
            return
        except Exception as e:  # broad: urllib + network errors
            last_err = e
            log.warning("download attempt %d failed: %s", attempt, e)
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_err}")


def _download_synapse_pickle(syn_id: str, dest: Path) -> bool:
    """Download a Synapse File entity. Returns True on success.

    Tries (in order):
      1. synapseclient with cached login (~/.synapseConfig)
      2. synapseclient login via DILI_V05_SYNAPSE_AUTH env var (PAT)

    If neither works, returns False and the caller is expected to print
    manual-download instructions and exit 2.
    """
    try:
        import synapseclient
    except ImportError:
        log.error(
            "synapseclient is not installed. Run `pip install synapseclient` "
            "in the dili_v04_env conda env, then re-run this driver."
        )
        return False

    syn = synapseclient.Synapse()

    # Path 1: cached login
    try:
        syn.login(silent=True)
        log.info("synapse: logged in via cached config")
    except Exception:
        # Path 2: PAT from env
        token = os.environ.get("DILI_V05_SYNAPSE_AUTH")
        if not token:
            log.warning(
                "synapse: no cached creds and DILI_V05_SYNAPSE_AUTH env var not set; "
                "cannot authenticate"
            )
            return False
        try:
            syn.login(authToken=token, silent=True)
            log.info("synapse: logged in via DILI_V05_SYNAPSE_AUTH PAT")
        except Exception as e:
            log.error("synapse: PAT login failed: %s", e)
            return False

    # Now download
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        e = syn.get(syn_id, downloadFile=True, downloadLocation=str(dest.parent))
        if not getattr(e, "path", None):
            log.error(
                "synapse: get returned without a path (likely READ permission "
                "but no DOWNLOAD permission). Manual download required."
            )
            return False
        downloaded = Path(e.path)
        # Synapse downloads with the original filename — rename if needed.
        if downloaded.resolve() != dest.resolve():
            downloaded.replace(dest)
        log.info("synapse: downloaded %s (%d bytes)", dest, dest.stat().st_size)
        return True
    except Exception as ex:
        log.error("synapse: download failed: %s", ex)
        return False


def _print_synapse_manual_instructions(dest: Path) -> None:
    msg = f"""
============================================================
  MANUAL DOWNLOAD REQUIRED — Synapse pickle (DATA-02)
============================================================

The Wang/Li drug-split pickle is hosted on Synapse and requires authenticated
access to download. Anonymous access has READ permission but NOT DOWNLOAD.

Fix one of:

OPTION A — Synapse Personal Access Token (preferred, scriptable)
  1. Go to https://www.synapse.org/  -> Account Settings -> Personal Access Tokens
  2. Create a new PAT with "view" + "download" scopes (call it "dili_v05")
  3. Export it: export DILI_V05_SYNAPSE_AUTH="<your-token>"
  4. Re-run: python scripts/build_phase1_wangli.py

OPTION B — synapseclient cached login
  1. conda run -n dili_v04_env python -c "import synapseclient; \\
       synapseclient.Synapse().login('<email>', '<password>', rememberMe=True)"
  2. Re-run: python scripts/build_phase1_wangli.py

OPTION C — Manual web download
  1. Log into https://www.synapse.org/  in your browser
  2. Visit https://www.synapse.org/Synapse:{PICKLE_SYN_ID}
  3. Click the download button to download `{PICKLE_NAME}`
  4. Place the file at: {dest}
  5. Re-run: python scripts/build_phase1_wangli.py
     (driver is idempotent; will skip download if file already present)

Project landing page: https://www.synapse.org/Synapse:{PICKLE_SYN_PROJECT}
File entity:          https://www.synapse.org/Synapse:{PICKLE_SYN_ID}

This driver will exit with code 2.
============================================================
"""
    print(msg)


# ---------------------------------------------------------------------------
# inst_id parser → brd_id, cell_id, time_h, dose_str
# ---------------------------------------------------------------------------
# inst_id pattern (from h5 + load_inst_ids regex):
#   {plate}_{cell}_{time}H:{BRD-K######## or DMSO}:{dose}
# Examples:
#   CGS001_MCF7_24H:BRD-K12345678:10
#   CPC020_HEPG2_6H:BRD-K01234567:3.33333
_INST_ID_PARSE_RE = re.compile(
    r"^(?P<plate>[A-Z0-9]+)_(?P<cell>[A-Z0-9]+)_(?P<time>\d+)H:(?P<brd>[A-Z0-9\-]+):(?P<dose>\S+)$"
)


def _parse_inst_id(inst_id: str) -> tuple[str, str, int, str]:
    """Return (brd_id, cell_id, time_h, dose_str) from a LINCS sig_id."""
    m = _INST_ID_PARSE_RE.match(inst_id)
    if not m:
        raise ValueError(f"Cannot parse LINCS inst_id: {inst_id!r}")
    return m.group("brd"), m.group("cell"), int(m.group("time")), m.group("dose")


def _tuple_key(smiles: str, cell_id: str, dose_str: str) -> str:
    """sha256(smiles|cell_id|dose_str)[:16]"""
    return hashlib.sha256(
        f"{smiles}|{cell_id}|{dose_str}".encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# MANIFEST update (idempotent)
# ---------------------------------------------------------------------------


_V05_SECTION_HEADER = "## v0.5 Phase 1 (Wang/Li data acquisition)"
_V05_SECTION_OUTPUTS_HEADER = (
    "### v0.5 Phase 1 outputs (Producer: scripts/build_phase1_wangli.py; "
    "Consumers: P2, P3, P4, P5)"
)


def _append_v05_manifest(
    manifest_path: Path,
    xlsx_sha: str,
    xlsx_bytes: int,
    pickle_sha: str,
    pickle_bytes: int,
    h5_sha: str,
    h5_bytes: int,
    canonical_sha: str,
    canonical_bytes: int,
) -> None:
    """Append the v0.5 Phase 1 section to MANIFEST.md (idempotent — replaces
    any existing v0.5 Phase 1 section, preserves all v0.4 content)."""
    text = manifest_path.read_text() if manifest_path.exists() else ""

    # Strip any prior v0.5 section (everything from `## v0.5 Phase 1 ...` to EOF
    # OR to the next `---\n\n## ` boundary).
    marker_idx = text.find(_V05_SECTION_HEADER)
    if marker_idx != -1:
        # Keep everything up to (but not including) the marker line and any
        # preceding `---\n\n` separator. Find the start of the line containing
        # the marker, then walk back over a possible `---\n\n` block.
        line_start = text.rfind("\n", 0, marker_idx) + 1
        # Walk back over a preceding `---\n\n`
        prefix = text[:line_start].rstrip("\n")
        if prefix.endswith("\n\n---"):
            prefix = prefix[: -len("\n\n---")]
        text = prefix.rstrip() + "\n"

    # Build new section
    new_section = f"""
---

{_V05_SECTION_HEADER}

| File | Path | SHA256 | Bytes | Source |
|---|---|---|---|---|
| Wang/Li 6000 inst IDs | data/raw/wangli_2020/{XLSX_NAME} | `{xlsx_sha}` | {xlsx_bytes} | https://github.com/TingLi2016/L1000_DILI |
| Wang/Li drug-split + expressions pickle | data/raw/wangli_2020/{PICKLE_NAME} | `{pickle_sha}` | {pickle_bytes} | Synapse {PICKLE_SYN_ID} (project {PICKLE_SYN_PROJECT}) |
| LINCS Level 5 (local Bayesian) | {LINCS_H5} | `{h5_sha}` | {h5_bytes} | Local file; verified vs pickle expressions per-gene Pearson on 100-sample (see results/tables/P1_wangli_data_summary.md) |
| dili_canonical.csv (v0.4 P1, carryover) | data/processed/dili_canonical.csv | `{canonical_sha}` | {canonical_bytes} | v0.4 P1 output |

{_V05_SECTION_OUTPUTS_HEADER}

| File | Path | Shape | dtype |
|---|---|---|---|
| Profile join table | data/processed/wangli_profiles.csv | (N, 11) | mixed |
| Measured DE matrix | data/processed/wangli_measured_de.npy | (N, 978) | float32 |
| 50 drug splits | data/processed/wangli_drug_splits.npz | 50 keys, each (N,) | bool |
| Phase 1 deliverable | results/tables/P1_wangli_data_summary.md | — | markdown |
"""
    final = text.rstrip() + "\n" + new_section
    manifest_path.write_text(final)


# ---------------------------------------------------------------------------
# Summary md renderer
# ---------------------------------------------------------------------------


def _render_summary_md(
    *,
    xlsx_sha: str,
    xlsx_bytes: int,
    pickle_sha: str,
    pickle_bytes: int,
    h5_sha: str,
    h5_bytes: int,
    canonical_sha: str,
    canonical_bytes: int,
    n_input: int,
    n_found: int,
    n_missing: int,
    inst_id_pass: bool,
    n_resolved: int,
    n_dropped: int,
    n_retained: int,
    retained_pass: bool,
    n_pos: int,
    n_neg: int,
    pn_ratio: float,
    pearson_mean: float,
    pearson_median: float,
    pearson_n_low: int,
    pearson_pass: bool,
    pearson_sample_n: int,
    cell_counts: pd.Series,
    dose_counts: pd.DataFrame,
    drift_within_5pct: bool,
    n_disagree: int,
    top_disagree_names: list[str],
) -> str:
    def pct(n: int, total: int) -> str:
        return f"{100.0 * n / total:.2f}%" if total else "n/a"

    def verdict(b: bool) -> str:
        return "PASS" if b else "FAIL"

    cell_table_lines = ["| cell_id | count | pct |", "|---|---|---|"]
    cell_total = int(cell_counts.sum())
    for cid, cnt in cell_counts.items():
        cell_table_lines.append(f"| {cid} | {int(cnt)} | {pct(int(cnt), cell_total)} |")
    cell_table = "\n".join(cell_table_lines)

    dose_table_lines = ["| dose_str | dose_um | count |", "|---|---|---|"]
    for _, row in dose_counts.iterrows():
        dose_table_lines.append(
            f"| {row['dose_str']} | {row['dose_um']} | {int(row['count'])} |"
        )
    dose_table = "\n".join(dose_table_lines)

    if pearson_pass:
        pearson_postscript = (
            "Local Bayesian h5 numerically equivalent to Wang/Li's pickle expressions "
            "→ h5 is usable as the LINCS source for v0.5 P1+."
        )
    else:
        pearson_postscript = (
            "Local h5 deviates from Wang/Li's pickle expressions on the 100-profile "
            "sample. If mean Pearson < 0.95 the discrepancy is too large; if "
            "0.95 ≤ Pearson < 0.99 we proceed with a flag and prefer pickle "
            "expressions for cross-validation purposes. Manual review required."
        )

    if drift_within_5pct:
        drift_verdict = "within 5% — no halt"
    else:
        drift_verdict = "outside 5% — flag for re-discuss"

    top_names_block = (
        ", ".join(top_disagree_names) if top_disagree_names else "(none)"
    )

    return f"""# Phase 1 Wang/Li Data Acquisition Summary

## Source files

| File | Path | SHA256 | Bytes |
|---|---|---|---|
| xlsx (DATA-01) | data/raw/wangli_2020/{XLSX_NAME} | `{xlsx_sha}` | {xlsx_bytes} |
| pickle (DATA-02) | data/raw/wangli_2020/{PICKLE_NAME} | `{pickle_sha}` | {pickle_bytes} |
| LINCS h5 (DATA-03) | {LINCS_H5} | `{h5_sha}` | {h5_bytes} |
| dili_canonical.csv (v0.4 P1 carryover) | data/processed/dili_canonical.csv | `{canonical_sha}` | {canonical_bytes} |

## Profile resolution

- Input inst_ids from xlsx: {n_input}
- Found in local h5: {n_found} ({pct(n_found, n_input)})
- Missing in local h5: {n_missing} ({pct(n_missing, n_input)}) → **{verdict(inst_id_pass)}** against 1% threshold
- SMILES resolved via dili_canonical.csv: {n_resolved} ({pct(n_resolved, n_found)})
- SMILES unresolved (dropped): {n_dropped}
- Final retained rows in wangli_profiles.csv: {n_retained} → **{verdict(retained_pass)}** against [{RETAINED_ROW_LO}, {RETAINED_ROW_HI}] gate

## Class balance

- DILI-positive (Wang/Li label = 1): {n_pos}
- DILI-negative (Wang/Li label = 0): {n_neg}
- P/N ratio: {pn_ratio:.3f} (Wang/Li paper reports {PAPER_PN_RATIO} = {PAPER_POS}/{PAPER_NEG})

## Cell-line distribution

{cell_table}

## Dose distribution

{dose_table}

Note: P3 will map these to MultiDCP's grid [0.04, 0.12, 0.37, 1.11, 3.33, 10.0] µM via nearest-neighbor.

## Numerical sanity check

- Pearson cross-validation method: per-gene Pearson on {pearson_sample_n} random profiles (seed={SEED})
- Mean per-gene Pearson: {pearson_mean:.6f}
- Median per-gene Pearson: {pearson_median:.6f}
- Genes below 0.99: {pearson_n_low}
- **Verdict: {verdict(pearson_pass)}** against {PEARSON_THRESHOLD} threshold
- {pearson_postscript}

## Drift vs Wang/Li paper

- Paper: 6,000 profiles, P/N={PAPER_PN_RATIO}, {PAPER_POS}/{PAPER_NEG}
- Our: {n_retained} profiles, P/N={pn_ratio:.3f}, {n_pos}/{n_neg}
- Profiles lost to LINCS-h5 lookup: {n_missing}
- Profiles lost to SMILES resolution: {n_dropped}
- Drift verdict: {drift_verdict}

## DILI label disagreements

- Wang/Li-vs-canonical disagreements (where compound_name resolved to a dili_canonical.csv row): {n_disagree}
- Per CONTEXT.md decision: Wang/Li wins; this section is informational only.
- Top 10 disagreement names: {top_names_block}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build v0.5 Phase 1 Wang/Li artifacts end-to-end."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download attempts; assume xlsx + pickle are already in data/raw/wangli_2020/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    xlsx_path = RAW_DIR / XLSX_NAME
    pickle_path = RAW_DIR / PICKLE_NAME

    # ------------------------------------------------------------------ 1
    # Ensure raw files present.
    # ------------------------------------------------------------------
    if not xlsx_path.exists():
        if args.no_download:
            log.error("xlsx missing and --no-download set: %s", xlsx_path)
            return EXIT_SYNAPSE_AUTH_REQUIRED
        log.info("xlsx not found; downloading from %s", XLSX_URL)
        _download_with_retry(XLSX_URL, xlsx_path)

    if not pickle_path.exists():
        if args.no_download:
            log.error(
                "pickle missing and --no-download set: %s. "
                "Run without --no-download or download manually.",
                pickle_path,
            )
            _print_synapse_manual_instructions(pickle_path)
            return EXIT_SYNAPSE_AUTH_REQUIRED
        log.info(
            "pickle not found; attempting Synapse download (entity %s)",
            PICKLE_SYN_ID,
        )
        ok = _download_synapse_pickle(PICKLE_SYN_ID, pickle_path)
        if not ok or not pickle_path.exists():
            _print_synapse_manual_instructions(pickle_path)
            return EXIT_SYNAPSE_AUTH_REQUIRED

    # ------------------------------------------------------------------ 2
    # SHA256 bookkeeping
    # ------------------------------------------------------------------
    log.info("computing SHA256 for raw + canonical files")
    if not LINCS_H5.exists():
        log.error("local LINCS h5 missing: %s", LINCS_H5)
        return EXIT_INST_ID_MISS_OVER_LIMIT
    if not CANONICAL.exists():
        log.error(
            "dili_canonical.csv missing: %s — run v0.4 P1 first.", CANONICAL
        )
        return EXIT_INST_ID_MISS_OVER_LIMIT

    xlsx_sha = _compute_sha256(xlsx_path)
    pickle_sha = _compute_sha256(pickle_path)
    h5_sha = _compute_sha256(LINCS_H5)
    canonical_sha = _compute_sha256(CANONICAL)
    log.info("SHA256 xlsx=%s", xlsx_sha)
    log.info("SHA256 pickle=%s", pickle_sha)
    log.info("SHA256 lincs_h5=%s", h5_sha)
    log.info("SHA256 canonical=%s", canonical_sha)

    _append_v05_manifest(
        MANIFEST,
        xlsx_sha=xlsx_sha,
        xlsx_bytes=xlsx_path.stat().st_size,
        pickle_sha=pickle_sha,
        pickle_bytes=pickle_path.stat().st_size,
        h5_sha=h5_sha,
        h5_bytes=LINCS_H5.stat().st_size,
        canonical_sha=canonical_sha,
        canonical_bytes=CANONICAL.stat().st_size,
    )
    log.info("MANIFEST.md updated with v0.5 Phase 1 section")

    # ------------------------------------------------------------------ 3
    # Load Wave-1 outputs.
    # ------------------------------------------------------------------
    log.info("loading inst_ids from xlsx")
    ids = load_inst_ids(xlsx_path)
    log.info("loaded %d inst_ids", len(ids))

    log.info("loading drug-split pickle")
    ds = load_split_pickle(pickle_path)
    log.info(
        "pickle: N=%d compounds, dili_binary dtype=%s, expressions shape=%s, splits shape=%s, inst_ids=%s",
        len(ds.compound_names),
        ds.dili_binary.dtype,
        ds.expressions.shape,
        ds.split_flags.shape,
        "present" if ds.inst_ids is not None else "absent",
    )

    if len(ds.compound_names) != len(ids):
        log.warning(
            "pickle has N=%d profiles but xlsx has N=%d inst_ids — "
            "joining by row order (per CONTEXT.md decision when inst_ids absent in pickle)",
            len(ds.compound_names),
            len(ids),
        )
        if len(ds.compound_names) < len(ids):
            log.error(
                "FATAL: pickle has fewer profiles than xlsx inst_ids; "
                "cannot align by row order"
            )
            return EXIT_INST_ID_MISS_OVER_LIMIT

    # ------------------------------------------------------------------ 4
    # h5 lookup
    # ------------------------------------------------------------------
    log.info("looking up %d inst_ids in local h5", len(ids))
    matrix, found_ids, missing_ids = lookup_inst_ids(LINCS_H5, ids)
    n_input = len(ids)
    n_found = len(found_ids)
    n_missing = len(missing_ids)
    miss_frac = n_missing / max(n_input, 1)
    inst_id_pass = miss_frac <= INST_ID_MISS_FRACTION_LIMIT
    log.info(
        "h5 lookup: %d/%d found (%.2f%% miss); pass=%s",
        n_found,
        n_input,
        100.0 * miss_frac,
        inst_id_pass,
    )

    if not inst_id_pass:
        log.error(
            "HALT: %.2f%% inst_ids missing exceeds %.2f%% threshold. "
            "Per CONTEXT.md, fall back to fresh GEO download — out of scope "
            "for this driver. Write HALT_REASON.md and exit %d.",
            100.0 * miss_frac,
            100.0 * INST_ID_MISS_FRACTION_LIMIT,
            EXIT_INST_ID_MISS_OVER_LIMIT,
        )

    # ------------------------------------------------------------------ 5
    # Cross-validation Pearson on 100-profile sample
    # ------------------------------------------------------------------
    # We need the pickle's expressions for the same profiles we pulled from h5.
    # Both are aligned to the input order of `ids`; `found_ids` is the subset
    # that h5 had. Pickle row i corresponds to xlsx-index i (same order).
    # Build a parallel "found-position-in-input" mapping so we can index
    # pickle.expressions by xlsx-index.
    found_input_positions: list[int] = []
    j = 0  # pointer into found_ids
    for i, qid in enumerate(ids):
        if j < n_found and found_ids[j] == qid:
            found_input_positions.append(i)
            j += 1
    assert j == n_found, "found_ids ordering mismatch vs ids"

    # Random sample of min(SAMPLE_N_FOR_PEARSON, n_found) indices into found_ids
    sample_n = min(SAMPLE_N_FOR_PEARSON, n_found)
    if sample_n < 2:
        log.error(
            "Cannot run Pearson cross-validation with sample_n=%d (<2)", sample_n
        )
        return EXIT_INST_ID_MISS_OVER_LIMIT

    sample_idx_in_found = rng.choice(n_found, size=sample_n, replace=False)
    sample_h5 = matrix[sample_idx_in_found]  # (sample_n, 978)
    sample_input_positions = [found_input_positions[k] for k in sample_idx_in_found]
    sample_pickle = ds.expressions[sample_input_positions]  # (sample_n, 978)

    log.info(
        "running per-gene Pearson on %d-profile sample (h5 vs pickle expressions)",
        sample_n,
    )
    pearson_per_gene = crossvalidate_pearson(sample_h5, sample_pickle)
    pearson_finite = pearson_per_gene[np.isfinite(pearson_per_gene)]
    pearson_mean = float(pearson_finite.mean()) if pearson_finite.size else float("nan")
    pearson_median = float(np.median(pearson_finite)) if pearson_finite.size else float("nan")
    pearson_n_low = int((pearson_finite < PEARSON_THRESHOLD).sum())
    pearson_pass = pearson_mean >= PEARSON_THRESHOLD
    log.info(
        "Pearson: mean=%.6f median=%.6f n_low=%d pass=%s",
        pearson_mean,
        pearson_median,
        pearson_n_low,
        pearson_pass,
    )

    # ------------------------------------------------------------------ 6
    # SMILES resolution via dili_canonical.csv
    # ------------------------------------------------------------------
    log.info("loading dili_canonical.csv (%s)", CANONICAL)
    canonical_df = pd.read_csv(CANONICAL)
    log.info("canonical_df: %d rows, columns=%s", len(canonical_df), canonical_df.columns.tolist())

    # Resolve only the compound_names that survived h5 lookup
    found_compound_names = [ds.compound_names[i] for i in found_input_positions]
    found_dili_binary = ds.dili_binary[found_input_positions]
    found_split_flags = ds.split_flags[found_input_positions]

    log.info("resolving SMILES for %d compounds", len(found_compound_names))
    resolved = resolve_smiles(found_compound_names, canonical_df)
    n_resolved = sum(1 for s in resolved.smiles if s is not None)
    n_dropped = len(resolved.drop_indices)
    log.info(
        "SMILES: resolved=%d dropped=%d (rate=%.2f%%)",
        n_resolved,
        n_dropped,
        100.0 * n_resolved / max(len(found_compound_names), 1),
    )

    # ------------------------------------------------------------------ 7-9
    # Build retained mask and three primary outputs
    # ------------------------------------------------------------------
    retained_mask = np.array(
        [s is not None for s in resolved.smiles], dtype=bool
    )
    retained_idx = np.where(retained_mask)[0]
    n_retained = int(retained_idx.size)
    log.info("retained %d / %d profiles after SMILES resolution", n_retained, n_found)

    retained_pass = RETAINED_ROW_LO <= n_retained <= RETAINED_ROW_HI
    if not retained_pass:
        log.error(
            "HALT: retained rows %d outside [%d, %d] gate",
            n_retained,
            RETAINED_ROW_LO,
            RETAINED_ROW_HI,
        )

    # Build wangli_profiles.csv (11-col schema)
    rows: list[dict] = []
    for k, idx in enumerate(retained_idx):
        inst_id = found_ids[idx]
        brd_id, cell_id, time_h, dose_str = _parse_inst_id(inst_id)
        try:
            dose_um = float(dose_str)
        except ValueError:
            dose_um = float("nan")
        smiles = resolved.smiles[idx]
        severity = resolved.severity[idx]
        rows.append({
            "profile_id": inst_id,
            "compound_name": found_compound_names[idx],
            "brd_id": brd_id,
            "cell_id": cell_id,
            "time_h": time_h,
            "dose_str": dose_str,
            "dose_um": dose_um,
            "smiles": smiles,
            "dili_binary": int(found_dili_binary[idx]),
            "dili_severity": severity if severity is not None else "",
            "tuple_key": _tuple_key(smiles, cell_id, dose_str),
        })

    df_out = pd.DataFrame(rows, columns=[
        "profile_id", "compound_name", "brd_id", "cell_id", "time_h",
        "dose_str", "dose_um", "smiles", "dili_binary", "dili_severity", "tuple_key",
    ])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED / "wangli_profiles.csv"
    df_out.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(df_out))

    # Build wangli_measured_de.npy aligned to the retained rows.
    # We use the h5-extracted matrix (which is the "measured DE" in the v0.5
    # vocabulary; CONTEXT.md treats Wang/Li's published Level-5 expressions and
    # the local h5 z-scores as numerically equivalent post-cross-validation).
    # Per the project's "DE rule", these are already DE-style features (LINCS
    # Level-5 z-scores against control plate) — no further transform needed at
    # this stage.
    measured_de = matrix[retained_idx].astype(np.float32, copy=False)
    out_de = PROCESSED / "wangli_measured_de.npy"
    np.save(out_de, measured_de)
    log.info("wrote %s (shape=%s, dtype=%s)", out_de, measured_de.shape, measured_de.dtype)

    # Build wangli_drug_splits.npz: 50 boolean masks aligned to retained rows.
    splits_for_retained = found_split_flags[retained_idx]  # (n_retained, 50)
    splits_dict = {
        f"split_{k:02d}": splits_for_retained[:, k].astype(bool, copy=False)
        for k in range(50)
    }
    out_splits = PROCESSED / "wangli_drug_splits.npz"
    np.savez_compressed(out_splits, **splits_dict)
    log.info(
        "wrote %s (50 keys, each shape=(%d,) bool)",
        out_splits,
        n_retained,
    )

    # ------------------------------------------------------------------ 10
    # DILI label disagreements (informational)
    # ------------------------------------------------------------------
    n_disagree = 0
    disagree_names: list[str] = []
    canon_lower = {
        str(name).lower().strip(): int(b)
        for name, b in zip(canonical_df["drug_name"], canonical_df["dili_binary"])
        if isinstance(name, str)
    }
    for k, idx in enumerate(retained_idx):
        nm = found_compound_names[idx]
        wl = int(found_dili_binary[idx])
        canon = canon_lower.get(str(nm).lower().strip())
        if canon is not None and canon != wl:
            n_disagree += 1
            if len(disagree_names) < 10:
                disagree_names.append(nm)
    log.info(
        "Wang/Li-vs-canonical disagreements: %d (top 10 names captured)", n_disagree
    )

    # ------------------------------------------------------------------ 11
    # Class balance + drift
    # ------------------------------------------------------------------
    n_pos = int((df_out["dili_binary"] == 1).sum())
    n_neg = int((df_out["dili_binary"] == 0).sum())
    pn_ratio = n_pos / max(n_neg, 1)
    drift_within_5pct = (
        abs(n_retained - 6000) / 6000.0 < 0.05
        and PN_RATIO_LO <= pn_ratio <= PN_RATIO_HI
    )

    # Cell-line + dose distributions
    cell_counts = df_out["cell_id"].value_counts()
    dose_counts_df = (
        df_out.groupby("dose_str", as_index=False)
        .agg(dose_um=("dose_um", "first"), count=("dose_str", "size"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ 12
    # Render summary md
    # ------------------------------------------------------------------
    summary_md = _render_summary_md(
        xlsx_sha=xlsx_sha,
        xlsx_bytes=xlsx_path.stat().st_size,
        pickle_sha=pickle_sha,
        pickle_bytes=pickle_path.stat().st_size,
        h5_sha=h5_sha,
        h5_bytes=LINCS_H5.stat().st_size,
        canonical_sha=canonical_sha,
        canonical_bytes=CANONICAL.stat().st_size,
        n_input=n_input,
        n_found=n_found,
        n_missing=n_missing,
        inst_id_pass=inst_id_pass,
        n_resolved=n_resolved,
        n_dropped=n_dropped,
        n_retained=n_retained,
        retained_pass=retained_pass,
        n_pos=n_pos,
        n_neg=n_neg,
        pn_ratio=pn_ratio,
        pearson_mean=pearson_mean,
        pearson_median=pearson_median,
        pearson_n_low=pearson_n_low,
        pearson_pass=pearson_pass,
        pearson_sample_n=sample_n,
        cell_counts=cell_counts,
        dose_counts=dose_counts_df,
        drift_within_5pct=drift_within_5pct,
        n_disagree=n_disagree,
        top_disagree_names=disagree_names,
    )

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text(summary_md)
    log.info("wrote %s", SUMMARY_MD)

    # ------------------------------------------------------------------ 13
    # Final pass/fail summary
    # ------------------------------------------------------------------
    print(
        f"OK n_input={n_input} n_found={n_found} n_missing={n_missing} "
        f"inst_id_pass={inst_id_pass} n_resolved={n_resolved} n_dropped={n_dropped} "
        f"n_retained={n_retained} retained_pass={retained_pass} "
        f"n_pos={n_pos} n_neg={n_neg} pn_ratio={pn_ratio:.3f} "
        f"pearson_mean={pearson_mean:.6f} pearson_pass={pearson_pass} "
        f"n_disagree={n_disagree}"
    )

    # Halt-gate exit codes (per CONTEXT.md): inst_id miss → 3; pearson FAIL → 4;
    # retained out-of-range → 5. The driver writes outputs first so the user
    # can inspect them; the halt-gate exit code makes the CI / orchestrator
    # surface the failure.
    if not inst_id_pass:
        return EXIT_INST_ID_MISS_OVER_LIMIT
    if not retained_pass:
        return EXIT_RETAINED_OUT_OF_RANGE
    if not pearson_pass:
        # Per CONTEXT.md and the executor's halt-gate protocol:
        # - mean Pearson < 0.95 → halt (return EXIT_PEARSON_FAIL)
        # - 0.95 <= mean < 0.99 → flag in summary, continue
        if pearson_mean < 0.95:
            return EXIT_PEARSON_FAIL
        log.warning(
            "Pearson mean %.6f below %.2f but >= 0.95; flagged in summary, continuing",
            pearson_mean,
            PEARSON_THRESHOLD,
        )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
