#!/usr/bin/env python3
"""Build wangli_profiles.csv + wangli_measured_de.npy WITHOUT Synapse.

This is a trimmed variant of scripts/build_phase1_wangli.py that defers the
Synapse-gated wangli_drug_splits.npz (Plan 01-04) to a follow-up.

Rationale: the random-split deliverables (wangli_profiles.csv +
wangli_measured_de.npy) are sufficient for the Wang/Li model-reproduction
task (Step 2 of the v0.5 plan).  The Usage column in
6000_transcriptomic_profiles_id.xlsx provides the train/test split labels
directly (Training=4800, Test=1200), so we do not need the Synapse pickle's
50 drug-based split arrays.

Differences from build_phase1_wangli.py:
  - No Synapse download / pickle loading / drug_splits.npz
  - No Pearson cross-validation (no pickle reference expressions)
  - SMILES resolution still applied (needed for tuple_key)
  - Usage column propagated into wangli_profiles.csv as `usage` column
  - wangli_measured_de.npy aligned to retained rows, same as Phase 1
  - Exit 0 on success; exit 1 on any hard error

Usage:
    conda run -n dili_v04_env python scripts/build_wangli_random_only.py [--no-download]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.wangli_lincs_lookup import lookup_inst_ids
from src.data.wangli_smiles_resolver import resolve_smiles

# ── Source-of-truth paths ────────────────────────────────────────────────────
LINCS_H5 = Path(
    "/raid/home/joshua/data/L1000_and_CMap/Bayesian_GSE92742_Level5_COMPZ_n361481x978.h5"
)
CANONICAL = REPO / "data" / "processed" / "dili_canonical.csv"
RAW_DIR = REPO / "data" / "raw" / "wangli_2020"
PROCESSED = REPO / "data" / "processed"

XLSX_NAME = "6000_transcriptomic_profiles_id.xlsx"
XLSX_URL = "https://github.com/TingLi2016/L1000_DILI/raw/master/6000_transcriptomic_profiles_id.xlsx"

log = logging.getLogger("build_wangli_random_only")

_INST_ID_PARSE_RE = re.compile(
    # plate may contain dots (e.g. MUC.CP005) and BRD IDs may have multi-hyphen
    # compound IDs (e.g. BRD-K55696337-003-16-0), so use \S+ for brd token.
    r"^(?P<plate>[A-Z0-9.]+)_(?P<cell>[A-Z0-9]+)_(?P<time>\d+)H:(?P<brd>\S+):(?P<dose>\S+)$"
)


def _parse_inst_id(inst_id: str) -> tuple[str, str, int, str]:
    m = _INST_ID_PARSE_RE.match(inst_id)
    if not m:
        raise ValueError(f"Cannot parse LINCS inst_id: {inst_id!r}")
    return m.group("brd"), m.group("cell"), int(m.group("time")), m.group("dose")


def _tuple_key(smiles: str, cell_id: str, dose_str: str) -> str:
    return hashlib.sha256(
        f"{smiles}|{cell_id}|{dose_str}".encode("utf-8")
    ).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build wangli_profiles.csv + wangli_measured_de.npy (no Synapse)."
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download; assume xlsx is already in data/raw/wangli_2020/",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    xlsx_path = RAW_DIR / XLSX_NAME

    # ── 1. Ensure xlsx present ───────────────────────────────────────────────
    if not xlsx_path.exists():
        if args.no_download:
            log.error("xlsx missing and --no-download set: %s", xlsx_path)
            return 1
        log.info("xlsx not found; downloading from %s", XLSX_URL)
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(XLSX_URL, str(xlsx_path))

    if not LINCS_H5.exists():
        log.error("LINCS h5 missing: %s", LINCS_H5)
        return 1

    if not CANONICAL.exists():
        log.error("dili_canonical.csv missing: %s — run v0.4 P1 first.", CANONICAL)
        return 1

    # ── 2. Load xlsx — inst_ids + Usage + labels ─────────────────────────────
    log.info("loading xlsx: %s", xlsx_path)
    df_xlsx = pd.read_excel(xlsx_path, sheet_name=0)
    log.info("xlsx: %d rows, columns=%s", len(df_xlsx), df_xlsx.columns.tolist())

    # Validate Usage column
    usage_counts = df_xlsx["Usage"].value_counts()
    log.info("Usage value counts: %s", usage_counts.to_dict())
    if set(df_xlsx["Usage"].unique()) != {"Training", "Test"}:
        log.error("Unexpected Usage values: %s", df_xlsx["Usage"].unique())
        return 1

    # inst_ids directly from the 'sig_id' column.
    # Note: load_inst_ids() uses a narrow BRD-K regex that misses BRD-A / compound
    # IDs with hyphens (BRD-K55696337-003-16-0 etc.), so we read the column directly.
    # The pure lib is correct for its stated scope; the xlsx simply has a broader
    # sig_id format than the regex anticipated.
    inst_ids_all = df_xlsx["sig_id"].astype(str).str.strip().tolist()
    assert len(inst_ids_all) == 6000, f"Expected 6000 sig_ids, got {len(inst_ids_all)}"
    log.info("loaded %d inst_ids from sig_id column directly", len(inst_ids_all))

    # Build a parallel usage + label vector aligned to inst_ids_all
    usage_vec = df_xlsx["Usage"].tolist()          # "Training" / "Test"
    label_vec = df_xlsx["DILIst.1"].tolist()       # 0 / 1
    compound_vec = df_xlsx["CompoundName"].tolist()

    # ── 3. H5 lookup ─────────────────────────────────────────────────────────
    log.info("looking up %d inst_ids in local h5", len(inst_ids_all))
    matrix, found_ids, missing_ids = lookup_inst_ids(LINCS_H5, inst_ids_all)
    n_found = len(found_ids)
    n_missing = len(missing_ids)
    miss_frac = n_missing / max(len(inst_ids_all), 1)
    log.info("h5 lookup: %d found, %d missing (%.2f%%)", n_found, n_missing, 100 * miss_frac)

    if miss_frac > 0.01:
        log.warning(
            "WARN: %.2f%% inst_ids missing exceeds 1%% threshold — "
            "proceeding anyway (build_wangli_random_only does not halt on miss rate).",
            100 * miss_frac,
        )

    # Build found-position-in-input for alignment of parallel vectors
    found_id_set = set(found_ids)
    found_input_positions: list[int] = []
    j = 0
    for i, qid in enumerate(inst_ids_all):
        if j < n_found and found_ids[j] == qid:
            found_input_positions.append(i)
            j += 1

    found_compound_names = [compound_vec[i] for i in found_input_positions]
    found_usage = [usage_vec[i] for i in found_input_positions]
    found_labels = [label_vec[i] for i in found_input_positions]

    # ── 4. SMILES resolution ──────────────────────────────────────────────────
    log.info("loading dili_canonical.csv")
    canonical_df = pd.read_csv(CANONICAL)
    log.info("resolving SMILES for %d compounds", len(found_compound_names))
    resolved = resolve_smiles(found_compound_names, canonical_df)
    n_resolved = sum(1 for s in resolved.smiles if s is not None)
    n_dropped = len(resolved.drop_indices)
    log.info(
        "SMILES: resolved=%d dropped=%d (%.2f%%)",
        n_resolved, n_dropped, 100 * n_resolved / max(len(found_compound_names), 1),
    )

    # ── 5. Build retained mask ────────────────────────────────────────────────
    retained_mask = np.array([s is not None for s in resolved.smiles], dtype=bool)
    retained_idx = np.where(retained_mask)[0]
    n_retained = int(retained_idx.size)
    log.info("retained %d / %d profiles after SMILES resolution", n_retained, n_found)

    if not (5700 <= n_retained <= 6000):
        log.warning(
            "WARN: retained rows %d outside [5700, 6000] expected range — "
            "proceeding (no halt in this script).",
            n_retained,
        )

    # ── 6. Build wangli_profiles.csv ─────────────────────────────────────────
    rows: list[dict] = []
    for idx in retained_idx:
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
            "dili_binary": int(found_labels[idx]),
            "dili_severity": severity if severity is not None else "",
            "tuple_key": _tuple_key(smiles, cell_id, dose_str),
            "usage": found_usage[idx],  # "Training" or "Test"
        })

    df_out = pd.DataFrame(rows, columns=[
        "profile_id", "compound_name", "brd_id", "cell_id", "time_h",
        "dose_str", "dose_um", "smiles", "dili_binary", "dili_severity",
        "tuple_key", "usage",
    ])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_csv = PROCESSED / "wangli_profiles.csv"
    df_out.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(df_out))

    # ── 7. Build wangli_measured_de.npy ──────────────────────────────────────
    measured_de = matrix[retained_idx].astype(np.float32, copy=False)
    out_de = PROCESSED / "wangli_measured_de.npy"
    np.save(out_de, measured_de)
    log.info("wrote %s shape=%s dtype=%s", out_de, measured_de.shape, measured_de.dtype)

    # ── 8. Summary ────────────────────────────────────────────────────────────
    n_train = int((df_out["usage"] == "Training").sum())
    n_test = int((df_out["usage"] == "Test").sum())
    n_pos = int((df_out["dili_binary"] == 1).sum())
    n_neg = int((df_out["dili_binary"] == 0).sum())
    cell_dist = df_out["cell_id"].value_counts().to_dict()

    print(
        f"OK n_retained={n_retained} n_train={n_train} n_test={n_test} "
        f"n_pos={n_pos} n_neg={n_neg} pn_ratio={n_pos / max(n_neg, 1):.3f} "
        f"n_missing_h5={n_missing} n_smiles_dropped={n_dropped}"
    )
    print(f"Cell distribution: {cell_dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
