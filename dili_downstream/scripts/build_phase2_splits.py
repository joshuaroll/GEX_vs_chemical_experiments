#!/usr/bin/env python3
"""Build all Phase 2 splits from real Phase 1 data.

Wires the 5 pure-library outputs from Plans 01 and 02 (`scaffold_split`,
`cluster_split`, `tdc_dili_scaffold_split`, `filter_upstream_train`,
`compute_transfer_slices`) into a single CLI that runs on the canonical
`dili_canonical.csv` + PDG `all_drugs_pdg.csv`.

Outputs (paths relative to the dili_downstream repo root):
    data/splits/unified_dili_aware_scaffold.json
    data/splits/unified_dili_aware_cluster.json
    data/splits/tdc_dili_scaffold.json
    data/processed/p2_diagnostics.json   (consumed by scripts/summarize_phase2.py)

Halt gate:
    `len(slices['test_drug_novel'])` is checked against the threshold of 30
    per `02-CONTEXT.md` Halt gate 1. PASS → continue (Phase 3 unblocked).
    FAIL → write `HALT_REASON.md` to the phase planning dir AND emit the
    locked FAIL line in P2_split_summary.md.

Usage:
    cd /raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream
    conda run -n dili_v04_env python scripts/build_phase2_splits.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.scaffold_split import scaffold_split, _to_dilist_id  # noqa: E402
from src.data.cluster_split import cluster_split  # noqa: E402
from src.data.tdc_split import tdc_dili_scaffold_split  # noqa: E402
from src.data.upstream_filter import filter_upstream_train  # noqa: E402
from src.data.transfer_slices import compute_transfer_slices  # noqa: E402

# ----------------------------------------------------------------------
# Locked path defaults (per 02-CONTEXT.md + planner_prelim_findings #1)
# ----------------------------------------------------------------------
DEFAULT_CANONICAL_CSV = REPO_ROOT / "data" / "processed" / "dili_canonical.csv"
DEFAULT_PDG_DRUGS_CSV = Path(
    "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_DIAGNOSTICS_OUT = REPO_ROOT / "data" / "processed" / "p2_diagnostics.json"
HALT_REASON_PATH = (
    Path("/raid/home/joshua")
    / ".planning"
    / "phases"
    / "02-unified-split-construction"
    / "HALT_REASON.md"
)
HALT_GATE_THRESHOLD = 30

CANONICAL_COLUMNS = (
    "pert_id",
    "drug_name",
    "smiles",
    "canonical_smiles",
    "scaffold",
    "dili_binary",
    "dili_severity",
    "in_lincs",
    "in_pdg",
)


# ============================================================================
# Inline helpers (kept here to keep `src/data/` libraries pure)
# ============================================================================


def _from_dilist_id(s: str) -> int:
    """Inverse of `_to_dilist_id` — strip 'DILIST_' prefix and return int."""
    if not s.startswith("DILIST_"):
        raise ValueError(f"Expected DILIST_NNNN, got {s!r}")
    return int(s.removeprefix("DILIST_"))


def _derive_test_lincs_held_out(
    test_pert_id_strs: list[str], canon: pd.DataFrame
) -> list[str]:
    """Return the subset of `test_pert_id_strs` whose `in_lincs` flag is False.

    "Held out from LINCS" = drugs in test that DO NOT have a measured
    LINCS L1000 signature; this is the regime where MultiDCP predicted
    signatures must do work measured can't (per 02-CONTEXT.md
    "Implementation defaults" → split-file format).
    """
    out: list[str] = []
    for pid_str in test_pert_id_strs:
        pid_int = _from_dilist_id(pid_str)
        rows = canon.loc[canon["pert_id"] == pid_int]
        if len(rows) == 0:
            # Defensive — don't crash; just skip.
            continue
        if not bool(rows.iloc[0]["in_lincs"]):
            out.append(pid_str)
    return sorted(out)


def _morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _max_train_tanimoto_per_test_drug(
    canonical_df: pd.DataFrame,
    train_pid_strs: list[str],
    test_pid_strs: list[str],
) -> list[float]:
    """For each test drug, compute max Tanimoto similarity to any train drug."""
    canon = canonical_df.set_index("pert_id", drop=False)
    train_int = [_from_dilist_id(p) for p in train_pid_strs]
    test_int = [_from_dilist_id(p) for p in test_pid_strs]

    train_smis = [str(canon.loc[p]["canonical_smiles"]) for p in train_int]
    test_smis = [str(canon.loc[p]["canonical_smiles"]) for p in test_int]

    train_fps = [_morgan_fp(s) for s in train_smis]
    test_fps = [_morgan_fp(s) for s in test_smis]

    if not train_fps:
        return [0.0] * len(test_fps)

    out: list[float] = []
    for tfp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        out.append(float(max(sims)) if sims else 0.0)
    return out


def _bucket_tanimoto_histogram(
    sims: list[float],
) -> list[tuple[float, float, int]]:
    """Bucket `sims` into 10 bins of width 0.1 across [0.0, 1.0].

    The right edge 1.0 is INCLUSIVE in the last bucket to handle perfect-match
    cases (Tanimoto == 1.0).
    """
    edges = [round(i * 0.1, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
    counts = [0] * 10
    for s in sims:
        if s >= 1.0:
            counts[9] += 1
            continue
        idx = min(9, int(s * 10.0))
        counts[idx] += 1
    return [(edges[i], edges[i + 1], counts[i]) for i in range(10)]


def _class_balance(
    pid_strs: list[str], canon: pd.DataFrame
) -> tuple[float, int, int]:
    """Compute (positive_rate, n_positive, n_total) for a list of DILIST_NNNN ids."""
    if not pid_strs:
        return (0.0, 0, 0)
    int_ids = [_from_dilist_id(p) for p in pid_strs]
    sub = canon.loc[canon["pert_id"].isin(int_ids), "dili_binary"]
    n_total = int(len(sub))
    n_pos = int(sub.sum())
    rate = (n_pos / n_total) if n_total else 0.0
    return (rate, n_pos, n_total)


def _check_disjoint(
    split_dict: dict[str, list[str]],
    keys: tuple[str, str, str],
    label: str,
) -> None:
    """Assert the three lists at `keys` in `split_dict` are pairwise disjoint."""
    a, b, c = (set(split_dict[k]) for k in keys)
    if a & b:
        raise AssertionError(
            f"{label}: {keys[0]} ∩ {keys[1]} non-empty: {sorted(a & b)[:5]}..."
        )
    if a & c:
        raise AssertionError(
            f"{label}: {keys[0]} ∩ {keys[2]} non-empty: {sorted(a & c)[:5]}..."
        )
    if b & c:
        raise AssertionError(
            f"{label}: {keys[1]} ∩ {keys[2]} non-empty: {sorted(b & c)[:5]}..."
        )


def _train_scaffolds_from_split(
    train_pid_strs: list[str], canon: pd.DataFrame
) -> set[str]:
    """Build the set of scaffold strings appearing across train pert_ids."""
    out: set[str] = set()
    for pid_str in train_pid_strs:
        pid_int = _from_dilist_id(pid_str)
        rows = canon.loc[canon["pert_id"] == pid_int]
        if len(rows) == 0:
            continue
        scaffold = rows.iloc[0]["scaffold"]
        if isinstance(scaffold, str):
            out.add(scaffold)
        else:
            out.add("")  # NaN scaffold treated as acyclic
    return out


# ============================================================================
# Halt-gate FAIL writer
# ============================================================================


def _write_halt_reason(
    halt_value: int,
    upstream_diag: dict[str, int],
    transfer_slices: dict[str, list[str]],
) -> None:
    """Write the locked HALT_REASON.md per CLAUDE.md halt-gate convention."""
    HALT_REASON_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# HALT REASON: Phase 2 — Halt Gate 1 Fired

**Date:** {dt.datetime.now(dt.timezone.utc).isoformat()}
**Phase:** 02-unified-split-construction
**Halt gate:** 1 (`|D_DILI_test \\ D_PDG| < 30`)

## Reason

The drug-novel transfer slice is **below** the locked threshold of 30:

    |test_drug_novel| = {halt_value} (threshold {HALT_GATE_THRESHOLD})

Without ≥30 drug-novel test drugs, the headline transfer-test slice
(D_DILI_test \\ D_PDG) doesn't have enough cardinality to support
meaningful AUROC reporting in Phase 5.

## Slice composition (counts)

- |test_in_pdg| = {len(transfer_slices['test_in_pdg'])}
- |test_drug_novel| = {len(transfer_slices['test_drug_novel'])}  ← halt-gate input
- |test_drug_and_scaffold_novel| = {len(transfer_slices['test_drug_and_scaffold_novel'])}

## Upstream filter diagnostics

```
{json.dumps(upstream_diag, indent=2)}
```

## Suggested mitigations

1. **Increase D_DILI test fraction** — bump test from 10% to 15% or 20%
   (decreases train, but PDG isn't reduced so drug-novel count grows).
2. **Use cluster split as primary** — cluster split groups more aggressively
   and may push more drugs out of in-PDG bucket.
3. **Re-evaluate PDG snapshot** — verify `all_drugs_pdg.csv` is current; if a
   newer snapshot has fewer drugs, drug-novel count goes up.
4. **Discuss the leakage discipline** — Option (a) is conservative; option (b)
   (no scaffold exclusion) recovers some test-drug coverage at the cost of
   weaker upstream cleanliness.

## Action

Invoke `/gsd-discuss-phase 2 --reset` to re-discuss the split design before
attempting Phase 3 (upstream training).

— `scripts/build_phase2_splits.py` (CLAUDE.md halt-gate convention)
"""
    HALT_REASON_PATH.write_text(text)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build Phase 2 splits + halt-gate eval (SPLIT-01..05)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--canonical-csv",
        type=Path,
        default=DEFAULT_CANONICAL_CSV,
        help="Path to data/processed/dili_canonical.csv (Phase 1 output).",
    )
    ap.add_argument(
        "--pdg-drugs-csv",
        type=Path,
        default=DEFAULT_PDG_DRUGS_CSV,
        help="Path to MultiDCP/data/all_drugs_pdg.csv (PDG drug_name + cpd_smiles).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--diagnostics-out", type=Path, default=DEFAULT_DIAGNOSTICS_OUT
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("build_phase2_splits")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load canonical CSV (Phase 1 invariant: 1,118 rows × 9 cols)
    # ------------------------------------------------------------------
    if not args.canonical_csv.exists():
        raise FileNotFoundError(
            f"{args.canonical_csv} missing. Run scripts/build_dili_canonical.py first."
        )
    canon = pd.read_csv(args.canonical_csv)
    assert tuple(canon.columns) == CANONICAL_COLUMNS, (
        f"canonical CSV column mismatch: {canon.columns.tolist()}"
    )
    assert len(canon) == 1118, (
        f"Expected 1,118 rows in canonical CSV; saw {len(canon)}"
    )
    log.info("Loaded canonical CSV: %d rows", len(canon))

    # ------------------------------------------------------------------
    # 2. Load PDG drugs CSV
    # ------------------------------------------------------------------
    if not args.pdg_drugs_csv.exists():
        raise FileNotFoundError(
            f"{args.pdg_drugs_csv} missing. PDG SMILES source per "
            "02-CONTEXT.md planner_prelim_findings #1."
        )
    pdg_drugs = pd.read_csv(args.pdg_drugs_csv)
    if list(pdg_drugs.columns) != ["drug_name", "cpd_smiles"]:
        raise ValueError(
            f"Unexpected PDG drugs columns: {pdg_drugs.columns.tolist()}; "
            "expected ['drug_name', 'cpd_smiles']."
        )
    log.info("Loaded PDG drugs: %d rows", len(pdg_drugs))

    # ------------------------------------------------------------------
    # 3. Scaffold split (SPLIT-01)
    # ------------------------------------------------------------------
    log.info("Computing scaffold split (SPLIT-01)…")
    scaffold_dict = scaffold_split(canon, seed=args.seed)
    log.info(
        "Scaffold split sizes: train=%d val=%d test=%d",
        len(scaffold_dict["train"]),
        len(scaffold_dict["val"]),
        len(scaffold_dict["test"]),
    )
    _check_disjoint(scaffold_dict, ("train", "val", "test"), "scaffold split")

    # ------------------------------------------------------------------
    # 4. Cluster split (SPLIT-03)
    # ------------------------------------------------------------------
    log.info("Computing cluster split (SPLIT-03)…")
    cluster_dict = cluster_split(
        canon,
        seed=args.seed,
        tanimoto_threshold=0.4,
        cache_dir=REPO_ROOT / "data" / "processed",
    )
    log.info(
        "Cluster split sizes: train=%d val=%d test=%d",
        len(cluster_dict["train"]),
        len(cluster_dict["val"]),
        len(cluster_dict["test"]),
    )
    _check_disjoint(cluster_dict, ("train", "val", "test"), "cluster split")

    # ------------------------------------------------------------------
    # 5. TDC split (SPLIT-04)
    # ------------------------------------------------------------------
    log.info("Computing TDC-DILI scaffold split (SPLIT-04)…")
    tdc_dict = tdc_dili_scaffold_split(
        seed=args.seed, cache_dir=Path.home() / ".tdc"
    )
    log.info(
        "TDC split sizes: train=%d val=%d test=%d (tdc_version=%s, dataset_size=%d)",
        len(tdc_dict["train"]),
        len(tdc_dict["val"]),
        len(tdc_dict["test"]),
        tdc_dict["tdc_version"],
        tdc_dict["dataset_size"],
    )

    # ------------------------------------------------------------------
    # 6. Upstream filter (SPLIT-02)
    # ------------------------------------------------------------------
    log.info("Computing upstream-train filter (SPLIT-02)…")
    train_upstream, upstream_diag = filter_upstream_train(
        pdg_drugs, scaffold_dict, scaffold_dict["test"], canon
    )
    log.info(
        "Upstream filter: |D_PDG|=%d, excluded_scaffold=%d, excluded_drug_name=%d, "
        "intersection=%d, |D_PDG_train|=%d",
        upstream_diag["d_pdg_total"],
        upstream_diag["excluded_by_scaffold"],
        upstream_diag["excluded_by_pert_id"],
        upstream_diag["excluded_intersection"],
        upstream_diag["d_pdg_train_after_exclusion"],
    )

    # ------------------------------------------------------------------
    # 7. Transfer slices (SPLIT-05)
    # ------------------------------------------------------------------
    log.info("Computing three transfer slices (SPLIT-05)…")
    pdg_drug_names_lower = {
        str(n).strip().lower() for n in pdg_drugs["drug_name"].tolist()
    }
    train_scaffolds = _train_scaffolds_from_split(scaffold_dict["train"], canon)
    slices = compute_transfer_slices(
        scaffold_dict["test"], canon, pdg_drug_names_lower, train_scaffolds
    )
    log.info(
        "Transfer slices: |test_in_pdg|=%d, |test_drug_novel|=%d, "
        "|test_drug_and_scaffold_novel|=%d",
        len(slices["test_in_pdg"]),
        len(slices["test_drug_novel"]),
        len(slices["test_drug_and_scaffold_novel"]),
    )

    # ------------------------------------------------------------------
    # 8. test_lincs_held_out for scaffold + cluster splits
    # ------------------------------------------------------------------
    scaffold_lincs_held_out = _derive_test_lincs_held_out(scaffold_dict["test"], canon)
    cluster_lincs_held_out = _derive_test_lincs_held_out(cluster_dict["test"], canon)
    log.info(
        "test_lincs_held_out: scaffold=%d cluster=%d",
        len(scaffold_lincs_held_out),
        len(cluster_lincs_held_out),
    )

    # ------------------------------------------------------------------
    # 9. Tanimoto histograms
    # ------------------------------------------------------------------
    log.info("Computing Tanimoto train-test max-similarity histograms…")
    sc_sims = _max_train_tanimoto_per_test_drug(
        canon, scaffold_dict["train"], scaffold_dict["test"]
    )
    cl_sims = _max_train_tanimoto_per_test_drug(
        canon, cluster_dict["train"], cluster_dict["test"]
    )
    sc_hist = _bucket_tanimoto_histogram(sc_sims)
    cl_hist = _bucket_tanimoto_histogram(cl_sims)
    log.info(
        "Scaffold Tanimoto: mean=%.3f max=%.3f n=%d",
        float(np.mean(sc_sims)) if sc_sims else 0.0,
        float(max(sc_sims)) if sc_sims else 0.0,
        len(sc_sims),
    )

    # ------------------------------------------------------------------
    # 10. Class balance per slice
    # ------------------------------------------------------------------
    sc_cb_train = _class_balance(scaffold_dict["train"], canon)
    sc_cb_val = _class_balance(scaffold_dict["val"], canon)
    sc_cb_test = _class_balance(scaffold_dict["test"], canon)
    cl_cb_train = _class_balance(cluster_dict["train"], canon)
    cl_cb_val = _class_balance(cluster_dict["val"], canon)
    cl_cb_test = _class_balance(cluster_dict["test"], canon)

    # ------------------------------------------------------------------
    # 11. Halt-gate evaluation
    # ------------------------------------------------------------------
    halt_value = len(slices["test_drug_novel"])
    halt_passed = halt_value >= HALT_GATE_THRESHOLD
    if halt_passed:
        line = (
            f"HALT-GATE PASS: |D_DILI_test \\ D_PDG| = {halt_value} "
            f"(threshold {HALT_GATE_THRESHOLD})"
        )
    else:
        line = (
            f"HALT-GATE FAIL: |D_DILI_test \\ D_PDG| = {halt_value} "
            f"(threshold {HALT_GATE_THRESHOLD}) -- STOP and re-discuss before Phase 3"
        )
        _write_halt_reason(halt_value, upstream_diag, slices)
    print(line)
    log.info(line)

    # ------------------------------------------------------------------
    # 12. Determine cluster-split stratification status
    # ------------------------------------------------------------------
    # Per cluster_split's behavior: if max cluster size > 10% of n_total, it
    # logs a non-stratified warning. Detect by checking whether the per-slice
    # positive rates are wildly different (heuristic — but we honor the
    # stratification flag in the diagnostics).
    n_total = len(canon)
    # Recompute max cluster size cheaply: we can derive it from the cluster
    # split sizes — if any partition has wildly different rates it's
    # non-stratified. Simpler: check per-slice positive-rate spread > 5pp.
    cl_rates = [cl_cb_train[0], cl_cb_val[0], cl_cb_test[0]]
    cluster_stratified = (max(cl_rates) - min(cl_rates)) <= 0.05

    # ------------------------------------------------------------------
    # 13. Assemble + write JSONs
    # ------------------------------------------------------------------
    scaffold_out = {
        "train": scaffold_dict["train"],
        "val": scaffold_dict["val"],
        "test": scaffold_dict["test"],
        "test_lincs_held_out": scaffold_lincs_held_out,
        "test_in_pdg": slices["test_in_pdg"],
        "test_drug_novel": slices["test_drug_novel"],
        "test_drug_and_scaffold_novel": slices["test_drug_and_scaffold_novel"],
        "train_upstream": train_upstream,
    }
    cluster_out = {
        "train": cluster_dict["train"],
        "val": cluster_dict["val"],
        "test": cluster_dict["test"],
        "test_lincs_held_out": cluster_lincs_held_out,
    }
    tdc_out = {
        "train": tdc_dict["train"],
        "val": tdc_dict["val"],
        "test": tdc_dict["test"],
        "tdc_version": tdc_dict["tdc_version"],
        "dataset_size": tdc_dict["dataset_size"],
    }

    scaffold_path = args.out_dir / "unified_dili_aware_scaffold.json"
    cluster_path = args.out_dir / "unified_dili_aware_cluster.json"
    tdc_path = args.out_dir / "tdc_dili_scaffold.json"

    scaffold_path.write_text(json.dumps(scaffold_out, indent=2, sort_keys=False))
    cluster_path.write_text(json.dumps(cluster_out, indent=2, sort_keys=False))
    tdc_path.write_text(json.dumps(tdc_out, indent=2, sort_keys=False))
    log.info("Wrote %s", scaffold_path)
    log.info("Wrote %s", cluster_path)
    log.info("Wrote %s", tdc_path)

    # ------------------------------------------------------------------
    # 14. Assemble + write diagnostics for the summarize driver
    # ------------------------------------------------------------------
    diagnostics = {
        "scaffold_split": {
            "size_train": len(scaffold_dict["train"]),
            "size_val": len(scaffold_dict["val"]),
            "size_test": len(scaffold_dict["test"]),
            "class_balance_train": list(sc_cb_train),
            "class_balance_val": list(sc_cb_val),
            "class_balance_test": list(sc_cb_test),
            "stratified": True,
            "tanimoto_histogram_buckets": [list(b) for b in sc_hist],
        },
        "cluster_split": {
            "size_train": len(cluster_dict["train"]),
            "size_val": len(cluster_dict["val"]),
            "size_test": len(cluster_dict["test"]),
            "class_balance_train": list(cl_cb_train) if cluster_stratified else None,
            "class_balance_val": list(cl_cb_val) if cluster_stratified else None,
            "class_balance_test": list(cl_cb_test) if cluster_stratified else None,
            "stratified": cluster_stratified,
            "tanimoto_histogram_buckets": [list(b) for b in cl_hist],
        },
        "tdc_split": {
            "tdc_version": tdc_dict["tdc_version"],
            "dataset_size": tdc_dict["dataset_size"],
            "size_train": len(tdc_dict["train"]),
            "size_val": len(tdc_dict["val"]),
            "size_test": len(tdc_dict["test"]),
        },
        "transfer_slices": {
            "test_in_pdg": len(slices["test_in_pdg"]),
            "test_drug_novel": len(slices["test_drug_novel"]),
            "test_drug_and_scaffold_novel": len(slices["test_drug_and_scaffold_novel"]),
        },
        "upstream_filter": {
            "d_pdg_total": upstream_diag["d_pdg_total"],
            "excluded_by_scaffold": upstream_diag["excluded_by_scaffold"],
            "excluded_by_pert_id": upstream_diag["excluded_by_pert_id"],
            "excluded_intersection": upstream_diag["excluded_intersection"],
            "d_pdg_train_after_exclusion": upstream_diag["d_pdg_train_after_exclusion"],
        },
        "halt_gate": {
            "value": halt_value,
            "passed": halt_passed,
            "threshold": HALT_GATE_THRESHOLD,
        },
        "metadata": {
            "seed": args.seed,
            "dili_canonical_rows": len(canon),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }
    args.diagnostics_out.write_text(json.dumps(diagnostics, indent=2))
    log.info("Wrote %s", args.diagnostics_out)

    # ------------------------------------------------------------------
    # 15. Print summary line
    # ------------------------------------------------------------------
    print(
        f"OK n_canonical={len(canon)} "
        f"scaffold={len(scaffold_dict['train'])}/{len(scaffold_dict['val'])}/"
        f"{len(scaffold_dict['test'])} "
        f"cluster={len(cluster_dict['train'])}/{len(cluster_dict['val'])}/"
        f"{len(cluster_dict['test'])} "
        f"tdc={len(tdc_dict['train'])}/{len(tdc_dict['val'])}/{len(tdc_dict['test'])} "
        f"halt_gate={'PASS' if halt_passed else 'FAIL'}({halt_value})"
    )


if __name__ == "__main__":
    main()
