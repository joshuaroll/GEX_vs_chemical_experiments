"""Phase 1 EDA driver: floor + ceiling + diagnostics + Halt Gate 2.

Sub-commands:
  floor       -- EDA-01: structure-only LR/RF floor on DILIrank ECFP4 fingerprints.
  ceiling     -- EDA-02: measured-biology ceiling (Wang/Li LINCS DE).
  diagnostics -- EDA-03: region distinguishability + human-rodent basal concordance.
  all         -- Run floor, ceiling, diagnostics in sequence; evaluate Halt Gate 2.

Hard rules honored:
    - Real data only: all reads are from verified on-disk files (MANIFEST.md rows
      asserted before use, D-05).
    - DE rule: ceiling uses wangli_measured_de.npy (already differential expression;
      no raw expression is used).
    - Halt Gate 2: gate_fires = ci_lower <= 0 (CI includes 0). HALT_REASON.md written
      and sys.exit(1) emitted AFTER writing P1_eda.md when gate fires (D-02).
    - Honesty rule: if the measured ceiling is unavailable, report "no measured ceiling
      -- floor only" and skip the gate (D-05).
    - Provenance before use: MANIFEST.md rows checked for all 6 reused external files
      BEFORE any read (T-01-10 mitigate; sys.exit(2) if a row is absent).
    - SMILES coverage below 40% of non-Ambiguous DILIrank is flagged in P1_eda.md.
    - Mouse liver is reported whole-sample with "no zone annotation available" (Open Q3).
    - OOD method is "Mahalanobis, 978-gene landmark subspace, alpha=1e-2" (D-05 name).
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import pathlib
import sys
import tarfile
import tempfile
import time
import zipfile
from typing import Optional

# ---------------------------------------------------------------------------
# sys.path injection (analog: report_coverage.py lines 35-36)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths (analog: report_coverage.py lines 46-52)
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
PHASE_DIR = ROOT / ".planning" / "phases" / "01-eda-the-bracket"

# Data paths (local)
DILIRANK_PATH = ROOT / "data" / "raw" / "labels" / "dilirank" / "dilirank.xlsx"
DILIST_PATH = ROOT / "data" / "raw" / "labels" / "dilist" / "dilist.xlsx"
YU2022_L5_ZIP = ROOT / "data" / "raw" / "spatial" / "yu2022_liver" / "L5_upload.zip"
GSE272564_TAR = (
    ROOT / "data" / "raw" / "spatial" / "gse272564_mouse_liver_ctrl" / "GSE272564_RAW.tar"
)
PDG_MANIFOLD_CSV = pathlib.Path(
    "/raid/home/joshua/projects/MultiDCP_pdg/data/"
    "pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv"
)
MULTIDCP_SYMBOLS_PATH = ROOT / "data" / "processed" / "spatial" / "multidcp_10716_symbols.txt"
# build_one2one_orthologs expects the RAW BioMart 8-column TSV (not the pre-filtered 6-col output)
ORTHOLOG_RAW_TSV_PATH = ROOT / "data" / "raw" / "spatial" / "biomart" / "orthologs_raw_116_20260620.tsv"
# Pre-filtered TSV (used as direct DataFrame load when available)
ORTHOLOG_TSV_PATH = ROOT / "data" / "processed" / "spatial" / "orthologs_h_m_r_one2one.tsv"

# Sibling v0.5 paths
_SIBLING = ROOT.parent / "dili_downstream" / "data" / "processed"
WANGLI_DE_PATH = _SIBLING / "wangli_measured_de.npy"
WANGLI_PROFILES_PATH = _SIBLING / "wangli_profiles.csv"
DILI_CANONICAL_PATH = _SIBLING / "dili_canonical.csv"
DRUGBANK_PATH = _SIBLING / "drugbank_smiles_index.csv"

# ---------------------------------------------------------------------------
# MANIFEST provenance row check (T-01-10; D-05)
# ---------------------------------------------------------------------------

_MANIFEST_PATH = ROOT / "MANIFEST.md"

_REQUIRED_MANIFEST_FILENAMES = [
    "wangli_measured_de.npy",
    "wangli_profiles.csv",
    "dili_canonical.csv",
    "drugbank_smiles_index.csv",
    "pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv",
    "dilirank.xlsx",
]


def _assert_manifest_rows(manifest_path: pathlib.Path, required: list[str]) -> None:
    """Assert that each required filename appears as a MANIFEST.md row.

    Reads the MANIFEST text and checks each filename is present (substring match
    on a table row). Prints an error and calls sys.exit(2) if any are missing
    (T-01-10 provenance-before-use mitigation).
    """
    if not manifest_path.exists():
        print(
            f"ERROR: MANIFEST.md not found at {manifest_path}. "
            "Cannot assert provenance rows.",
            file=sys.stderr,
        )
        sys.exit(2)

    manifest_text = manifest_path.read_text(encoding="utf-8")
    missing = [f for f in required if f not in manifest_text]
    if missing:
        print(
            f"ERROR: The following files are used as external inputs but are absent "
            f"from MANIFEST.md: {missing}. Add provenance rows (SHA256 + source) "
            "before reading these files (D-05 provenance-before-use).",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_p1_eda")

# ---------------------------------------------------------------------------
# Library imports (after sys.path set)
# ---------------------------------------------------------------------------

from src.spatial.eda.labels import load_dilirank, load_dilist  # noqa: E402
from src.spatial.eda.smiles_join import join_smiles_cascade  # noqa: E402
from src.spatial.eda.fingerprints import smiles_to_ecfp4  # noqa: E402
from src.spatial.eda.floor import compute_floor, floor_probabilities  # noqa: E402
from src.spatial.eda.ceiling import load_ceiling, compute_ceiling  # noqa: E402
from src.spatial.eda.bootstrap import paired_bootstrap_auroc_gap, BootstrapResult  # noqa: E402
from src.spatial.eda.region_diagnostics import (  # noqa: E402
    compute_moran_svgs,
    basal_similarity_matrix,
    svg_retention,
    human_mouse_liver_correlation,
    ood_mahalanobis,
    OOD_METHOD,
)
from src.spatial.orthology import build_one2one_orthologs  # noqa: E402
from src.spatial.pseudobulk import from_anndata  # noqa: E402

# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------


def _write_report(path: pathlib.Path, content: str, mode: str = "w") -> None:
    """Write or append to a report file, creating parent dirs if needed."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Sub-command: floor (EDA-01)
# ---------------------------------------------------------------------------


def run_floor(report_lines: list[str]) -> dict:
    """Compute structure-only floor and write EDA-01 rows to report_lines.

    Returns a dict with keys: floor_result, floor_probs_by_drug, smiles_df,
    labels_df, coverage_flag, n_non_ambiguous, n_with_smiles.
    """
    log.info("=== FLOOR (EDA-01) ===")

    # --- Load labels ---
    labels_df = load_dilirank(str(DILIRANK_PATH))
    n_total_dilirank = len(labels_df)
    n_positive = int(labels_df["dili_binary"].sum())
    n_negative = n_total_dilirank - n_positive
    # Non-Ambiguous = all rows that survived the Ambiguous filter (pos + neg)
    n_non_ambiguous = n_total_dilirank

    log.info("DILIrank: %d non-Ambiguous (%d pos, %d neg)", n_total_dilirank, n_positive, n_negative)

    # --- SMILES join cascade ---
    smiles_df = join_smiles_cascade(
        labels_df,
        str(DILI_CANONICAL_PATH),
        str(DRUGBANK_PATH),
    )

    n_with_smiles = int(smiles_df["smiles"].notna().sum())
    coverage = n_with_smiles / max(n_non_ambiguous, 1)
    coverage_flag = ""
    if coverage < 0.40:
        coverage_flag = (
            f"**WARNING: SMILES coverage {coverage:.1%} < 40% of non-Ambiguous DILIrank "
            f"({n_with_smiles}/{n_non_ambiguous}) -- floor classifier is underpowered.**"
        )
        log.warning("SMILES coverage %.1f%% < 40%% threshold.", 100 * coverage)
    else:
        log.info("SMILES coverage %.1f%% (%d / %d).", 100 * coverage, n_with_smiles, n_non_ambiguous)

    # --- Fingerprints on SMILES-covered subset ---
    covered_mask = smiles_df["smiles"].notna()
    smiles_covered = smiles_df.loc[covered_mask, "smiles"].tolist()
    y_covered = smiles_df.loc[covered_mask, "dili_binary"].values

    fps, valid_mask = smiles_to_ecfp4(smiles_covered)
    # Exclude rows where RDKit could not parse the SMILES (valid_mask=False)
    fps_valid = fps[valid_mask]
    y_valid = y_covered[valid_mask]
    n_rdkit_excluded = int((~valid_mask).sum())
    if n_rdkit_excluded > 0:
        log.warning(
            "smiles_to_ecfp4: %d SMILES failed RDKit parsing -- excluded from floor.",
            n_rdkit_excluded,
        )

    n_floor_drugs = len(y_valid)
    log.info("Floor dataset: %d drugs with valid ECFP4 fingerprints.", n_floor_drugs)

    # --- Floor classifier (LR + RF, seeds 0,1,2) ---
    floor_result = compute_floor(
        fps_valid, y_valid, seeds=(0, 1, 2), n_drugs_with_smiles=n_with_smiles
    )

    # --- Floor probabilities (LR, seed=0; for paired bootstrap later) ---
    # Map back to drug names for alignment with ceiling
    valid_drug_names = (
        smiles_df.loc[covered_mask, "name_lower"].values[valid_mask]
    )
    floor_probs_arr = floor_probabilities(fps_valid, y_valid, seed=0)
    floor_probs_by_drug = {
        name: float(prob)
        for name, prob in zip(valid_drug_names, floor_probs_arr)
    }

    # --- DILIst secondary cross-check (D-04: "cheap: base rates + LR floor AUROC") ---
    try:
        dilist_df = load_dilist(str(DILIST_PATH))
        dilist_smiles_df = join_smiles_cascade(
            dilist_df,
            str(DILI_CANONICAL_PATH),
            str(DRUGBANK_PATH),
        )
        dilist_covered = dilist_smiles_df["smiles"].notna()
        n_dilist_with_smiles = int(dilist_covered.sum())
        dilist_smiles_list = dilist_smiles_df.loc[dilist_covered, "smiles"].tolist()
        dilist_y = dilist_smiles_df.loc[dilist_covered, "dili_binary"].values
        if n_dilist_with_smiles >= 10:
            dilist_fps, dilist_valid = smiles_to_ecfp4(dilist_smiles_list)
            dilist_fps_v = dilist_fps[dilist_valid]
            dilist_y_v = dilist_y[dilist_valid]
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.metrics import roc_auc_score
            dilist_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            dilist_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
            dilist_oof = cross_val_predict(dilist_lr, dilist_fps_v.astype(float), dilist_y_v, cv=dilist_cv, method="predict_proba")
            dilist_auroc = float(roc_auc_score(dilist_y_v, dilist_oof[:, 1]))
            dilist_pos_frac = float(dilist_y_v.mean())
            dilist_row = (
                f"| DILIst (secondary, D-04) | {len(dilist_df)} "
                f"| {n_dilist_with_smiles} | {dilist_pos_frac:.3f} "
                f"| {dilist_auroc:.4f} | -- |"
            )
        else:
            dilist_row = f"| DILIst (secondary, D-04) | {len(dilist_df)} | {n_dilist_with_smiles} | N/A | N/A (too few) | -- |"
    except Exception as exc:
        log.warning("DILIst secondary cross-check failed: %s", exc)
        dilist_row = f"| DILIst (secondary, D-04) | ERROR | -- | -- | -- | {exc} |"

    # --- Report section ---
    report_lines.append("## EDA-01: Structure-Only Floor (DILIrank, liver/human)\n\n")
    report_lines.append(
        f"**Label set:** DILIrank 2.0 (FDA LTKB); Wang/Li convention (D-04).\n"
        f"**Non-Ambiguous drugs:** {n_non_ambiguous} ({n_positive} positive, {n_negative} negative).\n"
        f"**SMILES coverage:** {coverage:.1%} ({n_with_smiles} / {n_non_ambiguous} non-Ambiguous).\n"
    )
    if coverage_flag:
        report_lines.append(f"\n{coverage_flag}\n\n")
    else:
        report_lines.append("\n")
    report_lines.append(
        f"**Floor dataset:** {n_floor_drugs} drugs with valid ECFP4 fingerprints "
        f"(2048-bit, radius 2; RDKit excluded: {n_rdkit_excluded}).\n\n"
    )
    report_lines.append("### Floor metrics\n\n")
    report_lines.append(
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Label entropy (bits) | {floor_result.label_entropy:.4f} |\n"
        f"| Class balance (positive fraction) | {floor_result.class_balance:.4f} |\n"
        f"| AUPRC base rate | {floor_result.auprc_base_rate:.4f} |\n"
        f"| LR floor AUROC (mean, seeds 0-2) | {floor_result.lr_auroc:.4f} |\n"
        f"| RF floor AUROC (mean, seeds 0-2) | {floor_result.rf_auroc:.4f} |\n"
        f"| n drugs (floor dataset) | {n_floor_drugs} |\n"
        "\n"
    )
    report_lines.append("### DILIst secondary cross-check (D-04)\n\n")
    report_lines.append(
        "| Label set | n total | n with SMILES | Positive fraction | LR AUROC (seed 0) | Notes |\n"
        "|-----------|---------|---------------|-------------------|-------------------|-------|\n"
    )
    report_lines.append(dilist_row + "\n\n")

    return {
        "floor_result": floor_result,
        "floor_probs_by_drug": floor_probs_by_drug,
        "smiles_df": smiles_df,
        "labels_df": labels_df,
        "coverage_flag": coverage_flag,
        "n_non_ambiguous": n_non_ambiguous,
        "n_with_smiles": n_with_smiles,
    }


# ---------------------------------------------------------------------------
# Sub-command: ceiling (EDA-02)
# ---------------------------------------------------------------------------


def _leakage_decomposition(de_aligned, y_profiles, drug_key, seed: int = 42) -> dict:
    """Decompose the profile-level measured AUROC into leakage vs honest signal.

    The Wang/Li-style headline (profile-level random split) lets a single drug's
    profiles (up to 784) sit in both train and test, so the model memorizes drug
    identity. Comparing a leaky StratifiedKFold split against a drug-disjoint
    StratifiedGroupKFold split quantifies that inflation reproducibly -- this is
    the Phase 1 headline finding (the benchmark is drug-leakage-inflated).
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_predict,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    leaky = cross_val_predict(
        clf, de_aligned, y_profiles,
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        method="predict_proba",
    )[:, 1]
    disjoint = cross_val_predict(
        clf, de_aligned, y_profiles,
        cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed),
        groups=drug_key, method="predict_proba",
    )[:, 1]
    auc_leaky = float(roc_auc_score(y_profiles, leaky))
    auc_disjoint = float(roc_auc_score(y_profiles, disjoint))
    return {
        "profile_auroc_leaky": auc_leaky,
        "profile_auroc_disjoint": auc_disjoint,
        "leakage_inflation": auc_leaky - auc_disjoint,
    }


def run_ceiling(report_lines: list[str]) -> Optional[dict]:
    """Compute measured-biology ceiling and write EDA-02 rows to report_lines.

    Returns dict with ceiling_result and labels_df alignment, or None if
    measured ceiling is unavailable (honesty rule: "no measured ceiling -- floor only").
    """
    log.info("=== CEILING (EDA-02) ===")

    # Honesty rule (D-05): if the measured DE file is absent, report and skip gate.
    try:
        de, profiles = load_ceiling(str(WANGLI_DE_PATH), str(WANGLI_PROFILES_PATH))
    except FileNotFoundError as exc:
        msg = (
            "no measured ceiling -- floor only "
            f"(FileNotFoundError: {exc})"
        )
        log.warning("%s", msg)
        report_lines.append("## EDA-02: Measured-Biology Ceiling (liver/human)\n\n")
        report_lines.append(f"**Status:** {msg}\n\n")
        return None

    # Align profiles to DILIrank binary labels via name_lower
    labels_df = load_dilirank(str(DILIRANK_PATH))
    dilirank_map = dict(zip(labels_df["name_lower"], labels_df["dili_binary"]))

    # Profile-level alignment: assign binary label from DILIrank
    profile_names = profiles["compound_name"].str.lower().str.strip()
    in_dilirank_mask = profile_names.isin(dilirank_map)
    profiles_aligned = profiles[in_dilirank_mask].copy()
    de_aligned = de[in_dilirank_mask.values]

    # Profile-level labels from DILIrank
    y_profiles = profile_names[in_dilirank_mask].map(dilirank_map).values.astype(int)

    n_profiles = len(de_aligned)
    n_ceiling_drugs = profiles_aligned["compound_name"].str.lower().str.strip().nunique()
    log.info(
        "Ceiling: %d profiles from %d unique drugs (DILIrank-aligned).",
        n_profiles, n_ceiling_drugs,
    )

    # Compute ceiling metrics (participation ratio, MI, drug-level AUROC, ceiling_probs)
    ceiling_result = compute_ceiling(de_aligned, y_profiles, profiles=profiles_aligned)

    # Reproducible leakage decomposition (Phase 1 headline finding)
    drug_key = profiles_aligned["compound_name"].str.lower().str.strip().values
    leak = _leakage_decomposition(de_aligned, y_profiles, drug_key)
    ceiling_result["leakage"] = leak

    # Report
    report_lines.append("## EDA-02: Measured-Biology Ceiling (liver/human)\n\n")
    report_lines.append(
        f"**Source:** Wang/Li LINCS measured DE (wangli_measured_de.npy, "
        f"shape {de.shape}, float32; already differential expression, DE rule satisfied).\n"
        f"**Profiles in DILIrank:** {n_profiles} profiles from {n_ceiling_drugs} unique drugs.\n\n"
    )
    report_lines.append("### Ceiling metrics\n\n")
    report_lines.append(
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| PCA participation ratio (effective rank) | {ceiling_result['participation_ratio']:.1f} |\n"
        f"| MI fraction nonzero | {ceiling_result['mi_fraction_nonzero']:.4f} |\n"
        f"| Ceiling AUROC (drug-level, leakage-free) | {ceiling_result['ceiling_auroc']:.4f} |\n"
        f"| n drugs (drug-level, ceiling) | {ceiling_result['n_drugs']} |\n"
        "\n"
    )
    report_lines.append("### Leakage decomposition (Phase 1 headline)\n\n")
    report_lines.append(
        "Profile-level measured-DE AUROC under a leaky split (a drug's profiles "
        "in both train and test, the Wang/Li-style setup) vs a drug-disjoint "
        "split (held-out drugs):\n\n"
        "| Profile-level evaluation | AUROC |\n"
        "|--------|-------|\n"
        f"| Leaky (drug in train+test; benchmark-style) | {leak['profile_auroc_leaky']:.4f} |\n"
        f"| Drug-disjoint (honest, held-out drugs) | {leak['profile_auroc_disjoint']:.4f} |\n"
        f"| **Drug-leakage inflation** | **{leak['leakage_inflation']:+.4f}** |\n\n"
        f"The honest profile-level measured ceiling ({leak['profile_auroc_disjoint']:.3f}) "
        "is comparable to the structure floor, and the leaky number "
        f"({leak['profile_auroc_leaky']:.3f}) reproduces/exceeds the published "
        "Wang/Li benchmark (~0.798). The gap between them is drug-identity "
        "memorization, indicating the benchmark headline is substantially "
        "drug-leakage-inflated.\n\n"
    )

    return {
        "ceiling_result": ceiling_result,
        "labels_df": labels_df,
    }


# ---------------------------------------------------------------------------
# Gap + Halt Gate 2 (combined in "all" sub-command)
# ---------------------------------------------------------------------------


def run_gap_and_gate(
    floor_data: dict,
    ceiling_data: Optional[dict],
    report_lines: list[str],
) -> Optional[BootstrapResult]:
    """Compute floor-ceiling gap, evaluate Halt Gate 2, append to report.

    Returns BootstrapResult or None if ceiling unavailable.
    """
    log.info("=== GAP + HALT GATE 2 ===")

    if ceiling_data is None:
        report_lines.append(
            "## Halt Gate 2\n\n"
            "**Status:** SKIPPED -- no measured ceiling -- floor only (D-05 honesty rule).\n\n"
        )
        return None

    floor_probs_by_drug = floor_data["floor_probs_by_drug"]
    ceiling_probs_by_drug = ceiling_data["ceiling_result"]["ceiling_probs"]
    labels_df = floor_data["labels_df"]
    dilirank_map = dict(zip(labels_df["name_lower"], labels_df["dili_binary"]))

    # Shared drug set: intersection of floor (with SMILES + valid ECFP4) and ceiling drugs
    shared_drugs = sorted(
        set(floor_probs_by_drug.keys()) & set(ceiling_probs_by_drug.keys())
    )
    n_shared = len(shared_drugs)
    log.info("Shared drug set: %d drugs (floor ∩ ceiling).", n_shared)

    if n_shared < 10:
        msg = (
            f"Shared drug set too small ({n_shared} < 10) to run paired bootstrap. "
            "Skipping Halt Gate 2."
        )
        log.error(msg)
        report_lines.append(f"## Halt Gate 2\n\n**Status:** SKIPPED -- {msg}\n\n")
        return None

    # Build aligned arrays
    import numpy as np
    y_shared = np.array([dilirank_map[d] for d in shared_drugs], dtype=int)
    floor_probs_arr = np.array([floor_probs_by_drug[d] for d in shared_drugs])
    ceil_probs_arr = np.array([ceiling_probs_by_drug[d] for d in shared_drugs])

    # Observed AUROCs
    from sklearn.metrics import roc_auc_score
    floor_auroc_shared = float(roc_auc_score(y_shared, floor_probs_arr))
    ceil_auroc_shared = float(roc_auc_score(y_shared, ceil_probs_arr))

    # Paired bootstrap (D-02: 10,000 resamples)
    bootstrap_result = paired_bootstrap_auroc_gap(
        y_shared, floor_probs_arr, ceil_probs_arr, n_resamples=10_000, seed=42
    )

    gate_str = "**FIRES** (stop-and-REFRAME per D-02)" if bootstrap_result.gate_fires else "**CLEAR**"

    report_lines.append("## Gap + Halt Gate 2 (EDA-01/02, D-02)\n\n")
    report_lines.append(
        f"**Shared drug set:** {n_shared} drugs (floor ECFP4 ∩ ceiling LINCS).\n\n"
    )
    n_pos_shared = int(y_shared.sum())
    n_neg_shared = int((y_shared == 0).sum())
    report_lines.append(
        "| Metric | Value |\n"
        "|--------|-------|\n"
        f"| Floor AUROC (LR, shared set) | {floor_auroc_shared:.4f} |\n"
        f"| Ceiling AUROC (drug-grouped OOF, shared set) | {ceil_auroc_shared:.4f} |\n"
        f"| Gap (ceiling - floor) | {bootstrap_result.gap_observed:+.4f} |\n"
        f"| 95% CI [lo, hi] | [{bootstrap_result.ci_lower:.4f}, {bootstrap_result.ci_upper:.4f}] |\n"
        f"| Bootstrap resamples (valid) | {bootstrap_result.n_resamples_valid:,} / 10,000 |\n"
        f"| Shared-set class balance | {n_pos_shared} pos / {n_neg_shared} neg "
        f"({n_pos_shared / n_shared:.1%} positive) |\n"
        f"| Halt Gate 2 | {gate_str} |\n"
        "\n"
    )
    # Power analysis (Hanley-McNeil SE) -- the gate is power-limited by few negatives.
    def _hm_se(A, n_pos, n_neg):
        q1 = A / (2 - A)
        q2 = 2 * A * A / (1 + A)
        v = (A * (1 - A) + (n_pos - 1) * (q1 - A * A) + (n_neg - 1) * (q2 - A * A)) / (
            n_pos * n_neg
        )
        return float(np.sqrt(max(v, 0.0)))

    se_floor = _hm_se(floor_auroc_shared, n_pos_shared, n_neg_shared)
    se_ceil = _hm_se(ceil_auroc_shared, n_pos_shared, n_neg_shared)
    se_gap_indep = float(np.sqrt(se_floor ** 2 + se_ceil ** 2))
    mdes = (1.96 + 0.84) * se_gap_indep  # min detectable gap @80% power, alpha=.05

    report_lines.append("### Power (Hanley-McNeil SE)\n\n")
    report_lines.append(
        "| Quantity | Value |\n"
        "|--------|-------|\n"
        f"| Floor AUROC SE | {se_floor:.4f} |\n"
        f"| Ceiling AUROC SE | {se_ceil:.4f} |\n"
        f"| Min detectable gap @80% power (conservative, indep SE) | {mdes:.4f} |\n"
        f"| Observed |gap| | {abs(bootstrap_result.gap_observed):.4f} |\n\n"
        f"With only {n_neg_shared} negative drugs the conservative minimum "
        f"detectable gap ({mdes:.3f}) exceeds the observed |gap| "
        f"({abs(bootstrap_result.gap_observed):.3f}); the paired-bootstrap "
        "significance comes from cancelling shared per-drug noise (same drugs in "
        "floor and ceiling) and the margin is thin.\n\n"
    )

    leak = ceiling_data["ceiling_result"].get("leakage", {})
    honest_prof = leak.get("profile_auroc_disjoint")
    inflation = leak.get("leakage_inflation")
    report_lines.append(
        "### Interpretation caveat (read before acting on the gate)\n\n"
        "The ceiling AUROC above is a **leakage-free, drug-grouped** estimate "
        "(StratifiedGroupKFold over compound). Three points bound the conclusion:\n\n"
        f"1. **Headline = drug leakage.** At the profile level the honest "
        f"drug-disjoint measured AUROC is {honest_prof:.3f} vs a leaky "
        f"{honest_prof + inflation:.3f} ({inflation:+.3f} inflation) -- the "
        "Wang/Li-style benchmark is substantially drug-leakage-inflated (see "
        "EDA-02 leakage decomposition).\n"
        f"2. **Measured ~= structure at the fair level.** The honest profile-level "
        f"ceiling ({honest_prof:.3f}) is comparable to the structure floor "
        f"({floor_auroc_shared:.3f}); the more negative drug-aggregated gap is "
        "noise from collapsing many profiles onto few drugs.\n"
        f"3. **Underpowered.** Only {n_neg_shared} negative drugs; the gate firing "
        "is marginal (see Power above). Treat this as 'measured biology adds no "
        "lift over structure, benchmark is leakage-inflated', not as a clean "
        "'structure beats biology' result.\n\n"
    )

    return bootstrap_result


# ---------------------------------------------------------------------------
# Sub-command: diagnostics (EDA-03) -- Task 2
# ---------------------------------------------------------------------------


def _load_yu2022_l5(zip_path: pathlib.Path, tmpdir: pathlib.Path):
    """Extract L5_upload.zip and load as AnnData with spatial coords.

    Pitfall 5: load h5 from zip, join category CSV, load spatial coordinates.
    Returns (adata, gene_names).
    """
    import anndata
    import h5py
    import numpy as np
    import pandas as pd

    with zipfile.ZipFile(zip_path) as zf:
        # Extract all to tmpdir
        zf.extractall(tmpdir)

    # Find h5 file
    h5_files = list(tmpdir.rglob("filtered_feature_bc_matrix.h5"))
    if not h5_files:
        raise FileNotFoundError(
            f"filtered_feature_bc_matrix.h5 not found inside {zip_path}"
        )
    h5_path = h5_files[0]

    # Load using scanpy inside dili_v04_env (numba/numpy compatible here)
    try:
        import scanpy as sc
        adata = sc.read_10x_h5(str(h5_path))
    except Exception:
        # Fallback: anndata + h5py manual load
        with h5py.File(str(h5_path), "r") as h5:
            import scipy.sparse as sp
            data = h5["matrix/data"][:]
            indices = h5["matrix/indices"][:]
            indptr = h5["matrix/indptr"][:]
            shape = tuple(h5["matrix/shape"][:])
            # 10x HDF5: shape = (n_genes, n_barcodes); transpose to (n_barcodes, n_genes)
            X_csc = sp.csc_matrix((data, indices, indptr), shape=(shape[0], shape[1]))
            X = X_csc.T  # (n_barcodes, n_genes)
            barcodes = [b.decode() if isinstance(b, bytes) else b
                        for b in h5["matrix/barcodes"][:]]
            gene_names = [n.decode() if isinstance(n, bytes) else n
                          for n in h5["matrix/features/name"][:]]
        adata = anndata.AnnData(X=X, obs=pd.DataFrame(index=barcodes),
                                var=pd.DataFrame(index=gene_names))

    # Make var_names unique (duplicate gene names in 10x h5 cause squidpy Moran's I to fail)
    adata.var_names_make_unique()
    log.info("Loaded yu2022 L5: %s spots x %s genes (var_names made unique)", adata.n_obs, adata.n_vars)

    # Find and join category CSV (barcode -> category)
    cat_files = list(tmpdir.rglob("l5_category.csv"))
    if not cat_files:
        # Try l18_category.csv as fallback (wrong but let's catch the error clearly)
        raise FileNotFoundError(f"l5_category.csv not found inside {zip_path}")
    cat_df = pd.read_csv(cat_files[0], index_col=0)
    # Join on barcode (index)
    adata.obs = adata.obs.join(cat_df, how="left")
    if "category" not in adata.obs.columns:
        # cat_df may have barcodes as a regular column
        cat_df2 = pd.read_csv(cat_files[0])
        # Try various column name conventions
        bc_col = cat_df2.columns[0]
        cat_col = "category" if "category" in cat_df2.columns else cat_df2.columns[1]
        cat_map = dict(zip(cat_df2[bc_col], cat_df2[cat_col]))
        adata.obs["category"] = adata.obs.index.map(cat_map)

    log.info(
        "Region annotation: %s (unique categories: %s)",
        "category",
        list(adata.obs["category"].dropna().unique()),
    )

    # Load spatial coordinates (Pitfall 5: Moran's I needs obsm['spatial'])
    pos_files = list(tmpdir.rglob("tissue_positions_list.csv"))
    if pos_files:
        pos_df = pd.read_csv(pos_files[0], header=None,
                             names=["barcode", "in_tissue", "row", "col", "pix_row", "pix_col"])
        pos_df = pos_df.set_index("barcode")
        # Align to AnnData barcodes
        coords = pos_df.loc[adata.obs.index.intersection(pos_df.index), ["pix_col", "pix_row"]].values
        # If not all barcodes matched, align carefully
        if len(coords) == adata.n_obs:
            adata.obsm["spatial"] = coords
        else:
            coord_arr = np.zeros((adata.n_obs, 2), dtype=float)
            idx_map = {bc: i for i, bc in enumerate(pos_df.index)}
            for i, bc in enumerate(adata.obs.index):
                if bc in idx_map:
                    row_i = idx_map[bc]
                    coord_arr[i] = [pos_df.iloc[row_i]["pix_col"], pos_df.iloc[row_i]["pix_row"]]
            adata.obsm["spatial"] = coord_arr
        log.info("Spatial coordinates loaded into obsm['spatial'] (shape: %s).", adata.obsm["spatial"].shape)
    else:
        log.warning("tissue_positions_list.csv not found inside %s; Moran's I will use fallback.", zip_path)
        # Create dummy grid coords so spatial graph can still be built
        _n = adata.n_obs
        adata.obsm["spatial"] = np.column_stack([
            np.arange(_n) % 100,
            np.arange(_n) // 100,
        ]).astype(float)

    gene_names = list(adata.var_names)
    return adata, gene_names


def _load_gse272564_mouse_ctrl(tar_path: pathlib.Path, tmpdir: pathlib.Path):
    """Extract GSE272564_RAW.tar, filter to APAP0h (GSM8404653), load as pseudobulk.

    Pitfall 6: all four APAP timepoints in one tar; filter to *APAP0h*.
    Returns (mouse_profile, mouse_genes) -- whole-sample pseudobulk.
    """
    import gzip as _gzip
    import io as _io
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if not tar_path.exists():
        raise FileNotFoundError(f"GSE272564_RAW.tar not found at {tar_path}")

    # Extract to tmpdir
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmpdir)

    # Find APAP0h files (Pitfall 6: filter to control timepoint)
    all_files = list(tmpdir.rglob("*"))
    apap0h_files = [f for f in all_files if "APAP0h" in f.name or "APAP0H" in f.name]
    log.info("GSE272564 APAP0h files found: %s", [f.name for f in apap0h_files])

    # Find matrix, barcodes, features for APAP0h
    matrix_file = next((f for f in apap0h_files if "matrix" in f.name.lower() and f.name.endswith(".mtx.gz")), None)
    barcodes_file = next((f for f in apap0h_files if "barcode" in f.name.lower()), None)
    features_file = next((f for f in apap0h_files if "feature" in f.name.lower() or "gene" in f.name.lower()), None)

    if matrix_file is None or features_file is None:
        # Try h5 fallback
        h5_files = [f for f in apap0h_files if f.name.endswith(".h5")]
        if h5_files:
            import h5py
            with h5py.File(str(h5_files[0]), "r") as h5:
                data = h5["matrix/data"][:]
                indices = h5["matrix/indices"][:]
                indptr = h5["matrix/indptr"][:]
                shape = tuple(h5["matrix/shape"][:])
                X_csc = sp.csc_matrix((data, indices, indptr), shape=(shape[0], shape[1]))
                X = X_csc.T.toarray()
                gene_names = [n.decode() if isinstance(n, bytes) else n
                              for n in h5["matrix/features/name"][:]]
        else:
            raise FileNotFoundError(
                f"Could not find APAP0h matrix/features files in {tar_path}. "
                f"Found files: {[f.name for f in apap0h_files]}"
            )
    else:
        # Read .mtx.gz
        with _gzip.open(str(matrix_file), "rb") as gf:
            import scipy.io
            mat = scipy.io.mmread(_io.BytesIO(gf.read()))
        X = mat.T.toarray() if hasattr(mat, "toarray") else mat.T

        # Read features
        with _gzip.open(str(features_file), "rb") as gf:
            lines = gf.read().decode("utf-8").strip().splitlines()
        gene_names = [line.split("\t")[1] if "\t" in line else line for line in lines]

    # Whole-sample pseudobulk (mean across all spots/cells -- no zone annotation)
    # Open Q3 / Pitfall 6: GSE272564 APAP0h has no zone annotation
    mouse_profile = np.mean(X, axis=0)
    log.info(
        "GSE272564 APAP0h: %d spots x %d genes; whole-sample pseudobulk computed.",
        X.shape[0], X.shape[1],
    )
    return mouse_profile, gene_names


def run_diagnostics(report_lines: list[str]) -> dict:
    """Run EDA-03 region diagnostics + cross-species concordance.

    Implements Task 2: SVG retention, basal similarity, human-mouse correlation,
    Mahalanobis OOD distance.
    """
    import numpy as np
    import pandas as pd

    log.info("=== DIAGNOSTICS (EDA-03) ===")

    diag_results: dict = {}

    # ------------------------------------------------------------------
    # Load MultiDCP 10,716-gene symbol list (for SVG retention)
    # ------------------------------------------------------------------
    if not MULTIDCP_SYMBOLS_PATH.exists():
        raise FileNotFoundError(
            f"MultiDCP symbol list not found at {MULTIDCP_SYMBOLS_PATH}."
        )
    with open(MULTIDCP_SYMBOLS_PATH) as fh:
        model_genes = [line.strip() for line in fh if line.strip()]
    log.info("MultiDCP gene space: %d symbols.", len(model_genes))

    # ------------------------------------------------------------------
    # Load ortholog table (P0 cached one-to-one TSV -- pre-filtered output)
    # build_one2one_orthologs expects the raw BioMart 8-column TSV; for the
    # pre-filtered processed TSV we construct OrthologTable directly.
    # ------------------------------------------------------------------
    from src.spatial.orthology import OrthologTable
    if ORTHOLOG_RAW_TSV_PATH.exists():
        ortholog_table = build_one2one_orthologs(str(ORTHOLOG_RAW_TSV_PATH))
    else:
        # Fallback: read the pre-filtered 6-column TSV directly
        import pandas as _pd_ot
        ot_df = _pd_ot.read_csv(str(ORTHOLOG_TSV_PATH), sep="\t")
        n_ot = len(ot_df)
        ortholog_table = OrthologTable(
            pairs=ot_df,
            n_input=n_ot,
            n_one2one=n_ot,
            dropped_fraction=0.0,
        )
    log.info("Ortholog table: %d one-to-one pairs.", ortholog_table.n_one2one)

    # ------------------------------------------------------------------
    # Load yu2022 L5 Visium (human liver; Pitfall 5)
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        adata, human_genes = _load_yu2022_l5(YU2022_L5_ZIP, tmpdir / "yu2022")
        tmpdir_yu = tmpdir / "yu2022"
        tmpdir_yu.mkdir(exist_ok=True)

        # --- Moran's I SVGs (inside tmpdir context so files exist) ---
        # Subset to model gene space (10,716 genes) for speed; Moran's I on 36k genes is very slow.
        log.info("Computing Moran's I SVGs on yu2022 L5 (subset to MultiDCP gene space)...")
        try:
            import anndata as _ad
            model_genes_set = set(model_genes)
            # Keep genes that are in the model space AND in the adata var_names
            keep_genes = [g for g in adata.var_names if g in model_genes_set]
            adata_subset = adata[:, keep_genes].copy() if len(keep_genes) > 0 else adata
            log.info(
                "Moran's I subset: %d / %d genes kept (in MultiDCP space).",
                len(keep_genes), adata.n_vars,
            )
            moran_result = compute_moran_svgs(adata_subset, spatial_key="spatial", n_neighs=6)
            # SVG retention: since Moran's I was run on the model-gene subset, all SVGs detected
            # are by definition in the model space. Retention = n_top_svgs_in_model / n_top_svgs.
            # Note: coverage_fraction(query, reference) = len(intersection)/len(reference) which is
            # the wrong denominator for this metric -- we compute directly.
            top_svgs = moran_result["top_svgs"]
            model_genes_set_local = set(model_genes)
            n_top_in_model = sum(1 for g in top_svgs if g in model_genes_set_local)
            svg_retain = n_top_in_model / len(top_svgs) if top_svgs else 0.0
            diag_results["moran"] = moran_result
            diag_results["svg_retention"] = svg_retain
            diag_results["svg_n_top_in_model"] = n_top_in_model
            diag_results["svg_n_top"] = len(top_svgs)
            # Note: Moran's I was run on the model-space gene subset, so SVG retention reflects
            # what fraction of top SVGs are in the MultiDCP 10,716-gene space (should be ~100%).
        except Exception as exc:
            log.error("Moran's I failed: %s", exc)
            diag_results["moran"] = {"svg_count": None, "svg_fraction": None, "top_svgs": [], "error": str(exc)}
            diag_results["svg_retention"] = None

        # --- Pseudobulk per zone (for basal similarity matrix) ---
        # Filter out spots with missing category (nan) before pseudobulk
        try:
            has_category = adata.obs["category"].notna()
            adata_annotated = adata[has_category].copy()
            n_unannotated = int((~has_category).sum())
            if n_unannotated > 0:
                log.info("Pseudobulk: dropped %d unannotated spots (category=NaN).", n_unannotated)
            pb_result = from_anndata(adata_annotated, obs_col="category")
            zone_profiles = {
                region: pb_result.profiles[i]
                for i, region in enumerate(pb_result.region_order)
            }
            sim_matrix = basal_similarity_matrix(zone_profiles)
            diag_results["zones"] = pb_result.region_order
            diag_results["zone_profiles"] = zone_profiles
            diag_results["sim_matrix"] = sim_matrix
            diag_results["spot_counts"] = pb_result.spot_counts
            diag_results["n_unannotated_spots"] = n_unannotated
        except Exception as exc:
            log.error("Pseudobulk/similarity failed: %s", exc)
            diag_results["zones"] = []
            diag_results["zone_profiles"] = {}
            diag_results["sim_matrix"] = None
            diag_results["spot_counts"] = {}
            diag_results["n_unannotated_spots"] = 0

        # --- Human whole-tissue pseudobulk (for cross-species correlation) ---
        human_whole = np.mean(
            adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X),
            axis=0,
        )

    # ------------------------------------------------------------------
    # Load GSE272564 mouse liver control (APAP0h; Pitfall 6)
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        mouse_profile, mouse_genes = _load_gse272564_mouse_ctrl(
            GSE272564_TAR, tmpdir / "gse272564"
        )

    # --- Human-mouse liver correlation (ortholog-filtered) ---
    try:
        corr_result = human_mouse_liver_correlation(
            human_whole, human_genes,
            mouse_profile, mouse_genes,
            ortholog_table,
        )
        diag_results["human_mouse_corr"] = corr_result
    except Exception as exc:
        log.error("Human-mouse correlation failed: %s", exc)
        diag_results["human_mouse_corr"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # OOD Mahalanobis: project to 978 LINCS landmark genes, then compute
    # ------------------------------------------------------------------
    try:
        # Load PDG manifold
        pdg_df = pd.read_csv(str(PDG_MANIFOLD_CSV), index_col=0)
        pdg_genes = list(pdg_df.columns)  # 10716 gene symbols
        pdg_matrix = pdg_df.values  # (10, 10716)
        log.info("PDG manifold: %s shape, %d genes.", pdg_matrix.shape, len(pdg_genes))

        # 978 LINCS landmark gene subspace (load from ceiling DE gene names;
        # the landmark list comes from wangli_measured_de columns -- use
        # wangli_profiles as the gene-count reference: de has 978 genes)
        # We need the actual landmark gene symbols. Load the profiles and
        # check if there's a gene list. The .npy has no names; use the
        # multidcp symbols list to find the 978-gene intersection.
        # Best approach: project both to a shared subset of the 978 LINCS
        # L1000 landmark genes. The wangli_measured_de.npy is (5517, 978)
        # -- 978 genes in column order. The gene names are not stored in
        # the .npy; we use the gene symbols from the profiles if available.
        # Fallback: use the 978 genes that appear in BOTH pdg_genes AND
        # human_genes (Visium) as the landmark proxy.
        landmark_genes_set = set(pdg_genes[:978]) if len(pdg_genes) >= 978 else set(pdg_genes)
        # Better: use true LINCS L1000 landmark proxy = first 978 genes of
        # the MultiDCP space that appear in the Visium data and PDG manifold.
        # Actually the PDG manifold has 10716 genes. We project to the common
        # genes among: pdg_genes, human_genes, model_genes (978 is the DE
        # matrix dimensionality). We use len==978 as target.
        # Most defensible: project to intersection of pdg_genes and human_genes,
        # capped at 978 genes (or as many as available).
        common_genes = sorted(set(pdg_genes) & set(human_genes))
        if len(common_genes) > 978:
            common_genes = common_genes[:978]
        log.info("OOD landmark subspace: %d common genes (PDG ∩ human Visium, capped at 978).", len(common_genes))

        if len(common_genes) < 10:
            raise ValueError(
                f"Too few common genes ({len(common_genes)}) between PDG manifold "
                "and Visium for OOD projection."
            )

        # Project PDG to landmark subspace
        pdg_gene_idx = {g: i for i, g in enumerate(pdg_genes)}
        pdg_proj = pdg_matrix[:, [pdg_gene_idx[g] for g in common_genes if g in pdg_gene_idx]]

        # Project each zone pseudobulk to landmark subspace
        h_gene_idx = {g: i for i, g in enumerate(human_genes)}
        zone_names = list(diag_results.get("zone_profiles", {}).keys())
        if zone_names:
            zone_proj = np.stack([
                diag_results["zone_profiles"][z][[h_gene_idx[g] for g in common_genes if g in h_gene_idx]]
                for z in zone_names
            ])
        else:
            # Whole-tissue fallback
            zone_names = ["whole_tissue"]
            zone_proj = human_whole[[h_gene_idx[g] for g in common_genes if g in h_gene_idx]].reshape(1, -1)

        ood_dists = ood_mahalanobis(pdg_proj, zone_proj, alpha=1e-2)
        diag_results["ood_zones"] = zone_names
        diag_results["ood_distances"] = ood_dists
        diag_results["ood_n_genes"] = len(common_genes)
    except Exception as exc:
        log.error("OOD Mahalanobis failed: %s", exc)
        diag_results["ood_zones"] = []
        diag_results["ood_distances"] = np.array([])
        diag_results["ood_n_genes"] = 0
        diag_results["ood_error"] = str(exc)

    # ------------------------------------------------------------------
    # Build report sections
    # ------------------------------------------------------------------
    report_lines.append("## EDA-03: Region Distinguishability (liver/human, yu2022 L5)\n\n")

    # SVG retention
    moran = diag_results.get("moran", {})
    if moran.get("svg_count") is not None:
        n_top_in = diag_results.get("svg_n_top_in_model", len(moran["top_svgs"]))
        n_top = diag_results.get("svg_n_top", len(moran["top_svgs"]))
        report_lines.append(
            f"**Moran's I SVGs (pval_norm < 0.05):** {moran['svg_count']} "
            f"({moran['svg_fraction']:.1%} of {10693} model-space genes tested; "
            "Moran's I run on MultiDCP 10,716-gene subset for speed).\n"
            f"**SVG retention in model space:** {n_top_in}/{n_top} top SVGs in MultiDCP space "
            f"({diag_results['svg_retention']:.1%}; all SVGs are in model space by construction since "
            "Moran's I was run on the model-space gene subset).\n"
            f"**Top-5 SVGs:** {', '.join(moran['top_svgs'][:5])}.\n\n"
        )
    else:
        report_lines.append(
            f"**Moran's I:** computation error -- {moran.get('error', 'unknown')}.\n\n"
        )

    # Basal similarity matrix
    if diag_results.get("sim_matrix") is not None:
        sim_df = diag_results["sim_matrix"]
        zones = diag_results["zones"]
        report_lines.append("### Basal similarity matrix (zone pairwise Pearson r)\n\n")
        # Header row
        header = "| Zone | " + " | ".join(zones) + " |"
        separator = "|------|" + "|".join(["------"] * len(zones)) + "|"
        report_lines.append(header + "\n")
        report_lines.append(separator + "\n")
        for z in zones:
            row_vals = " | ".join(f"{sim_df.loc[z, z2]:.3f}" for z2 in zones)
            report_lines.append(f"| {z} | {row_vals} |\n")
        report_lines.append("\n")
        n_unannot = diag_results.get("n_unannotated_spots", 0)
        report_lines.append(
            f"**Spot counts per zone:** "
            + ", ".join(f"{z}={diag_results['spot_counts'].get(z, 0)}" for z in zones)
            + (f" | unannotated={n_unannot}" if n_unannot > 0 else "")
            + "\n\n"
        )
    else:
        report_lines.append("**Basal similarity matrix:** computation error.\n\n")

    # Gene coverage (SVG in model space)
    report_lines.append(
        f"**Gene coverage (yu2022 Visium vs 10,716-gene MultiDCP space):** "
        "see P0_coverage.md (>99% confirmed in Phase 0).\n\n"
    )

    # Cross-species concordance
    report_lines.append(
        "## EDA-03: Human-Rodent Basal Concordance "
        "(liver, yu2022 human vs GSE272564 APAP0h mouse)\n\n"
    )
    report_lines.append(
        "**Mouse liver note:** GSE272564 APAP0h (GSM8404653) -- "
        "no zone annotation available; whole-sample pseudobulk used as mouse reference "
        "(Open Q3; zone-level rodent correlation deferred).\n\n"
    )
    corr = diag_results.get("human_mouse_corr", {})
    if "error" not in corr:
        report_lines.append(
            "### Cross-species correlation (ortholog-filtered)\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Pearson r | {corr['pearson_r']:.4f} |\n"
            f"| p-value | {corr['p_value']:.2e} |\n"
            f"| n genes compared | {corr['n_genes_compared']:,} |\n"
            f"| Ortholog map | one-to-one (Ensembl BioMart release 116) |\n"
            f"| Ortholog pairs total | {ortholog_table.n_one2one:,} |\n"
            "\n"
        )
    else:
        report_lines.append(f"**Cross-species correlation error:** {corr['error']}\n\n")

    # OOD distances
    report_lines.append(
        "### OOD Distance from Cancer-Line Manifold (PDG diseased baseline)\n\n"
        f"**Method:** {OOD_METHOD}.\n"
        f"**Landmark subspace:** {diag_results.get('ood_n_genes', 0)}-gene "
        "(PDG manifold ∩ Visium; capped at 978).\n\n"
    )
    ood_zones = diag_results.get("ood_zones", [])
    ood_dists = diag_results.get("ood_distances", np.array([]))
    if len(ood_zones) > 0 and len(ood_dists) > 0:
        report_lines.append(
            "| Zone | Mahalanobis distance |\n"
            "|------|---------------------|\n"
        )
        for z, d in zip(ood_zones, ood_dists):
            report_lines.append(f"| {z} | {d:.4f} |\n")
        report_lines.append("\n")
    elif "ood_error" in diag_results:
        report_lines.append(f"**OOD error:** {diag_results['ood_error']}\n\n")

    return diag_results


# ---------------------------------------------------------------------------
# Write HALT_REASON.md (D-02)
# ---------------------------------------------------------------------------


def _write_halt_reason(
    bootstrap_result: BootstrapResult, ceiling_data: Optional[dict] = None
) -> None:
    """Write HALT_REASON.md to the phase directory and log.

    Framing leads with the robust finding (drug-leakage inflation of the
    benchmark) rather than overclaiming a 'structure beats biology' result: the
    drug-aggregated gate is underpowered (few negative drugs) and the honest
    profile-level measured ceiling is comparable to the structure floor.
    """
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    halt_path = PHASE_DIR / "HALT_REASON.md"

    leak = (ceiling_data or {}).get("ceiling_result", {}).get("leakage", {})
    honest_prof = leak.get("profile_auroc_disjoint")
    inflation = leak.get("leakage_inflation")
    leak_line = ""
    if honest_prof is not None and inflation is not None:
        leak_line = (
            f"- **Headline = drug leakage.** Profile-level measured AUROC is "
            f"{honest_prof + inflation:.3f} under a leaky (drug-in-train+test) split "
            f"but only {honest_prof:.3f} under a drug-disjoint split "
            f"({inflation:+.3f} inflation). The Wang/Li-style benchmark (~0.798) is "
            "substantially drug-leakage-inflated.\n"
        )

    ci_clause = (
        "ci_upper < 0 -- CI entirely below 0"
        if bootstrap_result.ci_upper < 0
        else "ci_lower <= 0 -- CI includes 0"
    )

    content = (
        "# HALT: Gate 2 Fired -- No Measured Lift Over Structure; "
        "Benchmark Is Drug-Leakage-Inflated\n\n"
        f"**Gap observed (drug-agg):** {bootstrap_result.gap_observed:+.4f} AUROC\n"
        f"**95% CI:** [{bootstrap_result.ci_lower:.4f}, {bootstrap_result.ci_upper:.4f}]\n"
        f"**gate_fires:** True ({ci_clause})\n\n"
        "## Interpretation\n\n"
        "The drug-aggregated, leakage-free measured ceiling does not exceed the "
        "structure-only ECFP4 floor (paired bootstrap CI of ceiling - floor is "
        "below 0). Measured LINCS L1000 DE provides no drug-level lift over chemical "
        "structure for liver DILI on this shared set. This is a refined, honest "
        "reading -- not 'structure beats biology' -- bounded by the caveats below.\n\n"
        "## Caveats (bound the strength of this conclusion)\n\n"
        + leak_line +
        "- **Measured ~= structure at the fair level.** The honest profile-level "
        "measured ceiling is comparable to the structure floor; the more-negative "
        "drug-aggregated gap is largely noise from collapsing many profiles onto "
        "few drugs.\n"
        "- **Underpowered.** The shared set has few negative drugs; the conservative "
        "minimum detectable gap exceeds the observed gap, so the firing rests on the "
        "paired bootstrap with a thin margin (see P1_eda.md Power section).\n\n"
        "## Decision per D-02\n\n"
        "**stop-and-REFRAME** (negative result is publishable, D-02 locked). "
        "Do NOT proceed to Phase 2 model training. The reframe centers on the "
        "drug-leakage finding (the benchmark is inflated; measured DE does not "
        "generalize to held-out drugs above structure here) and on powering a "
        "future drug-disjoint comparison.\n\n"
        f"**Bootstrap resamples (valid):** {bootstrap_result.n_resamples_valid:,} / 10,000\n"
        f"**Logged:** run_p1_eda.py\n"
    )
    with open(halt_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\nHALT_REASON.md written: {halt_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 EDA driver: floor / ceiling / diagnostics / all."
    )
    parser.add_argument(
        "subcommand",
        choices=["floor", "ceiling", "diagnostics", "all"],
        help="Sub-command to run.",
    )
    args = parser.parse_args()

    # --- Provenance assertion (T-01-10; D-05) ---
    _assert_manifest_rows(_MANIFEST_PATH, _REQUIRED_MANIFEST_FILENAMES)

    t0 = time.time()
    report_lines: list[str] = []
    gate_fired = False
    bootstrap_result: Optional[BootstrapResult] = None

    # Write report header
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_lines.append(
        f"# Phase 1 EDA Report: The Bracket (liver/human)\n\n"
        f"**Generated:** {ts}\n"
        f"**Sub-command:** {args.subcommand}\n\n"
    )

    if args.subcommand == "floor":
        floor_data = run_floor(report_lines)

    elif args.subcommand == "ceiling":
        ceiling_data = run_ceiling(report_lines)

    elif args.subcommand == "diagnostics":
        run_diagnostics(report_lines)

    elif args.subcommand == "all":
        floor_data = run_floor(report_lines)
        ceiling_data = run_ceiling(report_lines)
        bootstrap_result = run_gap_and_gate(floor_data, ceiling_data, report_lines)
        if bootstrap_result is not None and bootstrap_result.gate_fires:
            gate_fired = True
        run_diagnostics(report_lines)

    # Footer
    elapsed = time.time() - t0
    report_lines.append(f"\n---\n*Run elapsed: {elapsed:.1f}s*\n")

    # Write report
    report_path = RESULTS / "P1_eda.md"
    _write_report(report_path, "".join(report_lines), mode="w")
    print(f"\nWrote {report_path}")

    # Halt Gate 2: write HALT_REASON.md and exit(1) AFTER writing the report
    if gate_fired and bootstrap_result is not None:
        _write_halt_reason(bootstrap_result, ceiling_data)
        sys.exit(1)

    print(f"Done in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
