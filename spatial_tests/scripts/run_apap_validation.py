"""APAP validity-anchor driver (WIRE-03 / Halt Gate 3).

Computes the predicted-vs-measured per-zone Pearson on the mouse APAP injury
anchor and enforces Halt Gate 3 (pericentral Pearson < 0.3 -> write
HALT_REASON.md, stop-and-REFRAME). Mirrors the run_p1_eda.py driver pattern
(sys.path inject, MANIFEST provenance assert, _write_report, gate -> sys.exit).

Predicted side  : the WIRE-01 rule-B MOUSE cache (D-02), acetaminophen drug,
                  per zone (pericentral / periportal), over the 10,716 PDG genes.
Measured side   : GSE272564 matched arms (D-06 primary). The series is an APAP
                  time course in ONE RAW.tar (Pitfall 6): APAP0h is the matched
                  pre-injury baseline (control arm); APAP24h is the classical
                  pericentral-necrosis injury timepoint (APAP arm). Measured DE
                  per zone = pseudobulk(APAP24h_zone) - pseudobulk(APAP0h_zone),
                  in mouse symbols, mapped mouse->human and reindexed to the
                  10,716 order with absent genes FLAGGED (present_mask, D-08).
GSE280652       : weaker partial replication (NO matched control arm, D-06
                  amendment) -- reported as a note, not an independent matched
                  anchor.

Gate            : keyed to the PERICENTRAL zone (D-08). A fired gate is the
                  designed, reportable stop-and-REFRAME outcome (D-09), NOT a
                  failure: 02-02 found the frozen backbone is near-zonal-invariant
                  so the predicted pericentral DE is expected to barely correlate
                  with the measured pericentral injury pattern. The rodent-anchor
                  asymmetry (gate is mouse, headline organ is human) is stated
                  honestly in the report (D-09).

Hard rules honored
------------------
    - Real data only: GSE272564 real Visium + the real mouse predicted cache.
      No fabricated measured data, no synthetic zonal variation to force a pass.
    - DE rule: measured DE is a differential (APAP24h - APAP0h); raw expression
      is never the comparison quantity.
    - present_mask flag-not-zero (D-08): absent ortholog/coverage genes excluded
      from the Pearson, never zero-filled.
    - Provenance-before-use: GSE272564 / GSE280652 RAW.tar + the gene-order file
      + the ortholog TSV are asserted as MANIFEST rows before any read.
    - scanpy/numpy only on this path -- no torch, no CUDA-hygiene block needed
      (the predicted side is read from cache, not re-run).
    - Pitfall 6: the APAP arm is never mixed into the control reference; arms are
      split by GEO sample (GSM) before pseudobulking.
"""

from __future__ import annotations

import gzip
import io
import logging
import pathlib
import sys
import tarfile
import tempfile
import time
from typing import Optional

# ---------------------------------------------------------------------------
# sys.path injection (analog: run_p1_eda.py lines 43-44)
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Paths (analog: run_p1_eda.py lines 50-77)
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
PHASE_DIR = ROOT / ".planning" / "phases" / "02-multidcp-wiring-tox-head"

# APAP anchors (validation-only; NEVER basal input).
GSE272564_TAR = (
    ROOT / "data" / "raw" / "spatial" / "gse272564_apap_liver" / "GSE272564_RAW.tar"
)
GSE280652_TAR = (
    ROOT / "data" / "raw" / "spatial" / "gse280652_apap_liver" / "GSE280652_RAW.tar"
)

# Gene order (the 10,716 human PDG symbols -- the cache's column order).
GENE_ORDER_PATH = ROOT / "data" / "processed" / "spatial" / "multidcp_10716_symbols.txt"

# Ortholog map (P0 output: human_symbol / mouse_symbol pairs).
ORTHOLOG_TSV_PATH = ROOT / "data" / "processed" / "spatial" / "orthologs_h_m_r_one2one.tsv"

# Predicted-DE source: the WIRE-01 rule-B MOUSE cache (D-02).
MOUSE_CACHE_DIR = ROOT / "data" / "processed" / "spatial" / "region_de_cache" / "mouse"

# The APAP drug as named in the cache's pert_id list.
APAP_PERT_ID = "acetaminophen"

# GSE272564 time-course samples: APAP0h is the matched pre-injury baseline
# (control arm); APAP24h is the classical pericentral-necrosis injury timepoint
# (APAP arm). Split by these GSM tokens (Pitfall 6).
CTRL_SAMPLE_TOKEN = "APAP0h"
APAP_SAMPLE_TOKEN = "APAP24h"

# ---------------------------------------------------------------------------
# MANIFEST provenance row check (analog: run_p1_eda.py lines 82-118)
# ---------------------------------------------------------------------------

_MANIFEST_PATH = ROOT / "MANIFEST.md"

_REQUIRED_MANIFEST_FILENAMES = [
    "GSE272564_RAW.tar",
    "GSE280652_RAW.tar",
    "multidcp_10716_symbols.txt",
    "orthologs_h_m_r_one2one.tsv",
]


def _assert_manifest_rows(manifest_path: pathlib.Path, required: list[str]) -> None:
    """Assert each required filename appears as a MANIFEST.md row (sys.exit(2) if not)."""
    if not manifest_path.exists():
        print(
            f"ERROR: MANIFEST.md not found at {manifest_path}. Cannot assert provenance.",
            file=sys.stderr,
        )
        sys.exit(2)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    missing = [f for f in required if f not in manifest_text]
    if missing:
        print(
            f"ERROR: external inputs absent from MANIFEST.md: {missing}. Add "
            "provenance rows (SHA256 + source) before reading (provenance-before-use).",
            file=sys.stderr,
        )
        sys.exit(2)


# ---------------------------------------------------------------------------
# Logging (analog: run_p1_eda.py lines 125-130)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_apap_validation")

# ---------------------------------------------------------------------------
# Library imports (after sys.path set)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from src.spatial.apap_validation import (  # noqa: E402
    assign_zones,
    halt_gate_3_fires,
    measured_zone_de,
    zone_pearson,
)
from src.spatial.orthology import OrthologTable, build_one2one_orthologs  # noqa: E402
from src.spatial.pseudobulk import pseudobulk  # noqa: E402


# ---------------------------------------------------------------------------
# Report writer (analog: run_p1_eda.py lines 164-168)
# ---------------------------------------------------------------------------


def _write_report(path: pathlib.Path, content: str, mode: str = "w") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# GSE272564 arm loader (Pitfall 6: split by GSM sample, never mix arms)
# ---------------------------------------------------------------------------


def _load_gse272564_arm(tar_path: pathlib.Path, tmpdir: pathlib.Path, sample_token: str):
    """Load one GSE272564 arm (matched by ``sample_token``) as (X, spot_ids, genes).

    The series ships per-sample 10x triplets (matrix.mtx.gz / features.tsv.gz /
    barcodes.tsv.gz) inside one RAW.tar. We extract and read ONLY the files whose
    name carries ``sample_token`` -- so the APAP arm is never mixed into the
    control arm (Pitfall 6 / T-02-10).

    Returns
    -------
    (X, spot_ids, mouse_genes)
        X : np.ndarray (n_spots, n_genes) dense float; mouse_genes : list[str].
    """
    import scipy.io
    import scipy.sparse as sp

    if not tar_path.exists():
        raise FileNotFoundError(f"GSE272564_RAW.tar not found at {tar_path}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(tmpdir)

    all_files = list(tmpdir.rglob("*"))
    arm_files = [f for f in all_files if sample_token in f.name]
    log.info("GSE272564 arm %s: %d files matched.", sample_token, len(arm_files))

    matrix_file = next(
        (f for f in arm_files if "matrix" in f.name.lower() and f.name.endswith(".mtx.gz")),
        None,
    )
    features_file = next(
        (f for f in arm_files if ("feature" in f.name.lower() or "gene" in f.name.lower())
         and f.name.endswith(".tsv.gz")),
        None,
    )
    barcodes_file = next(
        (f for f in arm_files if "barcode" in f.name.lower() and f.name.endswith(".tsv.gz")),
        None,
    )

    if matrix_file is None or features_file is None:
        raise FileNotFoundError(
            f"Could not find matrix/features for arm {sample_token} in {tar_path}. "
            f"Arm files: {[f.name for f in arm_files]}"
        )

    # Read the matrix (.mtx is genes x spots in 10x convention -> transpose).
    with gzip.open(str(matrix_file), "rb") as gf:
        mat = scipy.io.mmread(io.BytesIO(gf.read()))
    X = mat.T.toarray() if hasattr(mat, "toarray") else np.asarray(mat).T
    X = np.asarray(X, dtype=np.float64)

    # Read features (col 1 = gene symbol in the 10x features.tsv).
    with gzip.open(str(features_file), "rb") as gf:
        lines = gf.read().decode("utf-8").strip().splitlines()
    mouse_genes = [ln.split("\t")[1] if "\t" in ln else ln for ln in lines]

    # Spot barcodes (for spot-order labels).
    if barcodes_file is not None:
        with gzip.open(str(barcodes_file), "rb") as gf:
            spot_ids = gf.read().decode("utf-8").strip().splitlines()
    else:
        spot_ids = [f"spot_{i}" for i in range(X.shape[0])]

    log.info(
        "GSE272564 arm %s: %d spots x %d genes.", sample_token, X.shape[0], X.shape[1]
    )
    return X, spot_ids, mouse_genes


class _MatAdata:
    """Minimal AnnData-like wrapper so assign_zones can score markers on a matrix."""

    def __init__(self, X, gene_names):
        self.X = X
        self.var_names = list(gene_names)
        self.n_obs = X.shape[0]
        self.n_vars = X.shape[1]


def _zone_pseudobulk(X, mouse_genes, zone_labels):
    """Pseudobulk a spots-by-genes matrix per assigned zone (drop 'unassigned')."""
    res = pseudobulk(X, zone_labels, mouse_genes, agg="mean")
    out = {}
    for i, zone in enumerate(res.region_order):
        if zone == "unassigned":
            continue
        out[zone] = res.profiles[i]
    return out, res.spot_counts


# ---------------------------------------------------------------------------
# HALT_REASON.md writer (mirror run_p1_eda.py _write_halt_reason + P1 format)
# ---------------------------------------------------------------------------


def _write_halt_reason(
    per_zone: dict, key_zone: str, threshold: float, n_genes: dict, ts: str
) -> None:
    """Write HALT_REASON.md (stop-and-REFRAME, D-09) and log. Mirrors the P1 format."""
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    halt_path = PHASE_DIR / "HALT_REASON.md"

    peri_r = per_zone[key_zone]["pearson_r"]

    table_rows = "\n".join(
        f"| {z} | {per_zone[z]['pearson_r']:+.4f} | {per_zone[z]['p_value']:.2e} | "
        f"{per_zone[z]['n_genes_compared']:,} |"
        for z in per_zone
    )

    content = (
        "# HALT: Gate 3 Fired -- Predicted Region-Resolved DE Does Not Recover the "
        "Pericentral APAP Injury Pattern\n\n"
        f"**Date:** {ts}\n"
        f"**Trigger:** pericentral predicted-vs-measured Pearson "
        f"{peri_r:+.4f} < {threshold} (Halt Gate 3, keyed to the pericentral zone, D-08).\n"
        f"**gate_fires:** True\n\n"
        "## Per-zone predicted-vs-measured Pearson\n\n"
        "| Zone | Pearson r | p-value | n genes compared |\n"
        "|------|-----------|---------|------------------|\n"
        f"{table_rows}\n\n"
        "## Interpretation\n\n"
        "The predicted, region-resolved DE for acetaminophen (WIRE-01 rule-B MOUSE "
        "cache) does not correlate with the measured pericentral APAP injury DE "
        "(GSE272564 APAP24h - APAP0h, matched arms) above the 0.3 gate. The likely "
        "mechanism (flagged in 02-02-SUMMARY): the frozen MultiDCP-CheMoE backbone is "
        "near-zonal-invariant on healthy-liver basals -- its per-zone predicted vectors "
        "differ by ~float32 epsilon, so the predicted side carries essentially no "
        "pericentral-vs-periportal contrast to match the measured injury gradient. The "
        "prediction is dominated by drug + dose, not by the regional basal context. "
        "This is the project's instrumented OOD hypothesis surfacing as the gate result, "
        "not a wiring bug.\n\n"
        "## Caveats (bound the conclusion)\n\n"
        "- **Rodent anchor for a human headline (D-09).** The only drug-perturbed "
        "spatial anchor that exists as of mid-2026 is rodent (mouse APAP); the headline "
        "organ is human. The gate is rodent; per-region predicted DE remains the "
        "primary, indirect validity path for the human claim.\n"
        "- **Pattern, not magnitude (D-05).** A fixed canonical dose was used for the "
        "predicted signature; the Pearson tests the DE pattern across genes, not "
        "magnitude. The null is about pattern recovery, not dose calibration.\n"
        "- **Partial replication only (GSE280652, D-06 amendment).** GSE280652 has no "
        "matched control arm, so it is a weaker partial replication, not a second "
        "matched-DE anchor.\n\n"
        "## Decision per D-09\n\n"
        "**stop-and-REFRAME the spatial claim** (NOT abandon). Consistent with P1's "
        "negative-is-publishable precedent (D-02) and DEC-negative-result-acceptable: "
        "the predicted region-resolved signature does not recover the pericentral APAP "
        "pattern under the frozen backbone, which is itself the reportable finding. The "
        "reframe centers on the near-zonal-invariance of the frozen model on healthy "
        "basals (the cell-context encoder contributes little for these inputs) and on "
        "what a region-sensitive signature would require.\n\n"
        "**Logged:** run_apap_validation.py\n"
    )
    with open(halt_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\nHALT_REASON.md written: {halt_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from datetime import datetime, timezone

    # --- Provenance assertion ---
    _assert_manifest_rows(_MANIFEST_PATH, _REQUIRED_MANIFEST_FILENAMES)

    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Load the target human gene order (10,716 PDG symbols) ---
    if not GENE_ORDER_PATH.exists():
        print(f"ERROR: gene-order file not found at {GENE_ORDER_PATH}.", file=sys.stderr)
        sys.exit(2)
    with open(GENE_ORDER_PATH) as fh:
        human_genes = [ln.strip() for ln in fh if ln.strip()]
    log.info("Target human gene space: %d symbols.", len(human_genes))

    # --- Load the ortholog table (human_symbol / mouse_symbol pairs) ---
    if not ORTHOLOG_TSV_PATH.exists():
        print(f"ERROR: ortholog TSV not found at {ORTHOLOG_TSV_PATH}.", file=sys.stderr)
        sys.exit(2)
    import pandas as pd

    ot_df = pd.read_csv(str(ORTHOLOG_TSV_PATH), sep="\t")
    if "human_symbol" in ot_df.columns and "mouse_symbol" in ot_df.columns:
        ortholog_table = OrthologTable(
            pairs=ot_df, n_input=len(ot_df), n_one2one=len(ot_df), dropped_fraction=0.0
        )
    else:
        ortholog_table = build_one2one_orthologs(str(ORTHOLOG_TSV_PATH))
    log.info("Ortholog table: %d one-to-one pairs.", ortholog_table.n_one2one)

    # --- Load the predicted DE from the WIRE-01 MOUSE cache (rule B, D-02) ---
    import json

    cache_manifest = json.load(open(MOUSE_CACHE_DIR / "manifest.json"))
    cache_pert_ids = cache_manifest["pert_ids"]
    cache_regions = cache_manifest["regions"]
    if APAP_PERT_ID not in cache_pert_ids:
        print(
            f"ERROR: '{APAP_PERT_ID}' not in mouse cache pert_ids "
            f"({len(cache_pert_ids)} drugs). Re-cache with APAP before the gate.",
            file=sys.stderr,
        )
        sys.exit(2)
    de_array = np.load(MOUSE_CACHE_DIR / "de_array.npy")  # (n_pert, n_regions, 10716)
    apap_idx = cache_pert_ids.index(APAP_PERT_ID)
    # zone -> predicted DE vector
    pred_de_by_zone = {
        cache_regions[j]: de_array[apap_idx, j, :].astype(np.float64)
        for j in range(len(cache_regions))
    }
    log.info(
        "Predicted DE (cache): %s, zones %s, %d genes each.",
        APAP_PERT_ID, list(pred_de_by_zone), de_array.shape[-1],
    )

    # --- Load + zone the GSE272564 matched arms (Pitfall 6: split, never mix) ---
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        X_ctrl, _, mouse_genes_ctrl = _load_gse272564_arm(
            GSE272564_TAR, tmp / "ctrl", CTRL_SAMPLE_TOKEN
        )
        X_apap, _, mouse_genes_apap = _load_gse272564_arm(
            GSE272564_TAR, tmp / "apap", APAP_SAMPLE_TOKEN
        )

    # The two arms share the same features.tsv gene order in this series; assert.
    if mouse_genes_ctrl != mouse_genes_apap:
        log.warning(
            "GSE272564 control/APAP arm gene orders differ; aligning APAP to control order."
        )
        m_idx_apap = {g: i for i, g in enumerate(mouse_genes_apap)}
        keep = [g for g in mouse_genes_ctrl if g in m_idx_apap]
        X_apap = X_apap[:, [m_idx_apap[g] for g in keep]]
        # reindex control to the same kept order
        m_idx_ctrl = {g: i for i, g in enumerate(mouse_genes_ctrl)}
        X_ctrl = X_ctrl[:, [m_idx_ctrl[g] for g in keep]]
        mouse_genes = keep
    else:
        mouse_genes = mouse_genes_ctrl

    # Zone each arm by canonical markers (no published per-spot annotation in
    # this series -> markers per D-07).
    ctrl_labels = assign_zones(_MatAdata(X_ctrl, mouse_genes))
    apap_labels = assign_zones(_MatAdata(X_apap, mouse_genes))

    ctrl_pb, ctrl_counts = _zone_pseudobulk(X_ctrl, mouse_genes, ctrl_labels)
    apap_pb, apap_counts = _zone_pseudobulk(X_apap, mouse_genes, apap_labels)

    log.info("Control arm spot counts per zone: %s", ctrl_counts)
    log.info("APAP arm spot counts per zone: %s", apap_counts)

    # --- Per-zone measured DE + Pearson against the predicted side ---
    zones_to_eval = [z for z in pred_de_by_zone if z in ctrl_pb and z in apap_pb]
    per_zone: dict = {}
    for zone in zones_to_eval:
        md = measured_zone_de(
            apap_zone=apap_pb[zone],
            ctrl_zone=ctrl_pb[zone],
            mouse_genes=mouse_genes,
            human_genes=human_genes,
            ortholog_table=ortholog_table,
        )
        try:
            pr = zone_pearson(pred_de_by_zone[zone], md["de"], md["present_mask"])
        except ValueError as exc:
            log.error("zone_pearson failed for %s: %s", zone, exc)
            pr = {"pearson_r": float("nan"), "p_value": float("nan"),
                  "n_genes_compared": int(md["n_genes_compared"])}
        per_zone[zone] = pr
        log.info(
            "Zone %s: Pearson r=%.4f (p=%.2e, n=%d).",
            zone, pr["pearson_r"], pr["p_value"], pr["n_genes_compared"],
        )

    # --- Halt Gate 3 (pericentral-keyed, <0.3) ---
    per_zone_r = {z: per_zone[z]["pearson_r"] for z in per_zone}
    gate_fires = halt_gate_3_fires(per_zone_r, threshold=0.3, key_zone="pericentral")

    # --- Build the report (all zones, intersection sizes, caveats) ---
    lines: list[str] = []
    lines.append("# P2 APAP Validity Anchor: Predicted-vs-Measured Per-Zone DE Pearson "
                 "(WIRE-03 / Halt Gate 3)\n\n")
    lines.append(f"**Generated:** {ts}\n")
    lines.append(f"**Predicted side:** WIRE-01 rule-B MOUSE cache (D-02), drug "
                 f"`{APAP_PERT_ID}`, {de_array.shape[-1]} PDG genes per zone.\n")
    lines.append("**Measured side (primary, D-06):** GSE272564 matched arms -- measured "
                 "DE per zone = pseudobulk(APAP24h_zone) - pseudobulk(APAP0h_zone), in "
                 "mouse symbols, mapped mouse->human (one-to-one orthologs) and reindexed "
                 "to the PDG order; absent genes FLAGGED (present_mask), never zero-filled "
                 "(D-08).\n\n")

    lines.append("## Per-zone predicted-vs-measured Pearson\n\n")
    lines.append("| Zone | Pearson r | p-value | n genes compared (intersection) |\n")
    lines.append("|------|-----------|---------|---------------------------------|\n")
    for zone in per_zone:
        pr = per_zone[zone]
        lines.append(
            f"| {zone} | {pr['pearson_r']:+.4f} | {pr['p_value']:.2e} | "
            f"{pr['n_genes_compared']:,} |\n"
        )
    lines.append("\n")

    lines.append("### Spot counts per zone (arms split by GSM, Pitfall 6)\n\n")
    lines.append("| Zone | control (APAP0h) spots | APAP (APAP24h) spots |\n")
    lines.append("|------|------------------------|----------------------|\n")
    all_zones = sorted(set(list(ctrl_counts) + list(apap_counts)))
    for z in all_zones:
        lines.append(f"| {z} | {ctrl_counts.get(z, 0)} | {apap_counts.get(z, 0)} |\n")
    lines.append("\n")

    peri_r = per_zone_r.get("pericentral", float("nan"))
    verdict = (
        "**FIRES** (pericentral r < 0.3 -> stop-and-REFRAME, D-09)"
        if gate_fires
        else "**CLEAR** (pericentral r >= 0.3)"
    )
    lines.append("## Halt Gate 3 verdict\n\n")
    lines.append(
        f"Gate is keyed to the **pericentral** zone (where APAP injury classically "
        f"acts, D-08). Pericentral Pearson = {peri_r:+.4f}; threshold 0.3. "
        f"Halt Gate 3: {verdict}.\n\n"
    )

    lines.append("## Caveats\n\n")
    lines.append(
        "- **Rodent anchor for a human headline (D-09).** The gate is computed on "
        "mouse APAP Visium -- the only drug-perturbed spatial anchor that exists as of "
        "mid-2026 -- while the headline organ is human. This asymmetry is accepted "
        "(D-09): per-region predicted DE remains the primary, indirect validity path "
        "for the human claim; the rodent APAP gate is the direct stand-in.\n"
        "- **GSE280652 is a partial replication only (D-06 amendment).** GSE280652 has "
        "NO matched control arm, so it cannot supply a matched treated-minus-control "
        "DE; it is a weaker partial replication, not an independent matched anchor. The "
        "primary, matched anchor is GSE272564 (APAP0h baseline vs APAP24h injury).\n"
        "- **Pattern, not magnitude (D-05).** The predicted signature uses a fixed "
        "canonical dose; the Pearson tests the cross-gene DE pattern, not magnitude.\n"
        "- **Near-zonal-invariance of the frozen backbone (02-02).** The cached per-zone "
        "predicted DE vectors are near-identical across pericentral vs periportal "
        "(~float32 epsilon): the frozen CheMoE cell-context encoder contributes little "
        "for healthy-liver basals, so the predicted side carries little regional "
        "contrast. If the pericentral Pearson is ~0, this is the mechanism -- an honest "
        "negative, reported plainly, not a bug to chase.\n\n"
    )

    lines.append(f"\n---\n*Run elapsed: {time.time() - t0:.1f}s*\n")

    report_path = RESULTS / "P2_apap_validation.md"
    _write_report(report_path, "".join(lines), mode="w")
    print(f"\nWrote {report_path}")

    # --- Fire Halt Gate 3 AFTER writing the report (mirror P1) ---
    if gate_fires:
        n_genes = {z: per_zone[z]["n_genes_compared"] for z in per_zone}
        _write_halt_reason(per_zone, "pericentral", 0.3, n_genes, ts)
        sys.exit(1)

    print(f"Done in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
