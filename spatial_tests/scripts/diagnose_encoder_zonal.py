"""Read-only diagnostic: isolate INPUT-FLAT vs ENCODER-FLAT for Halt Gate 3.

Phase 2's per-zone predicted DE has ~zero zonal contrast (per-zone predicted
vectors differ by ~5.96e-08). Two distinct causes need different fixes:

  * INPUT-FLAT  — the periportal vs pericentral basals fed to the model are
    themselves ~identical (over-averaging / normalization flattening / markers
    zero-filled). Fix = basal pipeline (cheap, in-scope).
  * ENCODER-FLAT — the inputs genuinely differ but the frozen cell-context
    encoder maps them to (nearly) the same context. Fix = harder.

This script runs D1 (do INPUT basals differ?), D3 (did zonation markers survive
the gene space?), and D2 (does the encoder respond at all?) by REUSING the EXACT
``scripts/cache_region_de.py`` basal-construction + normalization functions and
the production ``RegionSignatureCacher`` load path. It does NOT modify any
production source. Real data only (Hard Rule 1).

CUDA hygiene (Hard Rule 5): ``--gpu`` parsed + ``CUDA_VISIBLE_DEVICES`` set
BEFORE importing torch; auto-detect a free GPU leaving one free.

Usage
-----
    conda run -n dili_v04_env python scripts/diagnose_encoder_zonal.py --gpu auto
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zonal-invariance root-cause diagnostic.")
    p.add_argument("--gpu", default="auto", help="GPU id, 'auto', or 'cpu'.")
    return p.parse_args()


_ARGS = _parse_cli()

# CUDA hygiene FIRST: reuse the driver's resolver, which sets
# CUDA_VISIBLE_DEVICES before torch is imported anywhere.
import pathlib  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING)

from scripts.cache_region_de import _resolve_device  # noqa: E402

DEVICE = _resolve_device(_ARGS.gpu)

# ---- heavy imports only after CUDA_VISIBLE_DEVICES is set ----
import json  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Reuse the EXACT cache pipeline functions (no re-implementation).
from scripts.cache_region_de import (  # noqa: E402
    _ZONE_MARKERS,
    _assign_zones_by_markers,
    _build_region_basal_map,
    _load_mouse_to_human_map,
    _normalize_to_manifold,
)
from scripts.run_p1_eda import (  # noqa: E402
    GSE272564_TAR,
    YU2022_L5_ZIP,
    _load_yu2022_l5,
)
from src.spatial.config import N_PDG  # noqa: E402
from src.spatial.gene_alignment import align_to_gene_space  # noqa: E402
from src.spatial.pseudobulk import pseudobulk  # noqa: E402
from src.spatial.region_signature import (  # noqa: E402
    INERT_CONTROL_SMILES,
    RegionSignatureCacher,
)

ROOT = _REPO_ROOT
CONFIG_PATH = ROOT / "configs" / "liver_p2.yaml"

# All eight canonical zonation markers asked for in the diagnostic spec.
_DIAG_MARKERS = {
    "pericentral": {
        "human": ["GLUL", "CYP2E1", "OAT", "SLC1A2"],
        "mouse": ["Glul", "Cyp2e1", "Oat", "Slc1a2"],
    },
    "periportal": {
        "human": ["SDS", "CYP2F2", "HAL", "ASS1"],
        "mouse": ["Sds", "Cyp2f2", "Hal", "Ass1"],
    },
}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64)))


def _load_species_raw_and_basal(species, gene_order, norm_lo, norm_hi):
    """Return raw pseudobulk + normalized aligned basal per zone for one species.

    Reuses the cache driver's zoning/pseudobulk/align/normalize exactly. Returns
    a dict with raw pseudobulk profiles (source gene space), the source symbols
    (translated to human for mouse), and the production normalized basal map.
    """
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        if species == "human":
            adata, _ = _load_yu2022_l5(YU2022_L5_ZIP, tmp / "yu2022")
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            gene_names = list(adata.var_names)
            mouse_to_human = None
        else:
            import gzip as _gzip
            import io as _io
            import tarfile

            import scipy.io

            mouse_to_human = _load_mouse_to_human_map()
            with tarfile.open(GSE272564_TAR) as tar:
                tar.extractall(tmp / "gse272564")
            base = tmp / "gse272564"
            all_files = list(base.rglob("*"))
            apap0h = [f for f in all_files if "APAP0h" in f.name or "APAP0H" in f.name]
            matrix_file = next(
                (f for f in apap0h if "matrix" in f.name.lower() and f.name.endswith(".mtx.gz")),
                None,
            )
            features_file = next(
                (f for f in apap0h if "feature" in f.name.lower() or "gene" in f.name.lower()),
                None,
            )
            if matrix_file is None or features_file is None:
                raise RuntimeError("mouse APAP0h mtx/features not found; cannot run zoned diag.")
            with _gzip.open(str(matrix_file), "rb") as gf:
                mat = scipy.io.mmread(_io.BytesIO(gf.read()))
            X = mat.T.toarray() if hasattr(mat, "toarray") else np.asarray(mat).T
            with _gzip.open(str(features_file), "rb") as gf:
                lines = gf.read().decode("utf-8").strip().splitlines()
            gene_names = [ln.split("\t")[1] if "\t" in ln else ln for ln in lines]

        # Production zoning + pseudobulk (exact reuse).
        zone_labels = _assign_zones_by_markers(X, gene_names, species)
        keep = zone_labels != "unzoned"
        pb = pseudobulk(X[keep], zone_labels[keep], gene_names, agg="mean")

        # Production normalized basal map (exact reuse).
        basal_map = _build_region_basal_map(
            X, gene_names, species, gene_order, norm_lo, norm_hi,
            mouse_to_human=mouse_to_human,
        )

        if species == "mouse" and mouse_to_human is not None:
            source_symbols = [mouse_to_human.get(g, g) for g in gene_names]
        else:
            source_symbols = list(gene_names)

        return {
            "raw_pb": pb,                  # PseudobulkResult, source gene space
            "raw_gene_names": gene_names,  # native (mouse symbols for mouse)
            "source_symbols": source_symbols,  # human-namespace symbols
            "basal_map": basal_map,        # normalized, 10716 order
            "spot_counts": pb.spot_counts,
        }


def _marker_value_in_pb(pb, gene_names, symbol, zone):
    """Raw pseudobulk value of `symbol` in `zone` (source gene space)."""
    if symbol not in gene_names:
        return None
    gi = gene_names.index(symbol)
    if zone not in pb.region_order:
        return None
    ri = pb.region_order.index(zone)
    return float(pb.profiles[ri, gi])


def _marker_value_in_basal(basal_map, gene_order, human_symbol, zone):
    """Normalized basal value of `human_symbol` in `zone` (10716 order)."""
    if zone not in basal_map:
        return None
    if human_symbol not in gene_order:
        return None
    gi = gene_order.index(human_symbol)
    return float(basal_map[zone][gi])


def run_d1_d3(species, gene_order, norm_lo, norm_hi, lines):
    lines.append(f"\n### {species.upper()} — D1 (input basals) + D3 (marker survival)\n")
    data = _load_species_raw_and_basal(species, gene_order, norm_lo, norm_hi)
    pb = data["raw_pb"]
    raw_genes = data["raw_gene_names"]
    source_symbols = data["source_symbols"]
    basal_map = data["basal_map"]

    zones = [z for z in ("periportal", "pericentral") if z in basal_map]
    lines.append(f"- Zones present: {zones}; spot counts: "
                 f"{ {z: data['spot_counts'].get(z, 0) for z in zones} }\n")

    # --- D1: whole-vector contrast between the two normalized input basals ---
    if "periportal" in basal_map and "pericentral" in basal_map:
        pp = basal_map["periportal"]
        pc = basal_map["pericentral"]
        r = _pearson(pp, pc)
        l2 = _l2(pp, pc)
        maxabs = float(np.max(np.abs(pp.astype(np.float64) - pc.astype(np.float64))))
        n_diff = int(np.sum(pp != pc))
        lines.append(
            f"- **Input basal contrast (normalized, 10716):** "
            f"Pearson(periportal, pericentral) = **{r:.6f}**, "
            f"L2 = **{l2:.4f}**, max|Δ| = {maxabs:.4e}, "
            f"#genes differing = {n_diff}/{len(pp)}.\n"
        )
    else:
        lines.append("- Only one zone present; cannot compute zone contrast.\n")

    # --- D3 + marker table: raw vs normalized, per zone, presence/zeroed ---
    casing = "mouse" if species == "mouse" else "human"
    target_set = set(gene_order)
    src_set = set(source_symbols)
    covered = target_set & src_set

    lines.append("\n**Marker table (raw pseudobulk vs normalized basal):**\n")
    lines.append(
        "| zone | marker (native) | human sym | in 10716? | covered (not zero-fill)? "
        "| raw PP | raw PC | norm PP | norm PC |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|\n")

    mouse_to_human = _load_mouse_to_human_map() if species == "mouse" else None

    for zone_class in ("pericentral", "periportal"):
        native_markers = _DIAG_MARKERS[zone_class][casing]
        for nm in native_markers:
            # human symbol used for the 10716 lookup
            if species == "mouse":
                hsym = mouse_to_human.get(nm, nm) if mouse_to_human else nm
            else:
                hsym = nm
            in_space = "yes" if hsym in target_set else "NO"
            is_covered = "yes" if hsym in covered else "NO(zero-fill)"
            raw_pp = _marker_value_in_pb(pb, raw_genes, nm, "periportal")
            raw_pc = _marker_value_in_pb(pb, raw_genes, nm, "pericentral")
            norm_pp = _marker_value_in_basal(basal_map, gene_order, hsym, "periportal")
            norm_pc = _marker_value_in_basal(basal_map, gene_order, hsym, "pericentral")

            def _f(x):
                return "—" if x is None else f"{x:.4f}"

            lines.append(
                f"| {zone_class} | {nm} | {hsym} | {in_space} | {is_covered} "
                f"| {_f(raw_pp)} | {_f(raw_pc)} | {_f(norm_pp)} | {_f(norm_pc)} |\n"
            )

    return data


def _try_tap_cell_context(cacher, basal):
    """Best-effort: run the forward and capture the 50-d cell-context vector.

    The CheMoE forward returns ``(pred, cell_hidden)``; cell_hidden is the
    per-sample cell-context embedding. Returns (pred[N_PDG], cell_hidden_vec or None).
    """
    torch = cacher._torch
    device = torch.device(cacher.device)
    drug, mask = cacher._featurize_drug(INERT_CONTROL_SMILES)
    bt = torch.as_tensor(basal, dtype=torch.float64, device=device).unsqueeze(0)
    dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=device)
    with torch.no_grad():
        pred, cell_hidden = cacher._model(
            input_cell_gex=bt, input_drug=drug, input_gene=cacher._gene_tensor,
            mask=mask, input_pert_idose=dose, job_id="perturbed", epoch=0,
        )
    ch = None
    try:
        ch = np.asarray(cell_hidden.squeeze(0).float().cpu().numpy()).ravel()
    except Exception:
        ch = None
    return pred.squeeze(0).float().cpu().numpy(), ch


def run_d2(cacher, gene_order, human_data, mouse_data, norm_lo, norm_hi, lines):
    lines.append("\n### D2 — Does the encoder respond at all?\n")

    # (a) the two healthy human-liver zone basals (the real failing case).
    hbm = human_data["basal_map"]
    if "periportal" in hbm and "pericentral" in hbm:
        pred_pp, ch_pp = _try_tap_cell_context(cacher, hbm["periportal"])
        pred_pc, ch_pc = _try_tap_cell_context(cacher, hbm["pericentral"])
        out_r = _pearson(pred_pp, pred_pc)
        out_max = float(np.max(np.abs(pred_pp.astype(np.float64) - pred_pc.astype(np.float64))))
        lines.append(
            f"- **(a) Human zone basals → output:** "
            f"Pearson(treated_PP, treated_PC) = {out_r:.6f}, "
            f"max|Δ output| = **{out_max:.4e}** "
            f"(input Pearson was {_pearson(hbm['periportal'], hbm['pericentral']):.6f}).\n"
        )
        if ch_pp is not None and ch_pc is not None:
            ctx_max = float(np.max(np.abs(ch_pp.astype(np.float64) - ch_pc.astype(np.float64))))
            lines.append(
                f"  - 50-d cell-context: dim={ch_pp.shape}, "
                f"max|Δ context| = **{ctx_max:.4e}**, "
                f"Pearson = {_pearson(ch_pp, ch_pc):.6f}.\n"
            )
        else:
            lines.append("  - (cell-context vector not tappable from forward return)\n")

    # (b) genuinely-different reference basals the encoder SHOULD respond to.
    #   First try real training basals from the MultiDCP repo; else synthetic.
    ref_a, ref_b, ref_label = _load_two_reference_basals(gene_order, norm_lo, norm_hi)
    pred_a, ch_a = _try_tap_cell_context(cacher, ref_a)
    pred_b, ch_b = _try_tap_cell_context(cacher, ref_b)
    out_r = _pearson(pred_a, pred_b)
    out_max = float(np.max(np.abs(pred_a.astype(np.float64) - pred_b.astype(np.float64))))
    in_r = _pearson(ref_a, ref_b)
    lines.append(
        f"- **(b) {ref_label} → output:** input Pearson = {in_r:.6f}; "
        f"Pearson(treated_a, treated_b) = {out_r:.6f}, "
        f"max|Δ output| = **{out_max:.4e}**.\n"
    )
    if ch_a is not None and ch_b is not None:
        ctx_max = float(np.max(np.abs(ch_a.astype(np.float64) - ch_b.astype(np.float64))))
        lines.append(
            f"  - 50-d cell-context: max|Δ context| = **{ctx_max:.4e}**, "
            f"Pearson = {_pearson(ch_a, ch_b):.6f}.\n"
        )

    # (c) finite-difference sensitivity: perturb a basal and measure ||Δ out||.
    base = hbm.get("pericentral", ref_a)
    rng = np.random.default_rng(0)
    pert = base.astype(np.float64).copy()
    # 10% relative perturbation on a random half of expressed genes, clipped to range.
    idx = rng.choice(len(pert), size=len(pert) // 2, replace=False)
    pert[idx] = np.clip(pert[idx] * 1.10 + 0.05, norm_lo, norm_hi)
    pred_base, _ = _try_tap_cell_context(cacher, base.astype(np.float32))
    pred_pert, _ = _try_tap_cell_context(cacher, pert.astype(np.float32))
    fd_in = _l2(base, pert)
    fd_out = _l2(pred_base, pred_pert)
    lines.append(
        f"- **(c) Finite-difference sensitivity:** ‖Δ input‖={fd_in:.4f} → "
        f"‖Δ output‖={fd_out:.4e} "
        f"(ratio ‖Δout‖/‖Δin‖ = {fd_out / fd_in if fd_in else float('nan'):.4e}).\n"
    )


def _load_two_reference_basals(gene_order, norm_lo, norm_hi):
    """Two genuinely-different reference basals the model was trained on, if findable.

    Tries the MultiDCP repo's diseased/cell-line basal CSV (the training manifold
    source). Falls back to two synthetic strongly-different on-manifold basals.
    """
    import glob

    mdcp_data_globs = [
        "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/data/**/pdg_diseased*avg_over_celltype*.csv",
        "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/**/pdg_diseased*celltype*.csv",
    ]
    csv_path = None
    for g in mdcp_data_globs:
        hits = glob.glob(g, recursive=True)
        if hits:
            csv_path = hits[0]
            break
    if csv_path is not None:
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            # Expect rows = cell types, columns = gene symbols (10716/10717).
            # Build two rows aligned to gene_order.
            gene_cols = [c for c in df.columns if c in set(gene_order)]
            if len(gene_cols) > 1000 and len(df) >= 2:
                row_a = df.iloc[0]
                row_b = df.iloc[min(5, len(df) - 1)]
                a = np.array([float(row_a[c]) if c in gene_cols else norm_lo + (norm_hi - norm_lo) / 2
                              for c in gene_order], dtype=np.float32)
                b = np.array([float(row_b[c]) if c in gene_cols else norm_lo + (norm_hi - norm_lo) / 2
                              for c in gene_order], dtype=np.float32)
                label = (f"two real training cell-type basals "
                         f"(rows 0 & {min(5, len(df) - 1)} of {pathlib.Path(csv_path).name})")
                return a, b, label
        except Exception as e:  # noqa: BLE001
            logging.warning("Could not load real training basals (%s); using synthetic.", e)

    # Synthetic strongly-different on-manifold sensitivity probe.
    rng = np.random.default_rng(1)
    a = rng.uniform(norm_lo, hi := norm_hi, size=len(gene_order)).astype(np.float32)
    # b: invert the rank order of a -> maximally different but same marginal range.
    order = np.argsort(a)
    b = np.empty_like(a)
    b[order] = a[order][::-1]
    return a, b, "two synthetic strongly-different on-manifold basals (no training CSV found)"


def main():
    lines = ["# P2 encoder zonal-invariance root-cause diagnostic\n"]
    lines.append(f"\n_Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})_\n")
    lines.append(
        "\nReuses `scripts/cache_region_de.py` basal pipeline + the production "
        "`RegionSignatureCacher` load path exactly. Real data only. Read-only.\n"
    )

    cfg = yaml.safe_load(open(CONFIG_PATH))
    checkpoint_path = cfg["checkpoint_path"]
    norm_lo, norm_hi = cfg["basal_normalization"]["manifold_range"]
    gene_order = (ROOT / cfg["gene_order_file"]).read_text().split()
    assert len(gene_order) == N_PDG, f"gene order {len(gene_order)} != {N_PDG}"

    # D1 + D3 for both species (no model needed).
    human_data = run_d1_d3("human", gene_order, float(norm_lo), float(norm_hi), lines)
    mouse_data = run_d1_d3("mouse", gene_order, float(norm_lo), float(norm_hi), lines)

    # D2: load the real frozen backbone (production load path).
    cacher = RegionSignatureCacher(
        model_variant="multidcp_chemoe", gene_ids=tuple(gene_order), device=DEVICE,
    )
    cacher.load_model(checkpoint_path, verify_sha=True)
    run_d2(cacher, gene_order, human_data, mouse_data, float(norm_lo), float(norm_hi), lines)

    out_path = ROOT / "results" / "tables" / "P2_encoder_diagnostic.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The verdict block is appended by the caller after reviewing the numbers;
    # here we emit the raw evidence so the report is reproducible.
    out_path.write_text("".join(lines))
    print("WROTE", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
