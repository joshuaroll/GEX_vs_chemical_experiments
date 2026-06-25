"""Gate 4a — basal-sensitivity probe on the linear-encoder smoke checkpoint.

The frozen row-17 checkpoint's cell-context encoder was inert (a collapsed
TransformerEncoder: basal in, ~constant out, cell_hidden max|Δ| ~1.7e-8,
pred max|Δ| ~3e-10, Pearson(pred_PP,pred_PC) 1.000000). We retrained with
``--linear_encoder_flag`` (swaps the transformer for the existing
``LinearEncoder`` MLP). This script tests whether the new encoder now moves its
output when the basal changes.

It reuses the proven loading/probe path from
``scripts/diagnose_wiring_vs_weights.py`` verbatim where possible:
  - Build ``MultiDCP_CheMoE_AE`` the upstream way (registry), here with
    ``linear_encoder_flag=True`` so it constructs the ``LinearEncoder`` and the
    smoke checkpoint strict-loads 0 missing / 0 unexpected.
  - Run the SAME two-basal probe (real human liver periportal vs pericentral
    basal, identical 0-1 manifold normalization).
  - Repeat the maximal-contrast synthetic probe (Pearson ~ -1.0 inputs) as a
    sanity check that the encoder is not collapsed.

PASS if: cell_hidden max|Δ| >= 1e-2 AND prediction max|Δ| >= 1e-3 AND
Pearson(pred_PP, pred_PC) < 0.999. FAIL otherwise.

CUDA hygiene (Hard Rule 5): CUDA_VISIBLE_DEVICES set BEFORE torch. GPU 2.
Real data only. Read-only: does NOT modify any production source.

Usage:
    conda run -n dili_v04_env python scripts/gate4a_basal_sensitivity.py --gpu 2
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 4a basal-sensitivity probe.")
    p.add_argument("--gpu", default="2", help="GPU id, 'auto', or 'cpu'.")
    return p.parse_args()


_ARGS = _parse_cli()

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging  # noqa: E402

logging.basicConfig(level=logging.WARNING)

from scripts.cache_region_de import _resolve_device  # noqa: E402

DEVICE = _resolve_device(_ARGS.gpu)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = _REPO_ROOT
CONFIG_PATH = ROOT / "configs" / "liver_p2.yaml"
MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
SMOKE_CKPT = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model_linearenc_smoke.pt"


def _pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import torch

    lines = ["# P2 Gate 4a — basal-sensitivity (linear-encoder smoke checkpoint)\n"]
    lines.append(
        f"\n_Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})_\n"
    )
    lines.append(f"\n_Checkpoint: `{SMOKE_CKPT}`_\n")
    lines.append(
        "\nBuilds `MultiDCP_CheMoE_AE` the upstream way with "
        "`linear_encoder_flag=True` (LinearEncoder MLP), strict-loads the "
        "8-epoch best-val smoke checkpoint. Real human liver basals, frozen "
        "checkpoint, read-only. Same probe path as `diagnose_wiring_vs_weights.py`.\n"
    )

    cfg = yaml.safe_load(open(CONFIG_PATH))
    norm_lo, norm_hi = cfg["basal_normalization"]["manifold_range"]
    gene_order = (ROOT / cfg["gene_order_file"]).read_text().split()
    N_PDG = len(gene_order)

    # ------------------------------------------------------------------
    # Build model the UPSTREAM way + strict load — LINEAR encoder variant.
    # ------------------------------------------------------------------
    sys.path.insert(0, os.path.join(MDCP_SRC, "models"))
    sys.path.insert(0, os.path.join(MDCP_SRC, "utils"))
    from multidcp_ae_pdg_utils import initialize_model_registry  # noqa: E402
    import multidcp_chemoe_pdg as mc  # noqa: E402
    from data_utils_pdg import convert_smile_to_feature, create_mask_feature  # noqa: E402

    device = torch.device(DEVICE)
    reg = initialize_model_registry()
    reg.update({
        "num_gene": N_PDG, "pert_idose_input_dim": 2, "dropout": 0.3,
        "linear_encoder_flag": True,   # <-- the fix under test
    })
    model = mc.MultiDCP_CheMoE_AE(device=device, model_param_registry=reg)
    model.to(device)
    model = model.double()
    state = torch.load(SMOKE_CKPT, map_location=device, weights_only=False)
    res = model.load_state_dict(state, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    enc_type = type(model.model.cell_encoder).__name__
    lines.append("\n## 1. Strict load (linear-encoder variant)\n")
    lines.append(
        f"- cell_encoder class: `{enc_type}` (expect `LinearEncoder`)\n"
        f"- load_state_dict(strict=False): missing={len(missing)}, "
        f"unexpected={len(unexpected)} (must be 0/0 — else wrong encoder variant)\n"
    )
    if missing:
        lines.append(f"  - missing (first 10): {missing[:10]}\n")
    if unexpected:
        lines.append(f"  - unexpected (first 10): {unexpected[:10]}\n")
    strict_ok = (len(missing) == 0 and len(unexpected) == 0)

    # ------------------------------------------------------------------
    # Real human liver basals (periportal vs pericentral), production prep.
    # ------------------------------------------------------------------
    from scripts.cache_region_de import _build_region_basal_map  # noqa: E402
    from scripts.run_p1_eda import YU2022_L5_ZIP, _load_yu2022_l5  # noqa: E402
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        adata, _ = _load_yu2022_l5(YU2022_L5_ZIP, tmp / "yu2022")
        X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        gene_names = list(adata.var_names)
        basal_map = _build_region_basal_map(
            X, gene_names, "human", gene_order, float(norm_lo), float(norm_hi),
            mouse_to_human=None,
        )
    pp = basal_map["periportal"]
    pc = basal_map["pericentral"]
    in_r = _pearson(pp, pc)

    # Upstream-native forward (same helper as diagnose_wiring_vs_weights.py).
    drug = convert_smile_to_feature(["C"], device)
    mask = create_mask_feature(drug, device)
    dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=device)
    gene_t = torch.arange(N_PDG, device=device)

    def native_forward(basal_np):
        bt = torch.as_tensor(basal_np, dtype=torch.float64, device=device).unsqueeze(0)
        with torch.no_grad():
            pred, ch = model(
                input_drug=drug, input_gene=gene_t, mask=mask,
                input_cell_gex=bt, input_pert_idose=dose,
                job_id="perturbed", epoch=0,
            )
        return pred.squeeze(0).double().cpu().numpy(), ch.squeeze(0).double().cpu().numpy()

    pred_pp, ch_pp = native_forward(pp)
    pred_pc, ch_pc = native_forward(pc)
    pred_max = float(np.max(np.abs(pred_pp - pred_pc)))
    ch_max = float(np.max(np.abs(ch_pp - ch_pc)))
    pred_r = _pearson(pred_pp, pred_pc)
    ch_r = _pearson(ch_pp, ch_pc)

    lines.append("\n## 2. Real-basal probe (periportal vs pericentral human liver)\n")
    lines.append(
        f"- input basals: Pearson(PP,PC) = {in_r:.6f} (clearly different)\n"
        f"- **cell_hidden (50-d)**: max|Δ| = **{ch_max:.4e}**, Pearson = {ch_r:.6f}\n"
        f"- **prediction ({N_PDG}-d)**: max|Δ| = **{pred_max:.4e}**, "
        f"Pearson(pred_PP, pred_PC) = {pred_r:.6f}\n"
    )

    # ------------------------------------------------------------------
    # Maximal-contrast synthetic probe (Pearson ~ -1.0) — collapse sanity.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(17)
    base = rng.uniform(float(norm_lo), float(norm_hi), size=N_PDG)
    syn_a = base
    syn_b = float(norm_lo) + float(norm_hi) - base  # mirror -> Pearson ~ -1.0
    syn_r = _pearson(syn_a, syn_b)
    _, ch_sa = native_forward(syn_a)
    _, ch_sb = native_forward(syn_b)
    syn_ch_max = float(np.max(np.abs(ch_sa - ch_sb)))
    syn_ch_r = _pearson(ch_sa, ch_sb)
    lines.append("\n## 3. Synthetic maximal-contrast probe (Pearson ~ -1.0)\n")
    lines.append(
        f"- input Pearson(A,B) = {syn_r:.6f}\n"
        f"- **cell_hidden max|Δ|** = **{syn_ch_max:.4e}**, Pearson = {syn_ch_r:.6f} "
        f"(must NOT collapse to ~0)\n"
    )

    # ------------------------------------------------------------------
    # Verdict.
    # ------------------------------------------------------------------
    lines.append("\n## 4. Verdict\n")
    lines.append(
        "Baseline (dead transformer, row-17): cell_hidden max|Δ| ~1.7e-8, "
        "pred max|Δ| ~3e-10, Pearson(pred_PP,pred_PC) 1.000000.\n\n"
    )
    c1 = ch_max >= 1e-2
    c2 = pred_max >= 1e-3
    c3 = pred_r < 0.999
    lines.append("| criterion | threshold | observed | pass |\n")
    lines.append("|---|---|---|---|\n")
    lines.append(f"| cell_hidden max\\|Δ\\| | >= 1e-2 | {ch_max:.4e} | {c1} |\n")
    lines.append(f"| prediction max\\|Δ\\| | >= 1e-3 | {pred_max:.4e} | {c2} |\n")
    lines.append(f"| Pearson(pred_PP, pred_PC) | < 0.999 | {pred_r:.6f} | {c3} |\n")
    passed = bool(strict_ok and c1 and c2 and c3)
    verdict = "PASS" if passed else "FAIL"
    if not strict_ok:
        verdict = "FAIL"
        lines.append(
            "\n**Strict load was NOT 0/0 — wrong encoder variant built. "
            "Verdict FAIL (build error, not a model result).**\n"
        )
    lines.append(f"\n### VERDICT: **{verdict}**\n")
    if passed:
        impl = (
            "The linear-encoder fix makes the model respond to the tissue basal "
            "input: cell context now moves the prediction. The full ~8h F4 "
            "retrain is worth committing."
        )
    elif strict_ok:
        impl = (
            "Encoder still basal-insensitive after the linear-encoder swap: the "
            "signal is data/label-side, no architecture fix helps. Redirect to "
            "F3 / reframe rather than committing the full F4 retrain."
        )
    else:
        impl = (
            "Build error (strict load not 0/0): fix the encoder flag/variant and "
            "re-run before drawing any model conclusion."
        )
    lines.append(f"\n_Implication: {impl}_\n")

    out_path = ROOT / "results" / "tables" / "P2_gate4a_smoke.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print("WROTE", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
