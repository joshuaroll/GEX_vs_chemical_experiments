"""Read-only root-cause diagnostic: WIRING (W) vs ARCHITECTURE-MODE (M) vs DEAD-WEIGHTS (D).

A prior probe (results/tables/P2_encoder_diagnostic.md) showed the wired row-17
``MultiDCP_CheMoE_AE`` produces ~constant output regardless of the basal/cell
context (output moves <=5.96e-08 for basals down to Pearson -0.9999), while the
drug branch moves the output 0.08-0.13. That probe exercised OUR wiring. This
script separates the three possible causes:

  (W) WIRING  — region_signature._call_model passes the basal to the wrong
      forward argument / a default / a zero, so it never reaches the encoder.
  (M) ARCH/MODE — the basal IS consumed by some path (AE/reconstruction) but NOT
      the prediction head we call, or a mode/job_id gates cell-context usage.
  (D) DEAD WEIGHTS — the cell-context encoder (or its link into the head) is
      genuinely ~zero in the checkpoint.

Method (all on REAL data, frozen real checkpoint):
  1. Build the model exactly the way the UPSTREAM training stack does (registry +
     MultiDCP_CheMoE_AE), NOT via region_signature.py. Strict-load row-17.
  2. Run the upstream-native forward on two clearly-different real basals at the
     training input scale. Does the prediction move? (decisive W vs M/D test)
  3. Trace input_cell_gex layer-by-layer through the cell encoder to cell_hidden
     and into global_features -> prediction, recording where the signal dies.
  4. Probe whether any AE/reconstruction path exists that consumes the basal.
  5. Weight-magnitude audit: cell encoder + first head layer that consumes
     cell_hidden, vs the drug branch (known live).

CUDA hygiene (Hard Rule 5): --gpu parsed + CUDA_VISIBLE_DEVICES set BEFORE torch.
Real data only. Read-only: does NOT modify any production source.

Usage:
    conda run -n dili_v04_env python scripts/diagnose_wiring_vs_weights.py --gpu auto
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wiring-vs-weights root-cause diagnostic.")
    p.add_argument("--gpu", default="auto", help="GPU id, 'auto', or 'cpu'.")
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


def _pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _wstats(t):
    """mean|.|, max|.|, frac(|.|<1e-6) for a tensor."""
    import torch
    x = t.detach().abs().double()
    return (
        float(x.mean().cpu()),
        float(x.max().cpu()),
        float((x < 1e-6).double().mean().cpu()),
    )


def main():
    import torch

    lines = ["# P2 wiring-vs-weights root-cause diagnostic\n"]
    lines.append(
        f"\n_Device: {DEVICE} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})_\n"
    )
    lines.append(
        "\nBuilds the model the UPSTREAM way (registry + `MultiDCP_CheMoE_AE`), "
        "strict-loads row-17, bypassing `region_signature.py`. Real data, frozen "
        "checkpoint, read-only.\n"
    )

    cfg = yaml.safe_load(open(CONFIG_PATH))
    checkpoint_path = cfg["checkpoint_path"]
    norm_lo, norm_hi = cfg["basal_normalization"]["manifold_range"]
    gene_order = (ROOT / cfg["gene_order_file"]).read_text().split()
    N_PDG = len(gene_order)

    # ------------------------------------------------------------------
    # 0. Forward signature (documented from source).
    # ------------------------------------------------------------------
    lines.append("\n## 1. Forward signature & graph trace (from source)\n")
    lines.append(
        "`MultiDCP_CheMoEBase.forward(input_drug, input_gene, mask, "
        "input_cell_gex, input_pert_idose, job_id='perturbed', epoch=0)` "
        "(multidcp_chemoe_pdg.py:345) delegates UNCONDITIONALLY to "
        "`MultiDCP_CheMoE.forward(...)` (line 350) — `job_id` is accepted for "
        "training-script compatibility and IGNORED; there is **no autoencoder "
        "branch** (docstring line 348-349).\n\n"
    )
    lines.append(
        "`input_cell_gex` graph to the PREDICTION output "
        "(multidcp_chemoe_pdg.py):\n"
        "- L244 `cell_hidden = self.cell_encoder(input_cell_gex, epoch)` -> [B,50]\n"
        "- L257 `global_features = cat([drug_embed, cell_hidden, dose_embed])` -> [B,306]\n"
        "- L260 gating uses global_features; L268-272 expert input = "
        "global_features ⊕ gene_embed -> [B,num_gene,434]; L289 weighted expert "
        "sum -> predictions [B,num_gene]. **cell_hidden feeds the prediction "
        "head directly** (no detach, no *0, no gate). So if the basal moved "
        "cell_hidden, it would move the prediction.\n\n"
        "cell_encoder = TransformerEncoder (linear_encoder_flag=False), "
        "multidcp_pdg.py:103-128:\n"
        "- L108 `cell_id_embed = Linear(10716->200)->Linear(200->50)` (input_cell_gex)\n"
        "- L110-112 unsqueeze(-1).repeat(1,1,32): every one of 32 channels is a "
        "COPY of the 50-d embed\n"
        "- L119 PositionalEncoding; L120 `nn.Transformer(d_model=32)` self-attn "
        "(has internal LayerNorm); L126 `max` over the 32 dim -> cell_hidden [B,50].\n\n"
    )

    # ------------------------------------------------------------------
    # 1. Build model the UPSTREAM way + strict load (bypass region_signature).
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
        "linear_encoder_flag": False,
    })
    model = mc.MultiDCP_CheMoE_AE(device=device, model_param_registry=reg)
    model.to(device)
    model = model.double()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = [], []
    res = model.load_state_dict(state, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    lines.append("\n## 2. Upstream-native strict load (bypasses region_signature.py)\n")
    lines.append(
        f"- load_state_dict(strict=False): missing={len(missing)}, "
        f"unexpected={len(unexpected)} "
        f"(0/0 == the production strict load).\n"
    )

    # Two clearly-different real basals (human periportal vs pericentral), built
    # by the production basal pipeline, at the training input scale.
    from scripts.cache_region_de import (  # noqa: E402
        _build_region_basal_map,
    )
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

    # Upstream-native forward helper (NOT region_signature._call_model).
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
    out_max = float(np.max(np.abs(pred_pp - pred_pc)))
    ctx_max = float(np.max(np.abs(ch_pp - ch_pc)))
    lines.append("\n## 3. DECISIVE TEST — upstream-native forward responds to basal?\n")
    lines.append(
        f"- input basals: Pearson(PP,PC) = {in_r:.6f} (clearly different)\n"
        f"- **prediction**: max|Δ output| = **{out_max:.4e}**, "
        f"Pearson(pred_PP, pred_PC) = {_pearson(pred_pp, pred_pc):.6f}\n"
        f"- **cell_hidden (50-d)**: max|Δ| = **{ctx_max:.4e}**, "
        f"Pearson = {_pearson(ch_pp, ch_pc):.6f}\n"
    )
    same_as_wired = out_max < 1e-6
    behav = "INVARIANT to basal" if same_as_wired else "RESPONDS to basal"
    if same_as_wired:
        verdict_line = (
            "=> NOT a wiring bug (W ruled out): the basal reaches the same dead "
            "path whether called via region_signature or natively."
        )
    else:
        verdict_line = "=> region_signature wiring is the suspect (W)."
    lines.append(
        f"- Upstream-native call reproduces the wired behaviour ({behav}). "
        f"{verdict_line}\n"
    )

    # ------------------------------------------------------------------
    # 4. Layer-by-layer trace inside the cell encoder: where does it die?
    # ------------------------------------------------------------------
    lines.append("\n## 4. Layer-by-layer trace inside cell_encoder (where signal dies)\n")
    enc = model.model.cell_encoder
    bt_pp = torch.as_tensor(pp, dtype=torch.float64, device=device).unsqueeze(0)
    bt_pc = torch.as_tensor(pc, dtype=torch.float64, device=device).unsqueeze(0)

    def stage_delta(fn, a, b, label):
        with torch.no_grad():
            ya, yb = fn(a), fn(b)
        d = float((ya - yb).abs().max().cpu())
        ra = _pearson(ya.flatten().cpu().numpy(), yb.flatten().cpu().numpy())
        lines.append(f"- {label}: max|Δ| = {d:.4e}, Pearson = {ra:.6f}\n")
        return ya, yb

    # stage A: linear embed (10716->200->50)
    with torch.no_grad():
        emb_pp = enc.cell_id_embed(bt_pp)
        emb_pc = enc.cell_id_embed(bt_pc)
    lines.append(
        f"- after `cell_id_embed` Linear(10716->200->50): "
        f"max|Δ| = {float((emb_pp-emb_pc).abs().max().cpu()):.4e}, "
        f"Pearson = {_pearson(emb_pp.cpu().numpy(), emb_pc.cpu().numpy()):.6f}\n"
    )

    # stage B: unsqueeze+repeat to [B,50,32] then pos-encode + transformer + max
    def to_trans_input(emb):
        e = emb.unsqueeze(-1).repeat(1, 1, enc.trans_cell_embed_dim)
        return enc.pos_encoder(e)

    with torch.no_grad():
        ti_pp = to_trans_input(emb_pp)
        ti_pc = to_trans_input(emb_pc)
    lines.append(
        f"- after repeat-to-[B,50,32] + pos_encoder (pre-transformer): "
        f"max|Δ| = {float((ti_pp-ti_pc).abs().max().cpu()):.4e}, "
        f"Pearson = {_pearson(ti_pp.cpu().numpy(), ti_pc.cpu().numpy()):.6f}\n"
    )
    with torch.no_grad():
        tr_pp = enc.cell_id_transformer(ti_pp, ti_pp)
        tr_pc = enc.cell_id_transformer(ti_pc, ti_pc)
    lines.append(
        f"- after `nn.Transformer` (d_model=32, internal LayerNorm): "
        f"max|Δ| = {float((tr_pp-tr_pc).abs().max().cpu()):.4e}, "
        f"Pearson = {_pearson(tr_pp.cpu().numpy(), tr_pc.cpu().numpy()):.6f}\n"
    )
    with torch.no_grad():
        ch2_pp, _ = torch.max(tr_pp, -1)
        ch2_pc, _ = torch.max(tr_pc, -1)
    lines.append(
        f"- after `max(-1)` -> cell_hidden [B,50]: "
        f"max|Δ| = {float((ch2_pp-ch2_pc).abs().max().cpu()):.4e}, "
        f"Pearson = {_pearson(ch2_pp.cpu().numpy(), ch2_pc.cpu().numpy()):.6f}\n"
    )

    # ------------------------------------------------------------------
    # 5. AE / reconstruction path probe.
    # ------------------------------------------------------------------
    lines.append("\n## 5. AE / reconstruction path probe (M check)\n")
    methods = [m for m in dir(model) if not m.startswith("__")]
    ae_like = [m for m in methods if any(
        k in m.lower() for k in ("recon", "decode", "autoenc", "ae_", "_ae"))]
    inner = [m for m in dir(model.model) if any(
        k in m.lower() for k in ("recon", "decode", "autoenc", "ae_"))]
    lines.append(
        f"- AE/reconstruct-like methods on MultiDCP_CheMoE_AE: {ae_like or 'NONE'}\n"
        f"- on inner MultiDCP_CheMoE: {inner or 'NONE'}\n"
        f"- Source confirms (docstring L348-349): 'CheMoE doesn't use autoencoder "
        "mode, always predicts perturbed expression.' There is no second forward "
        "path / reconstruction head that consumes the basal differently. => (M) "
        "ruled out: no alternate mode would route cell context into a prediction.\n"
    )

    # ------------------------------------------------------------------
    # 6. Weight-magnitude audit: cell encoder vs drug branch.
    # ------------------------------------------------------------------
    lines.append("\n## 6. Weight-magnitude audit (D check)\n")
    lines.append("| param | shape | mean\\|w\\| | max\\|w\\| | frac\\|w\\|<1e-6 |\n")
    lines.append("|---|---|---|---|---|\n")
    audit_keys = [
        "model.cell_encoder.cell_id_embed.0.weight",
        "model.cell_encoder.cell_id_embed.0.bias",
        "model.cell_encoder.cell_id_embed.1.weight",
        "model.cell_encoder.cell_id_transformer.encoder.layers.0.self_attn.in_proj_weight",
        "model.cell_encoder.cell_id_transformer.encoder.layers.0.linear1.weight",
        "model.cell_encoder.cell_id_transformer.encoder.norm.weight",
        "model.cell_encoder.cell_id_transformer.encoder.norm.bias",
    ]
    sd_named = dict(model.named_parameters())
    sd_buf = dict(model.named_buffers())
    for k in audit_keys:
        t = sd_named.get(k, sd_buf.get(k))
        if t is None:
            lines.append(f"| {k} | MISSING | — | — | — |\n")
            continue
        m, mx, fz = _wstats(t)
        lines.append(f"| `{k.replace('model.','')}` | {tuple(t.shape)} | {m:.4e} | {mx:.4e} | {fz:.3f} |\n")

    # drug branch reference (known live)
    drug_keys = [k for k, _ in model.named_parameters() if "drug_fp" in k][:3]
    lines.append("\n_Drug branch reference (known live):_\n")
    lines.append("| param | shape | mean\\|w\\| | max\\|w\\| | frac\\|w\\|<1e-6 |\n")
    lines.append("|---|---|---|---|---|\n")
    for k in drug_keys:
        t = sd_named[k]
        m, mx, fz = _wstats(t)
        lines.append(f"| `{k.replace('model.','')}` | {tuple(t.shape)} | {m:.4e} | {mx:.4e} | {fz:.3f} |\n")

    # The slice of the gating/expert FIRST layer that reads cell_hidden.
    # global_features = [drug(0:128), cell(128:178), dose(178:306)].
    lines.append(
        "\n_First head layers that consume cell_hidden (global_features cols "
        "128:178):_\n"
    )
    lines.append("| layer | full mean\\|w\\| | cell-slice(128:178) mean\\|w\\| | drug-slice(0:128) mean\\|w\\| | dose-slice(178:306) mean\\|w\\| |\n")
    lines.append("|---|---|---|---|---|\n")
    gate_w = sd_named.get("model.gating_network.gate.0.weight")
    if gate_w is not None:
        full = float(gate_w.abs().mean().cpu())
        cell_s = float(gate_w[:, 128:178].abs().mean().cpu())
        drug_s = float(gate_w[:, 0:128].abs().mean().cpu())
        dose_s = float(gate_w[:, 178:306].abs().mean().cpu())
        lines.append(f"| gating.gate.0.weight {tuple(gate_w.shape)} | {full:.4e} | {cell_s:.4e} | {drug_s:.4e} | {dose_s:.4e} |\n")
    exp0 = sd_named.get("model.experts.0.expert_mlp.0.weight")
    if exp0 is not None:
        full = float(exp0.abs().mean().cpu())
        cell_s = float(exp0[:, 128:178].abs().mean().cpu())
        drug_s = float(exp0[:, 0:128].abs().mean().cpu())
        dose_s = float(exp0[:, 178:306].abs().mean().cpu())
        lines.append(f"| experts.0.mlp.0.weight {tuple(exp0.shape)} | {full:.4e} | {cell_s:.4e} | {drug_s:.4e} | {dose_s:.4e} |\n")

    out_path = ROOT / "results" / "tables" / "P2_wiring_vs_weights.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print("WROTE", out_path)
    print("".join(lines))


if __name__ == "__main__":
    main()
