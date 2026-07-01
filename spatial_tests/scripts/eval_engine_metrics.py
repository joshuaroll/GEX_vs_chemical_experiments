#!/usr/bin/env python
"""Standard MultiDCP expression metrics for the trained per-organ engines.

Loads each (organ, arch) checkpoint, runs held-out-drug test inference, and reports
the metrics MultiDCP reports (utils/metric.py, imported verbatim for comparability):
per-sample Pearson, Spearman, RMSE, precision@k. Computed on DE (pred-x2 vs x1-x2,
the perturbation) which is the meaningful signal; raw-treated Pearson shown too.

Sanity gate: the DE-Pearson here must reproduce each run's logged test_de_pearson.
"""
from __future__ import annotations
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--encoder", default="linear")
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
CHEMOE = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ORGANS = ["liver", "kidney", "brain"]
ARCHS = ["multidcp", "chemoe"]

sys.path.insert(0, f"{MULTIDCP}/utils")
from metric import precision_k, correlation, rmse  # MultiDCP's exact definitions


def build(arch):
    if arch == "multidcp":
        sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]
        import multidcp
        from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
        from multidcp_ae_utils import initialize_model_registry
        reg = initialize_model_registry()
        reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 1,
                    "dropout": 0.3, "linear_encoder_flag": args.encoder == "linear"})
        model = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
        gene_t = read_gene(GENE_VECTOR, DEVICE)

        def feat(s):
            d = convert_smile_to_feature(list(s), DEVICE); return d, create_mask_feature(d, DEVICE)

        def fwd(d, m, basal):
            dose = torch.ones(basal.shape[0], 1, dtype=torch.float64, device=DEVICE)
            o = model(d, gene_t, m, basal, dose, epoch=0)
            return o[0] if isinstance(o, tuple) else o
    else:
        sys.path[:0] = [f"{CHEMOE}/models", f"{CHEMOE}/utils"]
        import multidcp_chemoe_pdg as mc
        from multidcp_ae_pdg_utils import initialize_model_registry
        from data_utils_pdg import convert_smile_to_feature, create_mask_feature
        reg = initialize_model_registry()
        reg.update({"num_gene": 978, "cell_id_input_dim": 978, "pert_idose_input_dim": 2,
                    "dropout": 0.3, "linear_encoder_flag": args.encoder == "linear"})
        model = mc.MultiDCP_CheMoE_AE(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
        gene_t = torch.arange(978, device=DEVICE)

        def feat(s):
            d = convert_smile_to_feature(list(s), DEVICE); return d, create_mask_feature(d, DEVICE)

        def fwd(d, m, basal):
            dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=DEVICE).repeat(basal.shape[0], 1)
            o = model(input_cell_gex=basal, input_drug=d, input_gene=gene_t, mask=m,
                      input_pert_idose=dose, job_id="perturbed", epoch=0)
            return o[0] if isinstance(o, tuple) else o
    return model, feat, fwd


@torch.no_grad()
def predict_test(organ, arch):
    ckpt = f"{CKPT_DIR}/{arch}_{organ}_{args.encoder}_sd42.pt"
    if not os.path.exists(ckpt):
        return None
    model, feat, fwd = build(arch)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)); model.eval()
    d = np.load(f"{REPO}/data/processed/organ_train/{organ}.npz", allow_pickle=True)
    te = np.where(d["split"] == "test")[0]
    x1, x2, smiles = d["x1"][te], d["x2"][te], d["smiles"][te]
    P = []
    for i in range(0, len(te), args.batch):
        s = smiles[i:i + args.batch]
        try:
            drug, mask = feat(s)
        except Exception:  # skip an unfeaturizable batch member by dropping the batch's bad ones
            ok = [j for j, ss in enumerate(s) if _feat_ok(feat, ss)]
            s = s[ok]; drug, mask = feat(s)
        basal = torch.as_tensor(x2[i:i + args.batch][:len(s)], dtype=torch.float64, device=DEVICE)
        P.append(fwd(drug, mask, basal).cpu().numpy())
    P = np.vstack(P).astype(np.float64)
    return x1.astype(np.float64), x2.astype(np.float64), P


def _feat_ok(feat, s):
    try:
        feat([s]); return True
    except Exception:
        return False


def metrics(x1, x2, P):
    de_t, de_p = x1 - x2, P - x2
    m = {}
    m["pearson_DE"], _ = correlation(de_t, de_p, "pearson")
    m["spearman_DE"], _ = correlation(de_t, de_p, "spearman")
    m["pearson_treated"], _ = correlation(x1, P, "pearson")
    m["rmse_DE"] = rmse(de_t, de_p)
    m["rmse_treated"] = rmse(x1, P)
    for k in (50, 100, 200):
        neg, pos = precision_k(de_t, de_p, k)
        m[f"prec@{k}_pos"], m[f"prec@{k}_neg"] = pos, neg
    return m


def main():
    from pathlib import Path
    rows = []
    for organ in ORGANS:
        for arch in ARCHS:
            r = predict_test(organ, arch)
            if r is None:
                continue
            m = metrics(*r)
            m.update(organ=organ, arch=arch, n_test=len(r[2]))
            rows.append(m)
            print(f"[{organ}/{arch}] n={m['n_test']} pearsonDE={m['pearson_DE']:.4f} "
                  f"spearmanDE={m['spearman_DE']:.4f} rmseDE={m['rmse_DE']:.4f} "
                  f"prec@100 pos/neg={m['prec@100_pos']:.3f}/{m['prec@100_neg']:.3f} "
                  f"pearson_treated={m['pearson_treated']:.4f}")

    L = ["# Standard MultiDCP expression metrics — per-organ engines (held-out-drug test set)", "",
         "Metrics from MultiDCP's `utils/metric.py` (per-sample). DE = pred−x2 vs x1−x2 (the "
         "perturbation; the meaningful signal). precision@k = overlap of the true top/bottom-100 "
         "genes with the predicted top/bottom-k. raw-treated Pearson shown for reference (inflated "
         "by the shared basal). Linear encoder, seed 42.", "",
         "| Organ | Arch | n_test | Pearson DE | Spearman DE | RMSE DE | prec@100 pos | prec@100 neg | Pearson treated |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['arch']} | {r['n_test']} | {r['pearson_DE']:.4f} | "
                 f"{r['spearman_DE']:.4f} | {r['rmse_DE']:.4f} | {r['prec@100_pos']:.3f} | "
                 f"{r['prec@100_neg']:.3f} | {r['pearson_treated']:.4f} |")
    L += ["", "## precision@k (positive / negative genes), all k", "",
          "| Organ | Arch | p@50 pos | p@50 neg | p@100 pos | p@100 neg | p@200 pos | p@200 neg |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['organ']} | {r['arch']} | {r['prec@50_pos']:.3f} | {r['prec@50_neg']:.3f} | "
                 f"{r['prec@100_pos']:.3f} | {r['prec@100_neg']:.3f} | {r['prec@200_pos']:.3f} | {r['prec@200_neg']:.3f} |")
    L += ["", "Sanity: Pearson DE here should match each run's logged test_de_pearson "
          "(kidney/multidcp ≈ 0.671).", ""]
    Path(f"{REPO}/results/tables/P3_engine_metrics.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote results/tables/P3_engine_metrics.md")


if __name__ == "__main__":
    main()
