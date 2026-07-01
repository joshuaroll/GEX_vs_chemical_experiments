#!/usr/bin/env python
"""From-scratch per-organ MultiDCP training (978 L1000 genes, drug-disjoint).

Trains a drug->treated-expression model on one organ's LINCS subset
(scripts/build_organ_dataset.py output). The model predicts treated x1 from
basal x2 + drug; DE = pred - x2 vs true x1 - x2 is the eval signal.

--arch multidcp : CANONICAL base MultiDCP (drug-gene attention ACTIVE, projects a
                  fixed gene_vector prior) from /raid/home/joshua/projects/MultiDCP.
--arch chemoe   : MultiDCP-CheMoE (MoE core) from MultiDCP_CheMoE_pdg (added after
                  the canonical path is validated).

CUDA hygiene: --gpu set before torch import. Real data only.
"""
from __future__ import annotations
import argparse, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--organ", required=True, choices=["liver", "kidney", "brain"])
ap.add_argument("--arch", default="multidcp", choices=["multidcp", "chemoe"])
ap.add_argument("--gpu", default="1")
ap.add_argument("--encoder", default="linear", choices=["linear", "transformer"])
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--smoke", type=int, default=0, help="if >0, cap train/dev to N samples and run few steps")
ap.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # BEFORE torch
os.environ["WANDB_MODE"] = args.wandb

import numpy as np
import torch

REPO = "/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests"
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
CHEMOE = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
DATA = f"{REPO}/data/processed/organ_train/{args.organ}.npz"
CKPT_DIR = f"{REPO}/results/checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_multidcp():
    """Canonical base MultiDCP (attention active)."""
    sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]
    import multidcp
    from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
    from multidcp_ae_utils import initialize_model_registry

    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978,
                "pert_idose_input_dim": 1, "dropout": 0.3,
                "linear_encoder_flag": args.encoder == "linear"})
    model = multidcp.MultiDCPOriginal(device=DEVICE, model_param_registry=reg)
    model.init_weights(pretrained=None)
    model.to(DEVICE).double()
    gene_t = read_gene(GENE_VECTOR, DEVICE)  # [978,128] float64, row-aligned to x1/x2
    assert gene_t.shape[0] == 978, gene_t.shape

    def featurize(smis):
        drug = convert_smile_to_feature(list(smis), DEVICE)
        mask = create_mask_feature(drug, DEVICE)
        return drug, mask

    def forward(drug, mask, basal, epoch):
        dose = torch.ones(basal.shape[0], 1, dtype=torch.float64, device=DEVICE)  # dose-agnostic
        out = model(drug, gene_t, mask, basal, dose, epoch=epoch)
        return out[0] if isinstance(out, tuple) else out  # [B,978]

    return model, featurize, forward


def build_chemoe():
    """MultiDCP-CheMoE (MoE core; learnable gene embedding, mask ignored)."""
    sys.path[:0] = [f"{CHEMOE}/models", f"{CHEMOE}/utils"]
    import multidcp_chemoe_pdg as mc
    from multidcp_ae_pdg_utils import initialize_model_registry
    from data_utils_pdg import convert_smile_to_feature, create_mask_feature

    reg = initialize_model_registry()
    reg.update({"num_gene": 978, "cell_id_input_dim": 978,
                "pert_idose_input_dim": 2, "dropout": 0.3,
                "linear_encoder_flag": args.encoder == "linear"})
    model = mc.MultiDCP_CheMoE_AE(device=DEVICE, model_param_registry=reg)
    model.init_weights(pretrained=None)  # diverse expert init, from scratch
    model.to(DEVICE).double()
    gene_t = torch.arange(978, device=DEVICE)  # placeholder; CheMoE ignores input_gene

    def featurize(smis):
        drug = convert_smile_to_feature(list(smis), DEVICE)
        mask = create_mask_feature(drug, DEVICE)
        return drug, mask

    def forward(drug, mask, basal, epoch):
        B = basal.shape[0]
        dose = torch.tensor([[1.0, 0.0]], dtype=torch.float64, device=DEVICE).repeat(B, 1)
        out = model(input_cell_gex=basal, input_drug=drug, input_gene=gene_t,
                    mask=mask, input_pert_idose=dose, job_id="perturbed", epoch=epoch)
        return out[0] if isinstance(out, tuple) else out  # [B,978]

    return model, featurize, forward


def de_pearson(pred, x1, x2):
    """Mean per-sample Pearson of predicted DE vs true DE."""
    dp, dt = pred - x2, x1 - x2
    dp = dp - dp.mean(1, keepdim=True); dt = dt - dt.mean(1, keepdim=True)
    num = (dp * dt).sum(1)
    den = dp.norm(dim=1) * dt.norm(dim=1) + 1e-8
    return float((num / den).mean())


def main():
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(CKPT_DIR, exist_ok=True)
    d = np.load(DATA, allow_pickle=True)
    x1, x2, smiles, split = d["x1"], d["x2"], d["smiles"], d["split"]
    print(f"[{args.organ}] loaded {len(x1)} profiles; arch={args.arch} encoder={args.encoder}")

    model, featurize, forward = {"multidcp": build_multidcp, "chemoe": build_chemoe}[args.arch]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_params:,}")

    # pre-filter drugs that fail SMILES featurization (one-time, on the unique set)
    uniq = sorted(set(smiles)); bad = set()
    for s in uniq:
        try:
            featurize([s])
        except Exception:
            bad.add(s)
    keep = np.array([s not in bad for s in smiles])
    if bad:
        print(f"dropped {len(bad)} unfeaturizable drugs ({(~keep).sum()} profiles)")
    x1, x2, smiles, split = x1[keep], x2[keep], smiles[keep], split[keep]

    idx = {s: np.where(split == s)[0] for s in ("train", "dev", "test")}
    if args.smoke:
        idx = {k: v[:args.smoke] for k, v in idx.items()}
    print({k: len(v) for k, v in idx.items()})

    def to_t(a):
        return torch.as_tensor(a, dtype=torch.float64, device=DEVICE)

    @torch.no_grad()
    def evaluate(split_idx):
        model.eval(); ps, x1s, x2s = [], [], []
        for i in range(0, len(split_idx), args.batch):
            b = split_idx[i:i + args.batch]
            drug, mask = featurize(smiles[b])
            basal = to_t(x2[b])
            pred = forward(drug, mask, basal, 0)
            ps.append(pred); x1s.append(to_t(x1[b])); x2s.append(basal)
        P, X1, X2 = torch.cat(ps), torch.cat(x1s), torch.cat(x2s)
        mse = float(torch.nn.functional.mse_loss(P, X1))
        return de_pearson(P, X1, X2), mse

    import wandb
    wandb.init(project="MultiDCP_organ_scratch", mode=args.wandb,
               name=f"{args.arch}_{args.organ}_{args.encoder}_sd{args.seed}",
               config=vars(args) | {"n_params": n_params, "n_train": len(idx["train"])})

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_dev, best_ep = -1.0, -1
    ckpt = f"{CKPT_DIR}/{args.arch}_{args.organ}_{args.encoder}_sd{args.seed}.pt"
    max_steps = 3 if args.smoke else 10**9

    for epoch in range(args.epochs):
        model.train(); rng = np.random.RandomState(args.seed + epoch)
        tr = idx["train"].copy(); rng.shuffle(tr)
        ep_loss, nb, t0 = 0.0, 0, time.time()
        for step, i in enumerate(range(0, len(tr), args.batch)):
            if step >= max_steps: break
            b = tr[i:i + args.batch]
            drug, mask = featurize(smiles[b])
            basal = to_t(x2[b])
            pred = forward(drug, mask, basal, epoch)
            loss = model.loss(to_t(x1[b]), pred)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        dev_de, dev_mse = evaluate(idx["dev"])
        tr_loss = ep_loss / max(nb, 1)
        wandb.log({"train_mse": tr_loss, "dev_de_pearson": dev_de, "dev_mse": dev_mse,
                   "epoch": epoch, "sec": time.time() - t0}, step=epoch)
        flag = ""
        if dev_de > best_dev:
            best_dev, best_ep = dev_de, epoch
            torch.save(model.state_dict(), ckpt); flag = " *"
        print(f"ep {epoch:3d} | train_mse {tr_loss:.4f} | dev_DE_pearson {dev_de:.4f} | dev_mse {dev_mse:.4f}{flag}")
        if args.smoke and epoch >= 2: break

    test_de, test_mse = evaluate(idx["test"])
    print(f"\nbest dev DE-Pearson {best_dev:.4f} @ep{best_ep} | test DE-Pearson {test_de:.4f} | ckpt {ckpt}")
    wandb.log({"test_de_pearson": test_de, "test_mse": test_mse, "best_dev_de": best_dev})
    wandb.finish()


if __name__ == "__main__":
    main()
