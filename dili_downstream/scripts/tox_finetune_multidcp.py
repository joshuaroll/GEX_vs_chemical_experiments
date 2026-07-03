#!/usr/bin/env python
"""Tox-finetune the ORIGINAL pretrained MultiDCP on the Li/Tong 6,000 DILI benchmark (condition E/F).

Question: does starting from a PRETRAINED MultiDCP backbone (vs the from-scratch organ engine, which
HURT: structure > frozen > tuned) change the structure-vs-expression verdict on the benchmark's own data?

Feature conditions (all share ONE head: BatchNorm -> Linear(.,128) -> ReLU -> Dropout(0.3) -> Linear(128,1)):
  structure     ECFP4(2048) -> head                                   (baseline, static SGD head)
  measured      measured L5 MODZ (978, Entrez) -> head                (in-harness measured reference)
  frozen_pred   frozen pretrained MultiDCP output (977) -> head       (expression, no finetune)
  tuned_E       pretrained MultiDCP + head trained end-to-end on DILI (expression, finetuned)  <-- headline
  tuned_F       tuned_E + anchor MSE(pred, frozen pred)               (finetuned, biology-anchored)
  both          ECFP4 (+) tuned_E engine output -> head end-to-end    (structure + expression)

Splits (paper protocol: resample many times, report mean +/- std across resamples):
  test      Wang/Li profile-level Training/Test (their published split) -> compare to 0.798
  drug      drug(compound)-disjoint GroupShuffleSplit x N resamples (leakage-corrected)
  scaffold  Murcko-scaffold-disjoint GroupShuffleSplit x N resamples (strongest)
Metrics: profile-level AUROC (paper's unit) AND drug-level AUROC (mean prediction per drug).

Faithful pretrained encoding: basal = CCLE/TCGA log2-TPM per cell (978, input context, not subtracted);
dose = 6-way canonical one-hot; predicted feature = MultiDCP_AE perturbed output (977) = predicted DE
(LINCS L5 MODZ is z-scored differential natively). >=3 seeds; internal-val early stopping on AUROC.
"""
from __future__ import annotations
import argparse, os, sys, copy, json

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="1")
ap.add_argument("--ckpt", default="/raid/home/joshua/projects/MultiDCP/trained_models/L5_split1_05032023.pt",
                help="original pretrained MultiDCP_AE checkpoint")
ap.add_argument("--variants", default="structure,measured,frozen_pred,tuned_E,tuned_F,both")
ap.add_argument("--splits", default="test,drug,scaffold")
ap.add_argument("--seeds", type=int, default=3, help="repeats on the fixed profile Test split")
ap.add_argument("--n-resamples", type=int, default=10,
                help="resampled drug/scaffold-disjoint splits (paper uses 50 for the DNN)")
ap.add_argument("--folds", type=int, default=5, help="(unused; kept for CLI compat)")
ap.add_argument("--max-epochs", type=int, default=60)
ap.add_argument("--patience", type=int, default=10)
ap.add_argument("--anchor", type=float, default=1.0, help="tuned_F anchor weight")
ap.add_argument("--bs", type=int, default=16, help="engine minibatch (gene-attention is O(B*977^2))")
ap.add_argument("--head-epochs", type=int, default=300, help="static-head max epochs")
ap.add_argument("--permute-labels", action="store_true",
                help="negative control: shuffle the per-drug label map; everything should return ~0.5")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tag", default="")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
MULTIDCP = "/raid/home/joshua/projects/MultiDCP/MultiDCP"
GENE_VECTOR = f"{MULTIDCP}/data/gene_vector.csv"
NPZ = REPO / "data/processed/wangli_multidcp_finetune.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path[:0] = [f"{MULTIDCP}/models", f"{MULTIDCP}/utils"]

import multidcp
from data_utils import convert_smile_to_feature, create_mask_feature, read_gene
from multidcp_ae_utils import initialize_model_registry
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

GENE_T = read_gene(GENE_VECTOR, DEVICE)
NUM_GENE = GENE_T.shape[0]      # 977


def load_pretrained():
    """Instantiate MultiDCP_AE with the ORIGINAL training registry and load the pretrained weights."""
    reg = initialize_model_registry()
    reg.update({"num_gene": NUM_GENE, "pert_idose_input_dim": 6, "dropout": 0.3,
                "linear_encoder_flag": False})   # original L5 used the Transformer cell encoder
    m = multidcp.MultiDCP_AE(device=DEVICE, model_param_registry=reg).to(DEVICE).double()
    sd = torch.load(args.ckpt, map_location=DEVICE)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    m.eval()
    return m


def ecfp4(smis, n_bits=2048):
    out = np.zeros((len(smis), n_bits), np.float64)
    for i, s in enumerate(smis):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        out[i] = np.frombuffer(fp.ToBitString().encode(), "u1") - ord("0")
    return out


def can_feat(s):
    try:
        convert_smile_to_feature([s], DEVICE)
        return True
    except Exception:
        return False


class ToxHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(in_dim), nn.Linear(in_dim, 128), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(128, 1)).double()

    def forward(self, x):
        return self.net(x).squeeze(1)


def engine_forward(model, smis, basal, dose_oh):
    """MultiDCP_AE perturbed output [B,977] for a batch (one forward)."""
    d = convert_smile_to_feature(list(smis), DEVICE)
    mask = create_mask_feature(d, DEVICE)
    cell = torch.as_tensor(basal, dtype=torch.float64, device=DEVICE)
    dose = torch.as_tensor(dose_oh, dtype=torch.float64, device=DEVICE)
    out, _ = model(cell, d, GENE_T, mask, dose, job_id="perturbed")
    return out


# ---------------------------------------------------------------- static head (SGD)
def train_static(Xtr, ytr, Xva, yva, Xte, seed):
    torch.manual_seed(seed)
    head = ToxHead(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float64, device=DEVICE)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float64, device=DEVICE)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float64, device=DEVICE)
    Xte_t = torch.as_tensor(Xte, dtype=torch.float64, device=DEVICE)
    best, best_state, bad = -1, None, 0
    for ep in range(args.head_epochs):
        head.train(); opt.zero_grad()
        loss = lossf(head(Xtr_t), ytr_t); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            va = roc_auc_score(yva, head(Xva_t).cpu().numpy())
        if va > best + 1e-4:
            best, best_state, bad = va, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        return head(Xte_t).cpu().numpy()


# ---------------------------------------------------------------- end-to-end finetune
BS = args.bs


@torch.no_grad()
def predict_tuned(model, head, idx, D, ecfp_all, use_both):
    out = []
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        de = engine_forward(model, D["smiles"][b], D["cell_basal"][b], D["dose_onehot"][b])
        feat = de if not use_both else torch.cat(
            [de, torch.as_tensor(ecfp_all[b], dtype=torch.float64, device=DEVICE)], dim=1)
        out.append(head(feat).cpu().numpy())
    return np.concatenate(out)


def train_tuned(D, ecfp_all, frozen_pred, tr, va, te, seed, anchor, use_both):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = load_pretrained()
    in_dim = NUM_GENE + (ecfp_all.shape[1] if use_both else 0)
    head = ToxHead(in_dim).to(DEVICE)
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": 1e-4},
                            {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    y = D["label"]
    best, best_state, bad = -1, None, 0
    order = np.arange(len(tr))
    _verbose = os.environ.get("FT_VERBOSE")
    _t0 = __import__("time").time()
    for ep in range(args.max_epochs):
        model.train(); head.train()
        rng.shuffle(order)
        for s in range(0, len(order), BS):
            bb = order[s:s + BS]
            if len(bb) < 2:
                continue  # BatchNorm needs >1 sample in train mode; drop a size-1 trailing minibatch
            gidx = tr[bb]
            opt.zero_grad()
            de = engine_forward(model, D["smiles"][gidx], D["cell_basal"][gidx], D["dose_onehot"][gidx])
            feat = de if not use_both else torch.cat(
                [de, torch.as_tensor(ecfp_all[gidx], dtype=torch.float64, device=DEVICE)], dim=1)
            yb = torch.as_tensor(y[gidx], dtype=torch.float64, device=DEVICE)
            loss = lossf(head(feat), yb)
            if anchor > 0:
                # anchor the finetuned output to the FROZEN pretrained output (keeps signature biology-shaped)
                fz = torch.as_tensor(frozen_pred[gidx], dtype=torch.float64, device=DEVICE)
                loss = loss + anchor * ((de - fz) ** 2).mean()
            loss.backward(); opt.step()
        model.eval(); head.eval()
        va_pred = predict_tuned(model, head, va, D, ecfp_all, use_both)
        va_auc = roc_auc_score(y[va], va_pred)
        if _verbose:
            print(f"    ep{ep:02d} va_auc={va_auc:.4f} best={max(best,va_auc):.4f} "
                  f"bad={bad} elapsed={__import__('time').time()-_t0:.0f}s", flush=True)
        if va_auc > best + 1e-4:
            best, best_state, bad = va_auc, (copy.deepcopy(model.state_dict()),
                                             copy.deepcopy(head.state_dict())), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    model.eval(); head.eval()
    return predict_tuned(model, head, te, D, ecfp_all, use_both)


# ---------------------------------------------------------------- pretrained STRUCTURE encoder (ChemBERTa)
# Fair counterpart to the pretrained expression backbone: a pretrained chemical LM, used frozen
# (chemberta_frozen) or finetuned end-to-end on DILI (chemberta_ft). The chemberta_ft-vs-tuned_E
# contrast is the load-bearing ablation: does routing SMILES through a predicted-GEX bottleneck
# (MultiDCP) add anything over using the same SMILES directly (ChemBERTa)?
CHEMBERTA_ID = "DeepChem/ChemBERTa-77M-MLM"


def load_chemberta():
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(CHEMBERTA_ID)
    mdl = AutoModel.from_pretrained(CHEMBERTA_ID).to(DEVICE).double()
    return mdl, tok


def chemberta_embed(mdl, tok, smis):
    """Mean-pooled (over real tokens) last-hidden-state embedding [B, 384], grad-enabled."""
    enc = tok([str(s) for s in smis], padding=True, truncation=True, max_length=256,
              return_tensors="pt").to(DEVICE)
    h = mdl(**enc).last_hidden_state              # [B, L, 384]
    m = enc["attention_mask"].unsqueeze(-1).double()
    return (h * m).sum(1) / m.sum(1).clamp(min=1)


@torch.no_grad()
def predict_chemberta(mdl, tok, head, idx, smis, meta):
    """Chunked logits over global row indices idx; meta (or None) is [N, md] cell+dose metadata."""
    out = []
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        emb = chemberta_embed(mdl, tok, smis[b])
        feat = emb if meta is None else torch.cat(
            [emb, torch.as_tensor(meta[b], dtype=torch.float64, device=DEVICE)], dim=1)
        out.append(head(feat).cpu().numpy())
    return np.concatenate(out)


def train_chemberta(D, tr, va, te, seed, meta=None):
    """Finetune ChemBERTa end-to-end on DILI (same recipe as train_tuned: encoder lr 1e-4, head lr 1e-3,
    group-disjoint-val early stopping). If meta is given ([N, md] cell+dose one-hot), it is concatenated
    to the pooled embedding before the head -> the fair 'SMILES + same assay metadata as MultiDCP' arm.
    Returns test-set logits."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    mdl, tok = load_chemberta()
    md = 0 if meta is None else meta.shape[1]
    head = ToxHead(mdl.config.hidden_size + md).to(DEVICE)
    opt = torch.optim.Adam([{"params": mdl.parameters(), "lr": 1e-4},
                            {"params": head.parameters(), "lr": 1e-3}], weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    y = D["label"]
    smis = D["smiles"]
    best, best_state, bad = -1, None, 0
    order = np.arange(len(tr))
    for ep in range(args.max_epochs):
        mdl.train(); head.train()
        rng.shuffle(order)
        for s in range(0, len(order), BS):
            bb = order[s:s + BS]
            if len(bb) < 2:
                continue
            gidx = tr[bb]
            opt.zero_grad()
            emb = chemberta_embed(mdl, tok, smis[gidx])
            feat = emb if meta is None else torch.cat(
                [emb, torch.as_tensor(meta[gidx], dtype=torch.float64, device=DEVICE)], dim=1)
            yb = torch.as_tensor(y[gidx], dtype=torch.float64, device=DEVICE)
            loss = lossf(head(feat), yb)
            loss.backward(); opt.step()
        mdl.eval(); head.eval()
        va_auc = roc_auc_score(y[va], predict_chemberta(mdl, tok, head, va, smis, meta))
        if va_auc > best + 1e-4:
            best, best_state, bad = va_auc, (copy.deepcopy(mdl.state_dict()),
                                             copy.deepcopy(head.state_dict())), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    mdl.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    mdl.eval(); head.eval()
    return predict_chemberta(mdl, tok, head, te, smis, meta)


def inner_split(tr, y, groups, seed, frac=0.15):
    """Carve an early-stopping validation set from tr.

    Group-disjoint when `groups` is given (val holds out whole compounds/scaffolds) so val AUROC
    tracks generalization rather than within-group memorization -> early stopping fires meaningfully
    and finetunes don't grind to the epoch cap. Random stratified for the profile-level split, where
    leakage is inherent to the split design. Falls back to random if a group holdout leaves one class
    empty on either side.
    """
    if groups is None:
        return train_test_split(tr, test_size=frac, random_state=seed, stratify=y[tr])
    g = groups[tr]
    uniq = np.unique(g)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(round(frac * len(uniq))))
    val_groups = set(uniq[:n_val].tolist())
    va_mask = np.array([x in val_groups for x in g])
    va, tr2 = tr[va_mask], tr[~va_mask]
    if len(va) == 0 or len(tr2) == 0 or len(np.unique(y[va])) < 2 or len(np.unique(y[tr2])) < 2:
        return train_test_split(tr, test_size=frac, random_state=seed, stratify=y[tr])
    return tr2, va


def auroc_profile_and_drug(y_te, pred_te, drugs_te):
    """Profile-level AUROC (paper's unit) and drug-level AUROC (mean prediction per drug, one label
    per drug — the honest per-drug unit since DILI is a drug property). NaN if a level has one class."""
    prof = roc_auc_score(y_te, pred_te) if len(np.unique(y_te)) > 1 else np.nan
    df = pd.DataFrame({"d": drugs_te, "p": pred_te, "y": y_te})
    g = df.groupby("d").agg(p=("p", "mean"), y=("y", "first"))
    drug = roc_auc_score(g["y"].values, g["p"].values) if g["y"].nunique() > 1 else np.nan
    return prof, drug


def main():
    d = np.load(NPZ, allow_pickle=True)
    D = {k: d[k] for k in d.files}
    # keep only featurizable SMILES
    uniq = {s: can_feat(s) for s in np.unique(D["smiles"])}
    ok = np.array([uniq[s] for s in D["smiles"]])
    if not ok.all():
        print(f"[featurize] dropping {int((~ok).sum())} profiles with non-featurizable SMILES")
        for k in list(D.keys()):
            if isinstance(D[k], np.ndarray) and D[k].shape[:1] == ok.shape:
                D[k] = D[k][ok]
    y = D["label"].astype(int)
    N = len(y)
    if args.permute_labels:
        # negative control: shuffle labels at the DRUG level (preserve one-label-per-drug), so any
        # above-chance score reveals leakage through shared inputs or the model-selection machinery.
        rng = np.random.RandomState(0)
        udrugs = np.unique(D["compound_name"])
        drug_lab = {d: int(y[D["compound_name"] == d][0]) for d in udrugs}
        shuffled = rng.permutation(np.array([drug_lab[d] for d in udrugs]))
        newmap = dict(zip(udrugs, shuffled))
        y = np.array([newmap[d] for d in D["compound_name"]], dtype=int)
        D["label"] = y
        print("[permute] labels shuffled at drug level (negative control; expect ~0.5 everywhere)")
    print(f"[data] N={N} pos={int(y.sum())} neg={int((1-y).sum())} "
          f"cells={len(set(D['cell_id']))} compounds={len(set(D['compound_name']))}")

    if args.smoke:
        m = load_pretrained()
        idx = np.arange(8)
        with torch.no_grad():
            out = engine_forward(m, D["smiles"][idx], D["cell_basal"][idx], D["dose_onehot"][idx])
        print(f"[smoke] pretrained forward ok: out.shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
              f"mean={out.mean().item():.4f} std={out.std().item():.4f}")
        ef = ecfp4(list(D["smiles"][idx]))
        print(f"[smoke] ecfp4 shape={ef.shape} bits/row={ef.sum(1)[:4]}")
        print(f"[smoke] measured_modz finite for 8: {np.isfinite(D['measured_modz'][idx]).all()}")
        return

    ecfp_all = ecfp4(list(D["smiles"]))
    print(f"[features] ecfp4 {ecfp_all.shape}")
    # frozen predicted output for all rows (one pass) -> frozen_pred arm + tuned_F anchor
    m0 = load_pretrained()
    frozen_pred = np.vstack([engine_forward(m0, D["smiles"][i:i+64], D["cell_basal"][i:i+64],
                                            D["dose_onehot"][i:i+64]).detach().cpu().numpy()
                             for i in range(0, N, 64)])
    del m0; torch.cuda.empty_cache()
    print(f"[features] frozen predicted output {frozen_pred.shape}")
    _mnan = int(np.isnan(D["measured_modz"]).any(axis=1).sum())
    if _mnan:
        print(f"[WARN] measured arm: {_mnan}/{N} profiles have NaN MODZ -> zero-filled; this HANDICAPS "
              f"the measured reference. Investigate the L5 join before trusting measured-arm numbers.")
    measured = np.nan_to_num(D["measured_modz"].astype(np.float64))

    # cell+dose assay metadata (the shared-across-split inputs MultiDCP gets but ECFP/ChemBERTa don't):
    # dose one-hot (6) + cell identity one-hot (57). `cell_dose_only` = this alone (no SMILES, no GEX)
    # -> upper-bounds how much of any expression score is the assay-metadata confound (esp. dose).
    ucell = sorted(set(D["cell_id"].tolist()))
    cidx = {c: i for i, c in enumerate(ucell)}
    cell_oh = np.zeros((N, len(ucell)), np.float64)
    cell_oh[np.arange(N), [cidx[c] for c in D["cell_id"]]] = 1.0
    meta = np.hstack([D["dose_onehot"].astype(np.float64), cell_oh])   # [N, 6+57]
    print(f"[features] cell+dose meta {meta.shape}")

    STATIC = {"structure": ecfp_all, "measured": measured, "frozen_pred": frozen_pred,
              "cell_dose_only": meta}
    # Frozen ChemBERTa embeddings (pretrained structure, no finetune) — precompute once if requested.
    variants_req = args.variants.split(",")
    if "chemberta_frozen" in variants_req:
        cb_m, cb_tok = load_chemberta(); cb_m.eval()
        with torch.no_grad():
            cb_frozen = np.vstack([chemberta_embed(cb_m, cb_tok, D["smiles"][i:i+64]).cpu().numpy()
                                   for i in range(0, N, 64)]).astype(np.float64)
        del cb_m; torch.cuda.empty_cache()
        STATIC["chemberta_frozen"] = cb_frozen
        print(f"[features] frozen ChemBERTa embedding {cb_frozen.shape}")
    if "dist_shift" in variants_req:
        # Cheap scRatio-spirit probe: does a DISTRIBUTIONAL summary of the perturbation (beyond the
        # 978-d mean MODZ) predict DILI? Features per profile: magnitude/breadth summaries + a
        # classifier-trick density ratio log[p_treated/p_null] (Sugiyama), null = N(0,I) (MODZ is
        # per-gene z-scored). Label-free (the density-ratio classifier never sees the DILI label), so
        # compute-once-then-slice is not a leak. Null ignores gene-gene covariance (a known limitation).
        from sklearn.linear_model import LogisticRegression
        rng_ds = np.random.RandomState(0)
        null = rng_ds.randn(*measured.shape)
        lr = LogisticRegression(max_iter=300).fit(
            np.vstack([measured, null]), np.r_[np.ones(N), np.zeros(N)])
        dr = lr.decision_function(measured)                     # log density ratio, per profile
        dist_shift = np.column_stack([
            np.linalg.norm(measured, axis=1), np.abs(measured).sum(1), np.abs(measured).max(1),
            (np.abs(measured) > 2).mean(1), (np.abs(measured) > 4).mean(1), dr]).astype(np.float64)
        STATIC["dist_shift"] = dist_shift
        print(f"[features] distributional-shift probe {dist_shift.shape} "
              f"(L2,L1,maxabs,frac|z|>2,frac|z|>4,density-ratio)")
    drugs = D["compound_name"]
    rows = []
    per_rows = []

    def eval_arm(variant, tr, te, split_groups, seed):
        """Train one arm on tr, predict te, return (profile_auroc, drug_auroc)."""
        tr2, va = inner_split(tr, y, split_groups, seed)
        if variant in STATIC:
            X = STATIC[variant]
            pred = train_static(X[tr2], y[tr2], X[va], y[va], X[te], seed)
        elif variant == "chemberta_ft":
            pred = train_chemberta(D, tr2, va, te, seed, meta=None)
        elif variant == "chemberta_meta":
            # fair vs tuned_E: SMILES + the same cell+dose assay metadata MultiDCP receives
            pred = train_chemberta(D, tr2, va, te, seed, meta=meta)
        elif variant == "both":
            pred = train_tuned(D, ecfp_all, frozen_pred, tr2, va, te, seed, anchor=0.0, use_both=True)
        else:  # tuned_E / tuned_F
            anchor = args.anchor if variant == "tuned_F" else 0.0
            pred = train_tuned(D, ecfp_all, frozen_pred, tr2, va, te, seed, anchor=anchor, use_both=False)
        return auroc_profile_and_drug(y[te], pred, drugs[te])

    for split in args.splits.split(","):
        # Build resampled train/test splits (paper protocol: many resamples, mean +/- std across them).
        if split == "test":
            # Wang/Li's fixed profile-level Training/Test split; repeat over seeds for a small CI vs 0.798.
            tr_all = np.where(D["usage"] == "Training")[0]
            te_all = np.where(D["usage"] == "Test")[0]
            resamples = [(s, tr_all, te_all, None) for s in range(args.seeds)]
        else:
            # drug- or scaffold-disjoint: N resampled GroupShuffleSplits (held-out 20% of groups),
            # matching Li/Tong's 50 drug-based splits (fewer here for finetuning cost).
            split_groups = D["compound_name"] if split == "drug" else D["scaffold"]
            gss = GroupShuffleSplit(n_splits=args.n_resamples, test_size=0.2, random_state=0)
            resamples = [(s, tr, te, split_groups)
                         for s, (tr, te) in enumerate(gss.split(np.zeros(N), y, split_groups))]

        for variant in args.variants.split(","):
            profs, drugaucs = [], []
            for s, tr, te, split_groups in resamples:
                p, dd = eval_arm(variant, tr, te, split_groups, s)
                profs.append(p); drugaucs.append(dd)
                per_rows.append(dict(split=split, variant=variant, resample=s,
                                     profile_auroc=p, drug_auroc=dd))
            rows.append(dict(
                split=split, variant=variant, n_resamples=len(resamples),
                profile_auroc=float(np.nanmean(profs)), profile_sd=float(np.nanstd(profs)),
                drug_auroc=float(np.nanmean(drugaucs)), drug_sd=float(np.nanstd(drugaucs))))
            print(f"  {split:9s} {variant:12s} "
                  f"profile={np.nanmean(profs):.4f}±{np.nanstd(profs):.4f}  "
                  f"drug={np.nanmean(drugaucs):.4f}±{np.nanstd(drugaucs):.4f}", flush=True)
            write(rows, per_rows)  # incremental save so a late crash preserves completed cells

    write(rows, per_rows)


def write(rows, per_rows=None):
    tag = f"_{args.tag}" if args.tag else ""
    csv = REPO / f"results/tables/P_tox_finetune_multidcp{tag}.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    if per_rows:
        pd.DataFrame(per_rows).to_csv(
            REPO / f"results/tables/P_tox_finetune_multidcp{tag}_per_resample.csv", index=False)


if __name__ == "__main__":
    main()
