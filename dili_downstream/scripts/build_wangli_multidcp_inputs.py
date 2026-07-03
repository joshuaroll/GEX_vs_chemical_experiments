#!/usr/bin/env python
"""Build the cached MultiDCP finetune input table for the Wang/Li 6,000 DILI benchmark.

Faithful to the ORIGINAL pretrained MultiDCP encoding:
  - basal `input_cell_gex` = adjusted_ccle_tcga_ad_tpm_log2.csv row per cell_id (978-d, file column order).
  - dose one-hot = 6-way over the canonical LINCS doses, indexed in the lexicographically sorted
    string order the original `data_utils` used (dict(zip(sorted(set(...)), range))). Each Wang/Li dose
    is mapped to the nearest canonical dose in log10-concentration space.
  - predicted feature downstream = MultiDCP_AE perturbed output [B,977] directly (LINCS L5 MODZ is a
    z-scored differential, so the model output IS predicted DE natively; no basal subtraction).

Inputs already resolved by Phase 1: data/processed/wangli_profiles.csv (SMILES, cell_id, dose_um, label).
Drops profiles whose cell_id lacks a basal row. Also joins the measured L5 MODZ (978-d, Entrez order)
from wangli_6000_landmark.npz by profile_id for an optional in-harness measured reference arm.

Output: data/processed/wangli_multidcp_finetune.npz
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
BASAL = "/raid/home/joshua/projects/MultiDCP/MultiDCP/data/adjusted_ccle_tcga_ad_tpm_log2.csv"
PROFILES = REPO / "data/processed/wangli_profiles.csv"
MEASURED = REPO / "data/processed/wangli_6000_landmark.npz"
OUT = REPO / "data/processed/wangli_multidcp_finetune.npz"

# The 6 canonical LINCS doses the original MultiDCP was trained on (DATA_FILTER in multidcp_ae.py).
# One-hot index = position in the lexicographically sorted STRING list, matching data_utils'
# `dict(zip(sorted(set(pert_idose)), range(...)))`.
CANON_STRINGS = ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]
CANON_VAL = {"0.04 um": 0.04, "0.12 um": 0.12, "0.37 um": 0.37,
             "1.11 um": 1.11, "3.33 um": 3.33, "10.0 um": 10.0}
SORTED_STRINGS = sorted(CANON_STRINGS)           # ['0.04 um','0.12 um','0.37 um','1.11 um','10.0 um','3.33 um']
STR_TO_IDX = {s: i for i, s in enumerate(SORTED_STRINGS)}
CANON_ARR = np.array([CANON_VAL[s] for s in SORTED_STRINGS])   # values aligned to one-hot index


def dose_onehot(dose_um: float) -> np.ndarray:
    """6-way one-hot: map dose (uM) to nearest canonical dose in log10 space, at that dose's sorted idx."""
    oh = np.zeros(6, dtype=np.float64)
    j = int(np.argmin(np.abs(np.log10(CANON_ARR) - np.log10(dose_um))))
    oh[j] = 1.0
    return oh


def murcko(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m) or smiles
    except Exception:
        return smiles


def main():
    p = pd.read_csv(PROFILES)
    print(f"loaded {len(p)} Phase-1 profiles (100% SMILES-resolved)")

    # basal: load once, keep only needed cells
    basal_df = pd.read_csv(BASAL, index_col=0)
    basal_cells = set(basal_df.index.astype(str))
    keep = p["cell_id"].astype(str).isin(basal_cells)
    dropped = p.loc[~keep, "cell_id"].value_counts().to_dict()
    p = p[keep].reset_index(drop=True)
    print(f"kept {len(p)} profiles with basal coverage; dropped cells: {dropped}")

    # per-profile basal matrix (float64, 978, basal-file column order)
    gene_cols = list(basal_df.columns)
    basal_mat = basal_df.loc[p["cell_id"].astype(str).values, :].to_numpy(np.float64)
    assert basal_mat.shape == (len(p), 978), basal_mat.shape
    assert np.isfinite(basal_mat).all(), "non-finite basal"

    # dose one-hot
    dose_oh = np.vstack([dose_onehot(float(d)) for d in p["dose_um"].values])
    # report how many map exactly to a canonical dose
    exact = sum(any(abs(float(d) - v) < 1e-6 for v in CANON_ARR) for d in p["dose_um"].values)
    print(f"dose: {exact}/{len(p)} profiles land exactly on a canonical dose; rest mapped to nearest (log10)")

    # scaffolds (per unique SMILES, then broadcast)
    uniq = {s: murcko(s) for s in p["smiles"].unique()}
    scaffold = p["smiles"].map(uniq).values

    # optional measured L5 MODZ (Entrez order), joined by profile_id == sig_id
    meas = np.load(MEASURED, allow_pickle=True)
    sig_to_row = {s: i for i, s in enumerate(meas["sig_ids"])}
    X = meas["X"]
    measured = np.full((len(p), X.shape[1]), np.nan, dtype=np.float32)
    hit = 0
    for i, pid in enumerate(p["profile_id"].values):
        r = sig_to_row.get(pid)
        if r is not None:
            measured[i] = X[r]
            hit += 1
    print(f"measured L5 MODZ joined for {hit}/{len(p)} profiles (Entrez gene order)")

    np.savez_compressed(
        OUT,
        profile_id=p["profile_id"].values.astype(str),
        smiles=p["smiles"].values.astype(str),
        compound_name=p["compound_name"].values.astype(str),
        cell_id=p["cell_id"].values.astype(str),
        dose_um=p["dose_um"].values.astype(np.float64),
        dose_onehot=dose_oh,
        cell_basal=basal_mat,
        scaffold=scaffold.astype(str),
        label=p["dili_binary"].values.astype(np.int64),
        usage=p["usage"].values.astype(str),
        time_h=p["time_h"].values.astype(np.int64),
        measured_modz=measured,
        basal_gene_cols=np.array(gene_cols, dtype=object),
    )
    print(f"wrote {OUT}")
    print(f"  N={len(p)}  pos={int(p['dili_binary'].sum())} neg={int((1-p['dili_binary']).sum())}")
    print(f"  usage: {p['usage'].value_counts().to_dict()}")
    print(f"  n_compounds={p['compound_name'].nunique()} n_scaffolds={len(set(scaffold))} n_cells={p['cell_id'].nunique()}")


if __name__ == "__main__":
    main()
