#!/usr/bin/env python
"""Extract the Li/Tong 2020 (fbioe.2020.562677) exact 6,000-profile x 978-landmark matrix.

Follows their read_gctx_data.ipynb recipe, but slices the GCTX with h5py directly (cmapPy is
fragile under numpy 2). Inputs (all public, already downloaded):
  - data/raw/L1000_DILI/6000_transcriptomic_profiles_id.xlsx  (sig_id, Usage, CompoundName, DILIst.1)
  - data/raw/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz    (pr_gene_id, pr_is_lm)
  - data/raw/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx  (standard MODZ)
Output:
  - data/processed/wangli_6000_landmark.npz  (X [n,978] float32, sig_ids, gene_ids, label, usage)
  - data/processed/wangli_6000_landmark.csv  (sig_id, label, usage) index for reference
"""
from __future__ import annotations
import numpy as np, pandas as pd, h5py
from pathlib import Path

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream")
XLSX = REPO / "data/raw/L1000_DILI/6000_transcriptomic_profiles_id.xlsx"
GINFO = REPO / "data/raw/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz"
GCTX = REPO / "data/raw/GSE92742/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
OUT = REPO / "data/processed"; OUT.mkdir(parents=True, exist_ok=True)


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def main():
    drugs = pd.read_excel(XLSX, sheet_name="supplementary1_all")
    print(f"ID table: {drugs.shape}  cols={list(drugs.columns)}")
    want_sig = drugs["sig_id"].astype(str).tolist()

    gi = pd.read_csv(GINFO, sep="\t", dtype=str)
    landmark = set(gi.loc[gi["pr_is_lm"] == "1", "pr_gene_id"])
    print(f"landmark genes in gene_info: {len(landmark)}")

    with h5py.File(GCTX, "r") as f:
        mat = f["/0/DATA/0/matrix"]
        col_ids = _dec(f["/0/META/COL/id"][:])   # sig_ids
        row_ids = _dec(f["/0/META/ROW/id"][:])   # gene_ids (Entrez as str)
        print(f"gctx matrix shape={mat.shape}  n_col_ids={len(col_ids)} n_row_ids={len(row_ids)}")
        sample_axis = 0 if mat.shape[0] == len(col_ids) else 1
        gene_axis = 1 - sample_axis

        col_pos = {s: i for i, s in enumerate(col_ids)}
        found = [s for s in want_sig if s in col_pos]
        missing = [s for s in want_sig if s not in col_pos]
        print(f"sig_ids found in gctx: {len(found)}/{len(want_sig)}  (missing {len(missing)})")

        lm_rows = np.array([i for i, g in enumerate(row_ids) if g in landmark])
        print(f"landmark rows located in gctx: {len(lm_rows)}")

        sel = np.array([col_pos[s] for s in found])
        order = np.argsort(sel); inv = np.argsort(order)
        sel_sorted = sel[order]
        # h5py fancy-index requires increasing indices on the sliced axis
        if sample_axis == 0:
            block = mat[sel_sorted, :][:, lm_rows]          # (n, 978)
        else:
            block = mat[:, sel_sorted][lm_rows, :].T        # (n, 978)
        X = np.asarray(block, np.float32)[inv]              # restore ID-table order
        gene_ids = row_ids[lm_rows]

    lab_map = dict(zip(drugs["sig_id"].astype(str), drugs["DILIst.1"].astype(int)))
    use_map = dict(zip(drugs["sig_id"].astype(str), drugs["Usage"].astype(str)))
    labels = np.array([lab_map[s] for s in found], int)
    usage = np.array([use_map[s] for s in found])

    np.savez_compressed(OUT / "wangli_6000_landmark.npz",
                        X=X, sig_ids=np.array(found), gene_ids=gene_ids,
                        label=labels, usage=usage)
    pd.DataFrame({"sig_id": found, "label": labels, "usage": usage}).to_csv(
        OUT / "wangli_6000_landmark.csv", index=False)

    print(f"\nX shape={X.shape}  finite={np.isfinite(X).all()}  std={X.std():.3f}")
    print(f"labels: {int((labels==1).sum())} pos / {int((labels==0).sum())} neg")
    u, c = np.unique(usage, return_counts=True)
    print("usage split:", dict(zip(u.tolist(), c.tolist())))
    print(f"wrote {OUT/'wangli_6000_landmark.npz'}")


if __name__ == "__main__":
    main()
