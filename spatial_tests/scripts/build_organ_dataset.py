#!/usr/bin/env python
"""Build per-organ 978-gene LINCS training datasets with drug-disjoint splits.

Source: the broad 978-modz LINCS corpus (processed_data_978modz_detplate),
91,657 profiles x 978 L1000 landmark genes, 164 cell lines. It carries paired
treated (x1) and control (x2) profiles in modz space, so DE = x1 - x2 natively
and x2 is a per-profile basal context (input_cell_gex). Cell line = sig prefix.

Per organ we subset to that organ's tissue-of-origin cell lines, then split
DRUG-DISJOINT (no InChIKey in more than one split) into train/dev/test.

Heart has no LINCS cell line -> 0 profiles -> not buildable from this corpus.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import h5py

REPO = Path("/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests")
sys.path.insert(0, str(REPO))
from scripts.three_way_comparison import smi2ikey14  # InChIKey14 for drug-disjoint grouping

H5 = Path("/raid/home/joshua/data/L1000_and_CMap/final_data/processed_data_978modz_detplate_09012024.h5")
OUT = REPO / "data/processed/organ_train"
SEED = 42
FRACS = (0.70, 0.15, 0.15)  # train/dev/test, drug-disjoint

ORGAN_LINES = {
    "liver":  ["PHH", "HEPG2", "JHH5", "HUH7"],
    "kidney": ["HA1E", "HEK293", "HEK293T"],
    "brain":  ["NPC", "NEU", "SHSY5Y", "U251MG", "GI1"],
    "heart":  [],  # no LINCS cardiac line
}


def drug_disjoint_split(ikeys, rng):
    """Assign whole drugs (InChIKey14) to train/dev/test by fraction."""
    uniq = np.array(sorted(set(ikeys)))
    rng.shuffle(uniq)
    n = len(uniq); n_tr = int(FRACS[0] * n); n_dv = int(FRACS[1] * n)
    bucket = {}
    for i, k in enumerate(uniq):
        bucket[k] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
    split = np.array([bucket[k] for k in ikeys])
    return split, len(uniq)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)

    with h5py.File(H5, "r") as f:
        sig = np.array([s.decode() for s in f["sig"][:]])
        cells = np.array([s.split("_")[0] for s in sig])
        smiles = np.array([s.decode() for s in f["canonical_smiles"][:]])
        # x1 treated, x2 control — load lazily per organ to keep memory sane
        n_total = len(sig)
        print(f"corpus: {n_total} profiles, {len(set(cells))} cell lines\n")

        report = ["# Per-organ 978-gene LINCS training datasets (drug-disjoint splits)", "",
                  f"Source: `{H5.name}` | DE = x1(treated) - x2(control), 978 L1000 genes | "
                  f"basal = x2 | split seed {SEED}, drug-disjoint {FRACS}.", "",
                  "| Organ | Cell lines (present) | Profiles | Unique drugs | train/dev/test profiles | train/dev/test drugs |",
                  "|---|---|---|---|---|---|"]

        for organ, lines in ORGAN_LINES.items():
            present = [c for c in lines if c in set(cells)]
            mask = np.isin(cells, present) if present else np.zeros(n_total, bool)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                print(f"[{organ}] 0 profiles (lines={lines}) — SKIP (no LINCS data)")
                report.append(f"| {organ} | none | 0 | 0 | — | — |")
                continue

            smi_o = smiles[idx]
            ikeys = np.array([smi2ikey14(s) or "NA" for s in smi_o])
            keep = ikeys != "NA"
            idx, smi_o, ikeys = idx[keep], smi_o[keep], ikeys[keep]

            split, n_drugs = drug_disjoint_split(ikeys, rng)
            x1 = f["x1"][:][idx].astype(np.float32)   # treated target
            x2 = f["x2"][:][idx].astype(np.float32)   # control / basal

            np.savez(OUT / f"{organ}.npz",
                     x1=x1, x2=x2, smiles=smi_o, ikey=ikeys, cell=cells[idx],
                     split=split, genes=np.array([g.decode() for g in f["genes"][:]]))

            def cnt(s): return int((split == s).sum())
            def dcnt(s): return len(set(ikeys[split == s]))
            row_prof = f"{cnt('train')}/{cnt('dev')}/{cnt('test')}"
            row_drug = f"{dcnt('train')}/{dcnt('dev')}/{dcnt('test')}"
            print(f"[{organ}] lines={present} | {len(idx)} profiles, {n_drugs} drugs | "
                  f"profiles {row_prof} | drugs {row_drug}")
            # leakage check
            tr_d, dv_d, te_d = (set(ikeys[split==s]) for s in ("train","dev","test"))
            assert not (tr_d & dv_d) and not (tr_d & te_d) and not (dv_d & te_d), "DRUG LEAKAGE"
            report.append(f"| {organ} | {', '.join(present)} | {len(idx)} | {n_drugs} | "
                          f"{row_prof} | {row_drug} |")

    report += ["", "## Notes",
               "- Drug-disjoint verified: no InChIKey14 appears in more than one split (assert passed).",
               "- `x2` is the per-profile control (paired basal) -> used as input_cell_gex; target = `x1`.",
               "- Heart has no LINCS cell line; not buildable from this corpus (needs external cardiac data).",
               "- Lines are tissue-of-origin; PHH (primary hepatocytes) and NPC/NEU (iPSC neural) are the "
               "most organ-faithful; HEPG2/JHH5/HUH7 are hepatoma lines (volume vs fidelity tradeoff)."]
    (REPO / "results/tables/P3_organ_datasets.md").write_text("\n".join(report) + "\n")
    print(f"\nwrote {OUT}/*.npz + results/tables/P3_organ_datasets.md")


if __name__ == "__main__":
    main()
