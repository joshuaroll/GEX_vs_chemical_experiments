"""L1000-landmark gene overlap with each organ's spatial measured gene panel.

Answers the collaborator question: of the 978 L1000 landmark genes (the genes
MultiDCP predicts and the measured Wang/Li DE is defined over), how many are
actually present in each organ's spatial Visium transcriptome?

978 landmarks = GSE92742 gene_info rows with pr_is_lm == 1.
Organ measured genes = .var_names of the primary human Visium dataset per organ,
read with the same readers as report_coverage.py (real archives, no synthetic lists).
"""
from __future__ import annotations
import pathlib, sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.report_coverage import _read_genes_for_slug, RAW  # reuse readers

GENE_INFO = pathlib.Path(
    "/raid/home/joshua/data/L1000_and_CMap/GSE92742_Broad_LINCS_gene_info.txt"
)

# primary human Visium dataset per organ (the ones used in EDA / three-way)
PRIMARY = {
    "liver":  "yu2022_liver",
    "kidney": "lake_kpmp_kidney",
    "heart":  "kuppe_heart",
    "brain":  "chen_brain_mtg",
}


def landmark_symbols() -> set[str]:
    syms = set()
    with open(GENE_INFO) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_sym, i_lm = header.index("pr_gene_symbol"), header.index("pr_is_lm")
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if p[i_lm] == "1":
                syms.add(p[i_sym])
    return syms


def main() -> None:
    lm = landmark_symbols()
    print(f"L1000 landmark genes (pr_is_lm==1): {len(lm)}\n")

    rows = []
    for organ, slug in PRIMARY.items():
        genes = _read_genes_for_slug(slug, RAW / slug)
        if not genes:
            print(f"  {organ:6s} ({slug}): no readable matrix")
            rows.append((organ, slug, None, None, None))
            continue
        gset = set(genes)
        inter = lm & gset
        n_in, frac = len(inter), len(inter) / len(lm)
        missing = sorted(lm - gset)
        print(f"  {organ:6s} ({slug}): panel={len(gset)} genes | "
              f"landmark overlap = {n_in}/{len(lm)} ({frac:.1%}) | "
              f"missing={len(missing)}")
        rows.append((organ, slug, len(gset), n_in, frac))

    # markdown table
    L = ["# L1000-landmark overlap with organ spatial gene panels", "",
         f"Landmarks = {len(lm)} L1000 genes (GSE92742 `pr_is_lm==1`); the gene set "
         "MultiDCP predicts and the measured Wang/Li DE is defined over.", "",
         "Organ panel = `.var_names` of the primary human Visium dataset (whole-transcriptome).", "",
         "| Organ | Dataset | Panel size | Landmark overlap | % of 978 |",
         "|---|---|---|---|---|"]
    for organ, slug, panel, n_in, frac in rows:
        if panel is None:
            L.append(f"| {organ} | {slug} | N/A | N/A | N/A |")
        else:
            L.append(f"| {organ} | {slug} | {panel:,} | {n_in}/{len(lm)} | {frac:.1%} |")
    out = REPO / "results/tables/P1_landmark_overlap.md"
    out.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
