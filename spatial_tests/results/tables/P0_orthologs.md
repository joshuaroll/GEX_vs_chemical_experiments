# P0 Ortholog Report: Human↔Mouse↔Rat One-to-One Map

**Source:** Ensembl BioMart release 116 (query date: 2026-06-20)
**Raw TSV:** `data/raw/spatial/biomart/orthologs_raw_116_20260620.tsv`
**Filtered TSV:** `data/processed/spatial/orthologs_h_m_r_one2one.tsv`
**Cross-species discipline:** XC-08 (one2one only; many2many DROPPED)

## Metrics

| Metric | Value |
|--------|-------|
| n_input | 219938 |
| n_one2one | 15956 |
| dropped_fraction | 92.75% |

## Notes

The 92.75% dropped fraction is **expected and correct**. BioMart returns all human
genes including those with no mouse/rat homolog, many-to-many paralogs, and rows
with empty Ensembl IDs. Only rows where BOTH mouse AND rat `orthology_type ==
"ortholog_one2one"` are kept; this strict mutual one2one filter retains ~15,956
unambiguous cross-species gene pairs.

The 15,956 one2one pairs is the correct order of magnitude for human-mouse-rat
ortholog discipline (typically ~15k–17k for strict one2one in all three species).

## Provenance

- **Ensembl release:** 116 (confirmed from BioMart registry `ensembl_mart_116`)
- **Query date:** 2026-06-20 (XC-10 time-leakage discipline)
- **Raw TSV SHA256:** `607dd457581753ce2247905633f4fd8def813bfb42fba44322ca8ce30b7a5551`
- **One2one TSV SHA256:** `b25404a9d79dde3dadd94645426556bbe2df2ffc21cdf12a2f5bd05aed7ec12c`
- **Build script:** `scripts/build_orthologs.py` (BioMart REST fetch + cache)
- **Filter library:** `src/spatial/orthology.py` (`build_one2one_orthologs`)
