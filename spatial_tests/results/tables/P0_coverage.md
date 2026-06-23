# P0 Coverage Report: Visium Gene-Space vs MultiDCP 10,716-Gene Space

**MultiDCP gene space:** 10716 genes (source: `pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv`)

**Halt Gate 1:** Coverage > 0.80 required for every whole-transcriptome basal Visium dataset.

Panels (whole_transcriptome=False) and non-input entries are informational only — low coverage is expected for ~313/500/960-gene panels.

**Coverage column:** fraction of the human MultiDCP 10,716 genes present in each dataset's `.var_names`. Human datasets must exceed 0.80 (Halt Gate 1). Rodent datasets show low human-symbol coverage by design (mouse/rat gene symbols differ); their gate is genome-scale check (n_genes > 10,000). Cross-species alignment is handled via the ortholog map (Phase 2).

| dataset | organ | species | platform | n_genes | coverage (human) | gate |
|---------|-------|---------|----------|---------|------------------|-----------|
| yu2022_liver | liver | human | Visium | 36592 | 99.8% | PASS |
| andrews_liver | liver | human | Visium | 33514 | 99.5% | PASS |
| lake_kpmp_kidney | kidney | human | Visium | 33514 | 99.5% | PASS |
| abedini_kidney | kidney | human | Visium | N/A | N/A | N/A (no readable matrix) |
| canela_kidney | kidney | human | Visium | N/A | N/A | N/A (no readable matrix) |
| maynard_dlpfc | brain | human | Visium | N/A | N/A | N/A (no readable matrix) |
| chen_brain_mtg | brain | human | Visium | 36601 | 99.8% | PASS |
| kuppe_heart | heart | human | Visium | 15730 | 83.4% | PASS |
| kanemaru_heart | heart | human | Visium | N/A | N/A | N/A (EGA controlled-access; usable_as_input=False) |
| gse280652_apap_liver | liver | mouse | Visium | 32245 | 0.1% | INFO (rodent validation n_genes=32245) |
| gse272564_apap_liver | liver | mouse | Visium | 32245 | 0.1% | INFO (rodent validation n_genes=32245) |
| gse272564_mouse_liver_ctrl | liver | mouse | Visium | 32245 | 0.1% | PASS (rodent n_genes=32245) |
| gse252772_mouse_kidney | kidney | mouse | Visium | N/A | N/A | N/A (no readable matrix) |
| gse233983_mouse_brain | brain | mouse | Visium | 32245 | 0.1% | PASS (rodent n_genes=32245) |

## Gate Status

**Halt Gate 1: NOT FIRED.** All whole-transcriptome basal Visium datasets with readable feature matrices pass their respective gate:
- Human: coverage > 0.80 of MultiDCP 10,716 human symbols.
- Rodent: genome-scale gene count (n_genes > 10,000).
