# v0.4 DILI Downstream — Data and Code Manifest

**Version:** v0.4-rc1 (scaffolding)
**Generated:** 2026-05-05
**Last updated:** 2026-05-05 (scaffolding)

This manifest is the **single source of truth for what data, code, and weights are inside v0.4**. Update it at the end of every phase: Phase 1 fills in DILIst SHAs, Phase 3 fills in upstream checkpoint paths, Phase 4 fills in cache paths, etc. The data-path test (`tests/test_data_paths.py`) reads from this implicitly — keep them in sync.

---

## Source code (SHA-pinned)

| File | Source repo | Source SHA | Purpose |
|---|---|---|---|
| `src/models/upstream/multidcp_pdg.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | Baseline MultiDCP-PDG (Conditions B, E) |
| `src/models/upstream/multidcp_pdgrapher_fusion.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | StructuredMoE wrapper (Conditions C, F) |
| `src/models/upstream/fusion_moe_models.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | `StructuredSparseMoE` class definition |
| `src/models/upstream/neural_fingerprint.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | Drug encoder (NeuralFingerprint, internal to MultiDCP) |
| `src/models/upstream/drug_gene_attention.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | Drug-gene attention block |
| `src/models/upstream/multi_head_attention.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | MHA |
| `src/models/upstream/positionwide_feedforward.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | FFN block |
| `src/models/upstream/graph_degree_conv.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | Graph conv for fingerprint |
| `src/models/upstream/ltr_loss.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | LtR / homophily losses |
| `src/models/upstream/loss_utils.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | Generic loss utilities |
| `src/models/upstream/scheduler_lr.py` | `/raid/home/joshua/projects/MultiDCP` | `871b8de0332045a3ad5ab3a39e689014317dadfe` | LR scheduler helpers |

---

## Data files (SHA256-pinned)

### Raw downloads (Phase 1)

| File | Path | SHA256 | Source | Notes |
|---|---|---|---|---|
| DILIst | `data/raw/DILIst/dilist.xlsx` | `4331ee9d16ae7641488161e4dc2c603c29e06baa8667dd16e7f3d635366e7e5e` | FDA NCTR LTKB ([landing](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-severity-and-toxicity-dilist-dataset), [direct](https://www.fda.gov/media/160597/download?attachment)) | "DILIst Supplementary Table" — 1,279 drugs × 4 cols (`DILIST_ID`, `CompoundName`, `DILIst Classification` [1/0 binary], `Routs of Administration`). **No severity columns in this release** — use DILIrank 2.0 below for M10 stratification. Downloaded 2026-05-05. |
| DILIrank 2.0 | `data/raw/DILIrank/dilirank.xlsx` | `1ca1352ff727af68e68e250eae2ed775bca8492335140ac0afd2233248694993` | FDA NCTR LTKB ([landing](https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset), [direct](https://www.fda.gov/media/113052/download?attachment)) | 1,337 drugs × 6 cols across two sheets (`version 2`, `version 1`). **Header is on row index 1**, not 0 — load with `pd.read_excel(..., sheet_name='version 2', header=1)`. Severity columns: `SeverityClass` (numeric), `vDILI-Concern` (text 4-class: `vMOST-DILI-concern`, `vLESS-DILI-concern`, `vNo-DILI-concern`, `vAmbiguous-DILI-concern`). Downloaded 2026-05-05. |

### Processed (Phase 1+)

| File | Path | Producer | Consumer |
|---|---|---|---|
| Canonical DILI table | `data/processed/dili_canonical.csv` | Phase 1 | Phase 2, 5 |
| Resolution failures log | `data/processed/dili_smiles_resolution_failures.csv` | Phase 1 | manual review |
| Unified scaffold split | `data/splits/unified_dili_aware_scaffold.json` | Phase 2 | Phase 3 (upstream), Phase 5 (downstream) |
| Cluster split | `data/splits/unified_dili_aware_cluster.json` | Phase 2 | Phase 6 (M8 robustness) |
| TDC-DILI split | `data/splits/tdc_dili_scaffold.json` | Phase 2 | Phase 5 (γ secondary) |

### Existing (NOT produced by v0.4 — referenced from MultiDCP repo)

| File | Path | Notes |
|---|---|---|
| MultiDCP train pickle | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl` | PDG perturbation data, 9 cells × ~10,717 genes |
| MultiDCP diseased pickle | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_diseased_brddrugfiltered.pkl` | Diseased baseline for DE computation |
| LINCS L1000 level 5 | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pert_transcriptom/GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx` | 12,328 genes (978 landmark + 11,350 BING-imputed); subset to PDG ~10,717 for Condition D |
| DrugBank (SMILES source) | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/drugbank_data/` | SMILES resolution for DILIst drug names |
| ehill viability data | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/ehill_data/` | Source for Phase 6 / M7 Szalai viability ablation |

---

## Trained checkpoints (Phase 3 — populated after upstream training)

Path convention: `/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream/trained_models/{condition}_seed{seed}/best.pt`

Checkpoints are **gitignored** (`*.pt` rule) so they don't bloat the repo. Their existence + integrity is verified by `tests/test_data_paths.py` once Phase 3 completes.

| Condition | Variant | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| B | Frozen MultiDCP-PDG baseline | _Phase 3_ | _Phase 3_ | _Phase 3_ |
| C | Frozen MultiDCP-CheMoE-StructuredMoE | _Phase 3_ | _Phase 3_ | _Phase 3_ |
| E | DILI-tuned MultiDCP-PDG | _Phase 3_ | _Phase 3_ | _Phase 3_ |
| F | DILI-tuned MultiDCP-CheMoE | _Phase 3_ | _Phase 3_ | _Phase 3_ |

---

## External resources (public encoders + libs — Phase 4 pinning)

| Resource | Pin | Notes |
|---|---|---|
| ChemBERTa | `DeepChem/ChemBERTa-77M-MTR` (HF rev: _Phase 4_) | Frozen, d_chem = 768 |
| MolFormer | `ibm/MoLFormer-XL-both-10pct` (HF rev: _Phase 4_) | Frozen, d_chem = 768 |
| GIN | _Phase 4: pin specific repo + commit_ | Frozen, d_chem = 300 (default 5-layer) |
| UniMol | _Phase 4: pin unimol_tools version_ | Frozen, d_chem = 512 |
| pytdc | _Phase 1: record installed version_ | TDC-DILI split |
| gseapy | _Phase 8: record installed version_ | M11 GSEA |
| captum | _Phase 8: record installed version_ | M11 integrated gradients |
| transformers | _Phase 4: record installed version_ | ChemBERTa, MolFormer host |
| torch_geometric | _Phase 4: record installed version_ | GIN backend |
| unimol_tools | _Phase 4: record installed version_ | UniMol wrapper |

---

## Compute config

| Item | Value |
|---|---|
| WandB entity / project | `joshroll` / `DILI_Downstream_v04` |
| WandB run group | `dili_downstream` |
| Conda env | `dili_v04_env` (cloned from `mdcp_env` on _Phase 1 setup_) |
| GPU policy | Auto-detect via `scripts/auto_select_gpus.py` — **always reserve 1 free** |
| DataLoader workers | Cap at 4–6 (shared CPU constraint) |
| Bootstrap resamples (M6 paired CIs) | 10,000 |
| Pearson sanity threshold (M3) | 0.3 averaged across cell lines |

---

## Out-of-scope (deferred to v0.5+)

These items are **referenced** from the source-of-truth doc but explicitly excluded from v0.4. Listed here so the manifest doesn't go stale when v0.5 starts.

- Hepatocyte cell-line training data (Open TG-GATEs PHH, DrugMatrix rat hepatocytes, HEPG2 LINCS as 10th cell line)
- Tasks 1–4 from broader downstream suite: TDC ADME, PK multitask, MultiDCP cell viability, PLI binding affinity
- Kidney/heart/brain organ arms (DICTrank, DIRIL, Zhao et al. neurotoxicity)
- Joint training Design Y (encoder-in-upstream)
- Nested 5-outer × 3-inner CV in Phase 9 (optional, per Cattebeke et al. 2025 alignment)

---

## Update log

| Date | Phase | Change |
|---|---|---|
| 2026-05-05 | scaffolding | Initial template — source code SHAs pinned, all data/checkpoint rows pending |
