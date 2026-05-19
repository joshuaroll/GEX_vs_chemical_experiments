# MANIFEST — Multi-Head MultiDCP DILI

Data sources, model checkpoints, and code SHAs pinned for reproducibility.

## Data

| Item | Path | SHA256 | Date pinned |
|---|---|---|---|
| E-Hill train | `/raid/home/joshua/data/MultiDCP/data/ehill_data/high_confident_data_train.csv` | _Phase 0: record_ | _TBD_ |
| E-Hill dev | `.../high_confident_data_dev.csv` | _Phase 0: record_ | _TBD_ |
| E-Hill test | `.../high_confident_data_test.csv` | _Phase 0: record_ | _TBD_ |
| LINCS PDG-filtered | `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl` | _Phase 0: record_ | _TBD_ |
| DILIst canonical | `../dili_downstream/data/processed/dili_canonical.csv` | _Phase 0: record_ | _TBD_ |
| MolFormer model | HF `ibm/MoLFormer-XL-both-10pct` | _Phase 3: record_ | _TBD_ |
| CCLE HepG2 baseline | (UNUSED — design uses 9-cell mean-pool) | — | — |

## Code

| Item | Path | SHA / version | Date pinned |
|---|---|---|---|
| MultiDCP upstream | `/raid/home/joshua/projects/MultiDCP` | _Phase 1: record_ | _TBD_ |
| `ehill_multidcp_pretrain.py` | `MultiDCP/ehill_multidcp_pretrain.py` | (same SHA) | _TBD_ |
| `multidcp_ae_balanceloss.py` | `MultiDCP/MultiDCP/models/multidcp_ae_balanceloss.py` | (same SHA) | _TBD_ |
| MolFormer (HF) | `transformers >= 4.40` | _Phase 3: record_ | _TBD_ |
| `dili_v04_env` | conda env | _Phase 0: record `conda list` snapshot_ | _TBD_ |

## Checkpoints

| Item | Path | Date | Halt gate? |
|---|---|---|---|
| MODEL_DOSE | `multihead_dili/results/checkpoints/chkpt_dose.pt` | _Phase 1: record_ | HG1 |
| MODEL_GEX | `multihead_dili/results/checkpoints/chkpt_gex.pt` | _Phase 2: record_ | HG2 |
