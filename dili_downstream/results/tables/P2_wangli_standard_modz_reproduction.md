# Li/Tong 2020 reproduction on the EXACT dataset (standard MODZ) — halt-gate-1 PASS

**Date:** 2026-07-01
**Result:** their published DILI DNN reproduced to 3 decimals on their own split.

## Exact match

| metric | this reproduction | published (DNN.ipynb) |
|---|---|---|
| Test AUROC | **0.7976** | 0.798 |
| Sensitivity | 0.839 | 0.839 |
| Specificity | 0.603 | 0.603 |
| Accuracy | 0.743 | 0.743 |

Method: their published weights (`optimized_model.h5`) run as a NumPy ELU forward pass (978→512→256→128→64→32→16→8→1, sigmoid) on the exact 1,200-profile Test split, **raw MODZ, no scaling** (a StandardScaler variant gives 0.755, confirming their preprocessing was raw). **Halt-gate-1 PASS** (Test AUROC 0.7976 ∈ [0.78, 0.82]).

## What fixed the earlier 0.51

The prior attempt (P2_wangli_reproduction.md) scored 0.5136 with this same checkpoint because it fed a **Bayesian-shrinkage** MODZ variant (`Bayesian_GSE92742_...n361481x978.h5`) and only recovered 5,558/6,000 sig_ids. Using the **standard** GSE92742 COMPZ.MODZ resolves it: **6,000/6,000 sig_ids recovered**, and the metrics match exactly. The gap was entirely the data variant, not the pipeline.

## Provenance (all public — NO Synapse needed)

The paper's data availability directs to GitHub + GEO; the Synapse ID (`syn22910821`) referenced in `src/data/wangli_loader.py` is not required and was a wrong turn.

| artifact | source | sha256 (kept files) |
|---|---|---|
| profile IDs + labels + split | GitHub `TingLi2016/L1000_DILI` @ `010361f` `6000_transcriptomic_profiles_id.xlsx` | `4b9a8cc3…c40d26` |
| their trained DNN | same repo `optimized_model.h5` | `35e174d5…08fa` |
| gene_info (landmark flags) | GEO GSE92742 `..._gene_info.txt.gz` | `741216cc…711a` |
| sig_info | GEO GSE92742 `..._sig_info.txt.gz` | `19da29c0…1299` |
| expression (standard MODZ) | GEO `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz` (20 GB, 2017-12-13) | downloaded, sliced, then deleted to return disk — re-acquire via `scripts/download_gse92742.sh` |
| **extracted matrix** | `data/processed/wangli_6000_landmark.npz` (6000×978 f32 + sig_ids, gene_ids, label, usage) | `07aa3e69…1ca7` |

## Reproduce from scratch

```bash
git clone https://github.com/TingLi2016/L1000_DILI data/raw/L1000_DILI   # IDs, labels, split, checkpoint
bash scripts/download_gse92742.sh                                          # 20 GB GEO MODZ + metadata
gunzip data/raw/GSE92742/*.gctx.gz
python scripts/extract_gse92742_6000.py         # -> data/processed/wangli_6000_landmark.npz (6000x978)
python scripts/reproduce_wangli_standard_modz.py  # -> Test AUROC 0.7976
```

## Why this matters for the comparison

We now hold the benchmark's exact data (same 978-landmark space as our organ corpus, 974/978 symbol overlap). Two things this unlocks: (1) a faithful 0.798 anchor (halt-gate-1 properly closed with real MODZ, not the Bayesian proxy); (2) re-running our honest drug-disjoint / scaffold-disjoint analysis on *their* 6,000 profiles, so the leakage-inflation finding lands on the benchmark's own data rather than a proxy corpus.
