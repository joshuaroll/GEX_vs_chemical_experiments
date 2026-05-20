# Phase 3: MolFormer + Stage-2 Feature Caching — Research

**Phase:** 03 — MolFormer download + Stage-2 feature caching
**Researched:** 2026-05-20
**Researcher:** Orchestrator (inline — all facts confirmed by direct file inspection)

---

## RESEARCH COMPLETE

---

## Summary

Phase 3 requires three distinct components: (1) a frozen MolFormer SMILES→768-dim encoder wrapper, (2) a Stage-2 inference script that queries MODEL_DOSE and MODEL_GEX at all 10 cells for each of 1,118 DILIst drugs and mean-pools the results, and (3) a parquet cache output with feature stats summary. The main research findings concern the HF model ID correction, the actual GEX feature dimension (918, not 978), the `trust_remote_code=True` requirement for MolFormer, the canonical cell baseline file for MODEL_GEX inference, and the `dili_canonical.csv` SMILES completeness.

---

## 1. MolFormer HuggingFace Model

### Model ID Correction (IMPORTANT)

The design doc and CONTEXT.md specify `ibm/MoLFormer-XL-both-10pct`. The **actual HuggingFace ID is `ibm-research/MoLFormer-XL-both-10pct`** (with a dash). Using the wrong ID will cause a 404 on download. Plan must use the corrected ID.

**Confirmed via hub API:**
```
Model ID:    ibm-research/MoLFormer-XL-both-10pct
SHA:         7b12d946c181a37f6012b9dc3b002275de070314
hidden_size: 768  ← embedding output dimension
Tags:        ['transformers', 'pytorch', 'safetensors', 'molformer', 'chemistry', 'custom_code', ...]
```

### trust_remote_code Audit

**Result: `trust_remote_code=True` IS REQUIRED.** The model uses custom classes:
```json
"auto_map": {
  "AutoConfig": "configuration_molformer.MolformerConfig",
  "AutoModel": "modeling_molformer.MolformerModel",
  "AutoModelForMaskedLM": "modeling_molformer.MolformerForMaskedLM"
}
```

The model ships three custom Python files: `configuration_molformer.py`, `modeling_molformer.py`, `tokenization_molformer.py`. These are **Apache-2.0 licensed** IBM Research code. Security assessment:
- Standard HF model: IBM Research is a trusted publisher.
- No exec/eval calls or shell commands in the public model card.
- HF model tag confirms `custom_code` (expected for MolFormer).
- **Decision: Use `trust_remote_code=True`.** Document this in MANIFEST.md with the reason.

### Loading Pattern (confirmed working with transformers 5.8.1, torch 2.6.0+cu124)

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ibm-research/MoLFormer-XL-both-10pct",
    trust_remote_code=True,
)
model = AutoModel.from_pretrained(
    "ibm-research/MoLFormer-XL-both-10pct",
    trust_remote_code=True,
    deterministic_eval=True,
)
model.eval()  # freeze in eval mode — no grad updates
```

**Output extraction:** Use the `pooler_output` (768-dim CLS embedding) from the model's forward pass:
```python
with torch.no_grad():
    outputs = model(**inputs)
    embedding = outputs.pooler_output  # shape: [batch, 768]
```

If `pooler_output` is None (some model configs), fall back to `last_hidden_state[:, 0, :]` (CLS token).

---

## 2. DILIst Canonical Data

**File:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream/data/processed/dili_canonical.csv`

**Confirmed:**
```
Shape:    (1118, 9)
Columns:  ['pert_id', 'drug_name', 'smiles', 'canonical_smiles', 'scaffold',
           'dili_binary', 'dili_severity', 'in_lincs', 'in_pdg']
SMILES completeness: 1118/1118 (no missing SMILES)
Label distribution: {1: 685, 0: 433} (DILI-positive/negative)
```

**SMILES column to use:** `canonical_smiles` (prefer over `smiles` — RDKit-standardized).

---

## 3. Feature Dimension — CRITICAL CORRECTION

The design doc and CONTEXT.md state `feat_gex = 978-dim` and total = 1,747-dim. **This is incorrect.**

**Root cause:** MultiDCP landmark gene set has 977 genes (gene_vector.csv, 977 rows). The LINCS PDG-filtered parquet has 10,716 gene columns. The overlap between these sets is **918 genes**, not 977.

```
gene_vector.csv:        977 landmark genes (index = gene names)
lincs_train_safe.parquet: 10,716 gene columns
Overlap:                 918 genes → MODEL_GEX num_gene = 918
```

Same applies to the cell baseline file:
```
pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv:
  Shape: (10, 10,716) — 10 cells × 10,716 genes
  Overlap with landmark genes: 918
  → Baseline vectors for inference: 918-dim per cell
```

**Corrected feature vector:**
| Component | Dim | Source |
|-----------|-----|--------|
| `feat_dose` | 1 | MODEL_DOSE mean-pooled over 10 cells |
| `feat_gex` | 918 | MODEL_GEX mean-pooled over 10 cells, 918 landmark genes |
| `feat_embed` | 768 | MolFormer frozen embedding |
| **Total** | **1,687** | **NOT 1,747** |

**Implementation requirement:** Phase 3 must load `chkpt_gex.pt` and read `model_params['num_gene']` to confirm the actual dimension, then use that value consistently. Do not hardcode 978 anywhere.

---

## 4. Cell Baseline for MODEL_GEX Inference

**File:** `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv`

This is the canonical diseased/baseline expression file used by MultiDCP's datareader at training time. It has:
```
Shape: (10, 10716)
Index: ['A375', 'A549', 'BT20', 'HA1E', 'HELA', 'HT29', 'MCF7', 'MDAMB231', 'PC3', 'VCAP']
```

**At inference:** Subset to the 918 landmark gene overlap (same filter as training). Result: `baseline_tensor[cell] → np.array of shape [918]`.

```python
diseased_df = pd.read_csv(DISEASED_PATH, index_col=0)
# Subset to landmark gene overlap (same as training filter)
baseline_per_cell = {cell: torch.tensor(diseased_df.loc[cell, landmark_genes].values,
                                        dtype=torch.float32) for cell in CELLS_10}
```

---

## 5. Canonical Dose for Inference (OQ-2 Resolution)

**From Phase 2 research (confirmed):** All rows in `lincs_train_safe.parquet` have `pert_idose = 'x'`. This means the MODEL_GEX was trained with dose-aggregated signatures only — `pert_idose_input_dim` at training = 1 (single dose level `'x'`).

**For MODEL_DOSE:** The E-Hill train data uses E-Hill scalars aggregated per drug (one value per drug-cell combination). The `pert_idose` concept doesn't apply in the same way — MODEL_DOSE predicts E-Hill, not GEX. At inference, MODEL_DOSE gets: drug SMILES graph + cell baseline + gene topology. The "dose" is implicit in the model's E-Hill training.

**Inference strategy:**
- MODEL_GEX: pass `pert_idose_feature = torch.tensor([1.0]).float()` (single `'x'` level) per row, matching training.
- MODEL_DOSE: same architecture as MODEL_GEX fork (both MultiDCP-AE); at inference use same pattern with E-Hill as scalar output.

---

## 6. Inference Data Flow (22,360 Forward Passes)

For each of 1,118 DILIst drugs × 10 cells = 11,180 drug-cell pairs:

```
MODEL_GEX inference (11,180 forward passes):
  Input: drug_graph(SMILES), cell_baseline[cell, :918], gene_topology, pert_idose='x'
  Output: perturbed GEX prediction [918-dim]
  → DE = perturbed_GEX - baseline[cell]  (per DE rule)

MODEL_DOSE inference (11,180 forward passes):
  Input: drug_graph(SMILES), cell_baseline[cell, :918], gene_topology, pert_idose='x'
  Output: E-Hill scalar [1-dim]

Mean-pool:
  feat_gex[drug]   = mean(DE[cell] for cell in 10 cells)     → [918]
  feat_dose[drug]  = mean(E-Hill[cell] for cell in 10 cells) → [1]

MolFormer:
  feat_embed[drug] = MolFormer(canonical_smiles[drug])        → [768]

Concatenate:
  feature[drug] = concat([feat_dose, feat_gex, feat_embed])  → [1,687]
```

---

## 7. MODEL_DOSE Architecture — E-Hill Scalar Output

From Phase 1 research: MODEL_DOSE is a MultiDCP-AE fork that predicts E-Hill (scalar). At inference, it outputs a scalar per drug-cell pair. The `num_gene` in MODEL_DOSE's `model_params` is also likely 918 (same landmark filter). The plan must read `chkpt_dose.pt['model_params']['num_gene']` to confirm.

The model forward pass returns GEX predictions per gene, but the E-Hill head compresses to a scalar:
```python
# From ehill_multidcp_pretrain.py: the model returns a scalar E-Hill prediction
ehill_pred = model(...)  # shape: [batch]  (or [batch, 1])
```

At inference for Phase 3, batch = 1 drug × 1 cell → scalar E-Hill.

---

## 8. Gene Topology and Drug Graph

MODEL_GEX (and MODEL_DOSE) require:
1. **Drug graph:** `drug_feature_dict[drug_name]` from `all_drugs_l1000.csv` (SMILES → Morgan fingerprint or graph)
2. **Gene topology:** `gene_data.gene` tensor from `gene_vector.csv` (shape `[978, 128]` → effectively `[num_gene, 128]`)

At inference for Phase 3:
- Drug graph: use the same drug feature dict loaded from `all_drugs_l1000.csv` (maps drug name or SMILES → graph)
- Gene topology: load `gene_vector.csv`, filter to the 918 overlapping genes → `[918, 128]` tensor

**SMILES→drug_name mapping:** `dili_canonical.csv` has `drug_name` and `canonical_smiles`. The MultiDCP drug lookup uses `pert_id` (drug name string). Phase 3 must map `dili_canonical.drug_name → all_drugs_l1000.csv lookup` — if a drug name is missing from the drug file, the SMILES can be used directly to compute the fingerprint on-the-fly.

---

## 9. Package Versions (dili_v04_env)

```
torch:        2.6.0+cu124  (8 GPUs available)
transformers: 5.8.1
```

MolFormer should work with transformers >= 4.40. `trust_remote_code=True` is required (see section 1).

---

## 10. MANIFEST.md Update

Phase 3 must update MANIFEST.md:
```
| MolFormer model | HF `ibm-research/MoLFormer-XL-both-10pct` | SHA 7b12d946c181a37f6012b9dc3b002275de070314 | 2026-05-20 |
| MolFormer (HF) | transformers 5.8.1 | trust_remote_code=True (Apache-2.0, IBM Research) | 2026-05-20 |
```

---

## 11. Output Schema — dili_features.parquet

```
Rows:    1,118 (one per DILIst drug)
Columns:
  - drug_name             str   (from dili_canonical.drug_name)
  - pert_id               int   (from dili_canonical.pert_id)
  - dili_binary           int   (0/1 label)
  - dili_severity         str   (from dili_canonical.dili_severity)
  - in_lincs              bool
  - in_pdg                bool
  - feat_dose             float (E-Hill mean over 10 cells)
  - feat_gex_{0..917}     float (918 GEX DE dims, mean over 10 cells) — OR stored as a single column of list type
  - feat_embed_{0..767}   float (768 MolFormer dims) — OR stored as a single column of list type
```

**Preferred storage:** Flat columns (one per feature dim) for easy Phase 4 consumption with pandas/sklearn. Total = 6 metadata + 1 + 918 + 768 = 1,693 columns.

---

## 12. Validation Architecture (Nyquist)

Phase 3 produces `dili_features.parquet`. The minimal Nyquist sanity:

```python
df = pd.read_parquet('data/processed/dili_features.parquet')
assert len(df) == 1118, f"Expected 1118 rows, got {len(df)}"
assert df.isnull().sum().sum() == 0, "NaN values found"
feat_cols = [c for c in df.columns if c.startswith('feat_')]
assert len(feat_cols) == 1 + 918 + 768  # 1687 feature cols
```

EMBED-04 requires: no NaN in any feature column + mean/std sanity print per pathway block.
