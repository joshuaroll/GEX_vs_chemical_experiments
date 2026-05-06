# Upstream model files — SHA-pinned clone

These files are **frozen clones** from the live MultiDCP repository, pinned to a specific commit so v0.4 reproducibility is protected against upstream edits.

## Source repository

- **Path:** `/raid/home/joshua/projects/MultiDCP`
- **Subdir:** `MultiDCP/models/`
- **Pinned SHA:** `871b8de0332045a3ad5ab3a39e689014317dadfe`
- **Commit message:** `Merge pull request #3 from XieResearchGroup/master_lab`
- **Cloned:** 2026-05-05

## File-to-architecture mapping (for v0.4)

| File | Used by | Role in v0.4 |
|---|---|---|
| `multidcp_pdg.py` | Conditions B, E | Baseline MultiDCP-PDG (no MoE), 10,716-gene PDG scope |
| `multidcp_pdgrapher_fusion.py` | Conditions C, F | Wrapper that uses `StructuredSparseMoE` as fusion layer |
| `fusion_moe_models.py` | Conditions C, F (via wrapper) | `StructuredSparseMoE` class — factorized cross-modality MoE, 4 experts, top-k=1, L1 sparsity on gates |
| `neural_fingerprint.py` | All upstream | Drug encoder (1024 fingerprint → 128) — internal, not the downstream-classifier chemical encoder |
| `drug_gene_attention.py` | All upstream | Drug-gene attention block |
| `multi_head_attention.py` | All upstream | Generic MHA |
| `positionwide_feedforward.py` | All upstream | FFN block |
| `graph_degree_conv.py` | `neural_fingerprint.py` | Graph convolution for fingerprint |
| `ltr_loss.py` | All upstream | Listwise/pointwise loss utilities (homophily, MSE+homophily, ranknet etc.) |
| `loss_utils.py` | All upstream | Generic loss utilities |
| `scheduler_lr.py` | All upstream | LR scheduler helpers |

## Important: do NOT edit these files in v0.4

If the architecture needs modifications (e.g., to support the joint DILI head for conditions E/F), create **new files in `src/models/`** that *import from* `upstream/`, rather than editing the cloned upstream files. The pinned SHA is the contract.

If the upstream MultiDCP repo evolves and v0.4 needs a newer revision, that's a deliberate **re-clone with a new SHA** — record both the old and new SHAs in `MANIFEST.md` and document the diff in a v0.4 changelog entry.

## Verifying integrity

```bash
# Re-derive SHAs at any time
cd /raid/home/joshua/projects/MultiDCP && git rev-parse HEAD
# Should match: 871b8de0332045a3ad5ab3a39e689014317dadfe

# File-level checksum
sha256sum /raid/home/joshua/projects/PDGrapher_Baseline_Models/experiments/dili_downstream/src/models/upstream/*.py
```

If checksums drift after a fresh clone (which they shouldn't — files are read-only from the v0.4 perspective), investigate before continuing any phase.
