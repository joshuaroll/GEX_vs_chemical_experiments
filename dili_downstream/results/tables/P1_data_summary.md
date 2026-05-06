# Phase 1 Data Summary

Generated from `data/processed/dili_canonical.csv` (1,118 rows).

## Class balance

DILI-concern (`dili_binary == 1`): **685** (61.270%)

No DILI-concern (`dili_binary == 0`): **433** (38.730%)

Total: **1,118** small-molecule rows.

## SMILES resolution rate

**1,118 / 1,279 = 87.412%** of DILIst rows resolved to canonical SMILES (per Q13 small-molecule scoping).

The 161 unresolved entries (biologics, polymers/inorganics, obsolete or industrial chemicals) are logged in `data/processed/dili_smiles_resolution_failures.csv` and excluded from `dili_canonical.csv`.

## DILIrank severity populated

**767 / 1,118 = 68.605%** of canonical rows have a `dili_severity` label from DILIrank 2.0 (`vDILI-Concern`).

Severity is `NaN` for drugs that aren't in DILIrank 2.0; matching uses lowercased compound name with salt-suffix stripping (mirrors Plan 01's SMILES resolver tactic so 'abacavir' [DILIst] hits 'abacavir sulfate' [DILIrank]).

## D_DILI ∩ LINCS

**502 / 1,118 = 44.902%** of canonical rows have a drug name (lowercased) present in the LINCS L1000 Phase II `pert_iname` set (`GSE70138_Broad_LINCS_sig_info_2017-03-06.txt`, 1,826 unique drugs).

## D_DILI ∩ PDG

**892 / 1,118 = 79.785%** of canonical rows have a drug name (lowercased) present in the PDG perturbation set (`MultiDCP/data/all_drugs_pdg.csv` ≅ `pdg_brddrugfiltered.pkl` pert_id column).
