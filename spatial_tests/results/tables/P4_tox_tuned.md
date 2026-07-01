# Tox-tuned MultiDCP signature (conditions E/F) vs frozen DE vs structure

All variants share one head (BatchNorm -> 128 -> 1). structure=ECFP4; frozen_DE=frozen engine DE; tuned_E=engine+head end-to-end on tox; tuned_F=tuned_E + anchor to frozen predicted-treated (w=1.0). 5-fold x 3 seeds, internal-val early stopping. **Caveat:** tuned signature is still a function of SMILES; a win = useful bottleneck inductive bias, not structure-independent information.

## drug-disjoint

| organ | structure | frozen_DE | tuned_E | tuned_F |
|---|---|---|---|---|
| liver | 0.712±0.002 | 0.578±0.035 | 0.559±0.005 | 0.574±0.029 |
| kidney | 0.652±0.006 | 0.560±0.017 | 0.532±0.016 | 0.545±0.002 |

## scaffold-disjoint

| organ | structure | frozen_DE | tuned_E | tuned_F |
|---|---|---|---|---|
| liver | 0.709±0.012 | 0.608±0.050 | 0.560±0.007 | 0.572±0.012 |
| kidney | 0.668±0.015 | 0.602±0.013 | 0.533±0.016 | 0.550±0.023 |

## Reading
- tuned_E/F > structure on both organs and splits => the GEX bottleneck is a useful inductive bias for tox (worth pursuing).
- tuned ~ structure => tox-tuning does not help; the bottleneck adds nothing over structure even when optimized for the endpoint (HALT GATE).
- tuned > frozen_DE => tox gradient to the encoder helps vs frozen; isolates the tuning effect.

