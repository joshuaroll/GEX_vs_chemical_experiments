# Encoder sweep: structure vs DE vs DE+structure (does the negative survive the encoder?)

structure = swept encoder | expression = **multidcp** organ-engine predicted DE (DE = engine(smiles, organ-mean basal) - basal), IDENTICAL across encoders per organ | head = L2 logistic regression (balanced), drug-disjoint 5-fold x 5 seeds. Every encoder scored on the same drug set per organ (engine-featurizable AND valid for all encoders).

## liver (n=492, 307/185; expression AUROC 0.671±0.014, constant)

| encoder | dim | structure AUROC | both AUROC | struct − expr | +lift (both − struct) | 95% CI |
|---|---|---|---|---|---|---|
| chemberta | 384 | 0.714±0.017 | 0.705±0.013 | +0.043 | **-0.016** | [-0.046, +0.012] |
| ecfp6 | 2048 | 0.698±0.012 | 0.708±0.013 | +0.026 | **+0.013** | [-0.002, +0.030] |
| ecfp4 | 2048 | 0.684±0.008 | 0.685±0.005 | +0.013 | **+0.009** | [-0.004, +0.022] |
| unimol_v2 | 768 | 0.684±0.017 | 0.677±0.017 | +0.013 | **+0.005** | [-0.030, +0.039] |
| rdkit_fp | 2048 | 0.679±0.012 | 0.677±0.012 | +0.008 | **-0.001** | [-0.019, +0.016] |
| maccs | 167 | 0.674±0.010 | 0.670±0.015 | +0.003 | **+0.008** | [-0.024, +0.038] |
| atompair | 2048 | 0.672±0.011 | 0.674±0.015 | +0.001 | **+0.008** | [-0.007, +0.023] |
| topotorsion | 2048 | 0.630±0.006 | 0.636±0.010 | -0.041 | **+0.015** | [-0.003, +0.033] |
| unimol_v1 | 512 | 0.630±0.014 | 0.638±0.014 | -0.041 | **+0.020** | [-0.009, +0.047] |

## kidney (n=317, 171/146; expression AUROC 0.599±0.013, constant)

| encoder | dim | structure AUROC | both AUROC | struct − expr | +lift (both − struct) | 95% CI |
|---|---|---|---|---|---|---|
| ecfp4 | 2048 | 0.705±0.013 | 0.708±0.007 | +0.106 | **-0.002** | [-0.029, +0.025] |
| topotorsion | 2048 | 0.687±0.010 | 0.690±0.013 | +0.088 | **-0.002** | [-0.029, +0.023] |
| rdkit_fp | 2048 | 0.682±0.005 | 0.670±0.009 | +0.083 | **-0.012** | [-0.034, +0.010] |
| atompair | 2048 | 0.672±0.010 | 0.674±0.014 | +0.073 | **+0.000** | [-0.021, +0.022] |
| chemberta | 384 | 0.654±0.010 | 0.660±0.017 | +0.055 | **+0.014** | [-0.027, +0.055] |
| maccs | 167 | 0.648±0.016 | 0.637±0.019 | +0.049 | **-0.026** | [-0.070, +0.019] |
| ecfp6 | 2048 | 0.641±0.007 | 0.651±0.005 | +0.042 | **+0.016** | [-0.013, +0.044] |
| unimol_v2 | 768 | 0.639±0.014 | 0.647±0.011 | +0.040 | **+0.005** | [-0.042, +0.053] |
| unimol_v1 | 512 | 0.578±0.014 | 0.609±0.019 | -0.021 | **+0.022** | [-0.016, +0.061] |

## Reading
- **struct − expr** > 0 for every encoder = structure beats predicted expression regardless of representation; the expression ceiling is not a ChemBERTa artifact.
- **+lift (both − struct)** with a CI that includes 0 = adding predicted DE to structure does not help. A single encoder with a CI clearing 0 would be the counter-signal to chase.
- expression AUROC is constant within an organ by construction (same engine DE, same drug set, same head) — only the structure representation moves across rows.
- Encoders: chemberta (learned LM), unimol_v1/v2 (3D pretrained), ecfp4/ecfp6 (Morgan r2/r3), maccs (167 keys), atompair, topotorsion, rdkit_fp (topological).

