# Fusion diagnostics: is there complementary structure+omics signal to fuse?

Measured-DE overlap set per organ. structure = ChemBERTa / ECFP4. REDUNDANCY = 5-fold variance-weighted R^2 of a Ridge map structure->DE (how much of the omics vector is just structure). FUSION on the two OOF probability streams: late = mean prob; stacker = LR on [p_struct, p_meas] (legitimate); ORACLE = per-drug pick the prob closer to the label (unachievable upper bound on ANY fusion). 5 seeds.

## Redundancy (how much of DE is linearly recoverable from structure)

| organ | structure | R^2 struct->predicted DE | R^2 struct->measured DE |
|---|---|---|---|
| liver | chemberta | +0.507 | -0.867 |
| liver | ecfp4 | +0.534 | -0.250 |
| kidney | chemberta | +0.293 | -1.361 |
| kidney | ecfp4 | +0.482 | -0.229 |

> predicted-DE R^2 near 1 = predicted DE is structure re-encoded (the near-circular arm). measured-DE R^2 well below 1 = real biology carries structure-independent variance.

## Fusion of structure + MEASURED DE (the real test)

| organ | structure | n | structure | measured | late fusion | learned stacker | ORACLE (UB) |
|---|---|---|---|---|---|---|---|
| liver | chemberta | 197 | 0.533±0.024 | 0.513±0.026 | 0.540±0.009 | 0.499±0.065 | 0.859±0.014 |
| liver | ecfp4 | 197 | 0.610±0.033 | 0.513±0.026 | 0.575±0.030 | 0.550±0.049 | 0.890±0.019 |
| kidney | chemberta | 240 | 0.691±0.015 | 0.527±0.030 | 0.649±0.026 | 0.674±0.022 | 0.902±0.008 |
| kidney | ecfp4 | 240 | 0.687±0.017 | 0.527±0.030 | 0.639±0.025 | 0.671±0.019 | 0.898±0.009 |

## Complementarity (seed-0 OOF errors)

| organ | structure | struct wrong | meas wrong | both wrong | meas right where struct wrong | err corr |
|---|---|---|---|---|---|---|
| liver | chemberta | 59 | 73 | 29 | 30 | +0.16 |
| liver | ecfp4 | 43 | 73 | 28 | 15 | +0.31 |
| kidney | chemberta | 84 | 108 | 36 | 48 | -0.03 |
| kidney | ecfp4 | 89 | 108 | 41 | 48 | +0.02 |

## Reading
- **ORACLE ~ structure** => no per-drug combination of these two streams beats structure; the negative is real and fusion engineering cannot rescue it on this data.
- **ORACLE >> structure but stacker ~ structure** => complementary signal EXISTS but linear prob-fusion misses it => invest in fusion design (MLP / gating / interactions).
- **stacker > structure** => a better fusion already helps; report it.
- err_corr near 1 = the two models fail on the same drugs (no complementarity); near 0 = independent failures (fusable).

## CORRECTION (see P4_omics_arm.md) — the high ORACLE here is INFLATION, not signal
The oracle (0.86-0.90) looked like huge complementary signal. It is not. `P4_omics_arm.md`'s
permutation sanity shows oracle(structure + **PERMUTED** omics) = 0.901 (liver) / 0.894 (kidney),
essentially equal to oracle(structure + real omics). A per-drug label-informed selector reaches
~0.90 with *any* independent second stream, even random noise. The real marginal complementarity is
~+0.01-0.03, within noise. Combined with omics-alone staying at chance across 9 rep x model configs
and no fusion (late / stacker / nonlinear concat-MLP) beating structure, the honest conclusion is:
**no extractable complementary signal on this measured-omics data** — the near-zero `stacker` vs
`structure` result, not the oracle, is the trustworthy number.

