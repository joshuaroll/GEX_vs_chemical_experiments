# Transcriptomic response signatures do not beat chemical structure for organ toxicity

**Scope:** consolidated results writeup for the P4 structure vs expression vs structure+expression
comparison (liver, kidney) plus the nine-encoder structure sweep.
**Date:** 2026-06-30 (this draft), spatial_tests milestone.
**Artifacts:** `results/tables/P4_*.md`, commits `efa8dac` (engine + extraction), `d2efac6`
(six-probe comparison), `08909ae` (encoder sweep).
**Status:** draft for internal review. Negative-result framing is pending professor sign-off
(spatial_tests decision D3, "negative result is publishable").

---

## 1. Question

For drug-induced organ toxicity, does a molecular *response* signature add predictive signal
over chemical *structure* alone? We compare three feature sets under one fixed classifier:

- **structure** = a molecular encoder embedding of the drug's SMILES,
- **expression** = the organ engine's predicted differential expression, `DE = engine(SMILES, organ-mean basal) − basal` (978 genes, DE rule),
- **structure + expression** = their concatenation.

The two organs that have both a trained perturbation engine and toxicity labels are **liver**
(DILIst-derived binary DILI, n = 492, 307/185) and **kidney** (DIRIL binary DIKI, n = 317,
171/146). Brain is excluded because its label source (SIDER) is CID-keyed with no structure join
on disk; heart is excluded because LINCS has no cardiac perturbation line to train an engine.

**Evaluation is honest by construction.** One row per drug, drug-disjoint 5-fold cross-validation
× 5 seeds, small fixed head (L2 logistic regression, class-balanced), threshold-free AUROC, paired
bootstrap confidence intervals (10,000 resamples on seed-0 out-of-fold predictions). We also run a
scaffold-disjoint variant (Bemis-Murcko group split) as the harder out-of-distribution test.

**The engines are faithful expression predictors, so this is not a prediction-quality artifact.**
On held-out drugs the engines reach DE-Pearson 0.757 (liver) and 0.675 (kidney), both clearing the
honest measured drug-disjoint ceiling (~0.60 from the Phase-1 bracket). The question is therefore
sharp: does *well-predicted* expression carry toxicity information beyond structure?

---

## 2. Headline: expression is structure-ceilinged

`P4_stage2_toxicity`, ChemBERTa structure:

| Organ | structure | expression | both | +lift (both − structure) | 95% CI |
|---|---|---|---|---|---|
| liver | **0.714** | 0.671 | 0.705 | −0.016 | [−0.046, +0.012] |
| kidney | 0.654 | 0.599 | **0.660** | +0.014 | [−0.027, +0.055] |

Structure is the strongest single feature; predicted expression sits below it; combining the two
produces no lift whose confidence interval clears zero. This is a negative for the expression
modality on both organs.

---

## 3. The negative survives six independent probes

Each probe attacks the result from a different angle. All six agree.

| Probe (`P4_*`) | What it tests | Result |
|---|---|---|
| `stage2_toxicity` | the base three-way | structure > expression; both ≈ structure (no CI-clearing lift) |
| `scaffold_split_tox` | is structure's edge analog memorization? | edge is **stable or widens** under scaffold-disjoint (liver gap +0.043→+0.048; kidney +0.055→+0.071) → real generalization |
| `internal_rep_tox` | is the DE *output* discarding signal the hidden layer keeps? | the penultimate 128-d hidden is also structure-ceilinged (structure+internal lift −0.014 liver, −0.012 kidney) |
| `region_basal_tox` | can region-resolved basal conditioning manufacture new features? | engine is only weakly basal-sensitive; per-basal DE is a SMILES function; concat-across-basals stays ≤ structure |
| `improved_both` | was naive concat under uniform L2 diluting structure? | tuned/PCA/late fusion still ≤ tuned structure (liver tuned structure 0.769 vs best fusion 0.756) |
| `predicted_vs_measured_de` | is it a *predicted*-expression problem? | **measured** DE also fails to beat structure (see §5) |

The `internal_rep` and `improved_both` probes are the two that could have rescued expression, and
neither does: the signal is not hidden in a richer layer, and it is not lost to a lazy fusion rule.

---

## 4. Encoder sweep: the negative is not a ChemBERTa artifact

`P4_encoder_sweep` swaps the structure arm across nine encoders while holding the engine expression
arm and the drug set fixed per organ, so the only moving part is the structure representation.

**Adding predicted DE to structure helps in 0 of 18 cells** (9 encoders × 2 organs): every
`+lift (both − structure)` confidence interval includes zero.

**Structure beats predicted DE in 16 of 18 cells.** The two exceptions (liver `topotorsion` and
`unimol_v1`, both 0.630; kidney `unimol_v1`, 0.578) are the *weakest* structure encoders dropping
below the flat expression bar, not expression becoming informative. The expression AUROC is constant
within an organ by construction (0.671 liver, 0.599 kidney).

**3D pretrained embeddings do not beat cheap fingerprints.** UniMol-v1 is the worst structure
encoder in both organs; UniMol-v2 is mid-pack. Best per organ:

| | best structure encoder | AUROC | note |
|---|---|---|---|
| liver | ChemBERTa | 0.714 | learned LM wins |
| kidney | ECFP4 | 0.705 | plain Morgan r2 wins; ChemBERTa only 5th (0.654) |

The best encoder differs by organ, but the verdict does not: predicted expression is
structure-ceilinged regardless of how structure is represented.

---

## 5. The measured-DE control (the load-bearing result)

`P4_predicted_vs_measured_de` scores structure, predicted DE, and *measured* LINCS DE on the same
drugs (the LINCS-overlap subset), so the comparison is not confounded by which drugs each feature
covers.

| Organ | n (pos/neg) | structure | predicted DE | measured DE |
|---|---|---|---|---|
| kidney | 240 (132/108) | **0.691** | 0.603 | 0.527 |
| liver | 197 (154/43) | 0.533 | 0.513 | 0.513 |

On kidney, the informative organ here, **real measured expression is near chance (0.527) and well
below structure (0.691)**. So the ceiling is a property of expression as a toxicity feature on
honest drug-disjoint splits, not of the engine's prediction fidelity. This is the single most
important control: it converts "your predicted signatures are bad" into "there is no expression
lift to capture."

Caveat: the liver overlap subset is small and heavily positive-skewed (154/43); *every* feature
including structure is near chance there (0.533), so liver cannot separate anything on this subset
and only kidney carries the control's weight.

---

## 6. Convergent evidence from other expression instantiations

The P4 suite uses the engine's 978-gene predicted DE *output*. Two other definitions of the
expression feature, run earlier, agree:

- **Measured LINCS DE** (`P1_three_way_comparison`, drug-level): no CI-clearing +lift for liver
  (−0.048), kidney (−0.005), or heart (+0.042).
- **MultiDCP-CheMoE latent** (`P1_latent_three_way_chemberta`, `global_features`, 306-d): no lift
  for kidney (−0.000) or heart (+0.009), and a *significantly negative* lift for liver
  (−0.011, CI [−0.020, −0.002]). The latent's per-drug signal is validated to be MultiDCP's own
  drug encoder (cell/dose blocks constant within a basal), so it is effectively a second structure
  encoder carrying redundant information, which is why concatenating it dilutes rather than helps.

Output, latent, measured; six probes; nine encoders; two organs. The result does not move.

---

## 6b. Follow-up: is it the combiner, or the signal? (fusion + tox-tuning)

A fair objection: a negative from linear concatenation is the weakest test, and predicted DE may be
redundant with structure. We ran it down.

**Predicted DE is largely structure re-encoded.** A linear map from structure recovers 29-53% of
predicted DE, but has *negative* R^2 to *measured* DE. So the predicted-DE arm was near-circular
(SMILES -> DE, another function of SMILES); only measured DE tests the omics hypothesis. Tables:
`P4_fusion_diagnostics.md`, `P4_omics_arm.md`.

**On measured DE, fusion has nothing to extract.** The measured-omics arm is at chance on its own
across 3 representations (raw / PCA30 / supervised top-100) x 3 models (logistic / RF / MLP; best
kidney 0.557). No fusion beats structure: late fusion, a learned stacker, and a nonlinear joint
concat-MLP all tie or trail structure. The oracle upper bound looked high (0.90) but is a mirage:
oracle(structure + *permuted/random* omics) is essentially identical (0.89-0.90), so the apparent
complementarity is label-selection inflation, not signal. The trustworthy number is the learned
stacker, which ties structure.

**Tox-tuning the signature (conditions E/F) makes it worse.** Fine-tuning the engine end-to-end
toward the tox label (tuned_E), and with an anchor to the frozen biological prediction (tuned_F),
underperforms both structure and the frozen signature on both organs and both splits
(drug-disjoint: liver 0.559 / 0.574 vs structure 0.712; kidney 0.532 / 0.545 vs 0.652). Fine-tuning
a 978-gene bottleneck on ~340 binary labels overfits and distorts the signature; the anchor partially
protects it (F > E everywhere), which confirms the mechanism. Table: `P4_tox_tuned.md`.

Bottom line of the follow-up: the bottleneck is the omics *signal*, not the combiner. Every modeling
lever we have (frozen, tuned, fused, nine encoders, six probes, better representations) leaves
structure as the ceiling. The one untested lever is *tox-informative measured omics in the right cell
context*, which is a data problem (proximal-tubule / primary-hepatocyte perturbation data barely
exists at scale), not a modeling one.

## 7. Limitations

- **Small samples, wide intervals.** n = 492 (liver) and 317 (kidney); the paired-lift CIs are on
  the order of ±0.03–0.06, so we can rule out a *large* expression lift, not a tiny one. The claim
  is "no material lift," not "exactly zero."
- **Two organs.** Only liver and kidney support the engine-based comparison. The measured and latent
  three-way checks add heart, but heart has no trained engine.
- **Kidney fusion instability.** In `improved_both`, nested-CV-tuned structure (0.628) falls below
  naive concat (0.660) on kidney (n = 317); the C-selection overfits at this sample size. Treat the
  kidney `improved_both` row as noisy; the primary kidney result is `stage2` (structure 0.654).
- **One engine family, one basal.** The engine is MultiDCP-original with a linear cell encoder,
  queried at the organ-mean basal. `region_basal` shows the engine is only weakly basal-sensitive,
  so region-resolved conditioning cannot manufacture signal the mean-basal query lacks; this also
  closes the spatial "combine across locations" prerequisite negatively.

None of these weaken the core direction: across every honest cut we tried, structure is the ceiling.

---

## 8. What this means

On honest drug-disjoint and scaffold-disjoint splits, chemical structure is the ceiling for these
toxicity endpoints. Transcriptomic response signatures, whether predicted or measured, output or
latent, extracted six ways and paired with nine structure encoders, do not beat or augment it. The
engines predict expression well (DE-Pearson 0.68–0.76 on new drugs); that fidelity simply does not
translate into toxicity signal beyond what structure already encodes.

This is consistent with the two priors on record: the Phase-1 bracket (the Wang/Li ~0.798 benchmark
is largely profile-level drug-identity leakage that structure reproduces; the honest measured
ceiling ≈ the structure floor) and the multihead_dili v1 negative (predicted GEX + dose added
nothing over MolFormer chemistry).

The methodological contribution is twofold and worth stating explicitly, because both are common
ways the field overstates expression's value:

1. **Profile-level evaluation inflates the apparent signal.** Drug-disjoint evaluation removes it.
2. **A single-encoder structure baseline understates the structure ceiling.** Sweeping encoders
   shows the honest ceiling is the *best* of several cheap representations (ECFP4 alone reaches
   0.705 on kidney), which any real expression method must clear.

Framed positively for a paper: *when the split is honest and the structure baseline is strong,
a faithful transcriptomic response model does not add toxicity signal over chemical structure for
liver or kidney.* That is a clean negative with a reproducible bracket behind it.

The modeling side of this question is now exhausted: frozen predicted DE, nine structure encoders,
six extraction probes, measured DE, four fusion methods, and tox-tuned signatures (conditions E/F)
all leave structure as the ceiling, and tox-tuning actively hurts. The one lever we have not been
able to test is *tox-informative measured omics in the right cell context* (proximal tubule for
kidney, primary hepatocyte for liver). That is a data limitation, not a method choice, and it is the
honest boundary of this negative: we show the gene-expression channel as currently obtainable does
not help, not that no conceivable omics assay could.
