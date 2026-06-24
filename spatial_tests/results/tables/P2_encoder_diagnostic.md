# P2 encoder zonal-invariance root-cause diagnostic

_Device: cuda:0 (CUDA_VISIBLE_DEVICES='2')_

Reuses `scripts/cache_region_de.py` basal pipeline + the production `RegionSignatureCacher` load path exactly. Real data only. Read-only.

### HUMAN — D1 (input basals) + D3 (marker survival)
- Zones present: ['periportal', 'pericentral']; spot counts: {'periportal': 53, 'pericentral': 4124}
- **Input basal contrast (normalized, 10716):** Pearson(periportal, pericentral) = **0.332578**, L2 = **30.3900**, max|Δ| = 7.2916e-01, #genes differing = 9593/10716.

**Marker table (raw pseudobulk vs normalized basal):**
| zone | marker (native) | human sym | in 10716? | covered (not zero-fill)? | raw PP | raw PC | norm PP | norm PC |
|---|---|---|---|---|---|---|---|---|
| pericentral | GLUL | GLUL | yes | yes | 0.1321 | 0.9273 | 0.7727 | 0.9595 |
| pericentral | CYP2E1 | CYP2E1 | yes | yes | 3.0943 | 27.8426 | 0.9899 | 0.9987 |
| pericentral | OAT | OAT | yes | yes | 0.0943 | 0.3528 | 0.6882 | 0.9050 |
| pericentral | SLC1A2 | SLC1A2 | yes | yes | 0.0189 | 0.1062 | 0.1394 | 0.7197 |
| periportal | SDS | SDS | yes | yes | 2.5094 | 0.5902 | 0.9873 | 0.9391 |
| periportal | CYP2F2 | CYP2F2 | NO | NO(zero-fill) | — | — | — | — |
| periportal | HAL | HAL | yes | yes | 0.0943 | 0.1026 | 0.6882 | 0.7114 |
| periportal | ASS1 | ASS1 | yes | yes | 2.2642 | 2.6855 | 0.9853 | 0.9842 |

### MOUSE — D1 (input basals) + D3 (marker survival)
- Zones present: ['periportal', 'pericentral']; spot counts: {'periportal': 332, 'pericentral': 1151}
- **Input basal contrast (normalized, 10716):** Pearson(periportal, pericentral) = **0.866525**, L2 = **13.5517**, max|Δ| = 6.0200e-01, #genes differing = 8131/10716.

**Marker table (raw pseudobulk vs normalized basal):**
| zone | marker (native) | human sym | in 10716? | covered (not zero-fill)? | raw PP | raw PC | norm PP | norm PC |
|---|---|---|---|---|---|---|---|---|
| pericentral | Glul | GLUL | yes | yes | 2.7500 | 12.3962 | 0.9305 | 0.9827 |
| pericentral | Cyp2e1 | CYP2E1 | yes | yes | 32.9247 | 212.6160 | 0.9942 | 0.9991 |
| pericentral | Oat | OAT | yes | yes | 4.7922 | 25.3727 | 0.9567 | 0.9922 |
| pericentral | Slc1a2 | SLC1A2 | yes | yes | 0.6145 | 3.3545 | 0.7593 | 0.9372 |
| periportal | Sds | SDS | yes | yes | 4.0120 | 1.9505 | 0.9502 | 0.8977 |
| periportal | Cyp2f2 | CYP2F1 | yes | yes | 46.3193 | 27.4883 | 0.9956 | 0.9930 |
| periportal | Hal | HAL | yes | yes | 19.4819 | 12.5387 | 0.9908 | 0.9831 |
| periportal | Ass1 | ASS1 | yes | yes | 31.4819 | 34.7150 | 0.9938 | 0.9945 |

### D2 — Does the encoder respond at all?
- **(a) Human zone basals → output:** Pearson(treated_PP, treated_PC) = 1.000000, max|Δ output| = **5.9605e-08** (input Pearson was 0.332578).
  - 50-d cell-context: dim=(50,), max|Δ context| = **2.9802e-08**, Pearson = 0.965900.
- **(b) two synthetic strongly-different on-manifold basals (no training CSV found) → output:** input Pearson = -0.999917; Pearson(treated_a, treated_b) = 1.000000, max|Δ output| = **5.9605e-08**.
  - 50-d cell-context: max|Δ context| = **2.9802e-08**, Pearson = 0.965900.
- **(c) Finite-difference sensitivity:** ‖Δ input‖=7.0413 → ‖Δ output‖=6.6640e-08 (ratio ‖Δout‖/‖Δin‖ = 9.4641e-09).

### D2 confirmation — drug moves, basal does not (fixed-basal / fixed-drug probe)

| probe | max\|Δ output\| | Pearson |
|---|---|---|
| drug C vs acetaminophen (basal=0.62 const) | 0.1300 | 0.9440 |
| drug C vs benzene (basal=0.62 const) | 0.0799 | 0.9727 |
| basal 0.05 vs 0.98 (const) @ acetaminophen | **5.96e-08** | 1.000000 |

The output responds to the **drug** branch (max\|Δ\| ~0.08–0.13) but is invariant
to the **basal** branch to float32 epsilon. The 50-d cell-context vector itself
moves only 2.98e-08 across inputs whose Pearson ranges from +0.33 to **-0.9999**.
The cell-context (basal) input path is effectively inert in this checkpoint.

---

## VERDICT: ENCODER-FLAT (the basal/cell-context input path is inert)

The INPUT basals are NOT flat and the markers DID survive the gene space, so the
input-flat hypothesis is ruled out:

- **D1 (inputs differ):** normalized periportal-vs-pericentral input basals have
  Pearson **0.333** (human) / **0.867** (mouse), L2 **30.4** / **13.6**, with
  ~9,000+ of 10,716 genes differing. Zonal contrast is preserved through the
  rank-percentile normalization (e.g. human GLUL 0.77→0.96 PP→PC; mouse Cyp2e1,
  Glul, Oat, Slc1a2 all higher pericentral as expected; periportal Sds, Hal
  higher periportal). The normalization did NOT flatten zonal contrast.
- **D3 (markers survived):** 7/8 human and 8/8 mouse markers are present in the
  10,716 space and covered (not zero-filled). The one absence (human CYP2F2) is a
  true mouse-specific gene; its human ortholog CYP2F1 is present and mouse Cyp2f2
  maps to it correctly. Marker dropout is NOT the cause.
- **D2 (encoder is dead):** feeding inputs with Pearson from +0.33 down to
  **-0.9999** changes the predicted output by at most **5.96e-08** (float32
  epsilon) and the 50-d cell-context by **2.98e-08**. A finite-difference probe
  gives ‖Δout‖/‖Δin‖ = **9.5e-09**. Meanwhile the **drug** branch moves the
  output by 0.08–0.13. The model ignores `input_cell_gex` entirely; the
  prediction is a function of (drug, dose) only.

This is not the project's "OOD-saturation on healthy tissue" hypothesis (which
would predict the encoder responds to in-distribution cancer-line basals but
saturates on tissue). The encoder is invariant to **all** basals, including
maximally-different synthetic ones — the basal input path contributes nothing in
this row-17 `MultiDCP_CheMoE_AE` checkpoint.

### Single most actionable fix

A basal-pipeline fix (F1) cannot help: no change to the input can move an output
that is constant in the input. Within the frozen-baseline rule, the only fix that
makes zonal contrast appear in the prediction is **F3 — make the zonal difference
explicit downstream of the dead encoder**: condition the cached signature on a
**contrast/delta basal** that is computed OUTSIDE the model (e.g. use
`zone_basal − tissue_mean_basal`, or carry the measured per-zone basal contrast
as the region feature), rather than relying on the model's cell-context path to
extract it. No model weights change.

If a genuinely region-resolved *predicted* DE (not a passed-through input
contrast) is required, the deeper fix is **F4 — source/re-derive a non-collapsed
checkpoint whose cell-context encoder actually uses `input_cell_gex`** (the same
class of provenance problem already recorded for the row-18 S-B checkpoint). That
re-opens the checkpoint hunt rather than the basal pipeline. Fine-tuning the
encoder (F2) is blocked by the absence of healthy-tissue drug-perturbed targets
and would reverse `DEC-frozen-baseline`.

**Bottom line:** the fix is NOT in `cache_region_de.py`'s basal construction
(that pipeline is correct and preserves zonation). It is upstream/around the
frozen model — the wired row-17 checkpoint does not read the basal.
