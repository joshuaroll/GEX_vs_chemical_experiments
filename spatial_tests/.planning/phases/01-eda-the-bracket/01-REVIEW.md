---
phase: 01-eda-the-bracket
reviewed: 2026-06-23T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - src/spatial/eda/floor.py
  - scripts/run_p1_eda.py
  - tests/spatial/test_eda_floor.py
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues
---

# Phase 1: Code Review Report (gap-closure plan 01-08)

**Reviewed:** 2026-06-23
**Depth:** deep (cross-file: floor.py <-> run_p1_eda.py <-> bootstrap.py)
**Scope:** `git diff 5e052e9..3611f6c -- src/ tests/ scripts/` (b7f2dc5 RED test, 02dd37e floor.py impl, 3611f6c driver wiring)
**Status:** issues_found (2 warnings, 1 info; no blockers)

## Summary

The 01-08 changes are correct on every load-bearing axis I could verify by tracing the code. The headline risk for this plan was silent row-misalignment between the floor and ceiling OOF vectors feeding the paired bootstrap; that alignment is sound. The `floor_profile_disjoint_auroc` -> `floor_profile_disjoint_oof` refactor is a verbatim body move plus a `.astype(float64)`, and the existing disjoint regression test plus the new `test_floor_profile_disjoint_oof_matches_scalar` lock equivalence to 1e-9. All 4 tests pass in `dili_v04_env` (17s). seed=42 and the per-fold pipeline are consistent across floor-leaky, floor-disjoint-OOF, the driver's recomputed ceiling-disjoint OOF, and `_leakage_decomposition`.

The two warnings are real but bounded statistical/robustness subtleties, not defects in the wiring. The info item is cosmetic.

### What I verified explicitly (no findings)

1. **OOF alignment (the main risk) is correct.** `de_aligned`, `y_profiles`, and `drug_key` are all derived from the same `in_dilirank_mask` in the same row order (run_p1_eda.py:414-430). The driver loop (`for i in range(len(drug_key))`, 624-631) appends `kept_idx.append(i)`, `y_rows.append(y_profiles[i])`, `groups_rows.append(drug_key[i])`, `fps_rows.append(fp)` in lockstep, so `kept_idx`, `y_kept`, `groups_kept`, `fps_prof` index identical profile rows in identical order. `de_kept = de_aligned[np.asarray(kept_idx)]` (728) selects exactly those rows in the same order. `floor_oof` (from `fps_prof`) and `ceil_oof` (from `de_kept`) are therefore row-aligned to `y_kept` before `paired_bootstrap_auroc_gap(y_kept, floor_oof, ceil_oof, ...)`. No misalignment.

2. **Refactor preserves identical folds/results.** `floor_profile_disjoint_oof` (floor.py:386-425) is byte-for-byte the pre-refactor `floor_profile_disjoint_auroc` body (validation, `np.float32` cast, pipeline, `eff_splits = min(n_splits, n_unique)`, the `>=2` grouped path, the degenerate StratifiedKFold fallback), with the sole addition of `oof.astype(np.float64)`. The scalar wrapper now computes `roc_auc_score(np.asarray(y, int), oof)`. Confirmed against `5e052e9:floor.py`.

3. **Leaky function genuinely has no grouping.** `floor_profile_leaky_auroc` calls `cross_val_predict(clf, fps, y, cv=StratifiedKFold(...))` with no `groups=` argument (floor.py:482-485). The new `test_floor_profile_leaky_exploits_drug_identity` proves the leaky split memorizes per-drug identity (leaky > disjoint, leaky > 0.75) — a behavioral lock that it does not group.

4. **Fold/seed consistency for the paired bootstrap.** Floor: `StratifiedGroupKFold(n_splits=min(5,n_unique), shuffle=True, random_state=42)`. Ceiling (driver 730-738): `StratifiedGroupKFold(n_splits=min(5,len(set(groups_kept))), shuffle=True, random_state=42)` with the same `groups_kept`. Same n_splits, same shuffle seed, same groups -> identical partition -> a genuinely paired CV. Bootstrap seed=42, n_resamples=10_000.

5. **No dtype/index pitfall in `de_aligned[np.asarray(kept_idx)]`.** `kept_idx` is a list of Python ints, so `np.asarray(kept_idx)` is int64 -> valid fancy index. The empty-list float64 edge case (which would break indexing) is precluded by the `n_kept >= 10` gate at line 641.

6. **Ceiling-disjoint OOF must be recomputed, no cleaner reuse exists.** The canonical `profile_auroc_disjoint` from `_leakage_decomposition` is computed on the FULL ceiling set; the gap CI requires OOF on the KEPT subset only (different rows -> different folds -> different OOF). Recomputation is necessary, not wasteful. No finding.

## Warnings

### WR-01: Profile-level paired bootstrap treats correlated same-drug profiles as independent resampling units

**File:** `scripts/run_p1_eda.py:740-742` (the `paired_bootstrap_auroc_gap(y_kept, floor_oof, ceil_oof, ...)` call)
**Issue:** `paired_bootstrap_auroc_gap` resamples `n` rows with replacement (bootstrap.py:151-156). Here `n` = number of kept profiles, and many profiles share a drug (the leakage decomposition exists precisely because a single drug contributes up to ~784 profiles). Resampling profiles as if independent ignores intra-drug correlation, which tends to understate the CI width — the interval can read narrower (more "significant") than a drug-clustered bootstrap would. The function's own docstring (bootstrap.py:86-89) states the intended contract is "aligned over the SHARED DRUG set ... aggregating the ceiling to drug level before calling (Pitfall 8)," i.e. drug-level resampling; this call deliberately departs from that contract.
**Mitigating context:** This is labeled "profile-level" throughout the report and is explicitly "documentation for honesty, not a new hard gate" (the PRIMARY drug-level Halt Gate 2 is unchanged). So the inferential weight on this CI is intentionally low. The departure is a reasonable, documented choice — but the narrowing is not surfaced to the reader.
**Fix:** Add one sentence to the report block (after line 752) noting the caveat, e.g.: "Profiles of the same drug are resampled independently, so this profile-level CI is narrower than a drug-clustered bootstrap; read it as a lower bound on uncertainty, with the drug-level gate above as the authoritative test." No code-path change needed.

### WR-02: Leaky/disjoint CV can raise if a class has fewer members than the effective fold count on the kept subset

**File:** `src/spatial/eda/floor.py:482-485` (leaky `StratifiedKFold`) and the disjoint path at `floor.py:404-410`; entry guarded only by `scripts/run_p1_eda.py:641` (`n_kept >= 10 and n_kept_drugs >= 2`)
**Issue:** `StratifiedKFold(n_splits=k)` requires every class to have >= k members; `StratifiedGroupKFold(n_splits=k)` requires >= k groups per the splitter. The driver gate checks total kept profiles (>=10) and total kept drugs (>=2) but not per-class profile counts or per-class drug counts. If the kept intersection ever yields, say, 4 positive profiles with `n_splits` resolving to 5, `cross_val_predict` raises and aborts the whole EDA run rather than degrading gracefully.
**Mitigating context:** On the real Wang/Li ceiling set this cannot trigger (hundreds of profiles per class), and the pre-existing disjoint path already carried the identical exposure before this plan — the leaky cell merely inherits it. So this is a latent robustness gap, not a regression.
**Fix:** Either tighten the gate to require a minimum per-class count (e.g. `min(np.bincount(y_kept)) >= 5`) before entering the 2x2/CI block, or cap `n_splits` by the minority-class count in the leaky helper, mirroring how the disjoint helper already caps by `n_unique`. Lowest-effort option: add `min_class = int(np.bincount(y_kept).min())` and skip the new cells with a logged note when `min_class < 2`.

## Info

### IN-01: Markdown table cells carry inconsistent trailing whitespace

**File:** `scripts/run_p1_eda.py:687-690` (the completed 2x2 table f-string)
**Issue:** Cells use uneven padding before the closing `|` (e.g. `{ceiling_auroc_profile_leaky:.4f}    |` vs `{ceiling_auroc_profile_disjoint:.4f} |`). Renders fine in any CommonMark viewer (whitespace inside cells is trimmed), so this is purely cosmetic and does not affect the report's correctness.
**Fix:** Optional — normalize to a single trailing space per cell for source readability. No behavioral impact.

---

_Reviewed: 2026-06-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
