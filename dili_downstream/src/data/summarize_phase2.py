"""Render `results/tables/P2_split_summary.md` from a diagnostics dict.

Pure library. The CLI driver (`scripts/summarize_phase2.py`) handles file I/O
and the diagnostics-load step; here we only depend on a structured dict (so unit
tests don't need to touch real-data JSON).

Mirrors the Phase 1 pattern: `src/data/summarize_phase1.py`'s `render_markdown`
returns a string; the driver writes it.

Required markdown sections (LOCKED — used as grep-checks per 02-CONTEXT.md
§specifics + 02-03-PLAN.md acceptance_criteria):
    # Phase 2 Split Summary
    ## Scaffold split
    ## Cluster split
    ## TDC-DILI scaffold split
    ## Three transfer slices
    ## Upstream-train filter diagnostics
    ## Halt gate evaluation

Halt-gate emit format (LOCKED, MUST be greppable per 02-03-PLAN.md):
    PASS: `HALT-GATE PASS: |D_DILI_test \\ D_PDG| = 226 (threshold 30)`
    FAIL: `HALT-GATE FAIL: |D_DILI_test \\ D_PDG| = 27 (threshold 30) -- STOP and re-discuss before Phase 3`
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__all__ = ["render_markdown"]


def _format_class_balance(triple: tuple[float, int, int] | None) -> str:
    """Format a `(pos_rate, n_pos, n_total)` triple as `61.5% (550/894 positive)`.

    Returns `"n/a"` when triple is None (non-stratified clusters where
    per-slice rates are not reported).
    """
    if triple is None:
        return "n/a"
    pos_rate, n_pos, n_total = triple
    return f"{100.0 * pos_rate:.1f}% ({n_pos}/{n_total} positive)"


def _format_tanimoto_histogram(
    buckets: list[tuple[float, float, int]],
) -> list[str]:
    """Render a markdown table for the Tanimoto train-test max-similarity histogram.

    Each bucket is `(lo, hi, count)`. The bar uses `█` characters capped at 50
    so wide buckets don't blow up the line length.

    Returns a list of markdown lines (header + separator + 1 row per bucket).
    """
    lines: list[str] = []
    lines.append("| Tanimoto bucket | Count | Bar |")
    lines.append("| --- | ---: | --- |")
    for lo, hi, count in buckets:
        bar_len = min(int(count), 50)
        bar = "█" * bar_len
        lines.append(f"| {lo:.1f}–{hi:.1f} | {count} | {bar} |")
    return lines


def render_markdown(diagnostics: dict) -> str:
    """Render P2_split_summary.md content from a diagnostics dict.

    Expected diagnostics keys (assembled by ``scripts/build_phase2_splits.py``):

        scaffold_split: {
            class_balance_train, class_balance_val, class_balance_test
                (each: tuple(pos_rate, n_pos, n_total) or None),
            tanimoto_histogram_buckets: list[tuple[lo, hi, count]],
            stratified: bool,
            size_train, size_val, size_test (ints),
        }
        cluster_split: {
            class_balance_train, class_balance_val, class_balance_test (or None),
            stratified: bool,
            tanimoto_histogram_buckets, size_train, size_val, size_test,
        }
        tdc_split: {tdc_version: str, dataset_size: int,
                    size_train: int, size_val: int, size_test: int}
        transfer_slices: {test_in_pdg: int, test_drug_novel: int,
                          test_drug_and_scaffold_novel: int}
        upstream_filter: {d_pdg_total: int, excluded_by_scaffold: int,
                          excluded_by_pert_id: int, excluded_intersection: int,
                          d_pdg_train_after_exclusion: int}
        halt_gate: {value: int, passed: bool, threshold: int}
        metadata: {seed: int, dili_canonical_rows: int, generated_at: str}

    Returns
    -------
    str
        The fully-rendered markdown.
    """
    sc = diagnostics["scaffold_split"]
    cl = diagnostics["cluster_split"]
    td = diagnostics["tdc_split"]
    ts = diagnostics["transfer_slices"]
    uf = diagnostics["upstream_filter"]
    hg = diagnostics["halt_gate"]
    meta = diagnostics.get("metadata", {})

    seed = meta.get("seed", "?")
    n_canonical = meta.get("dili_canonical_rows", "?")
    generated_at = meta.get("generated_at", "")

    lines: list[str] = []

    # ----------------------------------------------------------------------
    # Title
    # ----------------------------------------------------------------------
    lines.append("# Phase 2 Split Summary")
    lines.append("")
    lines.append(
        f"Generated from `data/processed/dili_canonical.csv` "
        f"({n_canonical} rows) with seed={seed}."
    )
    if generated_at:
        lines.append(f"_generated_at: {generated_at}_")
    lines.append("")

    # ----------------------------------------------------------------------
    # Scaffold split
    # ----------------------------------------------------------------------
    lines.append("## Scaffold split")
    lines.append("")
    lines.append(
        f"Murcko scaffold split (80/10/10), stratified on `dili_binary`. "
        f"Sizes: train={sc['size_train']}, val={sc['size_val']}, test={sc['size_test']}."
    )
    lines.append("")
    lines.append("**Class balance per slice:**")
    lines.append("")
    lines.append(f"- train: {_format_class_balance(sc.get('class_balance_train'))}")
    lines.append(f"- val: {_format_class_balance(sc.get('class_balance_val'))}")
    lines.append(f"- test: {_format_class_balance(sc.get('class_balance_test'))}")
    lines.append("")
    lines.append(
        f"**Stratified:** {'yes' if sc.get('stratified') else 'no'}."
    )
    lines.append("")
    lines.append("**Tanimoto train-test max-similarity histogram (scaffold split):**")
    lines.append("")
    lines.extend(_format_tanimoto_histogram(sc.get("tanimoto_histogram_buckets", [])))
    lines.append("")

    # ----------------------------------------------------------------------
    # Cluster split
    # ----------------------------------------------------------------------
    lines.append("## Cluster split")
    lines.append("")
    lines.append(
        f"Tanimoto 0.4 single-linkage cluster split (80/10/10) on Morgan FPs "
        f"(radius=2, nBits=2048). Sizes: train={cl['size_train']}, "
        f"val={cl['size_val']}, test={cl['size_test']}."
    )
    lines.append("")
    if cl.get("stratified"):
        lines.append("**Class balance per slice (stratified):**")
        lines.append("")
        lines.append(f"- train: {_format_class_balance(cl.get('class_balance_train'))}")
        lines.append(f"- val: {_format_class_balance(cl.get('class_balance_val'))}")
        lines.append(f"- test: {_format_class_balance(cl.get('class_balance_test'))}")
    else:
        # Locked literal per 02-CONTEXT.md §specifics + 02-03-PLAN.md test_5.
        lines.append(
            "**Class balance per slice:** non-stratified — clusters too small "
            "to balance per-class within Tanimoto 0.4 single-linkage; "
            "see `cluster_split` warning log for details."
        )
    lines.append("")
    lines.append("**Tanimoto train-test max-similarity histogram (cluster split):**")
    lines.append("")
    lines.extend(_format_tanimoto_histogram(cl.get("tanimoto_histogram_buckets", [])))
    lines.append("")

    # ----------------------------------------------------------------------
    # TDC-DILI scaffold split
    # ----------------------------------------------------------------------
    lines.append("## TDC-DILI scaffold split")
    lines.append("")
    lines.append(
        f"`tdc.single_pred.Tox(name='DILI_Hong')` with scaffold split "
        f"(TDC default 70/10/20, NOT 80/10/10 — explicit divergence)."
    )
    lines.append("")
    lines.append(f"- pytdc version: `{td['tdc_version']}`")
    lines.append(f"- dataset size: {td['dataset_size']}")
    lines.append(
        f"- counts: train={td['size_train']}, val={td['size_val']}, test={td['size_test']}"
    )
    lines.append("")

    # ----------------------------------------------------------------------
    # Three transfer slices
    # ----------------------------------------------------------------------
    lines.append("## Three transfer slices")
    lines.append("")
    lines.append(
        "Slices nested inside `D_DILI_test` for per-slice transfer reporting "
        "in Phase 5 (per `02-CONTEXT.md` §\"Three transfer slices structure\")."
    )
    lines.append("")
    lines.append(f"- |test_in_pdg| = {ts['test_in_pdg']}")
    lines.append(f"- |test_drug_novel| = {ts['test_drug_novel']}")
    lines.append(
        f"- |test_drug_and_scaffold_novel| = {ts['test_drug_and_scaffold_novel']}"
    )
    lines.append("")
    lines.append(
        "_Halt-gate input is `|test_drug_novel|`; threshold is 30 — see Halt gate "
        "evaluation section below._"
    )
    lines.append("")

    # ----------------------------------------------------------------------
    # Upstream-train filter diagnostics
    # ----------------------------------------------------------------------
    lines.append("## Upstream-train filter diagnostics")
    lines.append("")
    lines.append(
        "Option (a) leakage discipline (`upstream_filter.filter_upstream_train`): "
        "exclude any PDG drug whose Murcko scaffold is in S_test OR whose "
        "lowercased drug_name appears in DILIst-test."
    )
    lines.append("")
    lines.append(f"- |D_PDG_total| = {uf['d_pdg_total']}")
    lines.append(f"- |excluded_by_scaffold| = {uf['excluded_by_scaffold']}")
    lines.append(
        f"- |excluded_by_pert_id| = {uf['excluded_by_pert_id']} "
        f"_(actual matching is by lowercased drug_name; "
        f"the legacy `pert_id` label is preserved per 02-CONTEXT.md prose)_"
    )
    lines.append(f"- |intersection| = {uf['excluded_intersection']}")
    lines.append(
        f"- |D_PDG_train_after_exclusion| = {uf['d_pdg_train_after_exclusion']}"
    )
    lines.append("")

    # ----------------------------------------------------------------------
    # Halt gate evaluation
    # ----------------------------------------------------------------------
    lines.append("## Halt gate evaluation")
    lines.append("")
    if hg["passed"]:
        # Locked PASS format — grep-checked.
        lines.append(
            f"HALT-GATE PASS: |D_DILI_test \\ D_PDG| = {hg['value']} "
            f"(threshold {hg['threshold']})"
        )
        lines.append("")
        lines.append(
            "The drug-novel slice has enough drugs to support meaningful "
            "transfer-test reporting in Phase 5. Proceed to Phase 3 (upstream "
            "training)."
        )
    else:
        # Locked FAIL format — grep-checked. Two-line emit so the second line
        # carries the STOP directive for the test_3 assertion.
        lines.append(
            f"HALT-GATE FAIL: |D_DILI_test \\ D_PDG| = {hg['value']} "
            f"(threshold {hg['threshold']}) -- STOP and re-discuss before Phase 3"
        )
        lines.append("")
        lines.append(
            "The drug-novel slice is below the locked threshold of 30. STOP "
            "and re-discuss before Phase 3 — running upstream training without "
            "a meaningful transfer-test slice would invalidate the headline "
            "claim. See `HALT_REASON.md` in the phase directory for context."
        )
    lines.append("")

    return "\n".join(lines)
