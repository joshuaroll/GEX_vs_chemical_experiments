"""Three transfer slices on D_DILI_test (SPLIT-05).

Pure library. The Wave 3 CLI driver wires the real data; this module is the
algorithm only.

Per `02-CONTEXT.md` §"Three transfer slices structure" (locked by Q6
leakage discipline) — Phase 5 reports per-condition AUROC on each of:

  * test_in_pdg                : test drugs whose drug_name IS in PDG
                                  ("easy" — biology already saw the drug;
                                   classifier B/C should do well here).
  * test_drug_novel            : test drugs whose drug_name is NOT in PDG
                                  (the headline transfer test — B/C must
                                   work without having seen the drug).
  * test_drug_and_scaffold_novel : drug_novel ∩ {scaffold ∉ train_scaffolds}
                                   (strongest test — neither drug nor
                                    scaffold seen upstream).

Halt gate 1 input (CONTEXT.md): `|test_drug_novel| ≥ 30`. Verifier
predicted ~226 candidates from Phase 1; the gate is unlikely to fire.

Acyclic-scaffold semantics:
  At slice time we treat the empty-string scaffold "" as a SINGLE bucket.
  Whether an acyclic test drug is "scaffold-novel" depends on whether any
  train drug also has scaffold == "". This is a conservative interpretation
  — different from scaffold_split's per-row singleton handling, which is a
  partition-time concern (preventing all acyclic drugs from collapsing into
  one giant cluster). At slice time, the question is "did the upstream
  classifier ever see a drug with this scaffold string?" — and "" is one
  such string.

Reuses `_to_dilist_id` from `scaffold_split` for ID-format symmetry; defines
the `_from_dilist_id` inverse locally (parsing helper not worth re-exporting
from scaffold_split).
"""

from __future__ import annotations

import logging

import pandas as pd

# Re-import for symmetry / consumer convenience (Wave 3 CLI driver may want it).
from .scaffold_split import _to_dilist_id  # noqa: F401

log = logging.getLogger(__name__)

__all__ = ["compute_transfer_slices"]


REQUIRED_COLUMNS: frozenset[str] = frozenset({"pert_id", "drug_name", "scaffold"})


def _from_dilist_id(s: str) -> int:
    """Inverse of `_to_dilist_id` — strip 'DILIST_' prefix and return int."""
    if not s.startswith("DILIST_"):
        raise ValueError(
            f"Expected DILIST_NNNN format, got {s!r}. "
            "Pass scaffold_split() output directly without modification."
        )
    return int(s.removeprefix("DILIST_"))


def compute_transfer_slices(
    test_pert_ids: list[str],
    dili_canonical_df: pd.DataFrame,
    pdg_drug_names: set[str],
    train_scaffolds: set[str],
) -> dict[str, list[str]]:
    """Compute the three transfer slices on D_DILI_test (SPLIT-05).

    Parameters
    ----------
    test_pert_ids : list[str]
        `scaffold_split_dict['test']` — list of `DILIST_NNNN` strings.
    dili_canonical_df : pd.DataFrame
        Phase 1 output `dili_canonical.csv`. Must have columns
        ['pert_id', 'drug_name', 'scaffold']. (`in_pdg` is also expected but
        we don't rely on it here — see "Note on `in_pdg`" below.)
    pdg_drug_names : set[str]
        Lowercased PDG drug_names (from `all_drugs_pdg.csv → drug_name`,
        lowercased). Source of truth for "is in PDG"; Phase 1 used the same
        comparison method per `01-02-SUMMARY.md`.
    train_scaffolds : set[str]
        Set of scaffold strings present across `scaffold_split_dict['train']`.
        Caller (Wave 3 driver) builds this by looking up each train pert_id
        in dili_canonical_df and collecting its `scaffold` value. Empty
        string "" is included if any train drug is acyclic.

    Returns
    -------
    dict[str, list[str]]
        Three lists of `DILIST_NNNN` strings, each sorted lexicographically:
          {
            "test_in_pdg": [...],
            "test_drug_novel": [...],
            "test_drug_and_scaffold_novel": [...],
          }

    Note on `in_pdg`
    ----------------
    Phase 1 set a boolean `in_pdg` column on the canonical CSV via
    `drug_name.lower() in pdg_drug_names`. We re-derive the membership here
    rather than reading the column for two reasons: (a) PDG snapshots may
    have refreshed between Phase 1 and Phase 2, and (b) keeping the join
    method explicit makes Phase 5's halt-gate evaluation legible. We do
    log a debug warning if Phase 1's `in_pdg` disagrees with our computed
    membership for any test row, which would surface a snapshot drift.
    """
    # --------------------------------------------------------------
    # 0. Validate inputs
    # --------------------------------------------------------------
    missing = REQUIRED_COLUMNS - set(dili_canonical_df.columns)
    if missing:
        raise KeyError(
            f"compute_transfer_slices: dili_canonical_df missing columns "
            f"{sorted(missing)}. Saw: {dili_canonical_df.columns.tolist()}"
        )

    # Build a fast lookup from pert_id (int) → row.
    canon = dili_canonical_df.set_index("pert_id", drop=False)

    # Defensive lowercasing of the PDG name set — caller is supposed to pass
    # already-lowercased names but be tolerant of the alternative.
    pdg_set = {str(n).strip().lower() for n in pdg_drug_names}

    # --------------------------------------------------------------
    # 1. Walk test_pert_ids; classify each into in_pdg vs drug_novel
    # --------------------------------------------------------------
    in_pdg: list[str] = []
    drug_novel: list[str] = []
    drug_and_scaffold_novel: list[str] = []

    has_in_pdg_col = "in_pdg" in dili_canonical_df.columns
    n_in_pdg_drift = 0
    n_unresolved = 0

    for pid_str in test_pert_ids:
        pid_int = _from_dilist_id(pid_str)
        if pid_int not in canon.index:
            n_unresolved += 1
            log.warning(
                "compute_transfer_slices: pert_id %d (%s) not found in "
                "canonical CSV; skipping for slicing.",
                pid_int, pid_str,
            )
            continue
        row = canon.loc[pid_int]
        drug_name_lower = str(row["drug_name"]).strip().lower()
        scaffold = row["scaffold"]
        if not isinstance(scaffold, str):
            # Defensive: NaN scaffold treated like acyclic ""
            scaffold = ""

        is_in_pdg = drug_name_lower in pdg_set

        # Drift check — flag (don't fix) if Phase 1's in_pdg disagrees.
        if has_in_pdg_col:
            phase1_flag = bool(row["in_pdg"])
            if phase1_flag != is_in_pdg:
                n_in_pdg_drift += 1

        if is_in_pdg:
            in_pdg.append(pid_str)
        else:
            drug_novel.append(pid_str)
            # Strongest slice: also scaffold-novel?
            # Empty-scaffold "" is treated as a single bucket — if "" is in
            # train_scaffolds (any train drug acyclic), then this acyclic
            # test drug is NOT scaffold-novel. See module docstring.
            if scaffold not in train_scaffolds:
                drug_and_scaffold_novel.append(pid_str)

    if n_unresolved:
        log.warning(
            "compute_transfer_slices: %d/%d test pert_ids did not resolve in "
            "canonical CSV.", n_unresolved, len(test_pert_ids),
        )
    if n_in_pdg_drift:
        log.debug(
            "compute_transfer_slices: %d test rows where canonical CSV "
            "`in_pdg` disagreed with computed membership against pdg_drug_names "
            "— possible snapshot drift between Phase 1 and Phase 2.",
            n_in_pdg_drift,
        )

    # --------------------------------------------------------------
    # 2. Sort + return
    # --------------------------------------------------------------
    out: dict[str, list[str]] = {
        "test_in_pdg": sorted(in_pdg),
        "test_drug_novel": sorted(drug_novel),
        "test_drug_and_scaffold_novel": sorted(drug_and_scaffold_novel),
    }

    log.info(
        "compute_transfer_slices: |test_in_pdg|=%d, |test_drug_novel|=%d, "
        "|test_drug_and_scaffold_novel|=%d (halt-gate input is "
        "|test_drug_novel|; threshold is 30)",
        len(out["test_in_pdg"]),
        len(out["test_drug_novel"]),
        len(out["test_drug_and_scaffold_novel"]),
    )

    return out
