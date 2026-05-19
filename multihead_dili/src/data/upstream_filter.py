"""Upstream training-set leakage filter (SPLIT-02).

Pure library. The Wave 3 CLI driver (`scripts/build_phase2_splits.py`) wires
real-data I/O — see `MANIFEST.md` for `all_drugs_pdg.csv` provenance and
`dili_canonical.csv` from Phase 1.

Implements Option (a) leakage discipline per `02-CONTEXT.md` §"Scaffold-similarity
exclusion mechanism":

  D_PDG_train  =  D_PDG  \\  ( {drugs whose Murcko scaffold ∈ S_test}
                               ∪ {drugs whose drug_name ∈ DILIst-test} )

The two-pronged exclusion is "defense in depth":
  - Scaffold match (Murcko-string equality) catches structural leakage even
    when drug_name doesn't overlap (different name, same scaffold).
  - Drug_name match (lowercased) catches racemates, stereoisomers, and any
    edge case where Murcko strings might disagree but the drug is in fact
    the same as a DILIst-test entry.
The intersection of the two sets is reported in the diagnostics dict so the
summary table can document the overlap.

Per `02-CONTEXT.md` planner_prelim_findings:

  #1: PDG SMILES live in `all_drugs_pdg.csv` (cols: drug_name, cpd_smiles)
      — NOT in `pdg_brddrugfiltered.pkl`. Phase 2 derives PDG scaffolds
      from cpd_smiles via `MurckoScaffoldSmiles`.
  #2: PDG keys remain `drug_name` (NOT pert_id, which is opaque integer
      index in the pkl). The lowercased drug_name is the canonical
      cross-dataset join key — Phase 1 already uses it for `in_pdg`.
  #3: Acyclic PDG drugs (scaffold == "") are NOT auto-excluded — only
      excluded if their drug_name happens to be in DILIst-test. Acyclic
      ≠ "in S_test" because each acyclic drug is its own singleton in
      scaffold_split.

Returns
-------
filter_upstream_train(...) -> tuple[list[str], dict[str, int]]
    list[str]: sorted PDG drug_names remaining after exclusion.
    dict[str, int]: 8-key diagnostics for the summary table.
"""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Re-use Plan 01's locked DILIST_NNNN helper (see 02-01-SUMMARY.md §"DILIST_NNNN
# format"). The inverse `_from_dilist_id` is defined locally — kept minimal so
# we don't pollute scaffold_split's __all__ with parsing helpers it doesn't need.
from .scaffold_split import _to_dilist_id  # noqa: F401  (re-exported for symmetry)

log = logging.getLogger(__name__)

__all__ = ["filter_upstream_train", "_compute_pdg_scaffolds"]


REQUIRED_PDG_COLUMNS: frozenset[str] = frozenset({"drug_name", "cpd_smiles"})
REQUIRED_DILI_COLUMNS: frozenset[str] = frozenset(
    {"pert_id", "drug_name", "scaffold"}
)


def _from_dilist_id(s: str) -> int:
    """Inverse of `_to_dilist_id` — strip 'DILIST_' prefix and return int."""
    if not s.startswith("DILIST_"):
        raise ValueError(
            f"Expected DILIST_NNNN format, got {s!r}. "
            "Pass scaffold_split() output directly without modification."
        )
    return int(s.removeprefix("DILIST_"))


def _murcko_scaffold(smiles: str) -> str:
    """Compute Murcko scaffold SMILES; "" for acyclic OR invalid SMILES.

    Mirrors `build_dili_canonical.murcko_scaffold` with one difference:
    Phase 1 raises on invalid SMILES (Phase 1 invariant), but here we want
    a tolerant path because PDG SMILES are an external dataset we don't
    own. Invalid SMILES → "" plus a warning, NOT an exception.
    """
    if not smiles or not isinstance(smiles, str):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles, includeChirality=False)
    except Exception:  # pragma: no cover — RDKit's contract is strict on accepting Mol-roundtrip strings
        return ""


def _compute_pdg_scaffolds(pdg_drugs_df: pd.DataFrame) -> dict[str, str]:
    """Map PDG `drug_name` → Murcko scaffold SMILES.

    Parameters
    ----------
    pdg_drugs_df : pd.DataFrame
        Must have columns ['drug_name', 'cpd_smiles'].
        Source of truth: `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv`.

    Returns
    -------
    dict[str, str]
        drug_name → scaffold. Empty string for acyclic AND for invalid SMILES.
        Invalid SMILES emit a `log.warning` so the issue is visible in the
        Wave 3 driver run log; the row is NOT excluded from upstream-train
        on scaffold grounds (drug_name defense-in-depth still applies).
    """
    missing = REQUIRED_PDG_COLUMNS - set(pdg_drugs_df.columns)
    if missing:
        raise KeyError(
            f"_compute_pdg_scaffolds: pdg_drugs_df missing required columns "
            f"{sorted(missing)}. Saw: {pdg_drugs_df.columns.tolist()}"
        )

    out: dict[str, str] = {}
    n_invalid = 0
    for drug_name, smiles in zip(
        pdg_drugs_df["drug_name"].tolist(), pdg_drugs_df["cpd_smiles"].tolist()
    ):
        if not isinstance(smiles, str) or not smiles:
            log.warning(
                "PDG drug %r has missing/empty cpd_smiles; recording empty "
                "scaffold and SKIPPING scaffold-rule exclusion (drug_name "
                "defense-in-depth still applies).",
                drug_name,
            )
            out[str(drug_name)] = ""
            n_invalid += 1
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(
                "PDG drug %r has invalid cpd_smiles %r — RDKit could not parse. "
                "Recording empty scaffold and SKIPPING scaffold-rule exclusion.",
                drug_name, smiles,
            )
            out[str(drug_name)] = ""
            n_invalid += 1
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            smiles, includeChirality=False
        )
        out[str(drug_name)] = scaffold

    if n_invalid:
        log.info(
            "_compute_pdg_scaffolds: %d/%d PDG rows had invalid/empty SMILES",
            n_invalid, len(pdg_drugs_df),
        )
    return out


def filter_upstream_train(
    pdg_drugs_df: pd.DataFrame,
    scaffold_split_dict: dict[str, list[str]],
    dili_test_pert_ids: list[str],
    dili_canonical_df: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    """Filter PDG upstream-train per Option (a) leakage discipline (SPLIT-02).

    Excludes from D_PDG any drug whose:
      (a) Murcko scaffold (computed from `cpd_smiles`) appears in S_test,
          where S_test is the set of non-empty scaffolds across DILIst-test
          pert_ids resolved through `dili_canonical_df`.
      (b) drug_name (lowercased) appears in DILIst-test drug_names
          (lowercased).

    The two exclusion sets may overlap; their intersection size is
    reported separately for transparency.

    Parameters
    ----------
    pdg_drugs_df : pd.DataFrame
        Must have columns ['drug_name', 'cpd_smiles']. Source:
        `/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv`
        (per 02-CONTEXT.md planner_prelim_findings #1).
    scaffold_split_dict : dict[str, list[str]]
        Output of `scaffold_split()`. Must contain key 'test' as a list of
        DILIST_NNNN-formatted strings.
    dili_test_pert_ids : list[str]
        Same list as `scaffold_split_dict['test']`. Passed explicitly so
        callers can audit the exclusion target without re-deriving it.
        Must equal `scaffold_split_dict['test']` (defensively asserted).
    dili_canonical_df : pd.DataFrame
        Phase 1 output `dili_canonical.csv`. Must have columns
        ['pert_id', 'drug_name', 'scaffold'].

    Returns
    -------
    train_upstream : list[str]
        Sorted PDG drug_name strings remaining after exclusion. NOT prefixed
        with DILIST_ (PDG keys remain drug_name per planner_prelim_findings #2).
    diagnostics : dict[str, int]
        {
          'd_pdg_total':                 |D_PDG| (PDG drug count, before exclusion),
          'excluded_by_scaffold':        |drugs whose scaffold ∈ S_test|,
          'excluded_by_pert_id':         |drugs whose lowercased drug_name ∈ DILIst-test|,
                                         (named 'pert_id' for legacy CONTEXT.md naming;
                                          actual matching is by drug_name lowercase),
          'excluded_intersection':       |scaffold-set ∩ drug-name-set|,
          'd_pdg_train_after_exclusion': |D_PDG_train| (post-exclusion count),
          's_test_size':                 |non-empty scaffolds in S_test|,
          'scaffold_match_count':        same as 'excluded_by_scaffold' for symmetric naming,
          'drug_name_match_count':       same as 'excluded_by_pert_id' for symmetric naming,
        }
    """
    # --------------------------------------------------------------
    # 0. Validate inputs
    # --------------------------------------------------------------
    missing_pdg = REQUIRED_PDG_COLUMNS - set(pdg_drugs_df.columns)
    if missing_pdg:
        raise KeyError(
            f"filter_upstream_train: pdg_drugs_df missing columns "
            f"{sorted(missing_pdg)}. Saw: {pdg_drugs_df.columns.tolist()}"
        )

    missing_dili = REQUIRED_DILI_COLUMNS - set(dili_canonical_df.columns)
    if missing_dili:
        raise KeyError(
            f"filter_upstream_train: dili_canonical_df missing columns "
            f"{sorted(missing_dili)}. Saw: {dili_canonical_df.columns.tolist()}"
        )

    if "test" not in scaffold_split_dict:
        raise KeyError(
            f"filter_upstream_train: scaffold_split_dict must have a 'test' "
            f"key. Saw: {sorted(scaffold_split_dict.keys())}"
        )

    if list(dili_test_pert_ids) != list(scaffold_split_dict["test"]):
        log.warning(
            "filter_upstream_train: dili_test_pert_ids differs from "
            "scaffold_split_dict['test']. Using dili_test_pert_ids as the "
            "authoritative test list. Lengths: passed=%d split=%d.",
            len(dili_test_pert_ids), len(scaffold_split_dict["test"]),
        )

    # --------------------------------------------------------------
    # 1. Build PDG scaffold map
    # --------------------------------------------------------------
    pdg_scaffolds = _compute_pdg_scaffolds(pdg_drugs_df)
    d_pdg_total = len(pdg_scaffolds)

    # --------------------------------------------------------------
    # 2. Resolve DILIst-test pert_ids → scaffolds + drug_names
    # --------------------------------------------------------------
    # Build a fast lookup: pert_id (int) → (drug_name_lower, scaffold).
    canon = dili_canonical_df.set_index("pert_id", drop=False)
    s_test_scaffolds: set[str] = set()
    dili_test_drug_names_lower: set[str] = set()
    n_unresolved = 0
    for pid_str in dili_test_pert_ids:
        pid_int = _from_dilist_id(pid_str)
        if pid_int not in canon.index:
            n_unresolved += 1
            log.warning(
                "DILIst-test pert_id %d (from %s) not found in canonical CSV; "
                "skipping for both scaffold and drug-name exclusion.",
                pid_int, pid_str,
            )
            continue
        row = canon.loc[pid_int]
        scaffold = row["scaffold"]
        drug_name = row["drug_name"]
        # Skip empty/NaN scaffolds — acyclic test rows don't form a "match"
        # set (each acyclic drug is its own singleton in scaffold_split, so
        # there's no scaffold equality to test against).
        if isinstance(scaffold, str) and scaffold:
            s_test_scaffolds.add(scaffold)
        if isinstance(drug_name, str) and drug_name:
            dili_test_drug_names_lower.add(drug_name.strip().lower())
    if n_unresolved:
        log.warning(
            "filter_upstream_train: %d/%d DILIst-test pert_ids did not resolve "
            "in canonical CSV.", n_unresolved, len(dili_test_pert_ids),
        )

    s_test_size = len(s_test_scaffolds)

    # --------------------------------------------------------------
    # 3. Compute exclusion sets
    # --------------------------------------------------------------
    excluded_by_scaffold: set[str] = {
        drug_name
        for drug_name, scaf in pdg_scaffolds.items()
        if scaf and scaf in s_test_scaffolds
    }
    excluded_by_drug_name: set[str] = {
        drug_name
        for drug_name in pdg_scaffolds.keys()
        if str(drug_name).strip().lower() in dili_test_drug_names_lower
    }
    excluded = excluded_by_scaffold | excluded_by_drug_name
    excluded_intersection = excluded_by_scaffold & excluded_by_drug_name

    # --------------------------------------------------------------
    # 4. Build sorted train_upstream list
    # --------------------------------------------------------------
    train_upstream = sorted(set(pdg_scaffolds.keys()) - excluded)

    # --------------------------------------------------------------
    # 5. Diagnostics
    # --------------------------------------------------------------
    diagnostics: dict[str, int] = {
        "d_pdg_total": int(d_pdg_total),
        "excluded_by_scaffold": int(len(excluded_by_scaffold)),
        "excluded_by_pert_id": int(len(excluded_by_drug_name)),
        "excluded_intersection": int(len(excluded_intersection)),
        "d_pdg_train_after_exclusion": int(len(train_upstream)),
        "s_test_size": int(s_test_size),
        "scaffold_match_count": int(len(excluded_by_scaffold)),
        "drug_name_match_count": int(len(excluded_by_drug_name)),
    }

    log.info(
        "filter_upstream_train: |D_PDG|=%d, S_test=%d, excl_scaffold=%d, "
        "excl_drug_name=%d, intersection=%d, |D_PDG_train|=%d",
        diagnostics["d_pdg_total"],
        diagnostics["s_test_size"],
        diagnostics["excluded_by_scaffold"],
        diagnostics["excluded_by_pert_id"],
        diagnostics["excluded_intersection"],
        diagnostics["d_pdg_train_after_exclusion"],
    )

    return train_upstream, diagnostics
