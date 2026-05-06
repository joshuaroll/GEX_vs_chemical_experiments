"""Build the canonical 9-column DILI table per DATA-05 (locked schema).

Pure library. No file I/O or CLI here — see `scripts/build_dili_canonical.py` for
the driver that wires DILIst.xlsx + DILIrank.xlsx + resolved-SMILES CSV + LINCS GCTX
+ PDG pickle into this function.

Schema (locked in 01-CONTEXT.md):
    pert_id, drug_name, smiles, canonical_smiles, scaffold,
    dili_binary, dili_severity, in_lincs, in_pdg

Invariants enforced:
- Exactly the 9 columns above, in that order.
- `pert_id` taken from DILIst's `DILIST_ID` column; unique.
- `dili_binary` integer 0/1 with no NaN (rows with non-binary classification dropped).
- `canonical_smiles` non-nullable (rows without resolved SMILES dropped).
- `dili_severity` populated only for the DILIst ∩ DILIrank-2.0 intersection by
  lowercased CompoundName match (per 01-CONTEXT.md: "normalize compound names to lowercase").
- `scaffold` from `MurckoScaffold.MurckoScaffoldSmiles`; "" allowed (acyclic).
- `in_lincs` / `in_pdg` are bool dtype reflecting lowercased drug-name membership.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)


CANONICAL_COLUMNS: list[str] = [
    "pert_id",
    "drug_name",
    "smiles",
    "canonical_smiles",
    "scaffold",
    "dili_binary",
    "dili_severity",
    "in_lincs",
    "in_pdg",
]


# DILIrank lists many entries in salted form (e.g. "abacavir sulfate") while
# DILIst uses the parent name ("abacavir"). Plan 01's resolver uses the same
# tactic for SMILES lookup; reusing it here lifts severity coverage from
# ~536 to ~1,037 entries (the value 01-CONTEXT.md projects).
_SALT_SUFFIXES: frozenset[str] = frozenset({
    "hydrochloride", "dihydrochloride", "hydrobromide", "sulfate", "sulphate",
    "phosphate", "diphosphate", "bromide", "iodide", "chloride",
    "acetate", "tartrate", "citrate", "fumarate", "succinate", "maleate",
    "mesylate", "tosylate", "besylate", "lactate", "malate",
    "sodium", "potassium", "calcium", "magnesium", "lithium",
    "olamine", "pamoate", "stearate", "palmitate",
    "trihydrate", "dihydrate", "monohydrate", "hydrate",
    "edisylate", "gluconate", "glucuronate", "nitrate",
    "dimeglumine", "meglumine", "hyclate", "valerate", "propionate",
    "decanoate", "benzoate", "furoate", "enanthate",
    "isethionate", "estolate", "glycinate", "succinylsulfate",
    "diethanolamine", "ethanolamine", "embonate", "edetate",
    "pivoxil", "fosamil",
})


def _strip_salt_suffix(name_lower: str) -> str:
    """Return `name_lower` with up to TWO trailing salt-suffix tokens removed.

    Mirrors `src/data/resolve_smiles._strip_salt_suffix` but extends to handle
    multi-token suffixes like "X sodium glycinate" → "X glycinate" → "X" or
    "X phosphate complex" (where 'complex' is an additional token DILIrank uses).
    Returns the original string if no salt suffix is present, so callers can
    build a "best effort" lower-name column in one pass.
    """
    parts = name_lower.split()
    # Single-token strip
    if len(parts) >= 2 and parts[-1] in _SALT_SUFFIXES:
        candidate_parts = parts[:-1]
        # Try a second strip pass for multi-token suffixes ("X sodium glycinate")
        if len(candidate_parts) >= 2 and candidate_parts[-1] in _SALT_SUFFIXES:
            candidate_parts = candidate_parts[:-1]
        candidate = " ".join(candidate_parts).strip()
        if candidate:
            return candidate
    # Special-case "X phosphate complex" / "X sodium complex" patterns
    if len(parts) >= 3 and parts[-1] == "complex" and parts[-2] in _SALT_SUFFIXES:
        candidate = " ".join(parts[:-2]).strip()
        if candidate:
            return candidate
    return name_lower


def murcko_scaffold(canonical_smiles: str) -> str:
    """Return the Murcko scaffold SMILES; empty string for acyclic input.

    Caller should pass an RDKit-canonicalized SMILES (Plan 01 already does this).
    Raises whatever RDKit raises on unparseable input — we treat that as a bug
    upstream rather than silently dropping rows.
    """
    return MurckoScaffold.MurckoScaffoldSmiles(canonical_smiles, includeChirality=False)


def _find_classification_column(dilist_df: pd.DataFrame) -> str:
    """Locate the `DILIst Classification` column, tolerating trailing whitespace.

    Real-world DILIst.xlsx has a trailing space ('DILIst Classification ').
    """
    for col in dilist_df.columns:
        if col.strip() == "DILIst Classification":
            return col
    raise KeyError(
        "DILIst input is missing the 'DILIst Classification' column "
        f"(saw: {dilist_df.columns.tolist()})"
    )


def _find_dilirank_name_column(dilirank_df: pd.DataFrame) -> str:
    """Locate the DILIrank compound-name column (real label is 'CompoundName')."""
    for candidate in ("CompoundName", "Compound Name", "compound_name"):
        if candidate in dilirank_df.columns:
            return candidate
    # Allow a stripped variant in case of trailing whitespace.
    stripped = {c.strip(): c for c in dilirank_df.columns}
    for candidate in ("CompoundName", "Compound Name", "compound_name"):
        if candidate in stripped:
            return stripped[candidate]
    raise KeyError(
        "DILIrank input is missing a recognizable compound-name column "
        f"(saw: {dilirank_df.columns.tolist()})"
    )


def build_canonical(
    dilist_df: pd.DataFrame,
    dilirank_df: pd.DataFrame,
    resolved_df: pd.DataFrame,
    lincs_inames_lower: Iterable[str],
    pdg_inames_lower: Iterable[str],
) -> pd.DataFrame:
    """Produce the 9-column canonical DILI table.

    Parameters
    ----------
    dilist_df : raw DILIst (cols: DILIST_ID, CompoundName, DILIst Classification[, Routs...])
    dilirank_df : DILIrank 2.0 'version 2' sheet, header=1 (cols include CompoundName,
        vDILI-Concern, LTKBID)
    resolved_df : Plan 01 output (cols: DILIST_ID, drug_name, smiles, canonical_smiles,
        and possibly name_lower, source)
    lincs_inames_lower : set/iterable of lowercased LINCS drug names (pert_iname)
    pdg_inames_lower : set/iterable of lowercased PDG drug names (pert_id in PDG pickle)

    Returns
    -------
    pd.DataFrame with columns CANONICAL_COLUMNS in that order.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs and locate columns
    # ------------------------------------------------------------------
    classification_col = _find_classification_column(dilist_df)

    required_dilist = {"DILIST_ID", "CompoundName"}
    missing = required_dilist - set(dilist_df.columns)
    if missing:
        raise KeyError(f"DILIst missing columns: {sorted(missing)}")

    required_resolved = {"DILIST_ID", "canonical_smiles", "smiles"}
    missing = required_resolved - set(resolved_df.columns)
    if missing:
        raise KeyError(f"resolved_df missing columns: {sorted(missing)}")

    if "vDILI-Concern" not in dilirank_df.columns:
        raise KeyError(
            "DILIrank missing 'vDILI-Concern' column (this is the locked severity column "
            f"per 01-CONTEXT.md). Saw: {dilirank_df.columns.tolist()}"
        )
    dilirank_name_col = _find_dilirank_name_column(dilirank_df)

    # ------------------------------------------------------------------
    # 2. Filter DILIst to rows with binary classification in {0, 1}
    # ------------------------------------------------------------------
    dilist = dilist_df.copy()
    dilist["_class"] = pd.to_numeric(dilist[classification_col], errors="coerce")
    valid_mask = dilist["_class"].isin([0, 1])
    n_dropped_class = int((~valid_mask).sum())
    if n_dropped_class:
        log.warning(
            "Dropping %d DILIst row(s) with non-binary 'DILIst Classification' "
            "(values not in {0, 1})", n_dropped_class
        )
    dilist = dilist.loc[valid_mask, ["DILIST_ID", "CompoundName", "_class"]].copy()
    dilist["dili_binary"] = dilist["_class"].astype(int)
    dilist = dilist.drop(columns=["_class"])

    # ------------------------------------------------------------------
    # 3. Inner-merge with resolved SMILES (drops rows that didn't resolve)
    # ------------------------------------------------------------------
    resolved_keep = resolved_df[["DILIST_ID", "smiles", "canonical_smiles"]].copy()
    pre_n = len(dilist)
    df = dilist.merge(resolved_keep, on="DILIST_ID", how="inner")
    n_dropped_unresolved = pre_n - len(df)
    if n_dropped_unresolved:
        log.info(
            "Dropping %d DILIst row(s) without a resolved SMILES (would produce "
            "NaN canonical_smiles which violates the non-nullable contract)",
            n_dropped_unresolved,
        )

    # ------------------------------------------------------------------
    # 4. Severity merge (left join on lowercased compound name with
    #    salt-suffix-stripped fallback — same tactic Plan 01's resolver uses
    #    for SMILES lookup. Without the salt-stripped fallback, severity
    #    coverage drops from ~1,037 to ~536 because DILIrank lists salted
    #    forms and DILIst uses parent names.)
    # ------------------------------------------------------------------
    df["_name_lower"] = df["CompoundName"].astype(str).str.strip().str.lower()

    sev = dilirank_df[[dilirank_name_col, "vDILI-Concern"]].copy()
    sev["_name_lower"] = sev[dilirank_name_col].astype(str).str.strip().str.lower()
    sev["_name_lower_stripped"] = sev["_name_lower"].apply(_strip_salt_suffix)

    # Build a name -> severity lookup, preferring exact match over stripped match.
    sev_exact = (
        sev.drop_duplicates(subset=["_name_lower"], keep="first")
           .set_index("_name_lower")["vDILI-Concern"]
           .to_dict()
    )
    sev_stripped = (
        sev.drop_duplicates(subset=["_name_lower_stripped"], keep="first")
           .set_index("_name_lower_stripped")["vDILI-Concern"]
           .to_dict()
    )

    def _lookup_severity(name_lower: str) -> str | float:
        # 1. Direct match (DILIrank parent already in unsalted form)
        if name_lower in sev_exact:
            return sev_exact[name_lower]
        # 2. Stripped DILIst name → DILIrank parent (e.g. "lonidamine sodium" → "lonidamine")
        stripped_dilist = _strip_salt_suffix(name_lower)
        if stripped_dilist != name_lower and stripped_dilist in sev_exact:
            return sev_exact[stripped_dilist]
        # 3. DILIst parent → DILIrank-stripped name (e.g. DILIst "abacavir" matches
        #    DILIrank "abacavir sulfate" stripped to "abacavir").
        if name_lower in sev_stripped:
            return sev_stripped[name_lower]
        return float("nan")

    df["dili_severity"] = df["_name_lower"].apply(_lookup_severity)

    # ------------------------------------------------------------------
    # 5. Murcko scaffold
    # ------------------------------------------------------------------
    df["scaffold"] = df["canonical_smiles"].apply(murcko_scaffold)

    # ------------------------------------------------------------------
    # 6 + 7. in_lincs / in_pdg flags
    # ------------------------------------------------------------------
    lincs_set = {str(x).strip().lower() for x in lincs_inames_lower}
    pdg_set = {str(x).strip().lower() for x in pdg_inames_lower}
    # Use the DILIst CompoundName as the canonical drug_name (the resolver
    # lowercased it; we want the human-readable form here).
    drug_name_lower = df["CompoundName"].astype(str).str.strip().str.lower()
    df["in_lincs"] = drug_name_lower.isin(lincs_set).astype(bool)
    df["in_pdg"] = drug_name_lower.isin(pdg_set).astype(bool)

    # ------------------------------------------------------------------
    # 8. Final renaming + column ordering
    # ------------------------------------------------------------------
    df = df.rename(columns={"DILIST_ID": "pert_id", "CompoundName": "drug_name"})

    # Make sure dili_binary is integer (merge sometimes upcasts).
    df["dili_binary"] = df["dili_binary"].astype(int)

    out = df[CANONICAL_COLUMNS].copy()

    # Defensive: drop any duplicate pert_id (DILIst should already be unique, but
    # if upstream drift creates dupes, we surface it loudly rather than silently).
    if not out["pert_id"].is_unique:
        n_before = len(out)
        out = out.drop_duplicates(subset=["pert_id"], keep="first")
        log.warning(
            "Dropped %d duplicate pert_id rows (DILIst was expected to be unique)",
            n_before - len(out),
        )

    return out.reset_index(drop=True)
