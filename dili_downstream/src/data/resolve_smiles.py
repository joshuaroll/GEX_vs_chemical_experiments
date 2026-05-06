"""DILIst → DrugBank SMILES resolver + RDKit canonicalizer.

Pure-Python library used by ``scripts/resolve_smiles.py``. No I/O at import
time; all paths are caller-supplied.

Public API
----------
- ``canonicalize(smiles)`` → ``(canonical_smiles, error)``
- ``resolve_dilist(dilist_df, index_df)`` → ``(resolved_df, failures_df)``

Resolution pipeline (per DATA-04, locked in 01-CONTEXT.md):
  1. Lowercase + strip DILIst's ``CompoundName`` → ``name_lower``.
  2. Left-merge with the DrugBank index on ``name_lower``.
  3. Rows whose merged ``smiles`` is null → failures with reason
     ``name_not_in_drugbank_index``.
  4. Remaining rows → RDKit ``MolFromSmiles`` → ``MolToSmiles``. Parse failures
     → failures with reason ``rdkit_parse_failure``.
  5. Round-trip check ``MolToSmiles(MolFromSmiles(canonical)) == canonical`` —
     a row that fails round-trip is moved to failures with reason
     ``rdkit_parse_failure`` (defensive; this should be near-zero on a clean
     index).

The 90% gate (≥ 1,151 / 1,279 resolved) is enforced by the CLI driver
``scripts/resolve_smiles.py``, NOT here — this module returns frames and lets
the caller decide what to do with the rate.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# RDKit's parser logs a noisy [00:00:00] WARNING for every malformed SMILES it
# rejects. Silencing here keeps the resolver run output readable; the failures
# CSV is the canonical record of every parse failure.
RDLogger.DisableLog("rdApp.*")

DILIST_NAME_COL = "CompoundName"
DILIST_ID_COL = "DILIST_ID"

REASON_NOT_IN_INDEX = "name_not_in_drugbank_index"
REASON_RDKIT_PARSE_FAILURE = "rdkit_parse_failure"

# Common pharmaceutical salt / counter-ion suffixes. DILIst frequently lists
# drugs with the salt-form suffix (e.g. ``Levocetirizine dihydrochloride``)
# while DrugBank indexes the parent compound (``Levocetirizine``). Stripping
# the suffix and retrying the lookup recovers ~25 DILIst rows. We do NOT use
# RDKit's salt-stripping here because we are matching by NAME, not by molecule;
# the SMILES we adopt comes from the parent compound entry in DrugBank.
SALT_SUFFIXES: tuple[str, ...] = (
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
)


def _strip_salt_suffix(name_lower: str) -> str | None:
    """Return ``name_lower`` with a single trailing salt-suffix token removed,
    or ``None`` if no recognized salt suffix is present.

    Only strips ONE token at the end — multi-suffix names ('X sodium hydrate')
    are not common enough in DILIst to justify the complexity, and would
    risk false positives.
    """
    parts = name_lower.split()
    if len(parts) < 2:
        return None
    if parts[-1] in SALT_SUFFIXES:
        candidate = " ".join(parts[:-1]).strip()
        return candidate or None
    return None

RESOLVED_COLUMNS = [
    "DILIST_ID",
    "drug_name",
    "name_lower",
    "smiles",
    "canonical_smiles",
    "source",
]
FAILURE_COLUMNS = ["DILIST_ID", "drug_name", "reason"]

# PubChem fallback NOT YET WIRED — add only if drugbank-only rate < 90% per D-04.
# (Locked in 01-CONTEXT.md: PubChem fallback is reserved for a future iteration
# only if we miss the 90% target on the real data. Do not stub it.)


def canonicalize(smiles: str) -> tuple[str | None, str | None]:
    """Run RDKit canonicalization on ``smiles``.

    Returns
    -------
    (canonical, error)
        ``canonical`` is the round-trip-safe canonical SMILES, or ``None`` if
        RDKit could not parse the input. ``error`` is ``None`` on success or
        a short reason string on failure.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return (None, REASON_RDKIT_PARSE_FAILURE)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, REASON_RDKIT_PARSE_FAILURE)
    canon = Chem.MolToSmiles(mol)
    # Defensive round-trip — if the canonical form doesn't itself round-trip,
    # treat it as a parse failure so the row goes to the failures log instead
    # of silently entering the resolved frame with a brittle SMILES.
    mol2 = Chem.MolFromSmiles(canon)
    if mol2 is None or Chem.MolToSmiles(mol2) != canon:
        return (None, REASON_RDKIT_PARSE_FAILURE)
    return (canon, None)


def resolve_dilist(
    dilist_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve DILIst rows against a DrugBank ``name_lower → smiles`` index.

    Parameters
    ----------
    dilist_df
        Must contain ``DILIST_ID`` and ``CompoundName`` columns.
    index_df
        Must contain ``name_lower`` and ``smiles`` columns (output of
        ``drugbank_smiles_index.build_index``).

    Returns
    -------
    (resolved_df, failures_df)
        See ``RESOLVED_COLUMNS`` and ``FAILURE_COLUMNS`` for shape.
    """
    if DILIST_ID_COL not in dilist_df.columns:
        raise KeyError(f"dilist_df missing {DILIST_ID_COL!r}")
    if DILIST_NAME_COL not in dilist_df.columns:
        raise KeyError(f"dilist_df missing {DILIST_NAME_COL!r}")
    for col in ("name_lower", "smiles"):
        if col not in index_df.columns:
            raise KeyError(f"index_df missing {col!r}")

    work = dilist_df[[DILIST_ID_COL, DILIST_NAME_COL]].copy()
    work = work.rename(columns={DILIST_NAME_COL: "drug_name"})
    work["name_lower"] = work["drug_name"].astype(str).str.strip().str.lower()

    # De-dup the index by name_lower (keep first) defensively, in case the
    # caller passed a non-deduped frame.
    idx = index_df[["name_lower", "smiles"]].drop_duplicates(
        subset=["name_lower"], keep="first"
    )
    # Build a fast dict lookup for the salt-stripping retry pass below.
    idx_lookup: dict[str, str] = dict(zip(idx["name_lower"], idx["smiles"]))

    merged = work.merge(idx, on="name_lower", how="left")

    failures: list[dict] = []
    resolved: list[dict] = []

    for row in merged.itertuples(index=False):
        smiles = row.smiles
        # `smiles` may be NaN for rows whose name_lower wasn't in the index.
        if not isinstance(smiles, str) or not smiles:
            # Salt-suffix retry: 'X hydrochloride' → 'X' lookup.
            stripped = _strip_salt_suffix(row.name_lower)
            if stripped and stripped in idx_lookup:
                smiles = idx_lookup[stripped]
            else:
                failures.append(
                    {
                        "DILIST_ID": row.DILIST_ID,
                        "drug_name": row.drug_name,
                        "reason": REASON_NOT_IN_INDEX,
                    }
                )
                continue

        canon, err = canonicalize(smiles)
        if canon is None:
            failures.append(
                {
                    "DILIST_ID": row.DILIST_ID,
                    "drug_name": row.drug_name,
                    "reason": err or REASON_RDKIT_PARSE_FAILURE,
                }
            )
            continue

        resolved.append(
            {
                "DILIST_ID": row.DILIST_ID,
                "drug_name": row.drug_name,
                "name_lower": row.name_lower,
                "smiles": smiles,
                "canonical_smiles": canon,
                "source": "drugbank",
            }
        )

    resolved_df = pd.DataFrame(resolved, columns=RESOLVED_COLUMNS)
    failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    return resolved_df, failures_df
