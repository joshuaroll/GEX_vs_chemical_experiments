"""Pure library: resolve Wang/Li `compound_name` -> SMILES + DILIrank severity.

Phase 1 / Plan 01-03 deliverable. The CLI driver in `scripts/build_phase1_wangli.py`
(Wave 2) wires this with the actual canonical CSV and the LINCS lookup library.

SMILES resolution flow (per CONTEXT.md §Decisions):
    Wang/Li `compound_name`  --strip salt suffix-->  parent token
                                                      |
            canonical drug_name  --strip salt suffix--+--> dict lookup
                                                      |
                                            -> (canonical_smiles, dili_severity)

Match policy is a single, locked invariant: **"always salt-strip both sides"**.
Both the canonical-side lookup key and the query-side compound name are run
through `_strip_salt_suffix` before dict lookup. This is strictly stronger than
the v0.4 P1 "lowercased-exact, then salt-strip fallback" two-pass scheme — the
salt-strip helper is a no-op when no recognized salt token is present, so the
exact-lowercased match is naturally subsumed. The simpler invariant is preferred
because it removes the resolution-order dependency surface from the contract.

Proven heuristic: the salt-suffix-stripping pattern lifted DILIst severity
coverage 433 -> 767 in v0.4 P1 (per
`.planning/archive/phases-v0_4/01-data-foundation/01-02-SUMMARY.md`); the same
mechanism applies here for SMILES resolution against the canonical drug-name table.

Hard rules honored:
    - Pure library: NO source-of-truth paths, NO hardcoded absolute paths.
      The caller passes a `pd.DataFrame` constructed from the canonical CSV.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Salt suffix list (proven in v0.4 P1, extended per its 01-02-SUMMARY)
# ---------------------------------------------------------------------------
#
# Verbatim from `src/data/build_dili_canonical.py::_SALT_SUFFIXES` (v0.4 P1
# Plan 02 deviation: salt-suffix-stripping lifted DILIrank severity coverage
# 433 -> 767). The constant is duplicated rather than imported so this module
# stays independently importable (matches the v0.4 pattern where the canonical
# file hardcoded its own copy).
SALT_SUFFIXES: frozenset[str] = frozenset({
    # Acid salts and conjugate-base anions
    "hydrochloride", "dihydrochloride", "hydrobromide",
    "sulfate", "sulphate", "phosphate", "diphosphate",
    "bromide", "iodide", "chloride",
    "acetate", "tartrate", "citrate", "fumarate", "succinate", "maleate",
    "mesylate", "tosylate", "besylate", "lactate", "malate",
    "edisylate", "gluconate", "glucuronate", "nitrate",
    "valerate", "propionate", "decanoate", "benzoate", "furoate", "enanthate",
    "stearate", "palmitate",
    "isethionate",
    "estolate",
    "glycinate",
    "succinylsulfate",
    "embonate",
    "edetate",
    "pamoate",
    # Counter-cations
    "sodium", "potassium", "calcium", "magnesium", "lithium",
    "olamine", "diethanolamine", "ethanolamine",
    "dimeglumine", "meglumine",
    # Hydrate / packaging tokens
    "trihydrate", "dihydrate", "monohydrate", "hydrate",
    "hyclate",
    # Prodrug/ester tokens commonly trailing
    "pivoxil", "fosamil",
    # Generic packaging / formulation noun (from v0.4-summary extensions)
    "complex",
})


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class ResolvedSmiles(NamedTuple):
    """Locked output of `resolve_smiles`.

    All four fields align positionally to the input `compound_names` list.

    Attributes
    ----------
    smiles : list[Optional[str]]
        `None` at positions where no canonical row matched.
    severity : list[Optional[str]]
        `None` at positions where no canonical row matched OR where the
        matched row had NaN/empty severity.
    drop_indices : list[int]
        Input indices where SMILES was unresolved (subset of
        `range(len(compound_names))`).
    drop_names : list[str]
        The original (unmodified) compound_names at the dropped indices, for
        the Wave-2 driver's per-name failure log in
        `results/tables/P1_wangli_data_summary.md`.
    """

    smiles: list[Optional[str]]
    severity: list[Optional[str]]
    drop_indices: list[int]
    drop_names: list[str]


# ---------------------------------------------------------------------------
# Salt-suffix stripper
# ---------------------------------------------------------------------------


def _strip_salt_suffix(name: str) -> str:
    """Lowercase + iteratively strip trailing tokens that are in `SALT_SUFFIXES`.

    The loop continues until the rightmost token is no longer a recognized
    salt suffix, OR no tokens remain. Matches the v0.4 P1 multi-token strip
    behavior ("X Sodium Glycinate" -> "X Sodium" -> "X") plus the
    "X Salt Complex" trailing-noun pattern.

    Returns the input lowercased + whitespace-trimmed if no salt token is
    present. This makes `_strip_salt_suffix` a no-op for plain names, which is
    the property that lets us use the same call on both lookup-build and query
    sides without a separate "exact-lowercased-only" first pass.
    """
    parts = name.lower().split()
    # Iteratively strip trailing salt tokens. We must not strip down to zero
    # tokens (a name that consists ONLY of salt tokens would otherwise become
    # the empty string, which then matches a different empty-string sentinel
    # and could create false hits).
    while len(parts) >= 2 and parts[-1] in SALT_SUFFIXES:
        parts = parts[:-1]
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------


_REQUIRED_COLUMNS: frozenset[str] = frozenset({"drug_name", "canonical_smiles", "dili_severity"})


def resolve_smiles(
    compound_names: list[str],
    canonical_df: pd.DataFrame,
) -> ResolvedSmiles:
    """Resolve each Wang/Li compound_name to SMILES + DILI severity.

    Parameters
    ----------
    compound_names : list[str]
        Wang/Li `compound_name` column from `wangli_loader.load_split_pickle`.
        Order is preserved in the output.
    canonical_df : pd.DataFrame
        Must contain columns `{'drug_name', 'canonical_smiles', 'dili_severity'}`.
        Other columns are ignored. Sourced from the v0.4 P1 canonical CSV.

    Returns
    -------
    ResolvedSmiles
        See class docstring. `smiles[i]` and `severity[i]` are positional with
        the input list. Unresolved positions carry `None` and have their
        index/name appended to the `drop_*` lists.

    Raises
    ------
    ValueError
        If `canonical_df` is missing any of the three required columns.
    """
    # --- Schema validation --------------------------------------------------
    missing_cols = _REQUIRED_COLUMNS - set(canonical_df.columns)
    if missing_cols:
        # Stable, deterministic order in the message for easier debugging.
        cols_str = ", ".join(sorted(missing_cols))
        raise ValueError(
            f"canonical_df missing required column(s): {cols_str}. "
            f"Required schema: drug_name, canonical_smiles, dili_severity."
        )

    # --- Build lookup dict --------------------------------------------------
    # Always salt-strip the canonical-side key. First-occurrence wins on
    # duplicate keys (matches the test_resolve_smiles_duplicate_canonical_drug_name_uses_first
    # contract).
    lookup: dict[str, tuple[Optional[str], Optional[str]]] = {}
    for _, row in canonical_df.iterrows():
        raw_drug_name = row["drug_name"]
        if not isinstance(raw_drug_name, str):
            # Skip non-string keys (e.g., NaN) to avoid AttributeError in
            # _strip_salt_suffix. The real canonical CSV has all-string
            # drug_name; this is defensive against future schema drift.
            continue
        key = _strip_salt_suffix(raw_drug_name)
        if not key:
            # All-salt-token canonical row collapses to "" — skip rather than
            # poison the dict.
            continue
        if key in lookup:
            # First-occurrence wins.
            continue
        smiles_val = row["canonical_smiles"]
        if not isinstance(smiles_val, str):
            # The real canonical CSV has non-null canonical_smiles, but be
            # defensive.
            smiles_val = None
        sev_val = row["dili_severity"]
        if not isinstance(sev_val, str) or not sev_val:
            # Treat NaN / empty as None.
            sev_val = None
        lookup[key] = (smiles_val, sev_val)

    # --- Resolve queries ---------------------------------------------------
    smiles_out: list[Optional[str]] = []
    severity_out: list[Optional[str]] = []
    drop_indices: list[int] = []
    drop_names: list[str] = []

    for i, raw_query in enumerate(compound_names):
        if not isinstance(raw_query, str):
            # Defensive: non-string queries are unresolvable.
            smiles_out.append(None)
            severity_out.append(None)
            drop_indices.append(i)
            drop_names.append(raw_query if raw_query is not None else "")
            continue
        query_key = _strip_salt_suffix(raw_query)
        hit = lookup.get(query_key)
        if hit is None:
            smiles_out.append(None)
            severity_out.append(None)
            drop_indices.append(i)
            drop_names.append(raw_query)
        else:
            smiles_out.append(hit[0])
            severity_out.append(hit[1])

    # --- Resolution-rate log ----------------------------------------------
    n_total = len(compound_names)
    n_resolved = n_total - len(drop_indices)
    if n_total > 0:
        rate_pct = 100.0 * n_resolved / n_total
        log.info(
            "wangli_smiles_resolver: resolved %d / %d compounds (%.2f%%); "
            "%d drops",
            n_resolved,
            n_total,
            rate_pct,
            len(drop_indices),
        )

    return ResolvedSmiles(
        smiles=smiles_out,
        severity=severity_out,
        drop_indices=drop_indices,
        drop_names=drop_names,
    )
