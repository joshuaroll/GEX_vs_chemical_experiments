"""Unit tests for `src/data/cluster_split.py` (SPLIT-03).

Tanimoto single-linkage cluster split at threshold 0.4 on Morgan
fingerprints (radius=2, nBits=2048). In-memory fixtures only.

Behaviors covered (per 02-01-PLAN.md):
  1. Output keys exactly {"train", "val", "test"} with DILIST_NNNN strings.
  2. Determinism — same args twice → identical dict.
  3. Disjointness — pairwise empty intersections.
  4. Cluster integrity — Tanimoto-similar drugs (≥0.4 single-linkage)
     land in the same partition.
  5. 80/10/10 ratios within ±10pp tolerance (cluster splits are coarser
     than scaffold splits).
  6. Stratification fallback — when a cluster is too large to stratify,
     a non-stratified warning is logged.
  7. Empty/invalid SMILES rejection — defensive ValueError on empty SMILES.
"""

from __future__ import annotations

import logging
import re

import pandas as pd
import pytest

from src.data.cluster_split import cluster_split

DILIST_RE = re.compile(r"^DILIST_\d{4}$")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_50_diverse() -> pd.DataFrame:
    """50 rows of structurally diverse SMILES so most pairs are below
    Tanimoto 0.4 → small clusters, ratio test exercisable.

    Structures cover several scaffold families: benzenoids, heterocycles,
    aliphatics, sulfonamides, fused rings, peptide-like, etc.
    """
    smiles_list = [
        # Benzene / phenol family (likely some sim ≥0.4 amongst these)
        "c1ccccc1",                      # benzene
        "c1ccccc1O",                     # phenol
        "Cc1ccccc1",                     # toluene
        "c1ccc(N)cc1",                   # aniline
        "c1ccc(Cl)cc1",                  # chlorobenzene
        # Aliphatic
        "CCCC",
        "CCCCC",
        "CCCCCC",
        "CCCCCCC",
        "CCC(=O)O",
        # Carbonyl / ester
        "CC(=O)OC",
        "CCOC(=O)C",
        "CC(=O)NC",
        "O=C1CCCCC1",
        "O=C(O)CCCC(=O)O",
        # Heterocycles
        "c1ccncc1",                      # pyridine
        "c1ccc2ncccc2c1",                # quinoline
        "c1ccc2[nH]ccc2c1",              # indole
        "c1ccc2nc[nH]c2c1",              # benzimidazole
        "c1ccoc1",                       # furan
        # Halogens / sulfonamides
        "FC(F)(F)C",
        "ClCCCl",
        "BrCCBr",
        "S(=O)(=O)(N)c1ccccc1",
        "S(=O)(=O)(N)c1ccc(N)cc1",       # sulfanilamide
        # Drug-like (mixed)
        "CN(C)CCC1=CC=C(O)C=C1",         # tyramine-like
        "CC1=CC=C(C=C1)C(=O)O",          # toluic acid
        "CC1CCC(C(=O)O)C1",
        "C1CCC(CC1)NC(=O)C",
        "CCOC(=O)c1ccccc1",
        # Steroid-like, fused rings
        "C1CC2CCC3(CCCCC3CC12)C",
        "C1=CC=C2C=CC=CC2=C1",           # naphthalene
        "C1=CC2=CC=CC=C2C=C1",
        "C1CCC2CCCCC2C1",                # decalin
        "O=C1Nc2ccccc2C1",
        # More aliphatics
        "CCN",
        "CCO",
        "CCS",
        "CCSC",
        "CC(C)CO",
        # Acids and amines
        "NC(=N)N",
        "NCCCN",
        "OCCO",
        "OCCCO",
        "OC(C(=O)O)C(=O)O",
        # Final misc
        "CC(=O)Nc1ccc(O)cc1",            # paracetamol
        "CC(=O)Oc1ccccc1C(=O)O",          # aspirin
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
        "CCN(CC)CC",
        "CCCCN",
    ]
    assert len(smiles_list) == 50
    rows = []
    for i, smi in enumerate(smiles_list, start=1):
        rows.append({
            "pert_id": i,
            "canonical_smiles": smi,
            "dili_binary": i % 2,  # alternate 1, 0, 1, 0...
        })
    return pd.DataFrame(rows)


@pytest.fixture
def df_with_known_pair() -> pd.DataFrame:
    """30 rows where rows (1, 2) and (3, 4) form known Tanimoto-similar pairs.
    Used to test cluster integrity.
    """
    rows = [
        # Pair A: phenol and methyl-phenol — expected Tanimoto ≥ 0.4
        {"pert_id": 1, "canonical_smiles": "c1ccccc1O", "dili_binary": 1},
        {"pert_id": 2, "canonical_smiles": "Cc1ccc(O)cc1", "dili_binary": 1},
        # Pair B: sulfanilamide and a close analog
        {"pert_id": 3, "canonical_smiles": "S(=O)(=O)(N)c1ccc(N)cc1", "dili_binary": 0},
        {"pert_id": 4, "canonical_smiles": "S(=O)(=O)(N)c1ccc(N)c(C)c1", "dili_binary": 0},
    ]
    # 26 diverse fillers so the partitioner has options.
    fillers = [
        "CCCC", "CCCCC", "CCCCCC", "CCCCCCC", "CCCCCCCC",
        "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
        "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
        "C1CCCCC1", "C1CCCC1", "C1CCCCCC1",
        "OC(=O)CCCC(=O)O", "OC(=O)CCCCC(=O)O",
        "NCCCN", "NCCCCN",
        "CCSC", "CCSCC",
        "ClCCCl", "BrCCBr", "FCCF",
        "OCCO", "OCCCO", "OCCCCO",
        "C1=CC2=CC=CC=C2C=C1",
    ]
    for i, smi in enumerate(fillers, start=5):
        rows.append({"pert_id": i, "canonical_smiles": smi, "dili_binary": i % 2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_1_keys_and_ids(df_50_diverse):
    out = cluster_split(df_50_diverse, seed=42, tanimoto_threshold=0.4)
    assert set(out.keys()) == {"train", "val", "test"}
    for sname in ("train", "val", "test"):
        assert isinstance(out[sname], list)
        for pid in out[sname]:
            assert isinstance(pid, str)
            assert DILIST_RE.match(pid), f"{pid!r} not a DILIST_NNNN string"


def test_2_determinism(df_50_diverse):
    a = cluster_split(df_50_diverse, seed=42, tanimoto_threshold=0.4)
    b = cluster_split(df_50_diverse, seed=42, tanimoto_threshold=0.4)
    assert a == b, "cluster_split must be deterministic given seed"


def test_3_disjointness(df_50_diverse):
    out = cluster_split(df_50_diverse, seed=42, tanimoto_threshold=0.4)
    train, val, test = set(out["train"]), set(out["val"]), set(out["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    all_ids = train | val | test
    expected = {f"DILIST_{p:04d}" for p in df_50_diverse["pert_id"].tolist()}
    assert all_ids == expected, "Some pert_ids missing or extra"


def test_4_cluster_integrity(df_with_known_pair):
    """Pairs of structurally similar drugs (Tanimoto ≥ 0.4) MUST land in the
    same partition because single-linkage clustering keeps them together.
    """
    out = cluster_split(df_with_known_pair, seed=42, tanimoto_threshold=0.4)
    # Build a pid -> partition map.
    pid_to_split: dict[str, str] = {}
    for sname in ("train", "val", "test"):
        for pid in out[sname]:
            pid_to_split[pid] = sname

    # Pair A: pert_id 1 (phenol) and pert_id 2 (cresol)
    assert pid_to_split["DILIST_0001"] == pid_to_split["DILIST_0002"], (
        f"Phenol/cresol pair split: {pid_to_split['DILIST_0001']} vs {pid_to_split['DILIST_0002']}"
    )
    # Pair B: pert_id 3 and pert_id 4 (sulfanilamide analogs)
    assert pid_to_split["DILIST_0003"] == pid_to_split["DILIST_0004"], (
        f"Sulfanilamide pair split: {pid_to_split['DILIST_0003']} vs {pid_to_split['DILIST_0004']}"
    )


def test_5_ratios_within_10pp(df_50_diverse):
    out = cluster_split(df_50_diverse, seed=42, tanimoto_threshold=0.4)
    n_total = sum(len(out[s]) for s in ("train", "val", "test"))
    assert n_total == 50
    train_frac = len(out["train"]) / n_total
    val_frac = len(out["val"]) / n_total
    test_frac = len(out["test"]) / n_total
    # Cluster splits are coarser; ±10pp is the locked tolerance per
    # CONTEXT.md ("Cluster-split stratification: stratify if cluster sizes
    # permit; else single-fold non-stratified with warning logged").
    assert 0.70 <= train_frac <= 0.90, f"train_frac={train_frac:.3f} out of [0.70, 0.90]"
    assert 0.00 <= val_frac <= 0.20, f"val_frac={val_frac:.3f} out of [0.00, 0.20]"
    assert 0.00 <= test_frac <= 0.20, f"test_frac={test_frac:.3f} out of [0.00, 0.20]"


def test_6_non_stratified_warning_when_oversize_cluster(caplog):
    """Force a single oversize cluster (>10% of N) by making most rows
    structurally identical — the implementation should log a non-stratified
    warning per CONTEXT.md ("Cluster-split stratification: stratify if cluster
    sizes permit (≥10 per class per fold); else single-fold non-stratified
    with warning logged in summary").
    """
    # 20 near-identical SMILES (all benzene) → one giant cluster.
    rows = []
    for i in range(1, 21):
        rows.append({
            "pert_id": i,
            "canonical_smiles": "c1ccccc1",
            "dili_binary": i % 2,
        })
    # Plus 5 distinct singletons.
    for i, smi in enumerate(["CCCC", "CCN", "CCO", "OCCO", "NCCN"], start=21):
        rows.append({"pert_id": i, "canonical_smiles": smi, "dili_binary": 0})
    df = pd.DataFrame(rows)

    with caplog.at_level(logging.WARNING, logger="src.data.cluster_split"):
        out = cluster_split(df, seed=42, tanimoto_threshold=0.4)

    # The implementation is required to log a warning containing "non-stratified"
    # (case-insensitive) when at least one cluster exceeds 10% of n_total.
    found = any(
        "non-stratified" in record.getMessage().lower()
        for record in caplog.records
    )
    assert found, (
        "Expected a 'non-stratified' warning when an oversize cluster forces "
        f"non-stratified assignment. Got log records: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    # Splits still disjoint and total preserved.
    train, val, test = set(out["train"]), set(out["val"]), set(out["test"])
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert len(train) + len(val) + len(test) == 25


def test_7_empty_smiles_rejection():
    """canonical_smiles == "" must raise ValueError (Phase 1 invariant)."""
    df = pd.DataFrame({
        "pert_id": [1, 2, 3],
        "canonical_smiles": ["CCCC", "", "c1ccccc1"],
        "dili_binary": [1, 0, 1],
    })
    with pytest.raises(ValueError, match="(canonical_smiles|empty|SMILES)"):
        cluster_split(df, seed=42, tanimoto_threshold=0.4)


def test_8_invalid_smiles_rejection():
    """canonical_smiles that RDKit cannot parse must raise ValueError."""
    df = pd.DataFrame({
        "pert_id": [1, 2, 3],
        "canonical_smiles": ["CCCC", "this-is-not-smiles[[[", "c1ccccc1"],
        "dili_binary": [1, 0, 1],
    })
    with pytest.raises(ValueError, match="(SMILES|parse|RDKit|MolFromSmiles)"):
        cluster_split(df, seed=42, tanimoto_threshold=0.4)
