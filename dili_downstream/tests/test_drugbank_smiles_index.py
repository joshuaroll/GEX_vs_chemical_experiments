"""Unit tests for ``src.data.drugbank_smiles_index``.

We do NOT exercise the real 1.5 GB ``full_database.xml`` here — these tests use
in-memory XML fixtures written to ``tmp_path`` so the unit-test loop is fast.
The real-data run lives in the CLI script (``scripts/resolve_smiles.py``) and is
exercised by ``test_resolve_smiles.py`` Test 6.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.drugbank_smiles_index import build_index


# Real DrugBank namespace — used in the fixture so the namespaced lookups
# inside ``build_index`` actually exercise.
NS = "http://www.drugbank.ca"


def _drug_xml(
    name: str | None,
    smiles: str | None,
    synonyms: list[str] | None = None,
    intl_brands: list[str] | None = None,
) -> str:
    """Return a single ``<drug>...</drug>`` XML element string."""
    name_block = f"<name>{name}</name>" if name is not None else ""
    if smiles is None:
        smiles_block = ""
    else:
        smiles_block = (
            "<calculated-properties>"
            f"<property><kind>SMILES</kind><value>{smiles}</value>"
            "<source>ChemAxon</source></property>"
            "<property><kind>logP</kind><value>1.23</value>"
            "<source>ChemAxon</source></property>"
            "</calculated-properties>"
        )
    syn_block = ""
    if synonyms:
        syn_block = (
            "<synonyms>"
            + "".join(f"<synonym>{s}</synonym>" for s in synonyms)
            + "</synonyms>"
        )
    brand_block = ""
    if intl_brands:
        brand_block = (
            "<international-brands>"
            + "".join(
                f"<international-brand><name>{b}</name></international-brand>"
                for b in intl_brands
            )
            + "</international-brands>"
        )
    return f"<drug>{name_block}{syn_block}{brand_block}{smiles_block}</drug>"


def _build_fixture(tmp_path: Path, drugs_xml: list[str]) -> Path:
    """Write a tiny DrugBank-shaped XML to ``tmp_path`` and return its path."""
    body = "".join(drugs_xml)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<drugbank xmlns="{NS}">{body}</drugbank>'
    )
    path = tmp_path / "tiny_drugbank.xml"
    path.write_text(xml, encoding="utf-8")
    return path


def test_build_index_returns_required_columns_and_one_row_per_smiles_drug(tmp_path: Path) -> None:
    """Test 1: shape + at-least-one row per drug with both <name> and SMILES."""
    xml_path = _build_fixture(
        tmp_path,
        [
            _drug_xml("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            _drug_xml("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            _drug_xml("Lactate", "CC(O)C(=O)O"),
        ],
    )
    df = build_index(xml_path)

    assert isinstance(df, pd.DataFrame)
    for col in ("name_lower", "name", "smiles"):
        assert col in df.columns, f"missing column {col} in {df.columns.tolist()}"
    assert len(df) == 3
    assert set(df["name_lower"]) == {"aspirin", "caffeine", "lactate"}


def test_build_index_skips_drugs_without_smiles(tmp_path: Path) -> None:
    """Test 2: biologics / drugs missing the SMILES property are silently skipped."""
    xml_path = _build_fixture(
        tmp_path,
        [
            _drug_xml("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            _drug_xml("Insulin glargine", None),  # biologic — no SMILES property
            _drug_xml("Etanercept", None),  # biologic — no SMILES property
            _drug_xml("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ],
    )
    df = build_index(xml_path)
    # Only the two small molecules should appear — and the function MUST NOT raise.
    assert len(df) == 2
    assert set(df["name_lower"]) == {"aspirin", "caffeine"}
    assert "insulin glargine" not in set(df["name_lower"])
    assert "etanercept" not in set(df["name_lower"])


def test_build_index_lowercases_and_strips_names(tmp_path: Path) -> None:
    """Test 3: name_lower is consistently lowercase and whitespace-stripped."""
    xml_path = _build_fixture(
        tmp_path,
        [
            _drug_xml("  Aspirin  ", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            _drug_xml("CAFFEINE", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
            _drug_xml("Acetaminophen", "CC(=O)NC1=CC=C(O)C=C1"),
        ],
    )
    df = build_index(xml_path)
    assert df["name_lower"].tolist() == ["aspirin", "caffeine", "acetaminophen"]
    for value in df["name_lower"]:
        assert value == value.strip().lower(), f"name_lower not normalized: {value!r}"


def test_build_index_raises_filenotfounderror_for_missing_path(tmp_path: Path) -> None:
    """Test 4: missing path raises FileNotFoundError that names the path."""
    missing = tmp_path / "does_not_exist.xml"
    with pytest.raises(FileNotFoundError) as excinfo:
        build_index(missing)
    assert str(missing) in str(excinfo.value)


def test_build_index_includes_synonyms_and_intl_brands_under_same_smiles(tmp_path: Path) -> None:
    """Critical: DILIst names like 'Aspirin' map to DrugBank primary 'Acetylsalicylic acid'
    only via the <synonyms> block. Without synonym indexing, hit rate drops by ~5%."""
    xml_path = _build_fixture(
        tmp_path,
        [
            _drug_xml(
                "Acetylsalicylic acid",
                "CC(=O)OC1=CC=CC=C1C(=O)O",
                synonyms=["Aspirin", "ASA"],
                intl_brands=["Bayer Aspirin"],
            ),
        ],
    )
    df = build_index(xml_path)

    indexed = set(df["name_lower"])
    # Primary, two synonyms, one international brand → 4 entries, all → same SMILES.
    assert {"acetylsalicylic acid", "aspirin", "asa", "bayer aspirin"}.issubset(indexed)
    # Every alias points at the SAME smiles
    aspirin_row = df[df["name_lower"] == "aspirin"].iloc[0]
    primary_row = df[df["name_lower"] == "acetylsalicylic acid"].iloc[0]
    assert aspirin_row["smiles"] == primary_row["smiles"]
    # The `name` column always shows the DrugBank primary, regardless of which
    # alias the row was indexed under.
    assert aspirin_row["name"] == "Acetylsalicylic acid"
    assert primary_row["name"] == "Acetylsalicylic acid"


def test_build_index_handles_nested_drug_interaction_drug_tags(tmp_path: Path) -> None:
    """Bonus: DrugBank nests ``<drug>`` inside ``<drug-interactions>`` blocks. Make sure
    we only collect the top-level ``<drug>`` records, not interaction targets."""
    body = (
        "<drug>"
        "<name>Aspirin</name>"
        "<drug-interactions>"
        "<drug-interaction>"
        # Nested drug-interaction → has its own <drug> child + <name> in real
        # DrugBank schema. We synthesize the same shape to confirm we don't
        # double-count.
        "<drug><name>NestedFakeDrug</name>"
        "<calculated-properties>"
        "<property><kind>SMILES</kind><value>FAKE</value><source>x</source></property>"
        "</calculated-properties>"
        "</drug>"
        "</drug-interaction>"
        "</drug-interactions>"
        "<calculated-properties>"
        "<property><kind>SMILES</kind><value>CC(=O)OC1=CC=CC=C1C(=O)O</value>"
        "<source>ChemAxon</source></property>"
        "</calculated-properties>"
        "</drug>"
    )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<drugbank xmlns="{NS}">{body}</drugbank>'
    )
    path = tmp_path / "nested.xml"
    path.write_text(xml, encoding="utf-8")

    df = build_index(path)
    # Only Aspirin (the top-level drug) should appear — not the nested fake.
    assert df["name_lower"].tolist() == ["aspirin"]
    assert "nestedfakedrug" not in set(df["name_lower"])
