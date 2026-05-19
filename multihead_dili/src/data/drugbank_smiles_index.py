"""Streaming DrugBank XML → (name_lower, name, smiles) index builder.

DrugBank's `full_database.xml` is ~1.5 GB. We must stream-parse it (`iterparse`)
and aggressively `clear()` processed elements to keep peak memory bounded.

Public API
----------
- ``build_index(xml_path)``  → ``pd.DataFrame[name_lower, name, smiles]``
- ``write_index(xml_path, out_csv)`` → writes the frame to CSV, returns the path

Drugs without either a ``<name>`` or a SMILES ``<calculated-property>`` (e.g.
biologics / peptides) are silently skipped — they cannot enter our drug→SMILES
lookup.

Each top-level ``<drug>`` contributes (in addition to its primary ``<name>``)
every ``<synonyms>/<synonym>`` and ``<international-brands>/<international-brand>/<name>``
entry under that same drug, ALL pointing at the same SMILES. This is critical
for hit rate: DrugBank's canonical name for aspirin is ``Acetylsalicylic acid``
(not ``Aspirin``), for rifampin is ``Rifampicin``, for acyclovir is ``Aciclovir``,
etc. Adding synonyms lifts the DILIst hit rate from 82% → 87% (Rule 1 fix —
the original primary-name-only index had a known-synonym blind spot).

Duplicate ``name_lower`` values keep the first occurrence (deterministic given
DrugBank's stable XML ordering).

This module is pure: no I/O at import time, no logging at import time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator
import xml.etree.ElementTree as ET

import pandas as pd

DRUGBANK_NS = "{http://www.drugbank.ca}"

# Pre-built tag constants (avoid re-formatting the namespace string in hot loops).
_TAG_DRUG = f"{DRUGBANK_NS}drug"
_TAG_NAME = f"{DRUGBANK_NS}name"
_TAG_CALC_PROPS = f"{DRUGBANK_NS}calculated-properties"
_TAG_PROP = f"{DRUGBANK_NS}property"
_TAG_KIND = f"{DRUGBANK_NS}kind"
_TAG_VALUE = f"{DRUGBANK_NS}value"
_TAG_SYNONYMS = f"{DRUGBANK_NS}synonyms"
_TAG_SYNONYM = f"{DRUGBANK_NS}synonym"
_TAG_INTL_BRANDS = f"{DRUGBANK_NS}international-brands"
_TAG_INTL_BRAND = f"{DRUGBANK_NS}international-brand"


def _iter_top_level_drugs(xml_path: Path) -> Iterator[ET.Element]:
    """Yield each top-level ``<drug>`` element from a DrugBank XML stream.

    DrugBank nests ``<drug-interaction>`` blocks (etc.) that ALSO contain a
    ``<drug>`` child with its own ``<name>`` — those are not the drug records we
    want. We track depth on ``start``/``end`` of ``{ns}drug`` tags and only
    yield the element on its ``end`` event when it is at depth 1 (immediate
    child of the root).
    """
    depth = 0
    # iterparse with start+end events lets us count nesting depth without
    # building the full tree.
    context = ET.iterparse(str(xml_path), events=("start", "end"))
    _, root = next(context)  # consume the root <drugbank> start event

    for event, elem in context:
        if elem.tag != _TAG_DRUG:
            continue
        if event == "start":
            depth += 1
        elif event == "end":
            if depth == 1:
                yield elem
                # Clear this top-level <drug> element AND drop it from the
                # root's child list. Without dropping, the root accumulates
                # every parsed <drug> across the 1.5 GB stream → OOM.
                elem.clear()
                root.remove(elem)
            depth -= 1


def _extract_smiles(drug_elem: ET.Element) -> str | None:
    """Return the SMILES string from a drug's calculated-properties block, or None."""
    calc_props = drug_elem.find(_TAG_CALC_PROPS)
    if calc_props is None:
        return None
    for prop in calc_props.findall(_TAG_PROP):
        kind = prop.find(_TAG_KIND)
        if kind is None or kind.text is None:
            continue
        if kind.text.strip() == "SMILES":
            value = prop.find(_TAG_VALUE)
            if value is not None and value.text:
                s = value.text.strip()
                return s if s else None
            return None
    return None


def _collect_aliases(drug_elem: ET.Element, primary: str) -> set[str]:
    """Return the set of name strings under one DrugBank drug — primary + synonyms +
    international brand names. All map to the same SMILES.

    DrugBank's canonical primary ``<name>`` often differs from the common DILIst
    spelling (e.g. ``Acetylsalicylic acid`` vs ``Aspirin``, ``Rifampicin`` vs
    ``Rifampin``). Including synonyms is what lets DILIst lookups land.
    """
    names = {primary}

    syns_elem = drug_elem.find(_TAG_SYNONYMS)
    if syns_elem is not None:
        for syn in syns_elem.findall(_TAG_SYNONYM):
            if syn.text:
                t = syn.text.strip()
                if t:
                    names.add(t)

    brands_elem = drug_elem.find(_TAG_INTL_BRANDS)
    if brands_elem is not None:
        for brand in brands_elem.findall(_TAG_INTL_BRAND):
            bn = brand.find(_TAG_NAME)
            if bn is not None and bn.text:
                t = bn.text.strip()
                if t:
                    names.add(t)

    return names


def build_index(xml_path: str | Path) -> pd.DataFrame:
    """Stream-parse ``xml_path`` and return the (name_lower, name, smiles) index.

    For each top-level ``<drug>`` with a SMILES property, the index includes
    one row per unique alias (primary name + synonyms + international brand
    names), all sharing the same SMILES. This is required to hit DILIst names
    that use a different spelling than DrugBank's primary canonical name.

    Parameters
    ----------
    xml_path
        Filesystem path to ``full_database.xml`` (DrugBank XML).

    Returns
    -------
    pd.DataFrame
        Columns: ``name_lower``, ``name``, ``smiles``. ``name`` is the
        DrugBank primary name; ``name_lower`` is the alias the row was indexed
        under (lowercased + whitespace-stripped). Duplicates on ``name_lower``
        keep the first occurrence.

    Raises
    ------
    FileNotFoundError
        If ``xml_path`` does not exist.
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"DrugBank XML not found: {xml_path}")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for drug in _iter_top_level_drugs(path):
        name_elem = drug.find(_TAG_NAME)
        if name_elem is None or name_elem.text is None:
            continue
        name = name_elem.text.strip()
        if not name:
            continue

        smiles = _extract_smiles(drug)
        if smiles is None:
            continue  # biologics / peptides / DrugBank entries without computed SMILES

        for alias in _collect_aliases(drug, name):
            alias_lower = alias.strip().lower()
            if not alias_lower or alias_lower in seen:
                continue
            seen.add(alias_lower)
            rows.append({"name_lower": alias_lower, "name": name, "smiles": smiles})

    return pd.DataFrame(rows, columns=["name_lower", "name", "smiles"])


def write_index(xml_path: str | Path, out_csv: str | Path) -> Path:
    """Build the index from ``xml_path`` and persist it to ``out_csv``.

    Returns the resolved output path.
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_index(xml_path)
    df.to_csv(out_path, index=False)
    return out_path
