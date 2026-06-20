#!/usr/bin/env python
"""Fetch human-mouse-rat ortholog table from Ensembl BioMart, cache it, and
write the filtered one-to-one TSV to data/processed/spatial/.

Side-effecting script — all network I/O and filesystem writes live here.
Pure filtering logic lives in src/spatial/orthology.py (offline-testable).

Usage
-----
    conda run -n dili_v04_env python scripts/build_orthologs.py [--raw-dir DIR]
              [--out-file FILE] [--release RELEASE]

Outputs
-------
data/raw/spatial/biomart/
    orthologs_raw_<ensembl_release>_<YYYYMMDD>.tsv   — verbatim BioMart TSV
    orthologs_raw_<ensembl_release>_<YYYYMMDD>.meta  — sidecar with provenance
        (Ensembl release, query date, endpoint, column names).  XC-10 time-
        leakage discipline: the release + date are baked into the filename so
        the annotation version is always unambiguous.
data/processed/spatial/
    orthologs_h_m_r_one2one.tsv  — filtered one-to-one pairs (tab-separated)

Exit codes
----------
0   — success; orthologs_h_m_r_one2one.tsv written.
1   — BioMart unavailable or empty response; HALT_REASON.md written to the
      planning phase directory; no partial output emitted (XC-01).

Cross-cutting rules honored
---------------------------
- No fabricated ortholog table: if BioMart is unavailable, write HALT_REASON.md
  and exit 1.  Never return a synthetic or partial table.
- XC-10: raw TSV filename includes Ensembl release number + query date.
- XC-08: many-to-many orthologs are DROPPED by the pure filter in orthology.py.
- Commit discipline: the raw BioMart TSV and the processed one2one TSV are
  tracked by git (data/raw/spatial/biomart/ and data/processed/spatial/
  are not .gitignored for ortholog files).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import datetime

# ---------------------------------------------------------------------------
# Resolve project root and import the pure library
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import requests  # noqa: E402  — network fetch; only in scripts/
import pandas as pd  # noqa: E402

from src.spatial.orthology import (  # noqa: E402
    build_one2one_orthologs,
    ortholog_report,
)

# ---------------------------------------------------------------------------
# BioMart XML query (load-bearing literal — XC-10: full-genome, no chr filter)
# ---------------------------------------------------------------------------
BIOMART_QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" datasetConfigVersion="0.6">
 <Dataset name="hsapiens_gene_ensembl" interface="default">
  <Attribute name="ensembl_gene_id"/>
  <Attribute name="external_gene_name"/>
  <Attribute name="mmusculus_homolog_ensembl_gene"/>
  <Attribute name="mmusculus_homolog_associated_gene_name"/>
  <Attribute name="mmusculus_homolog_orthology_type"/>
  <Attribute name="rnorvegicus_homolog_ensembl_gene"/>
  <Attribute name="rnorvegicus_homolog_associated_gene_name"/>
  <Attribute name="rnorvegicus_homolog_orthology_type"/>
 </Dataset>
</Query>"""

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

BIOMART_COLS = [
    "ensembl_gene_id",
    "external_gene_name",
    "mmusculus_homolog_ensembl_gene",
    "mmusculus_homolog_associated_gene_name",
    "mmusculus_homolog_orthology_type",
    "rnorvegicus_homolog_ensembl_gene",
    "rnorvegicus_homolog_associated_gene_name",
    "rnorvegicus_homolog_orthology_type",
]

# Phase directory for HALT_REASON.md (relative to project root)
_PHASE_DIR = _PROJECT_ROOT / ".planning" / "phases" / "00-dataset-acquisition-manifest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_halt_reason(reason: str) -> None:
    """Write HALT_REASON.md to the P0 phase directory and print a message."""
    halt_path = _PHASE_DIR / "HALT_REASON.md"
    halt_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# HALT_REASON: BioMart Unavailable\n\n"
        f"**Date:** {datetime.datetime.utcnow().isoformat()}Z\n\n"
        f"## Reason\n\n{reason}\n\n"
        "## Required Action\n\n"
        "- Verify Ensembl BioMart is reachable: "
        "`curl -s 'https://www.ensembl.org/biomart/martservice?query=...' | head`\n"
        "- If the service is temporarily down, retry later.\n"
        "- If the service URL has changed (Ensembl mirror), update `BIOMART_URL` in "
        "`scripts/build_orthologs.py`.\n"
        "- Do NOT fabricate an ortholog table (XC-01).\n"
    )
    halt_path.write_text(content)
    print(f"HALT_REASON.md written to: {halt_path}", file=sys.stderr)


def _probe_ensembl_release() -> str:
    """Query the BioMart REST API for the current Ensembl release number.

    Returns the release string (e.g., "111") or "unknown" if the registry
    endpoint is unreachable (the caller still proceeds with the main query).
    """
    try:
        registry_url = "https://www.ensembl.org/biomart/martservice?type=registry"
        resp = requests.get(registry_url, timeout=30)
        if resp.status_code == 200 and "database=" in resp.text:
            # BioMart registry XML contains strings like:
            #   database="ensembl_mart_116"  (human/multi-species mart)
            #   database="mmusculus_gene_ensembl_109"  (older format)
            # Extract the highest numeric suffix across either pattern.
            import re
            matches = re.findall(r"_(?:mart|gene_ensembl)_(\d+)", resp.text)
            if matches:
                return str(max(int(m) for m in matches))
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Ensembl BioMart orthologs and write one2one TSV."
    )
    parser.add_argument(
        "--raw-dir",
        type=pathlib.Path,
        default=_PROJECT_ROOT / "data" / "raw" / "spatial" / "biomart",
        help="Directory for cached raw BioMart TSV (default: data/raw/spatial/biomart/)",
    )
    parser.add_argument(
        "--out-file",
        type=pathlib.Path,
        default=_PROJECT_ROOT / "data" / "processed" / "spatial" / "orthologs_h_m_r_one2one.tsv",
        help="Output one2one TSV path.",
    )
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Ensembl release number to embed in the raw TSV filename. "
        "If not provided, the script queries the BioMart registry.",
    )
    args = parser.parse_args()

    raw_dir: pathlib.Path = args.raw_dir
    out_file: pathlib.Path = args.out_file
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Probe Ensembl release (XC-10 time-leakage discipline)
    # ------------------------------------------------------------------
    ensembl_release = args.release or _probe_ensembl_release()
    query_date = datetime.datetime.utcnow().strftime("%Y%m%d")
    raw_tsv = raw_dir / f"orthologs_raw_{ensembl_release}_{query_date}.tsv"
    meta_file = raw_dir / f"orthologs_raw_{ensembl_release}_{query_date}.meta"

    print(f"Ensembl release: {ensembl_release}")
    print(f"Query date: {query_date}")
    print(f"Raw cache path: {raw_tsv}")
    print(f"Processed output: {out_file}")

    # ------------------------------------------------------------------
    # 2. Fetch from BioMart (or re-use cached raw TSV if present)
    # ------------------------------------------------------------------
    if raw_tsv.exists():
        print(f"Re-using cached raw TSV: {raw_tsv}")
    else:
        print(f"Fetching from {BIOMART_URL} ...")
        try:
            resp = requests.get(
                BIOMART_URL,
                params={"query": BIOMART_QUERY},
                timeout=180,
            )
        except requests.exceptions.RequestException as exc:
            _write_halt_reason(
                f"Network error fetching BioMart: {exc}\n\n"
                f"URL: {BIOMART_URL}"
            )
            sys.exit(1)

        if resp.status_code != 200:
            _write_halt_reason(
                f"BioMart returned HTTP {resp.status_code}.\n\n"
                f"URL: {BIOMART_URL}\n"
                f"Response body (first 500 chars): {resp.text[:500]}"
            )
            sys.exit(1)

        body = resp.text.strip()
        if not body:
            _write_halt_reason(
                "BioMart returned an empty response body.\n\n"
                f"URL: {BIOMART_URL}"
            )
            sys.exit(1)

        # Check for BioMart error marker
        if body.lower().startswith("query error") or "error" in body[:100].lower():
            _write_halt_reason(
                f"BioMart returned an error response:\n\n{body[:500]}"
            )
            sys.exit(1)

        # Cache verbatim raw TSV
        raw_tsv.write_text(body, encoding="utf-8")
        print(f"Cached raw TSV ({len(body):,} bytes) to: {raw_tsv}")

        # Write sidecar metadata (XC-10 provenance)
        meta_lines = [
            f"ensembl_release={ensembl_release}",
            f"query_date={query_date}",
            f"biomart_url={BIOMART_URL}",
            f"columns={','.join(BIOMART_COLS)}",
            f"query_xml={BIOMART_QUERY[:200].replace(chr(10), ' ')}...",
        ]
        meta_file.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        print(f"Sidecar metadata written to: {meta_file}")

    # ------------------------------------------------------------------
    # 3. Parse raw TSV
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(
            raw_tsv,
            sep="\t",
            header=None,
            names=BIOMART_COLS,
            dtype=str,
            na_values=[""],
            keep_default_na=True,
        )
    except Exception as exc:
        _write_halt_reason(
            f"Failed to parse cached BioMart TSV at {raw_tsv}:\n\n{exc}"
        )
        sys.exit(1)

    print(f"Loaded {len(df):,} rows from raw TSV.")

    if len(df) == 0:
        _write_halt_reason(
            f"Raw BioMart TSV is empty (0 data rows) at {raw_tsv}."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Call the pure filter (src/spatial/orthology.py)
    # ------------------------------------------------------------------
    table = build_one2one_orthologs(df)

    # ------------------------------------------------------------------
    # 5. Write processed one2one TSV
    # ------------------------------------------------------------------
    table.pairs.to_csv(out_file, sep="\t", index=False)
    print(f"Wrote {table.n_one2one:,} one2one pairs to: {out_file}")

    # ------------------------------------------------------------------
    # 6. Print ortholog report (stdout; paste into P0_orthologs.md)
    # ------------------------------------------------------------------
    print("\n## Ortholog Report (paste into P0_orthologs.md)")
    print(ortholog_report(table))
    print(f"\nEnsembl release pinned: {ensembl_release} (query date: {query_date})")


if __name__ == "__main__":
    main()
