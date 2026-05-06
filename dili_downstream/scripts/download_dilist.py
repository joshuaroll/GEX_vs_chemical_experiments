#!/usr/bin/env python3
"""Download DILIst from FDA NCTR with SHA256 logging for MANIFEST.md.

DILIst (Drug-Induced Liver Injury severity dataset) is hosted on the FDA NCTR
LTKB portal. Direct download URLs occasionally change; this script tries a
list of known URLs and gives clear instructions for manual download if all
candidates fail.

DILIrank info is bundled inside DILIst (per the user's 9b decision: "use
available unless lacking"), so this script also extracts the severity columns
into a separate companion CSV so Phase 1 / Phase 8 can stratify on severity
without re-downloading.

Usage:
    python scripts/download_dilist.py
    python scripts/download_dilist.py --output-dir /tmp/dilist
    python scripts/download_dilist.py --url <override>
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "raw" / "DILIst"

# Known FDA NCTR DILIst URL candidates (verified working 2026-05-05).
# When all candidates 404, scrape these canonical landing pages for new media IDs:
#   DILIst:  https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-severity-and-toxicity-dilist-dataset
#   DILIrank: https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset
CANDIDATE_URLS: list[str] = [
    "https://www.fda.gov/media/160597/download?attachment",
]

# Sibling DILIrank 2.0 download — explicit 4-class severity source for M10
# (DILI severity stratification). Fetched after DILIst regardless of whether
# DILIst's Supplementary Table includes severity columns.
DILIRANK_URLS: list[str] = [
    "https://www.fda.gov/media/113052/download?attachment",
]

# A realistic User-Agent reduces the chance of being rate-limited or served a
# CAPTCHA page by FDA's CDN.
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def attempt_download(urls: Iterable[str], output_path: Path) -> tuple[bool, str | None]:
    """Try each URL; return (success, url_used)."""
    try:
        import requests
    except ImportError:
        print("requests is not installed. Activate dili_v04_env first.", file=sys.stderr)
        return False, None

    headers = {"User-Agent": DEFAULT_UA}
    for url in urls:
        print(f"→ Trying {url}", file=sys.stderr)
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers, allow_redirects=True) as resp:
                if resp.status_code != 200:
                    print(f"  [HTTP {resp.status_code}] skipping", file=sys.stderr)
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            print(f"  ✓ saved to {output_path}", file=sys.stderr)
            return True, url
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {e}", file=sys.stderr)
            continue
    return False, None


def manual_instructions(output_dir: Path) -> None:
    print(
        f"""
──────────────────────────────────────────────────────────────────────────────
  Could not auto-fetch DILIst from any of the candidate URLs.

  Manual fallback:
    1. Visit https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/dilist-and-related-resources
    2. Download the latest DILIst Excel (.xlsx) file.
    3. Place it at: {output_dir}/dilist.xlsx
    4. Re-run this script with --skip-download to compute its SHA256 and
       record it in MANIFEST.md.

  If FDA's URL has changed, please update the CANDIDATE_URLS list at the top
  of this script and submit a one-line PR so future Phase-1 runs are
  unblocked.
──────────────────────────────────────────────────────────────────────────────
""".strip(),
        file=sys.stderr,
    )


def parse_severity(xlsx_path: Path, csv_out: Path) -> bool:
    """Extract DILIst rows + DILIrank severity columns into a flat CSV.

    Soft-fails (returns False) if the columns aren't where we expect — Phase 1
    will then either skip severity stratification or fall back to a separate
    DILIrank fetch.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping severity extraction", file=sys.stderr)
        return False

    try:
        df = pd.read_excel(xlsx_path)
    except Exception as e:  # noqa: BLE001
        print(f"Could not parse {xlsx_path}: {e}", file=sys.stderr)
        return False

    # Heuristic column matching: DILIst sheets vary by release.
    cols_lower = {c.lower(): c for c in df.columns}
    severity_keys = ["dili severity", "dili-severity", "severity", "dili concern", "dili-concern"]
    severity_col = next(
        (cols_lower[k] for k in severity_keys if k in cols_lower),
        None,
    )

    if severity_col is None:
        print(
            "  Severity column not found in this DILIst release. "
            "Phase 8 (severity stratification) may need a separate DILIrank fetch.",
            file=sys.stderr,
        )
        df.to_csv(csv_out, index=False)
        return False

    df.to_csv(csv_out, index=False)
    print(f"  ✓ Severity column found: '{severity_col}' — saved {csv_out}", file=sys.stderr)
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Where to save DILIst files")
    p.add_argument("--url", type=str, default=None,
                   help="Override: try this URL only (instead of CANDIDATE_URLS)")
    p.add_argument("--skip-download", action="store_true",
                   help="Don't fetch — just compute SHA256 and parse an existing dilist.xlsx")
    p.add_argument("--skip-dilirank", action="store_true",
                   help="Skip the sibling DILIrank fetch")
    args = p.parse_args()

    output_dir: Path = args.output_dir
    xlsx_path = output_dir / "dilist.xlsx"

    if not args.skip_download:
        urls = [args.url] if args.url else CANDIDATE_URLS
        ok, url_used = attempt_download(urls, xlsx_path)
        if not ok:
            manual_instructions(output_dir)
            return 1
        print(f"# Source URL: {url_used}", file=sys.stderr)

    if not xlsx_path.exists():
        print(f"Expected file not present: {xlsx_path}", file=sys.stderr)
        manual_instructions(output_dir)
        return 1

    digest = sha256_file(xlsx_path)
    print(f"\nDILIst file:    {xlsx_path}")
    print(f"DILIst SHA256:  {digest}")
    print("\nAdd to MANIFEST.md:")
    print(f"| DILIst | data/raw/DILIst/dilist.xlsx | {digest} | FDA NCTR |")

    csv_out = output_dir / "dilist_with_severity.csv"
    parse_severity(xlsx_path, csv_out)

    # ── DILIrank 2.0 (sibling, optional) ──────────────────────────────────
    if not args.skip_download and not args.skip_dilirank:
        dilirank_dir = output_dir.parent / "DILIrank"
        dilirank_path = dilirank_dir / "dilirank.xlsx"
        print(f"\n→ Fetching sibling DILIrank 2.0 to {dilirank_path}", file=sys.stderr)
        ok, url_used = attempt_download(DILIRANK_URLS, dilirank_path)
        if ok:
            dr_digest = sha256_file(dilirank_path)
            print(f"\nDILIrank file:    {dilirank_path}")
            print(f"DILIrank SHA256:  {dr_digest}")
            print(f"DILIrank URL:     {url_used}")
            print("\nAdd to MANIFEST.md:")
            print(f"| DILIrank 2.0 | data/raw/DILIrank/dilirank.xlsx | {dr_digest} | FDA NCTR |")
        else:
            print("  (DILIrank skipped — DILIst Supplementary Table likely contains severity already)",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
