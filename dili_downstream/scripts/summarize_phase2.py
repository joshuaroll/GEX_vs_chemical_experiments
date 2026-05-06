#!/usr/bin/env python3
"""Render `results/tables/P2_split_summary.md` from the diagnostics JSON.

CLI driver — small. Mirrors `scripts/summarize_phase1.py` (Phase 1 pattern).

Reads `data/processed/p2_diagnostics.json` (produced by
`scripts/build_phase2_splits.py`) and renders the locked 7-section markdown
via the pure library `src.data.summarize_phase2.render_markdown`.

Usage:
    cd /raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream
    conda run -n dili_v04_env python scripts/summarize_phase2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.summarize_phase2 import render_markdown  # noqa: E402

DEFAULT_DIAGNOSTICS = REPO_ROOT / "data" / "processed" / "p2_diagnostics.json"
DEFAULT_OUT = REPO_ROOT / "results" / "tables" / "P2_split_summary.md"

REQUIRED_HEADERS = (
    "# Phase 2 Split Summary",
    "## Scaffold split",
    "## Cluster split",
    "## TDC-DILI scaffold split",
    "## Three transfer slices",
    "## Upstream-train filter diagnostics",
    "## Halt gate evaluation",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render P2_split_summary.md")
    ap.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.diagnostics.exists():
        raise FileNotFoundError(
            f"{args.diagnostics} missing. Run scripts/build_phase2_splits.py first."
        )

    diagnostics = json.loads(args.diagnostics.read_text())
    text = render_markdown(diagnostics)

    # Inline grep-check assertions (the verifier greps for these).
    for header in REQUIRED_HEADERS:
        n = text.count(header)
        if n != 1:
            raise RuntimeError(
                f"Header invariant failed: {header!r} appeared {n} times "
                f"(expected exactly 1). Check src.data.summarize_phase2."
            )

    halt_pass = "HALT-GATE PASS" in text
    halt_fail = "HALT-GATE FAIL" in text
    if not (halt_pass or halt_fail):
        raise RuntimeError(
            "Halt-gate emit invariant failed: neither 'HALT-GATE PASS' nor "
            "'HALT-GATE FAIL' found in rendered markdown."
        )
    if halt_pass and halt_fail:
        raise RuntimeError(
            "Halt-gate emit invariant failed: BOTH PASS and FAIL strings found."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)

    halt_label = "PASS" if halt_pass else "FAIL"
    print(f"wrote {args.out} ({len(text)} bytes, halt-gate {halt_label})")


if __name__ == "__main__":
    main()
