#!/usr/bin/env python3
"""Pick GPU IDs to use, always leaving at least one free.

Constraint per Joshua: this is a shared box where CPU/RAM is the real
bottleneck. Saturating all GPUs starves CPU resources. Leave one GPU free
so other processes (and other people) can breathe.

Usage:
    # Print a comma-separated list of GPUs to use (suitable for CUDA_VISIBLE_DEVICES)
    python auto_select_gpus.py
    python auto_select_gpus.py --max 4              # cap to at most 4 GPUs
    python auto_select_gpus.py --memory-min 5000    # require >=5GB free
    python auto_select_gpus.py --util-max 20        # require <=20% utilization
    python auto_select_gpus.py --reserve 2          # leave 2 GPUs free instead of 1

    # Use as a launcher gate
    export CUDA_VISIBLE_DEVICES=$(python auto_select_gpus.py --max 1)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def query_gpus() -> list[tuple[int, int, int]]:
    """Returns a list of (gpu_id, memory_free_mib, utilization_pct)."""
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found — cannot enumerate GPUs", file=sys.stderr)
        return []
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def select(
    gpus: list[tuple[int, int, int]],
    memory_min: int,
    util_max: int,
    reserve: int,
    cap: int | None,
) -> list[int]:
    eligible = [
        gid for gid, mem, util in gpus
        if mem >= memory_min and util <= util_max
    ]
    # Leave `reserve` GPUs free.
    if reserve >= len(eligible):
        return []
    pick = eligible[: len(eligible) - reserve]
    if cap is not None:
        pick = pick[:cap]
    return pick


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--memory-min", type=int, default=4000,
                   help="Minimum free GPU memory in MiB (default: 4000)")
    p.add_argument("--util-max", type=int, default=30,
                   help="Maximum current utilization in %% (default: 30)")
    p.add_argument("--reserve", type=int, default=1,
                   help="Number of GPUs to leave free (default: 1, per Joshua's shared-box constraint)")
    p.add_argument("--max", type=int, default=None, dest="cap",
                   help="Cap the number of GPUs returned (default: no cap)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress diagnostic stderr output")
    args = p.parse_args()

    gpus = query_gpus()
    if not gpus:
        if not args.quiet:
            print("No GPUs visible — exit 1", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"# GPUs visible: {len(gpus)}", file=sys.stderr)
        for gid, mem, util in gpus:
            print(f"#   GPU {gid}: {mem} MiB free, {util}%% util", file=sys.stderr)

    picks = select(gpus, args.memory_min, args.util_max, args.reserve, args.cap)
    if not picks:
        if not args.quiet:
            print("# No GPUs match constraints (after reservation) — exit 2", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"# Picked GPUs: {picks} (reserved {args.reserve} free)", file=sys.stderr)
    print(",".join(str(g) for g in picks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
