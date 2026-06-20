"""Streaming SHA-256 helper for MANIFEST data rows.

Computes per-file SHA-256 digests and prints them in the format used by
MANIFEST.md data rows: `<sha256>  <relpath>`.

Pure I/O script — no torch, no model code.

Hard rules honored:
    - No mock or synthetic data.
    - No hardcoded absolute paths; callers provide paths as arguments.
    - No file writes — output goes to stdout for redirection.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


def sha256_file(path: str) -> str:
    """Compute the SHA-256 digest of a file using streaming reads.

    Parameters
    ----------
    path : str
        Absolute or relative path to the file.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Print SHA-256 digests for the given files or directory.

    Usage
    -----
    python scripts/compute_sha256.py <file_or_dir> [<file_or_dir> ...]

    Output format (mirrors MANIFEST data rows):
        <sha256>  <relpath>
    """
    parser = argparse.ArgumentParser(
        description="Compute SHA-256 digests for files or directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="File(s) or directory(-ies) to hash. Directories are walked recursively.",
    )
    parser.add_argument(
        "--relative-to",
        default=".",
        help="Base directory for relative paths in output (default: cwd).",
    )
    args = parser.parse_args()

    base = Path(args.relative_to).resolve()
    targets: list[Path] = []
    for p in args.paths:
        target = Path(p)
        if target.is_dir():
            targets.extend(sorted(f for f in target.rglob("*") if f.is_file()))
        elif target.is_file():
            targets.append(target)
        else:
            print(f"WARNING: {p} does not exist or is not a file/directory", file=sys.stderr)

    for f in targets:
        digest = sha256_file(str(f))
        try:
            relpath = f.resolve().relative_to(base)
        except ValueError:
            relpath = f.resolve()
        print(f"{digest}  {relpath}")


if __name__ == "__main__":
    main()
