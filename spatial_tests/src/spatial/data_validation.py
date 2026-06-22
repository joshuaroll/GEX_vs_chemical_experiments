"""Content/shape validation for acquired spatial datasets.

Existence of a named file is NOT proof of usable data: a Visium ``*_RAW.tar`` may
hold only images, a "dataset" may be the wrong study, and a download may yield
just a metadata sample-sheet. Phase 0 hit all three. This module inspects what is
actually on disk and decides whether a dataset directory contains a gene-expression
COUNT artifact — the input the DE-rule pipeline needs.

Pure library: no network, no torch, no scanpy. Structural inspection only — it
recognises count artifacts by archive/file shape (10x Space Ranger triplets and
HDF5, AnnData ``.h5ad``, serialized Seurat objects), looking through loose files
and one level of archive nesting. It answers "is there a counts artifact here?",
not "load and validate the matrix"; the heavier load/coverage check belongs to the
Phase-1 loaders, which can build on the evidence returned here.

The single source of truth for a dataset's on-disk directory is ``entry.slug``
(see :mod:`src.spatial.datasets`); this module resolves ``raw_root / entry.slug``
and never re-slugifies a dataset name.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Filename fragments that indicate a gene-expression COUNT artifact. Images
# (.tif/.png/.jpg), spatial JSON, isoform .bb/.gtf, and plain metadata CSVs are
# intentionally excluded so the Phase-0 failure modes are rejected.
_COUNT_PATTERNS: tuple[str, ...] = (
    "matrix.mtx",
    "barcodes.tsv",
    "features.tsv",
    "genes.tsv",
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix",
    ".h5ad",
    "_counts.rds",
    "_obj.rds",          # per-sample Seurat objects (e.g. GSE252772 mouse kidney)
    "expression_matrix",
    "count_matrix",
)

_ARCHIVE_SUFFIXES: tuple[str, ...] = (".tar.gz", ".tgz", ".tar", ".zip")


def _name_is_count(name: str) -> bool:
    """True if a member/file name looks like a count artifact."""
    low = name.lower()
    if any(p in low for p in _COUNT_PATTERNS):
        return True
    # bare Space Ranger HDF5 (".h5" but not ".h5ad", which is matched above)
    base = low.rsplit("/", 1)[-1]
    return base.endswith(".h5")


def _stream_archive_has_counts(name: str, fileobj, depth: int) -> tuple[bool, str]:
    """Check a nested archive given as an open file object. Returns (found, evidence)."""
    low = name.lower()
    try:
        if low.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if low.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(fileobj=fileobj, mode=mode) as tf:
                for m in tf:  # streaming iteration, early-exit
                    if _name_is_count(m.name):
                        return True, m.name
        elif low.endswith(".zip"):
            # zip needs a seekable source; nested zips here are small (per-sample).
            with zipfile.ZipFile(io.BytesIO(fileobj.read())) as zf:
                for n in zf.namelist():
                    if _name_is_count(n):
                        return True, n
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError):
        return False, ""
    return False, ""


def _archive_has_counts(path: Path, depth: int = 1) -> tuple[bool, str]:
    """Inspect a tar/zip on disk for a count artifact, descending ``depth`` levels."""
    low = str(path).lower()
    try:
        if low.endswith((".tar.gz", ".tgz", ".tar")):
            mode = "r:gz" if low.endswith((".tar.gz", ".tgz")) else "r:"
            with tarfile.open(str(path), mode) as tf:
                for m in tf:
                    if _name_is_count(m.name):
                        return True, m.name
                    if (
                        depth > 0
                        and m.isfile()
                        and m.name.lower().endswith(_ARCHIVE_SUFFIXES)
                    ):
                        sub = tf.extractfile(m)
                        if sub is not None:
                            ok, ev = _stream_archive_has_counts(m.name, sub, depth - 1)
                            if ok:
                                return True, f"{m.name}::{ev}"
        elif low.endswith(".zip"):
            with zipfile.ZipFile(str(path)) as zf:
                names = zf.namelist()
                for n in names:
                    if _name_is_count(n):
                        return True, n
                if depth > 0:
                    for n in names:
                        if n.lower().endswith(_ARCHIVE_SUFFIXES):
                            with zf.open(n) as sub:
                                ok, ev = _stream_archive_has_counts(n, sub, depth - 1)
                                if ok:
                                    return True, f"{n}::{ev}"
    except (tarfile.TarError, zipfile.BadZipFile, OSError, EOFError):
        return False, ""
    return False, ""


@dataclass
class CountCheck:
    """Result of inspecting one dataset directory for expression counts."""

    slug: str
    has_counts: bool
    evidence: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)


def dataset_has_counts(entry, raw_root) -> CountCheck:
    """Decide whether ``raw_root/entry.slug`` holds a gene-expression count artifact.

    Scans loose files and looks one level inside archives (e.g. a ``*_RAW.tar``
    of per-sample ``.tar.gz``/``.zip``). Returns a :class:`CountCheck`; on the
    first count artifact found it short-circuits with ``has_counts=True`` and the
    evidence path.
    """
    d = Path(raw_root) / entry.slug
    checked: list[str] = []
    if not d.exists():
        return CountCheck(entry.slug, False, [], ["<directory missing>"])

    for p in sorted(d.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(d))
        checked.append(rel)
        if _name_is_count(p.name):
            return CountCheck(entry.slug, True, [rel], checked)
        if p.name.lower().endswith(_ARCHIVE_SUFFIXES):
            ok, ev = _archive_has_counts(p, depth=1)
            if ok:
                return CountCheck(entry.slug, True, [f"{rel}::{ev}"], checked)

    return CountCheck(entry.slug, False, [], checked)


def validate_usable_inputs(datasets, raw_root) -> dict[str, CountCheck]:
    """Return ``{slug: CountCheck}`` for usable_as_input datasets that are present
    on disk but lack a count artifact.

    Datasets not yet downloaded are skipped (absence is handled by the
    existence-level data-path test, not here). An empty dict means every present
    input has counts.
    """
    failures: dict[str, CountCheck] = {}
    root = Path(raw_root)
    for d in datasets:
        if not getattr(d, "usable_as_input", False):
            continue
        if not (root / d.slug).exists():
            continue
        res = dataset_has_counts(d, root)
        if not res.has_counts:
            failures[d.slug] = res
    return failures
