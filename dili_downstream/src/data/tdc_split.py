"""TDC-DILI scaffold split (SPLIT-04).

Wraps `tdc.single_pred.Tox(name='DILI_Hong')`. See
https://tdcommons.ai/single_pred_tasks/tox#dili for the dataset spec.

Locked by 02-CONTEXT.md §"Locked by Q6 / Q12":
    - Secondary split via `tdc.single_pred.Tox(name='DILI_Hong')`,
      scaffold split per TDC default (70/10/20, NOT 80/10/10 — explicit
      divergence from the primary scaffold split).
    - TDC version pinning: record `tdc.__version__` for MANIFEST traceability.

The DILI_Hong dataset is a separate corpus from DILIst (the primary corpus
in this project). Drug_IDs are TDC's identifiers (typically drug names or
DrugBank IDs), NOT prefixed with `DILIST_` — they're a different ID
namespace, so the prefix would mislead downstream consumers.

Pure library. The Wave 3 driver writes the resulting JSON to
`data/splits/tdc_dili_scaffold.json`.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["tdc_dili_scaffold_split"]


def _import_tdc():
    """Import `tdc` lazily; on failure raise a tagged ImportError.

    Tagged so callers (and tests) can recognize the wrapper-specific failure
    vs. an unrelated ImportError further down the call stack.
    """
    try:
        tdc = importlib.import_module("tdc")
    except ImportError as e:
        raise ImportError(
            "pytdc not importable. Run scripts/env_setup.sh to install into "
            "dili_v04_env. (SPLIT-04 requires pytdc.)"
        ) from e
    if tdc is None:
        # Defensive: monkeypatching to None (used by tests to simulate
        # absence) leaves a bare None entry in sys.modules.
        raise ImportError(
            "pytdc not importable (sys.modules['tdc'] is None). Run "
            "scripts/env_setup.sh to install into dili_v04_env. "
            "(SPLIT-04 requires pytdc.)"
        )
    return tdc


def _resolve_tdc_version(tdc) -> str:
    """Best-effort version resolution: __version__ first, then importlib.metadata."""
    version = getattr(tdc, "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import PackageNotFoundError, version as md_version
        try:
            return md_version("pytdc")
        except PackageNotFoundError:
            try:
                return md_version("PyTDC")
            except PackageNotFoundError:
                pass
    except ImportError:
        pass
    return "unknown"


def tdc_dili_scaffold_split(
    seed: int = 42,
    cache_dir: Path | None = None,
) -> dict[str, list[str] | str | int]:
    """TDC-DILI scaffold split (SPLIT-04).

    Wraps `tdc.single_pred.Tox(name='DILI_Hong').get_split(method='scaffold',
    seed=seed)`. Default ratios are TDC's 70/10/20 (NOT 80/10/10 — locked
    divergence per 02-CONTEXT.md "scaffold split per TDC default").

    Parameters
    ----------
    seed : int
        Passed to TDC's get_split for reproducibility.
    cache_dir : Path | None
        If provided, TDC's dataset download caches under this directory
        (instead of the default `~/.tdc/`). Useful for keeping per-project
        caches under `data/raw/` so they're SHA-traceable.

    Returns
    -------
    dict with keys:
        - "train": list[str]   (Drug_IDs cast to str)
        - "val":   list[str]   (renamed from TDC's "valid")
        - "test":  list[str]
        - "tdc_version": str   (installed pytdc version for MANIFEST traceability)
        - "dataset_size": int  (total rows in DILI_Hong)

    Raises
    ------
    ImportError
        If pytdc is not importable in the active env. Message identifies
        pytdc and dili_v04_env so the caller can recover via env_setup.sh.
    """
    tdc = _import_tdc()
    tdc_version = _resolve_tdc_version(tdc)

    # Lazy import of single_pred so the ImportError path above catches a
    # missing tdc cleanly.
    single_pred = importlib.import_module("tdc.single_pred")
    Tox = single_pred.Tox

    # In pytdc 0.4.x, the canonical DILI dataset is registered under the
    # name 'dili' (which is the Hong et al. 2014 DILI corpus — same data
    # as the doc's 'DILI_Hong' label). The old 'DILI_Hong' name is not
    # accepted by pytdc 0.4.17's fuzzy_search. Using 'dili' here keeps the
    # spec-mandated dataset (verified n=475 matches the doc).
    if cache_dir is not None:
        data = Tox(name="dili", path=str(cache_dir))
    else:
        data = Tox(name="dili")

    # TDC default fractions for scaffold split: [0.7, 0.1, 0.2].
    # Pass explicitly for documentation visibility AND to lock the divergence
    # from the primary 80/10/10 in this project.
    splits = data.get_split(method="scaffold", seed=seed, frac=[0.7, 0.1, 0.2])

    # TDC returns a dict of DataFrames keyed train/valid/test.
    train_df = splits["train"]
    valid_df = splits["valid"]
    test_df = splits["test"]

    train_ids = [str(x) for x in train_df["Drug_ID"].tolist()]
    val_ids = [str(x) for x in valid_df["Drug_ID"].tolist()]
    test_ids = [str(x) for x in test_df["Drug_ID"].tolist()]

    dataset_size = len(train_ids) + len(val_ids) + len(test_ids)

    log.info(
        "tdc_dili_scaffold_split: dataset_size=%d (train/val/test: %d/%d/%d) "
        "tdc_version=%s seed=%d",
        dataset_size, len(train_ids), len(val_ids), len(test_ids),
        tdc_version, seed,
    )

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "tdc_version": tdc_version,
        "dataset_size": dataset_size,
    }
