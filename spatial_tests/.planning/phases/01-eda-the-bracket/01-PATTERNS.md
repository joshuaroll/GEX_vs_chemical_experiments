# Phase 1: EDA (the bracket) — Pattern Map

**Mapped:** 2026-06-22
**Files analyzed:** 15 (8 library files, 1 script entry point, 6 test files)
**Analogs found:** 13 / 15

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/spatial/eda/__init__.py` | config | — | `src/spatial/pseudobulk.py` (module header) | role-match |
| `src/spatial/eda/labels.py` | utility | CRUD | `src/spatial/orthology.py` | exact |
| `src/spatial/eda/smiles_join.py` | utility | CRUD | `src/spatial/gene_alignment.py` | role-match |
| `src/spatial/eda/fingerprints.py` | utility | transform | `src/spatial/gene_alignment.py` | role-match |
| `src/spatial/eda/floor.py` | utility | batch | `src/spatial/orthology.py` + sibling `dili_downstream/src/data/wangli_lincs_lookup.py` | role-match |
| `src/spatial/eda/ceiling.py` | utility | batch | sibling `dili_downstream/src/data/wangli_lincs_lookup.py` | role-match |
| `src/spatial/eda/bootstrap.py` | utility | batch | `src/spatial/orthology.py` (NamedTuple result schema) | role-match |
| `src/spatial/eda/region_diagnostics.py` | utility | batch | `src/spatial/pseudobulk.py` + `src/spatial/orthology.py` | exact |
| `scripts/run_p1_eda.py` | script | request-response | `scripts/report_coverage.py` | exact |
| `tests/spatial/__init__.py` | config | — | `tests/__init__.py` | exact |
| `tests/spatial/test_eda_labels.py` | test | — | `tests/spatial/test_gene_alignment.py` | exact |
| `tests/spatial/test_eda_fingerprints.py` | test | — | `tests/spatial/test_gene_alignment.py` | exact |
| `tests/spatial/test_eda_floor.py` | test | — | `tests/spatial/test_pseudobulk.py` | role-match |
| `tests/spatial/test_eda_ceiling.py` | test | — | `tests/spatial/test_gene_alignment.py` | exact |
| `tests/spatial/test_eda_bootstrap.py` | test | — | `tests/spatial/test_pseudobulk.py` | role-match |
| `tests/spatial/test_eda_region.py` | test | — | `tests/spatial/test_pseudobulk.py` | role-match |

---

## Pattern Assignments

### `src/spatial/eda/__init__.py` (package init)

**Analog:** `src/spatial/pseudobulk.py` (module header convention)

This file is empty or minimal. The convention in this codebase is a blank `__init__.py`; the package docstring lives in the individual module files. Do not re-export symbols here — callers import from the submodule directly (e.g. `from src.spatial.eda.labels import load_dilirank`).

---

### `src/spatial/eda/labels.py` (utility, CRUD)

**Analog:** `src/spatial/orthology.py`

**Module docstring pattern** (lines 1-25 of orthology.py):
```python
"""Pure library: build a human-mouse-rat one-to-one ortholog map.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides a real cached TSV or
      an already-fetched DataFrame.
    - Network fetch lives behind a cached-TSV boundary; scripts/build_orthologs.py
      does the GET; this module only reads and filters.
"""

from __future__ import annotations

import logging
from typing import Final, NamedTuple, Union

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["build_one2one_orthologs", "ortholog_report", "OrthologTable"]
```

Copy this header structure verbatim for `labels.py`. Replace the content description but keep the hard-rules block, `from __future__ import annotations`, `logging.getLogger(__name__)`, and `__all__`.

**NamedTuple result schema** (lines 63-89 of orthology.py):
```python
class OrthologTable(NamedTuple):
    """Human-mouse-rat one-to-one ortholog map.

    Attributes
    ----------
    pairs : pd.DataFrame
        ...
    n_input : int
    n_one2one : int
    dropped_fraction : float
    """
    pairs: object
    n_input: int
    n_one2one: int
    dropped_fraction: float
```

Use the same NamedTuple pattern for `LabelTable` (fields: `df`, `n_total`, `n_positive`, `n_negative`, `n_excluded`, `source`).

**Input validation and warn-on-unexpected pattern** (lines 240-258 of orthology.py):
```python
    if dropped_fraction >= _HIGH_DROPPED_THRESHOLD:
        log.warning(
            "build_one2one_orthologs: high dropped fraction %.1f%% "
            "(%d input rows, %d one2one kept). ...",
            100.0 * dropped_fraction,
            n_input,
            n_one2one,
        )
    else:
        log.debug(
            "build_one2one_orthologs: %d input rows -> %d one2one kept "
            "(dropped %.1f%%)",
            n_input,
            n_one2one,
            100.0 * dropped_fraction,
        )
```

Apply same warn-on-high-exclusion pattern when DILIrank Ambiguous fraction exceeds an expected threshold (e.g. > 0.30).

**DILIrank-specific loading** (from RESEARCH.md Pattern 1):
```python
def load_dilirank(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)          # header=1: skip FDA title row
    concern = df["vDILI-Concern"].str.lower().str.strip()
    pos_mask = concern.isin({"vmost-dili-concern", "vless-dili-concern"})
    neg_mask = concern.isin({"vno-dili-concern"})
    df = df[pos_mask | neg_mask].copy()
    df["dili_binary"] = pos_mask[pos_mask | neg_mask].astype(int)
    df["name_lower"] = df["CompoundName"].str.lower().str.strip()
    return df[["LTKBID", "CompoundName", "name_lower", "dili_binary", "SeverityClass"]]
```

**DILIst-specific loading** (from RESEARCH.md pitfall 3):
```python
def load_dilist(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()         # pitfall 3: trailing space on column name
    df["dili_binary"] = df["DILIst Classification"].astype(int)
    df["name_lower"] = df["Compound"].str.lower().str.strip()
    return df[["name_lower", "dili_binary"]]
```

---

### `src/spatial/eda/smiles_join.py` (utility, CRUD)

**Analog:** `src/spatial/gene_alignment.py`

**Imports pattern** (lines 44-54 of gene_alignment.py):
```python
from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["align_to_gene_space", "coverage_fraction"]
```

Replace `numpy` import with `pandas` for the join logic. Keep `from __future__ import annotations`, logger, and `__all__`.

**Coverage-reporting pattern** (lines 230-240 of gene_alignment.py):
```python
    cov = coverage_fraction(list(resolved.keys()), target_genes)
    if cov < 0.5 and len(target_genes) > 0:
        log.warning(
            "align_to_gene_space: low coverage %.1f%% (%d / %d target genes "
            "found in source). ...",
            100.0 * cov,
            int(round(cov * len(target_genes))),
            len(target_genes),
            missing,
        )
```

Apply same coverage-warning pattern when SMILES join coverage falls below 40% of non-Ambiguous DILIrank drugs.

**SMILES cascade** (from RESEARCH.md Pattern 2):
```python
def join_smiles_cascade(labels_df: pd.DataFrame,
                        dili_canonical_path: str,
                        drugbank_path: str) -> pd.DataFrame:
    # Layer 1: dili_canonical (canonical_smiles column, join key: name_lower)
    c1 = pd.read_csv(dili_canonical_path)
    c1["name_lower"] = c1["drug_name"].str.lower().str.strip()
    merged = labels_df.merge(c1[["name_lower", "canonical_smiles"]].rename(
        columns={"canonical_smiles": "smiles"}), on="name_lower", how="left")

    # Layer 2: drugbank_smiles_index (name_lower is already the join key)
    missing_mask = merged["smiles"].isna()
    c2 = pd.read_csv(drugbank_path)  # cols: name_lower, name, smiles
    residual = merged[missing_mask][["name_lower"]].merge(
        c2[["name_lower", "smiles"]], on="name_lower", how="left")
    merged.loc[missing_mask, "smiles"] = residual["smiles"].values

    # Layer 3: TDC fallback for still-missing (pytdc available; implement only if gap > 10%)
    return merged
```

---

### `src/spatial/eda/fingerprints.py` (utility, transform)

**Analog:** `src/spatial/gene_alignment.py`

**Module header and validation pattern** (lines 1-54 of gene_alignment.py — structure only):
Same `from __future__ import annotations`, `logging`, `__all__` pattern. Validation raises `ValueError` for unparseable SMILES (RDKit `MolFromSmiles` returns `None`), not an exception — consistent with gene_alignment's policy of filling with a zero/nan vector rather than crashing on bad input.

**Fingerprint function** (from RESEARCH.md Pattern 3):
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_ecfp4(smiles_list: list[str],
                    n_bits: int = 2048,
                    radius: int = 2) -> tuple[np.ndarray, np.ndarray]:
    fps = []
    valid_mask = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            valid_mask.append(False)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.frombuffer(fp.ToBitString().encode(), dtype='u1') - ord('0'))
            valid_mask.append(True)
    return np.array(fps, dtype=np.uint8), np.array(valid_mask)
```

---

### `src/spatial/eda/floor.py` (utility, batch)

**Analog:** `src/spatial/orthology.py` (NamedTuple output schema) + sibling `dili_downstream/src/data/wangli_lincs_lookup.py` (batch array processing with logging)

**NamedTuple output schema** (lines 63-89 of orthology.py — copy structure):
```python
class FloorResult(NamedTuple):
    """Floor classifier results.

    Attributes
    ----------
    lr_auroc : float
    rf_auroc : float
    label_entropy : float
    class_balance : float  # positive fraction
    auprc_base_rate : float
    n_drugs : int
    n_drugs_with_smiles : int
    """
    lr_auroc: float
    rf_auroc: float
    label_entropy: float
    class_balance: float
    auprc_base_rate: float
    n_drugs: int
    n_drugs_with_smiles: int
```

**Logging convention** (lines 169-172 of wangli_lincs_lookup.py):
```python
    log.info(
        "lookup_inst_ids: query=%d found=%d missing=%d (h5=%d profiles x %d genes)",
        len(inst_ids), n_found, len(missing_ids), n_profiles, n_genes,
    )
```

Apply same `log.info` pattern at completion: report n_drugs, n_drugs_with_smiles, lr_auroc, rf_auroc.

---

### `src/spatial/eda/ceiling.py` (utility, batch)

**Analog:** sibling `dili_downstream/src/data/wangli_lincs_lookup.py`

**File I/O with path validation** (lines 64-111 of wangli_lincs_lookup.py):
```python
def open_lincs_h5(h5_path: str | Path) -> h5py.File:
    f = h5py.File(str(h5_path), "r")
    try:
        keys = list(f.keys())
        for required in ("colid", "rowid", "data"):
            if required not in f:
                raise ValueError(
                    f"Missing required dataset {required!r} in {h5_path}; "
                    f"got top-level keys {keys}."
                )
        ...
    except Exception:
        f.close()
        raise
    return f
```

Copy this "validate schema after open, close on failure" pattern for the ceiling loader. The ceiling loads `.npy` + `.csv` instead of h5, but the same "validate before proceeding" structure applies:

```python
import numpy as np
import pandas as pd
from pathlib import Path

def load_ceiling(de_path: str | Path,
                 profiles_path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load wangli_measured_de.npy + wangli_profiles.csv as the liver ceiling.
    Validates shape agreement before returning."""
    de = np.load(str(de_path))           # expected (5517, 978)
    prof = pd.read_csv(str(profiles_path))
    if de.shape[0] != len(prof):
        raise ValueError(
            f"ceiling: de.shape[0]={de.shape[0]} != len(profiles)={len(prof)}. "
            "Check that wangli_measured_de.npy and wangli_profiles.csv are in sync."
        )
    log.info("load_ceiling: de shape %s, %d profiles", de.shape, len(prof))
    return de, prof
```

**Participation ratio** (from RESEARCH.md Pattern 4):
```python
def participation_ratio(X: np.ndarray) -> float:
    X_centered = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    lambdas = s ** 2
    lambdas = lambdas[lambdas > 0]
    return float((lambdas.sum()) ** 2 / (lambdas ** 2).sum())
```

**Per-gene MI** (from RESEARCH.md Pattern 5):
```python
from sklearn.feature_selection import mutual_info_classif

def per_gene_mi(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> np.ndarray:
    return mutual_info_classif(X, y, discrete_features=False,
                               n_neighbors=3, random_state=random_state)
```

---

### `src/spatial/eda/bootstrap.py` (utility, batch)

**Analog:** `src/spatial/orthology.py` (NamedTuple schema) for the result container.

**NamedTuple output schema** (lines 63-89 of orthology.py — copy structure):
```python
class BootstrapResult(NamedTuple):
    """Paired bootstrap CI of the AUROC gap (ceiling - floor).

    Attributes
    ----------
    gap_observed : float
    ci_lower : float
    ci_upper : float
    gate_fires : bool       # True = Halt Gate 2 fires (CI includes 0)
    n_resamples_valid : int
    """
    gap_observed: float
    ci_lower: float
    ci_upper: float
    gate_fires: bool
    n_resamples_valid: int
```

**Bootstrap function** (from RESEARCH.md Pattern 7):
```python
import numpy as np
from sklearn.metrics import roc_auc_score

def paired_bootstrap_auroc_gap(y: np.ndarray,
                                floor_probs: np.ndarray,
                                ceil_probs: np.ndarray,
                                n_resamples: int = 10_000,
                                seed: int = 42) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps = np.empty(n_resamples)
    obs_gap = roc_auc_score(y, ceil_probs) - roc_auc_score(y, floor_probs)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        if y_b.sum() == 0 or y_b.sum() == n:
            gaps[i] = np.nan
            continue
        gaps[i] = roc_auc_score(y_b, ceil_probs[idx]) - roc_auc_score(y_b, floor_probs[idx])
    gaps = gaps[~np.isnan(gaps)]
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    return BootstrapResult(
        gap_observed=obs_gap,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        gate_fires=ci_lo <= 0,
        n_resamples_valid=len(gaps),
    )
```

---

### `src/spatial/eda/region_diagnostics.py` (utility, batch)

**Analog:** `src/spatial/pseudobulk.py` (pure library structure, `from_anndata` lazy import) + `src/spatial/orthology.py` (result NamedTuple)

**Pure library pattern with lazy AnnData import** (lines 233-297 of pseudobulk.py):
```python
def from_anndata(
    adata,
    obs_col: str,
    *,
    agg: AggMode = "mean",
    region_labels: Optional[Sequence[str]] = None,
    layer: Optional[str] = None,
) -> PseudobulkResult:
    """Extract matrix and labels from an AnnData object and call `pseudobulk`.

    This is a convenience wrapper; no AnnData is imported at module level.
    AnnData must be installed in the calling environment.
    """
    if obs_col not in adata.obs.columns:
        raise KeyError(
            f"from_anndata: obs_col={obs_col!r} not found in adata.obs. "
            f"Available columns: {adata.obs.columns.tolist()}"
        )
    ...
    # AnnData matrices may be scipy sparse — convert to dense ndarray.
    if hasattr(raw, "toarray"):
        mat = raw.toarray()
    else:
        mat = np.asarray(raw)
```

Apply same `hasattr(raw, "toarray")` sparse-matrix guard in any place `adata.X` is accessed.

**Moran's I API** (from RESEARCH.md Pattern 6):
```python
import squidpy as sq
import anndata as ad

def compute_moran_svgs(adata: ad.AnnData,
                       spatial_key: str = "spatial",
                       n_neighs: int = 6) -> dict:
    sq.gr.spatial_neighbors_knn(adata, spatial_key=spatial_key,
                                 n_neighs=n_neighs, key_added="spatial")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=None,
                            connectivity_key="spatial_connectivities",
                            transformation=True, copy=False)
    moran_df = adata.uns["moranI"]
    svgs = moran_df[moran_df["pval_norm"] < 0.05]
    return {
        "svg_count": len(svgs),
        "svg_fraction": len(svgs) / len(moran_df),
        "top_svgs": svgs.nlargest(20, "I").index.tolist(),
    }
```

**Ortholog-filtered cross-species correlation** (from RESEARCH.md Pattern 9):
```python
from src.spatial.orthology import OrthologTable
from scipy.stats import pearsonr

def human_mouse_liver_correlation(human_profile, human_genes,
                                   mouse_profile, mouse_genes,
                                   ortholog_table: OrthologTable) -> dict:
    ot = ortholog_table.pairs   # cols: human_symbol, mouse_symbol
    h_to_m = dict(zip(ot["human_symbol"], ot["mouse_symbol"]))
    h_idx = {g: i for i, g in enumerate(human_genes)}
    m_idx = {g: i for i, g in enumerate(mouse_genes)}
    h_vals, m_vals = [], []
    for h_gene, m_gene in h_to_m.items():
        if h_gene in h_idx and m_gene in m_idx:
            h_vals.append(human_profile[h_idx[h_gene]])
            m_vals.append(mouse_profile[m_idx[m_gene]])
    r, p = pearsonr(np.array(h_vals), np.array(m_vals))
    return {"pearson_r": r, "p_value": p, "n_genes_compared": len(h_vals)}
```

**OOD Mahalanobis** (from RESEARCH.md Pattern 8 — use 978-gene subspace):
```python
from scipy.spatial.distance import mahalanobis

def ood_mahalanobis(manifold: np.ndarray,   # (10, d) cancer-line matrix
                    query: np.ndarray,       # (n_regions, d) pseudobulk
                    alpha: float = 1e-2) -> np.ndarray:
    cov = np.cov(manifold.T)
    cov_reg = cov + alpha * np.eye(cov.shape[0])
    vi = np.linalg.inv(cov_reg)
    mean_ref = manifold.mean(axis=0)
    return np.array([mahalanobis(q, mean_ref, vi) for q in query])
```

---

### `scripts/run_p1_eda.py` (script, request-response)

**Analog:** `scripts/report_coverage.py`

**Path block and sys.path injection** (lines 35-52 of report_coverage.py):
```python
from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.spatial.config import N_PDG          # noqa: E402
from src.spatial.datasets import SPATIAL_DATASETS  # noqa: E402
from src.spatial.gene_alignment import coverage_fraction  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "spatial"
PROCESSED = ROOT / "data" / "processed" / "spatial"
RESULTS = ROOT / "results" / "tables"
PHASE_DIR = ROOT / ".planning" / "phases" / "00-dataset-acquisition-manifest"
```

Copy this exact pattern for `run_p1_eda.py`. Replace imported modules with the new `src.spatial.eda.*` modules. Set `PHASE_DIR = ROOT / ".planning" / "phases" / "01-eda-the-bracket"`.

**Sub-command dispatch** (adapt from report_coverage.py `main()` at line 224):
```python
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", choices=["floor", "ceiling", "diagnostics", "all"])
    args = parser.parse_args()
    if args.subcommand in ("floor", "all"):
        run_floor()
    if args.subcommand in ("ceiling", "all"):
        run_ceiling()
    if args.subcommand in ("diagnostics", "all"):
        run_diagnostics()

if __name__ == "__main__":
    main()
```

**Halt gate + HALT_REASON.md pattern** (lines 375-385 of report_coverage.py):
```python
    if halt_violations:
        halt_path = PHASE_DIR / "HALT_REASON.md"
        with open(halt_path, "w") as fh:
            fh.write("# HALT: Gate 2 Violation — Floor-Ceiling Gap Not Significant\n\n")
            for v in halt_violations:
                fh.write(f"- {v}\n")
        print(f"\nHALT_REASON.md written: {halt_path}", file=sys.stderr)
        sys.exit(1)
```

**Results table writing pattern** (lines 328-372 of report_coverage.py):
```python
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as fh:
        fh.write("# P1 EDA Report: Floor-Ceiling Bracket (Liver, Human)\n\n")
        fh.write("| organ | species | ... | gate |\n")
        fh.write("|-------|---------|-----|------|\n")
        for row in rows:
            fh.write(f"| {row['organ']} | ... |\n")
```

---

### `tests/spatial/__init__.py`

Empty file. Same as `tests/__init__.py` in the repo root (blank). Create with zero content.

---

### `tests/spatial/test_eda_labels.py` (test)

**Analog:** `tests/spatial/test_gene_alignment.py`

**File header and import pattern** (lines 1-58 of test_gene_alignment.py):
```python
"""Unit tests for `src/spatial/eda/labels.py`.

In-memory fixtures only — no real data files are read.

Behaviors covered:
  1. DILIrank binary encoding: Ambiguous excluded, vMOST/vMost both map to 1.
  2. DILIrank name_lower normalization.
  3. DILIst trailing space on column name handled by .str.strip().
  4. SMILES join cascade: dili_canonical layer first, drugbank fallback.
  5. SMILES join: drugs with no hit in either layer have smiles=NaN.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.spatial.eda.labels import load_dilirank, load_dilist
from src.spatial.eda.smiles_join import join_smiles_cascade
```

**In-memory fixture pattern** (lines 50-76 of test_gene_alignment.py):
```python
def _make_dilirank_df() -> pd.DataFrame:
    """Return a minimal synthetic dilirank DataFrame (no file I/O)."""
    return pd.DataFrame({
        "LTKBID": ["A", "B", "C", "D"],
        "CompoundName": ["DrugA", "DrugB", "DrugC", "DrugD"],
        "vDILI-Concern": ["vMost-DILI-concern", "vNo-DILI-Concern",
                           "Ambiguous-DILI-concern", "vMOST-DILI-concern"],
        "SeverityClass": ["1", "0", "A", "1"],
    })
```

**Test function signature pattern** (line 79 of test_gene_alignment.py):
```python
def test_coverage_exact_overlap() -> None:
    """Test 1: complete overlap returns 1.0."""
```

All test functions use `-> None` return annotation and a one-line docstring describing the behavior being tested.

---

### `tests/spatial/test_eda_fingerprints.py` (test)

**Analog:** `tests/spatial/test_gene_alignment.py`

Same header/import pattern as above. Use inline SMILES for fixtures (aspirin: `CC(=O)Oc1ccccc1C(=O)O`; invalid: `not_a_smiles`). Never read from disk. Test: shape == (n, 2048), valid_mask True for known-valid SMILES, False for invalid.

---

### `tests/spatial/test_eda_floor.py` (test)

**Analog:** `tests/spatial/test_pseudobulk.py`

**Synthetic data fixture pattern** (lines 50-76 of test_pseudobulk.py):
```python
def _make_matrix_and_labels() -> tuple[np.ndarray, list[str], list[str]]:
    """Return a 6-spot x 4-gene float matrix, labels, and gene_names."""
    matrix = np.array([...], dtype=np.float64)
    labels = ["A", "B", "A", "C", "B", "C"]
    gene_names = ["G1", "G2", "G3", "G4"]
    return matrix, labels, gene_names
```

For the floor smoke test, create a small synthetic fingerprint matrix (20 drugs, 2048 bits) and binary labels (10 positive, 10 negative). Assert `floor_auroc > 0.5`. This is the correct approach — no real DILIrank data, purely synthetic shape/value fixture.

---

### `tests/spatial/test_eda_ceiling.py` (test)

**Analog:** `tests/spatial/test_gene_alignment.py`

Test `participation_ratio` on a synthetic (100, 50) matrix where the true PR is known analytically (e.g., identity covariance has PR = 50). Test `per_gene_mi` returns shape `(n_genes,)`. Test ceiling AUROC on synthetic drug-level data.

---

### `tests/spatial/test_eda_bootstrap.py` (test)

**Analog:** `tests/spatial/test_pseudobulk.py`

Test that `paired_bootstrap_auroc_gap` returns a `BootstrapResult` with `ci_upper > ci_lower` (nonzero width). Use synthetic `y`, `floor_probs`, `ceil_probs` arrays (50 drugs). Test that `gate_fires=True` when floor and ceiling are identical (gap = 0 throughout).

---

### `tests/spatial/test_eda_region.py` (test)

**Analog:** `tests/spatial/test_pseudobulk.py` (duck-typed AnnData fixture)

**Duck-typed AnnData fixture pattern** (lines 35-43 of test_pseudobulk.py):
```python
from types import SimpleNamespace

# In test_pseudobulk.py this creates a duck-typed AnnData-like object:
adata_like = SimpleNamespace(
    obs=pd.DataFrame({"region": labels}, index=barcodes),
    X=matrix,
    var_names=pd.Index(gene_names),
    layers={},
)
```

For `test_eda_region.py`, use `SimpleNamespace` to create a minimal AnnData-like object with `adata.obsm["spatial"]` populated. Test that `human_mouse_liver_correlation` uses only genes in the ortholog table (check `n_genes_compared < len(human_genes)`).

---

## Shared Patterns

### Module header (apply to all `src/spatial/eda/*.py` files)

**Source:** `src/spatial/orthology.py` lines 1-37 and `src/spatial/pseudobulk.py` lines 1-43

```python
"""Pure library: <one-line description>.

Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides real data.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["<public_function>", "<ResultClass>"]
```

### Error handling (apply to all library functions)

**Source:** `src/spatial/gene_alignment.py` lines 172-212 and `src/spatial/pseudobulk.py` lines 90-122

Pattern: validate inputs at the top of every public function with `raise ValueError(f"<function>: <specific message>. Got {actual!r}.")`. Use f-strings with `!r` for repr of bad values. Do not catch and re-raise — let exceptions propagate.

```python
    if matrix.ndim != 2:
        raise ValueError(
            f"pseudobulk: matrix must be 2-D (spots x genes), got ndim={matrix.ndim}."
        )
```

### Logging convention (apply to all library functions)

**Source:** `src/spatial/pseudobulk.py` lines 216-219 and `src/spatial/orthology.py` lines 247-258

```python
    log.info(
        "pseudobulk: agg=%s n_spots=%d n_regions=%d (%d empty) n_genes=%d",
        agg, n_spots, n_regions, len(empty_regions), n_genes,
    )
```

Use `log.info` for completion messages with quantitative counts. Use `log.warning` for unexpected conditions (high exclusion fraction, low coverage). Use `log.debug` for per-item iteration detail.

### Path handling in scripts (apply to `scripts/run_p1_eda.py`)

**Source:** `scripts/report_coverage.py` lines 35-51

```python
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tables"
PHASE_DIR = ROOT / ".planning" / "phases" / "01-eda-the-bracket"
```

Never use `os.path` — always `pathlib.Path`. Always resolve from `__file__` so the script works from any working directory. `RESULTS.mkdir(parents=True, exist_ok=True)` before writing output files.

### Provenance discipline (apply to `scripts/run_p1_eda.py` and `MANIFEST.md` update step)

**Source:** CONTEXT.md D-05 and MANIFEST.md table structure (lines 30-60)

Before reading `wangli_measured_de.npy`, `wangli_profiles.csv`, `dili_canonical.csv`, `dilist_smiles_resolved.csv`, `drugbank_smiles_index.csv`, and the PDG manifold CSV, they must be recorded in `MANIFEST.md` with the SHA256 values from RESEARCH.md. The table structure follows the existing pattern:

```markdown
| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |
|------|------|--------|--------|---------|--------------------|----|
| wangli measured DE | `../dili_downstream/data/processed/wangli_measured_de.npy` | `aaeb07ce...` | dili_downstream Phase 1 output | see source | N/A | External input; shape (5517, 978) float32 |
```

### HALT_REASON.md writing pattern (apply to `scripts/run_p1_eda.py`)

**Source:** `scripts/report_coverage.py` lines 375-385

```python
    if gate_fires:
        halt_path = PHASE_DIR / "HALT_REASON.md"
        with open(halt_path, "w") as fh:
            fh.write("# HALT: Gate 2 Violation\n\n")
            fh.write(f"- Gap CI lower bound: {result.ci_lower:.4f} (includes 0)\n")
            fh.write("- Action: stop-and-REFRAME before proceeding to Phase 2.\n")
        print(f"\nHALT_REASON.md written: {halt_path}", file=sys.stderr)
        sys.exit(1)
```

### Test fixture convention (apply to all `tests/spatial/test_eda_*.py`)

**Source:** `tests/spatial/test_pseudobulk.py` lines 32-43 and `tests/spatial/test_gene_alignment.py` lines 1-48

All tests use in-memory synthetic fixtures only. No real data files are read. Fixtures are created inline as `np.ndarray`, `pd.DataFrame`, or `SimpleNamespace`. Test function names follow `test_<behavior>() -> None` with a one-line docstring.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.spatial.eda.<module> import <function>


def test_<behavior>() -> None:
    """<One sentence describing the expected behavior>."""
    # arrange
    ...
    # act
    result = <function>(...)
    # assert
    assert ...
```

### Environment invocation (apply to all scripts that run in dili_v04_env)

**Source:** RESEARCH.md "Environment Availability" and "Note on numba/numpy conflict"

All scripts must be run as:
```bash
conda run -n dili_v04_env python scripts/run_p1_eda.py <subcommand>
```

Within scripts, import `anndata` directly (not `scanpy`) to avoid the numba/numpy conflict in the base env.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/spatial/eda/floor.py` (sklearn LR+RF classifier fit) | utility | batch | No existing sklearn classification code in this repo; the classifier itself uses a well-established sklearn API (RESEARCH.md Pattern 3 has the full pattern) |

---

## Metadata

**Analog search scope:** `src/spatial/`, `scripts/`, `tests/spatial/`, `tests/`, `../dili_downstream/src/data/`
**Files scanned:** 9 source files, 3 test files
**Pattern extraction date:** 2026-06-22
