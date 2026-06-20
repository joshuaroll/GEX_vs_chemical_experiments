# Phase 0: Dataset acquisition & MANIFEST - Pattern Map

**Mapped:** 2026-06-20
**Files analyzed:** 8 (3 new src/script-ish, 1 registry extend, 1 MANIFEST, 1 test, 2 results tables)
**Analogs found:** 7 / 8 (1 partial — side-effecting download script has no in-repo precedent; `src/spatial/` is deliberately pure)

> All paths below are absolute under the single safe root
> `/raid/home/joshua/projects/GEX_vs_chemical_experiments/spatial_tests/`.
> The codebase convention is **pure libraries in `src/spatial/`** (no file I/O, no
> hardcoded paths, no real-data filenames) with side-effects pushed to `scripts/`.
> Every module opens with a docstring enumerating the "Hard rules honored". Reproduce that.

## File Classification

| New/Modified File (abs) | Role | Data Flow | Closest Analog (abs) | Match Quality |
|-------------------------|------|-----------|----------------------|---------------|
| `src/spatial/orthology.py` (NEW) | utility / pure library | transform (TSV → one2one map) + cached-file boundary | `src/spatial/gene_alignment.py` | exact (same pure-library + coverage-report shape) |
| `src/spatial/datasets.py` (EXTEND) | config / registry | declarative metadata | itself (extend in place) + `src/spatial/config.py` | exact (extend existing NamedTuple) |
| `scripts/download_*.py` + `scripts/build_orthologs.py` + `scripts/compute_sha256.py` (NEW) | script / driver | file-I/O, request-response (REST/HTTPS fetch) | **no in-repo analog** (src/spatial is pure by design) — use `region_signature.make_cache_key` hashlib idiom + RESEARCH Pattern-2 BioMart block | partial (role-only) |
| `MANIFEST.md` (NEW) | config / provenance record | record (paths+SHA+license) | `dili_downstream/MANIFEST.md` (sibling) | exact (sibling convention) |
| `tests/test_data_paths.py` (NEW) | test | unit / filesystem + network-gated | `tests/spatial/test_gene_alignment.py` | exact (style/fixtures) + `conftest.py` (`network` marker) |
| `results/tables/P0_coverage.md` (NEW) | results table | report (markdown table) | derived from `gene_alignment.coverage_fraction` output | role-match |
| `results/tables/P0_orthologs.md` (NEW) | results table | report (markdown table) | derived from `orthology.py` dropped-fraction | role-match |

## Pattern Assignments

### `src/spatial/orthology.py` (NEW — utility, transform + cached-file boundary)

**Analog:** `src/spatial/gene_alignment.py` (pure library, has `coverage_fraction`, emits a coverage/dropped-fraction style report). Cross-check: `src/spatial/region_signature.py` for the `__all__` + `Final` + `NamedTuple` result-container idiom and the `hashlib` provenance idiom.

**Module-header pattern** — copy the docstring shape from `gene_alignment.py:1-43`. It opens with a one-line "Pure library:" summary, then an explicit policy block, then a "Hard rules honored:" block. Reproduce all three sections. The non-negotiable lines to echo (from `gene_alignment.py:37-43`):
```python
"""Pure library: build a human↔mouse↔rat one-to-one ortholog map.
...
Hard rules honored:
    - Pure library: NO hardcoded absolute paths, NO real-data filenames.
    - No mock or synthetic labels — the caller provides real arrays / a cached TSV.
    - Network fetch lives behind a cached-TSV boundary (scripts/ does the GET).
"""
```

**Imports + module-logger + `__all__` pattern** (`gene_alignment.py:45-54`):
```python
from __future__ import annotations
import logging
from typing import Literal   # add: NamedTuple, Final as needed
import numpy as np
log = logging.getLogger(__name__)
__all__ = ["build_one2one_orthologs", "ortholog_report"]
```

**Result-container pattern** — for the dropped-fraction report use a frozen `NamedTuple` exactly like `RegionDE` / `PseudobulkResult` (`region_signature.py:77-106`, `pseudobulk.py:57-70`). Each field gets a numpydoc `Attributes` entry:
```python
class OrthologTable(NamedTuple):
    """Human↔mouse↔rat one-to-one ortholog map.

    Attributes
    ----------
    pairs : pd.DataFrame
        Columns: human_ensembl, human_symbol, mouse_ensembl, mouse_symbol,
        rat_ensembl, rat_symbol. One row per human gene kept (one2one in BOTH).
    n_input : int
        Rows in the raw BioMart TSV before filtering.
    n_one2one : int
        Rows kept (mouse_type == rat_type == "ortholog_one2one").
    dropped_fraction : float
        (n_input - n_one2one) / n_input  — reported per XC-08.
    """
    pairs: object
    n_input: int
    n_one2one: int
    dropped_fraction: float
```

**Core transform pattern** — mirror `coverage_fraction`'s shape (`gene_alignment.py:71-102`): a small pure function, set-based, deterministic, returns a float/typed result, with a `>>>` doctest in the docstring. The one2one filter is the core:
```python
# Keep rows where BOTH species are one2one; DROP many2many (XC-08, Pitfall 3).
mask = (df["mmusculus_homolog_orthology_type"] == "ortholog_one2one") \
     & (df["rnorvegicus_homolog_orthology_type"] == "ortholog_one2one")
kept = df[mask]
dropped_fraction = 1.0 - len(kept) / max(len(df), 1)
```

**Low-coverage / report-warning pattern** — reuse the `log.warning(...)` percentage-report idiom from `gene_alignment.py:231-240` when `dropped_fraction` is unexpectedly high, so the dropped fraction is surfaced, not silent.

**BioMart REST query** is the load-bearing literal from RESEARCH.md Pattern 2 (`00-RESEARCH.md:170-191`). The XML query string + the `requests.get("https://www.ensembl.org/biomart/martservice", params={"query": QUERY}, timeout=180)` call. **Per project purity rule this network GET belongs in `scripts/build_orthologs.py`, not in `orthology.py`** — `orthology.py` takes an already-fetched/cached TSV path or DataFrame and does the pure one2one filtering. (RESEARCH `00-RESEARCH.md:210` anti-pattern: "Putting network I/O inside `src/spatial/` pure functions".)

---

### `src/spatial/datasets.py` (EXTEND — config / registry, declarative metadata)

**Analog:** itself — extend the existing `SpatialDataset` NamedTuple in place (`datasets.py:27-53`). Do **not** rebuild; the 124 existing tests + this module's purity docstring (`datasets.py:1-16`) must stay intact.

**Extend the NamedTuple** per RESEARCH Pattern 1 (`00-RESEARCH.md:150-164`) — add fields to the existing schema at `datasets.py:48-53`:
```python
class SpatialDataset(NamedTuple):
    name: str
    organ: str
    species: str                      # NEW: "human" | "mouse" | "rat"
    platform: str
    accession: str                    # CORRECTED values (see below)
    access_mechanism: str             # NEW: "figshare_api"|"geo_supp"|"spatialLIBD"|"kpmp"|"url"
    expected_files: tuple[str, ...]   # NEW
    license: str                      # NEW
    whole_transcriptome: bool         # NEW: the Halt-Gate-1 flag
    usable_as_input: bool
    region_annotation_source: str
```
> Field-order note: existing entries are positional `SpatialDataset(...)` calls (`datasets.py:64-71`). Adding fields shifts positions — convert the existing entries to **keyword args** so the diff is safe, or append new fields at the tail. Prefer keyword args (matches the readable multi-line call style already used).

**Accession corrections** (the DATA-01 acceptance gate; do these BEFORE any download script — RESEARCH anti-pattern `00-RESEARCH.md:207`):
- Yu liver `datasets.py:68`: `GSE189994` + `figshare 17058105` → **`figshare 22321447`** (DOI `10.6084/m9.figshare.22321447.v1`, GPL-3.0+, files `L5_upload.zip` md5 `3abf91538674c3b0cbc3f55b9c5b6074`, `L18_upload.zip` md5 `31bbf855afd7e3ab26f65f8786575a0d`) — `00-RESEARCH.md:280-282`.
- Maynard brain `datasets.py:122`: `GSE144239` (unrelated SCC) → **spatialLIBD / LieberInstitute** — `00-RESEARCH.md:286-287`.
- Lake/KPMP kidney `datasets.py:95`: `GSE211785`-as-primary → **`GSE183456` + `GSE183279` primary**, demote `GSE211785` (Abedini) to substitute — `00-RESEARCH.md:284-285`.
- Siletti brain `datasets.py:127-132`: relabel — it is snRNA-seq, NOT human MERFISH (no public human-brain MERFISH exists) — `00-RESEARCH.md:331`.

**Docstring discipline** — keep/extend the module's "Hard rules enforced" block (`datasets.py:11-16`) to note `whole_transcriptome=True` ⇔ genome-scale Visium, panels `False`.

**Gene-space constants** are imported from `config.py` (`N_LANDMARK=978`, `N_PDG=10716`, `config.py:20-31`) — the coverage gate references `N_PDG`. Do not redefine; import.

---

### `scripts/download_*.py`, `scripts/build_orthologs.py`, `scripts/compute_sha256.py` (NEW — script / driver, file-I/O + request-response)

**Analog:** None in-repo — `src/spatial/` is pure by mandate, so there is no existing side-effecting fetcher to copy. Two idioms to lift instead:

1. **SHA helper** — copy the `hashlib` idiom verbatim-in-spirit from `region_signature.py:218`:
   ```python
   import hashlib
   def sha256_file(path: str) -> str:
       h = hashlib.sha256()
       with open(path, "rb") as f:
           for chunk in iter(lambda: f.read(1 << 20), b""):
               h.update(chunk)
       return h.hexdigest()
   ```
   (`region_signature.make_cache_key` uses `hashlib.sha256(...).hexdigest()` — same primitive, file-streaming variant.)

2. **Registry-driven dispatch** — RESEARCH Pattern 1 (`00-RESEARCH.md:146-148`): the driver imports `SPATIAL_DATASETS` from the corrected `datasets.py`, dispatches on `access_mechanism`, writes to `data/raw/spatial/<dataset>/`. Keep all `requests` / filesystem writes here (never in `src/spatial/`).

3. **BioMart fetch** lives in `scripts/build_orthologs.py` (the GET from RESEARCH `00-RESEARCH.md:188`), then calls the pure `orthology.py` filter and writes `data/processed/spatial/orthologs_h_m_r_one2one.tsv` + caches raw TSV to `data/raw/spatial/biomart/` (`00-RESEARCH.md:190`).

**CUDA hygiene N/A** — P0 is data-only, no torch import (RESEARCH `00-RESEARCH.md:439`).

---

### `MANIFEST.md` (NEW — provenance record)

**Analog:** `/raid/home/joshua/projects/GEX_vs_chemical_experiments/dili_downstream/MANIFEST.md` (sibling).

**Header + update-discipline pattern** (`dili_downstream/MANIFEST.md:1-7`):
```markdown
# Spatial Arm — Data and Code Manifest
**Version:** ...   **Generated:** 2026-06-20   **Last updated:** ...
This manifest is the single source of truth ... Update it at the end of every phase.
The data-path test (`tests/test_data_paths.py`) reads from this implicitly — keep them in sync.
```

**"Source code (SHA-pinned)" table** (`dili_downstream/MANIFEST.md:11-25`) — `| File | Source repo | Source SHA | Purpose |`. Record the two frozen checkpoints with paths+SHA (RESEARCH `00-RESEARCH.md:309-321`): MultiDCP-CheMoE at `/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt`, MultiDCP-PDG under `/raid/home/joshua/projects/MultiDCP/trained_models/`, MultiDCP repo SHA `871b8de0332045a3ad5ab3a39e689014317dadfe`.

**"Data files (SHA256-pinned)" table** (`dili_downstream/MANIFEST.md:29-36`) — extend the sibling columns with the P0-specific ones from RESEARCH `00-RESEARCH.md:313`: `| File | Path | SHA256 | Source | License | Whole-transcriptome | Notes |`. License column is mandatory (DATA-02; Yu = GPL-3.0+) — Pitfall 6 (`00-RESEARCH.md:269-272`).

**Environment row** — record `conda env export -n dili_v04_env` snapshot + installed `squidpy` version (DATA-02). Pin the squidpy version actually installed, not from training data (`00-RESEARCH.md:66`).

**Time-leakage discipline (XC-10)** — pin the Ensembl release number + BioMart query date in MANIFEST (`00-RESEARCH.md:192`).

---

### `tests/test_data_paths.py` (NEW — test, unit/filesystem + network-gated)

**Analog:** `tests/spatial/test_gene_alignment.py` (style) + `conftest.py` (`network` marker).

**Test-module docstring pattern** (`test_gene_alignment.py:1-48`) — open with `"""Unit tests for ..."""`, then an enumerated "Behaviors covered:" block listing each numbered test. Reproduce that enumerated style.

**Imports + helpers pattern** (`test_gene_alignment.py:50-71`):
```python
from __future__ import annotations
import math
import numpy as np
import pytest
from src.spatial.gene_alignment import coverage_fraction        # reuse for the gate
from src.spatial.config import N_PDG                            # 10716 denominator
from src.spatial.datasets import SPATIAL_DATASETS               # assert corrected accessions
```

**One-behavior-per-test function pattern** (`test_gene_alignment.py:79-90`): tiny functions, `-> None`, one-line docstring `"""Test N: ..."""`, single assertion, `pytest.approx` for floats.

**Whole-transcriptome gate test** — RESEARCH Pattern 3 (`00-RESEARCH.md:198-204`):
```python
cov = coverage_fraction(visium_var_names, multidcp_10716_symbols)
assert cov > 0.80, f"Halt Gate 1: {name} covers only {cov:.1%} — not whole-transcriptome"
```

**Accession-correction regression** — assert no entry's `accession` contains `GSE189994` or `GSE144239`, and Yu's contains `22321447` (DATA-01, test map `00-RESEARCH.md:416`).

**Network-gating pattern** — the `network` marker is REGISTERED in `conftest.py:10-16` but not yet used anywhere. Be the first user. Mark download/BioMart-touching tests `@pytest.mark.network` and skip-by-default with an env gate (the marker docstring names `TDC_NETWORK_TESTS=1`):
```python
import os, pytest
needs_net = pytest.mark.skipif(
    not os.environ.get("TDC_NETWORK_TESTS"),
    reason="network test; set TDC_NETWORK_TESTS=1 to enable",
)
@pytest.mark.network
@needs_net
def test_biomart_one2one_live() -> None: ...
```
Provide a small cached BioMart TSV fixture (or the chr21 sample, `00-RESEARCH.md:171-172`) so `test_ortholog_one2one` runs offline + deterministically (Wave-0 gap, `00-RESEARCH.md:432`).

**Location note:** create flat `tests/test_data_paths.py` (matches the DATA-01 acceptance wording and the sibling `dili_downstream` convention); leave `tests/spatial/` (124 tests) untouched. Open question flagged in RESEARCH `00-RESEARCH.md:370-371`.

---

### `results/tables/P0_coverage.md` + `P0_orthologs.md` (NEW — results tables)

**Analog:** no template file in-repo; format is a plain markdown table fed by computed values. P0_coverage: one row per Visium dataset with `coverage_fraction(var_names, MULTIDCP_10716)` (`gene_alignment.py:71-102`) and a pass/fail vs the 0.80 gate. P0_orthologs: `n_input`, `n_one2one`, `dropped_fraction` from `OrthologTable`. Normalize HGNC aliases before intersection or coverage is understated (Pitfall 4, `00-RESEARCH.md:257-261`).

## Shared Patterns

### Pure-library purity contract
**Source:** `src/spatial/gene_alignment.py:37-43`, `datasets.py:11-16`, `pseudobulk.py:1-11`, `region_signature.py:43-50`
**Apply to:** `orthology.py`, `datasets.py` (every `src/spatial/` file)
Every module docstring ends with an explicit "Hard rules honored:" / "Hard rules enforced:" block asserting NO hardcoded absolute paths, NO real-data filenames, no mock/synthetic data. Network and filesystem side-effects live only in `scripts/`.

### Module skeleton (imports → logger → `__all__` → constants → types → API)
**Source:** `gene_alignment.py:45-68`, `pseudobulk.py:32-49`, `region_signature.py:52-74`
**Apply to:** `orthology.py`
```python
from __future__ import annotations
import logging
from typing import Final, NamedTuple, Literal
import numpy as np
log = logging.getLogger(__name__)
__all__ = [...]
```
Gene-space constants (`N_LANDMARK`, `N_PDG`) imported from `config.py`, never redefined.

### Frozen typed result containers (numpydoc Attributes)
**Source:** `pseudobulk.PseudobulkResult` (`pseudobulk.py:57-70`), `region_signature.RegionDE` / `RegionSignatureManifest` (`region_signature.py:77-149`), `datasets.SpatialDataset` (`datasets.py:27-53`)
**Apply to:** `orthology.OrthologTable`, the extended `SpatialDataset`
All structured outputs are `NamedTuple`s with a full `Attributes` docstring block per field.

### SHA-256 / hashlib provenance
**Source:** `region_signature.make_cache_key` (`region_signature.py:54,211-218`)
**Apply to:** `scripts/compute_sha256.py`, MANIFEST data rows
`hashlib.sha256(...).hexdigest()` — file-streaming variant for the SHA256 column.

### Coverage/dropped-fraction reporting + warn-on-low
**Source:** `gene_alignment.coverage_fraction` (`gene_alignment.py:71-102`) + `log.warning` percentage report (`gene_alignment.py:231-240`)
**Apply to:** whole-transcriptome gate (P0_coverage), ortholog dropped-fraction (P0_orthologs)

### Network-test gating
**Source:** `conftest.py:10-16` (`network` marker, names `TDC_NETWORK_TESTS=1`)
**Apply to:** every download/BioMart test in `tests/test_data_paths.py` — `@pytest.mark.network` + `skipif` env gate.

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `scripts/download_*.py` (the actual fetchers) | script/driver | file-I/O + REST | `src/spatial/` is pure-by-mandate; no side-effecting fetcher exists in this repo. Use RESEARCH Pattern 1/2 (`00-RESEARCH.md:146-191`) + the `region_signature` hashlib idiom; keep all I/O out of `src/spatial/`. |

> The two results-table files are role-matches only (computed-markdown), not code analogs — listed under their assignments above rather than here.

## Metadata

**Analog search scope:** `src/spatial/` (datasets, gene_alignment, config, pseudobulk, region_signature, region_combiner), `tests/spatial/`, `conftest.py`, sibling `dili_downstream/MANIFEST.md`
**Files scanned:** 8 read in full/part + 2 sibling MANIFESTs located
**Pattern extraction date:** 2026-06-20
