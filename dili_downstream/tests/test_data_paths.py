"""Data-path sanity test — the only sanity test in the v0.4 pipeline.

Runs before every phase. Asserts that every path the current phase needs exists
and is readable. Paths are tagged by the phase that introduces them; phases not
yet started have their paths skipped (warned in the test report) so this file
remains green at the start of v0.4 and tightens monotonically as phases
complete.

Run with: pytest experiments/dili_downstream/tests/test_data_paths.py -v
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]  # …/dili_downstream/
UMBRELLA_ROOT = REPO_ROOT.parent  # …/GEX_vs_chemical_experiments/
MULTIDCP_DATA = Path("/raid/home/joshua/projects/MultiDCP/MultiDCP/data")
MULTIDCP_REPO = Path("/raid/home/joshua/projects/MultiDCP")


def _phase_active(phase_num: int) -> bool:
    """Read the current phase from .planning/STATE.md or fall back to env var.

    Returns True if a path tagged with `phase_num` should be checked. Paths
    from later phases are skipped (returned False) so this test remains green
    at the start of v0.4.
    """
    env = os.environ.get("DILI_V04_CURRENT_PHASE")
    if env:
        try:
            return phase_num <= int(env)
        except ValueError:
            pass
    # Default: only Phase 0 (pre-phase scaffolding) paths are required.
    return phase_num <= 0


# Manifest: every path tagged by the phase that introduces it.
# Phase 0 paths must exist after scaffolding (this commit).
# Phase 1+ paths will be added as their phases complete.
PATHS = [
    # ────────────────────────────────────────────────────────
    # Phase 0 — scaffolding (these MUST exist after the v0.4 init commit)
    # ────────────────────────────────────────────────────────
    (0, REPO_ROOT / "CLAUDE_DILI_DOWNSTREAM.md", "source-of-truth doc"),
    (0, REPO_ROOT / "MANIFEST.md", "v0.4 manifest"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "multidcp_pdg.py", "Cloned baseline MultiDCP-PDG architecture"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "multidcp_pdgrapher_fusion.py", "Cloned MultiDCP-PDG-Fusion wrapper"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "fusion_moe_models.py", "Cloned StructuredSparseMoE definitions"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "neural_fingerprint.py", "Cloned drug encoder"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "drug_gene_attention.py", "Cloned drug-gene attention"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "ltr_loss.py", "Cloned loss utilities"),
    (0, REPO_ROOT / "src" / "models" / "upstream" / "README.md", "Upstream-clone README with pinned SHA"),
    # MultiDCP repo and data (always available; not produced by v0.4)
    (0, MULTIDCP_REPO, "MultiDCP source repository"),
    (0, MULTIDCP_DATA, "MultiDCP data directory"),
    (0, MULTIDCP_DATA / "pdg_brddrugfiltered.pkl", "PDG perturbation pickle (Conditions B/E training data)"),
    (0, MULTIDCP_DATA / "pdg_diseased_brddrugfiltered.pkl", "PDG diseased baseline pickle (DE computation)"),
    (0, MULTIDCP_DATA / "drugbank_data", "DrugBank SMILES resolution source"),
    (0, MULTIDCP_DATA / "ehill_data", "ehill viability data (M7 Szalai control source)"),
    (0, MULTIDCP_DATA / "pert_transcriptom" / "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx",
     "LINCS L1000 Phase II level 5 (12,328 BING-imputed genes; subset to PDG ~10,717 for Condition D)"),

    # ────────────────────────────────────────────────────────
    # Phase 1 — Data foundation
    # ────────────────────────────────────────────────────────
    (1, REPO_ROOT / "data" / "raw" / "DILIst", "DILIst raw download dir (must contain at least one xlsx after Phase 1)"),
    (1, REPO_ROOT / "data" / "processed" / "dili_canonical.csv", "Phase 1 output: harmonized DILI table"),
    (1, REPO_ROOT / "results" / "tables" / "P1_data_summary.md", "Phase 1 deliverable: data summary"),

    # ────────────────────────────────────────────────────────
    # Phase 2 — Unified split construction
    # ────────────────────────────────────────────────────────
    (2, REPO_ROOT / "data" / "splits" / "unified_dili_aware_scaffold.json", "Primary unified DILI-aware scaffold split"),
    (2, REPO_ROOT / "data" / "splits" / "unified_dili_aware_cluster.json", "Cluster-based robustness split"),
    (2, REPO_ROOT / "data" / "splits" / "tdc_dili_scaffold.json", "TDC-DILI secondary leaderboard split"),
    (2, REPO_ROOT / "results" / "tables" / "P2_split_summary.md", "Phase 2 deliverable: split summary"),

    # ────────────────────────────────────────────────────────
    # Phase 3 — Upstream training (12 frozen checkpoints)
    # ────────────────────────────────────────────────────────
    *[
        (3, REPO_ROOT / "trained_models" / f"{cond}_seed{seed}" / "best.pt",
         f"Phase 3 frozen checkpoint: {cond} seed {seed}")
        for cond in ("B", "C", "E", "F")
        for seed in (1, 2, 3)
    ],

    # ────────────────────────────────────────────────────────
    # Phase 4 — Signature & encoder caching
    # ────────────────────────────────────────────────────────
    (4, REPO_ROOT / "data" / "processed" / "sig_multidcp_de.pt", "Phase 4: Condition B signature cache"),
    (4, REPO_ROOT / "data" / "processed" / "sig_chemoe_de.pt", "Phase 4: Condition C signature cache"),
    (4, REPO_ROOT / "data" / "processed" / "sig_multidcp_dilituned_de.pt", "Phase 4: Condition E signature cache"),
    (4, REPO_ROOT / "data" / "processed" / "sig_chemoe_dilituned_de.pt", "Phase 4: Condition F signature cache"),
    (4, REPO_ROOT / "data" / "processed" / "sig_lincs_de.pt", "Phase 4: Condition D measured-LINCS cache"),
    (4, REPO_ROOT / "data" / "processed" / "chem_embeddings_chemberta.pt", "Phase 4: ChemBERTa embeddings"),
    (4, REPO_ROOT / "data" / "processed" / "chem_embeddings_molformer.pt", "Phase 4: MolFormer embeddings"),
    (4, REPO_ROOT / "data" / "processed" / "chem_embeddings_gin.pt", "Phase 4: GIN embeddings"),
    (4, REPO_ROOT / "data" / "processed" / "chem_embeddings_unimol.pt", "Phase 4: UniMol embeddings"),
]


@pytest.mark.parametrize("phase, path, description", PATHS, ids=lambda p: str(p))
def test_path_exists(phase: int, path: Path, description: str) -> None:
    """Assert path exists and is readable, OR skip if its phase isn't active yet."""
    if not _phase_active(phase):
        pytest.skip(f"Phase {phase} not yet active (set DILI_V04_CURRENT_PHASE to enable)")
    assert path.exists(), f"Phase {phase} path missing: {path} ({description})"
    assert os.access(path, os.R_OK), f"Phase {phase} path not readable: {path} ({description})"


def test_pytdc_importable() -> None:
    """TDC-DILI is fetched via the pytdc library — must be importable in the env."""
    if not _phase_active(2):
        pytest.skip("pytdc check is a Phase 2+ requirement")
    try:
        importlib.import_module("tdc")
    except ImportError:
        pytest.fail(
            "pytdc not installed. Run `scripts/env_setup.sh` to install all v0.4 deps "
            "into the dili_v04_env conda env."
        )


def test_upstream_clone_sha() -> None:
    """Upstream model files are SHA-pinned. Ensure the README documents the pinned SHA."""
    readme = REPO_ROOT / "src" / "models" / "upstream" / "README.md"
    text = readme.read_text()
    assert "871b8de0332045a3ad5ab3a39e689014317dadfe" in text, (
        "Upstream-clone README does not document the pinned MultiDCP SHA. "
        "If the SHA changed, update the README and record the diff in MANIFEST.md."
    )
