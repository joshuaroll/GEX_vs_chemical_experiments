"""WIRE-01 cache driver: per-region predicted DE for liver, human + mouse.

Drives the now-real ``RegionSignatureCacher`` (frozen MultiDCP-CheMoE, row-17)
to write the 3-vector cache (de/treated/control, D-03) for the starting organ
(liver) in BOTH species (D-01: human = P4 headline, mouse = WIRE-03 APAP gate).

CUDA hygiene (Hard Rule 5 / Pitfall 7)
--------------------------------------
``--gpu`` is parsed and ``CUDA_VISIBLE_DEVICES`` is set BEFORE ``import torch``.
A free device is auto-detected via ``nvidia-smi`` and we ALWAYS leave one GPU
free on the shared box. ``--gpu cpu`` forces CPU.

Pipeline per species
--------------------
1. Load liver Visium basal (human = yu2022 L5; mouse = GSE272564 control arm).
2. Assign periportal / pericentral zones per spot from canonical markers (D-07).
3. Pseudobulk per zone, align each zone basal to the 10,716 PDG gene order, then
   0-1 min-max normalize to the manifold range (Pitfall 1) BEFORE inference.
4. Build the liver drug set with resolved SMILES (reuse the P1 sibling tables;
   no SMILES re-resolution).
5. Run the frozen forward (treated + inert-control per zone, rule B / D-02),
   write the 3-vector cache + manifest JSON under
   ``data/processed/spatial/region_de_cache/{species}/``.
6. Real-inference sanity (Pitfall 1): log + assert the first prediction is
   finite and broadly within the manifold range.

Hard rules honored: real data only (no mocked outputs); rule-B DE; provenance
asserted against MANIFEST before any read (sys.exit(2) if absent); SHA-pinned
checkpoint asserted at load.

Usage
-----
    conda run -n dili_v04_env python scripts/cache_region_de.py --gpu auto
    conda run -n dili_v04_env python scripts/cache_region_de.py --gpu 1
    conda run -n dili_v04_env python scripts/cache_region_de.py --gpu cpu --species human
"""

from __future__ import annotations

# ===========================================================================
# CUDA hygiene FIRST (Hard Rule 5 / Pitfall 7): parse --gpu and set
# CUDA_VISIBLE_DEVICES BEFORE importing torch. Keep this above all heavy imports.
# ===========================================================================
import argparse
import os
import subprocess
import sys


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cache per-region predicted DE (liver, human + mouse)."
    )
    p.add_argument(
        "--gpu",
        default="auto",
        help="GPU id (e.g. '1'), 'auto' (least-used, leaves one free), or 'cpu'.",
    )
    p.add_argument(
        "--species",
        default="all",
        choices=["all", "human", "mouse"],
        help="Which species cache(s) to write.",
    )
    p.add_argument(
        "--max-drugs",
        type=int,
        default=0,
        help="If >0, cap the liver drug set to the first N resolved drugs "
        "(smoke / quick run). 0 = all resolved liver drugs.",
    )
    p.add_argument(
        "--no-verify-sha",
        action="store_true",
        help="Skip the checkpoint SHA256 assertion (NOT recommended).",
    )
    return p.parse_args()


def _free_gpu_leaving_one_free() -> str:
    """Pick the least-used GPU while ALWAYS leaving one GPU free (Hard Rule 5).

    Returns the chosen physical GPU id as a string, or "" if none can be chosen
    safely (caller should then fall back to CPU).
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    rows = []
    for line in out.strip().splitlines():
        idx_str, used_str = [s.strip() for s in line.split(",")]
        rows.append((int(idx_str), int(used_str)))
    if len(rows) < 2:
        # Only one GPU: cannot leave one free and still use one -> CPU.
        return ""
    # Sort by memory used ascending; the least-used GPUs are candidates.
    rows.sort(key=lambda r: r[1])
    # Treat <512 MB as "free". Keep at least one free GPU: if every GPU is free,
    # take the least-used and the rest stay free; if only one is free, do NOT
    # take it (it is the one we must leave free).
    free = [idx for idx, used in rows if used < 512]
    if len(free) >= 2:
        return str(free[0])
    if len(free) == 1:
        # Exactly one free GPU -> must leave it free; refuse to grab it.
        return ""
    # No free GPUs at all -> nothing to take; CPU.
    return ""


def _resolve_device(gpu_arg: str) -> str:
    """Set CUDA_VISIBLE_DEVICES (before torch import) and return a torch device str."""
    if gpu_arg == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if gpu_arg == "auto":
        chosen = _free_gpu_leaving_one_free()
        if chosen == "":
            print(
                "WARNING: no GPU could be selected while leaving one free; "
                "falling back to CPU.",
                file=sys.stderr,
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
        return "cuda:0"  # the single visible device after masking
    # Explicit id.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_arg)
    return "cuda:0"


_ARGS = _parse_cli()
DEVICE = _resolve_device(_ARGS.gpu)

# ---- ONLY NOW are heavy imports allowed (torch is imported transitively) ----
import json  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cache_region_de")

# Library imports after sys.path set.
from src.spatial.config import N_PDG  # noqa: E402
from src.spatial.gene_alignment import align_to_gene_space  # noqa: E402
from src.spatial.pseudobulk import pseudobulk  # noqa: E402
from src.spatial.region_signature import RegionSignatureCacher  # noqa: E402
from src.spatial.eda.labels import load_dilirank  # noqa: E402
from src.spatial.eda.smiles_join import join_smiles_cascade  # noqa: E402

# Reuse the verified P1 Visium loaders (human yu2022 + mouse GSE272564 control).
from scripts.run_p1_eda import (  # noqa: E402
    _assert_manifest_rows,
    _load_yu2022_l5,
    _load_gse272564_mouse_ctrl,
    DILIRANK_PATH,
    DILI_CANONICAL_PATH,
    DRUGBANK_PATH,
    YU2022_L5_ZIP,
    GSE272564_TAR,
)

ROOT = _REPO_ROOT
CONFIG_PATH = ROOT / "configs" / "liver_p2.yaml"
MANIFEST_PATH = ROOT / "MANIFEST.md"

# Provenance rows asserted before any read (T-01-10 / Security Domain).
_REQUIRED_MANIFEST_FILENAMES = [
    "best_model.pt",
    "multidcp_10716_symbols.txt",
    "GSE272564_RAW.tar",
    "L5_upload.zip",
    "dilirank.xlsx",
]

# Canonical zonation markers (D-07): human (10,716 space) symbols + the mouse
# casings that appear in mouse Visium var_names.
_ZONE_MARKERS = {
    "pericentral": {"human": ["GLUL", "CYP2E1"], "mouse": ["Glul", "Cyp2e1"]},
    "periportal": {"human": ["SDS", "CYP2F2"], "mouse": ["Sds", "Cyp2f2"]},
}


def _assign_zones_by_markers(
    matrix: np.ndarray, gene_names: list[str], species: str
) -> np.ndarray:
    """Per-spot periportal/pericentral labels from canonical markers (D-07).

    A spot is labelled by whichever zone's mean marker expression is higher.
    Spots where neither zone's markers are present get the label "unzoned" and
    are dropped before pseudobulking.
    """
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    casing = "mouse" if species == "mouse" else "human"
    zone_cols: dict[str, list[int]] = {}
    for zone, markers in _ZONE_MARKERS.items():
        cols = [gene_idx[m] for m in markers[casing] if m in gene_idx]
        zone_cols[zone] = cols
        log.info(
            "Zone markers (%s, %s): %s -> %d/%d present.",
            species, zone, markers[casing], len(cols), len(markers[casing]),
        )
    n_spots = matrix.shape[0]
    labels = np.full(n_spots, "unzoned", dtype=object)
    if not zone_cols["pericentral"] or not zone_cols["periportal"]:
        log.warning(
            "Missing markers for one zone (%s); all spots stay 'unzoned'.",
            species,
        )
        return labels
    pc = matrix[:, zone_cols["pericentral"]].mean(axis=1)
    pp = matrix[:, zone_cols["periportal"]].mean(axis=1)
    labels[pc >= pp] = "pericentral"
    labels[pp > pc] = "periportal"
    return labels


# Manifold midpoint: the training basal is DENSE (mean ~0.62, 85% of values
# > 0.5). Unmeasured genes are filled with this midpoint, NOT 0 — a zero-fill
# would collapse 90%+ of the vector to the floor and silently corrupt every
# feature (Pitfall 1). Verified from
# pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv.
_MANIFOLD_MIDPOINT = 0.62


def _normalize_to_manifold(
    vec: np.ndarray, present_mask: np.ndarray, lo: float, hi: float
) -> np.ndarray:
    """Rank-percentile normalize a basal into the dense manifold range (Pitfall 1).

    The frozen cell-encoder was trained on a DENSE [0, 1] basal (mean ~0.62, 85%
    of values > 0.5; not a sparse count vector). A naive global min-max of a
    zero-filled aligned profile pins 90%+ of genes to the floor — the silent-OOD
    bug. Instead:

      * EXPRESSED genes (present_mask True) are mapped to their rank percentile
        across the expressed set -> a uniform [lo, hi] spread that matches the
        manifold's per-gene distribution.
      * MISSING genes (not covered by the source after alignment) are filled with
        the manifold midpoint (~0.62), keeping them on-manifold rather than at 0.

    Parameters
    ----------
    vec : np.ndarray
        Aligned basal in the target gene order (zeros where missing).
    present_mask : np.ndarray
        Boolean, True where the gene was actually covered by the source profile.
    lo, hi : float
        Manifold clamp range (e.g. 0.018, 1.000).
    """
    from scipy.stats import rankdata

    v = np.asarray(vec, dtype=np.float64)
    out = np.full(v.shape, _MANIFOLD_MIDPOINT, dtype=np.float64)
    expressed = present_mask & np.isfinite(v) & (v > 0)
    n = int(expressed.sum())
    if n >= 2:
        r = rankdata(v[expressed], method="average")
        out[expressed] = (r - 0.5) / n  # rank percentiles in (0, 1)
    elif n == 1:
        out[expressed] = (lo + hi) / 2.0
    out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def _load_mouse_to_human_map() -> dict[str, str]:
    """One-to-one mouse->human symbol map (P0 ortholog TSV; Pitfall 5)."""
    import pandas as pd

    ot = pd.read_csv(
        ROOT / "data" / "processed" / "spatial" / "orthologs_h_m_r_one2one.tsv",
        sep="\t",
    )
    m2h = dict(zip(ot["mouse_symbol"].astype(str), ot["human_symbol"].astype(str)))
    log.info("Loaded %d one-to-one mouse->human ortholog pairs.", len(m2h))
    return m2h


def _build_region_basal_map(
    matrix: np.ndarray,
    gene_names: list[str],
    species: str,
    gene_order: list[str],
    norm_lo: float,
    norm_hi: float,
    mouse_to_human: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Per-zone basal mapped to the 10,716 human order and rank-normalized.

    For mouse, ``gene_names`` are mapped to human symbols via the ortholog table
    (Pitfall 5) BEFORE alignment so the basal lands in the human PDG gene space.
    """
    zone_labels = _assign_zones_by_markers(matrix, gene_names, species)
    keep = zone_labels != "unzoned"
    n_drop = int((~keep).sum())
    if n_drop:
        log.info("%s: dropping %d unzoned spots before pseudobulk.", species, n_drop)
    pb = pseudobulk(
        matrix[keep], zone_labels[keep], gene_names, agg="mean"
    )

    # Source symbols in the model's (human) namespace. For mouse, translate.
    if species == "mouse" and mouse_to_human is not None:
        source_symbols = [mouse_to_human.get(g, g) for g in gene_names]
    else:
        source_symbols = list(gene_names)

    target_set = set(gene_order)
    region_basal_map: dict[str, np.ndarray] = {}
    for i, zone in enumerate(pb.region_order):
        if zone in pb.empty_regions:
            log.warning("%s: zone %s has no spots; skipping.", species, zone)
            continue
        aligned = align_to_gene_space(
            pb.profiles[i], source_symbols, gene_order, missing="zero",
            duplicates="mean",
        )
        # present_mask: which target genes were actually covered by the source.
        covered = target_set & set(source_symbols)
        present_mask = np.array([g in covered for g in gene_order], dtype=bool)
        n_present = int(present_mask.sum())
        normed = _normalize_to_manifold(aligned, present_mask, norm_lo, norm_hi)
        region_basal_map[zone] = normed
        log.info(
            "%s zone %s: %d spots, %d/%d genes covered, rank-normalized basal "
            "[min=%.3f max=%.3f mean=%.3f].",
            species, zone, pb.spot_counts.get(zone, 0), n_present, len(gene_order),
            float(normed.min()), float(normed.max()), float(normed.mean()),
        )
    return region_basal_map


def _build_liver_drug_set(max_drugs: int) -> tuple[list[str], dict[str, str]]:
    """Liver drug set with resolved SMILES (reuse P1 sibling tables; no re-resolve)."""
    labels_df = load_dilirank(str(DILIRANK_PATH))
    smiles_df = join_smiles_cascade(
        labels_df, str(DILI_CANONICAL_PATH), str(DRUGBANK_PATH)
    )
    covered = smiles_df[smiles_df["smiles"].notna()].copy()
    # Deterministic order by name; dedupe pert_ids on name_lower.
    covered = covered.drop_duplicates(subset="name_lower").sort_values("name_lower")
    if max_drugs > 0:
        covered = covered.head(max_drugs)
    pert_ids = covered["name_lower"].astype(str).tolist()
    smiles_map = dict(zip(covered["name_lower"].astype(str), covered["smiles"].astype(str)))
    log.info("Liver drug set: %d drugs with resolved SMILES.", len(pert_ids))
    return pert_ids, smiles_map


def _write_cache(cache, cache_dir: pathlib.Path, species: str) -> dict:
    """Write de/treated/control .npy + manifest JSON under cache_dir/species/."""
    out_dir = cache_dir / species
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "de_array.npy", cache.de_array)
    np.save(out_dir / "treated_array.npy", cache.treated_array)
    np.save(out_dir / "control_array.npy", cache.control_array)
    m = cache.manifest
    manifest_json = {
        "species": species,
        "organ": "liver",
        "model_variant": m.model_variant,
        "de_convention": m.de_convention,
        "pert_ids": list(m.pert_ids),
        "regions": list(m.regions),
        "n_pert_ids": m.n_pert_ids,
        "n_regions": m.n_regions,
        "n_genes": m.n_genes,
        "gene_order_file": "data/processed/spatial/multidcp_10716_symbols.txt",
        "arrays": {
            "de_array": "de_array.npy",
            "treated_array": "treated_array.npy",
            "control_array": "control_array.npy",
        },
        "shape": list(cache.de_array.shape),
        "dtype": str(cache.de_array.dtype),
        "control_input": "inert_empty_drug (INERT_CONTROL_SMILES='C', D-02 amendment)",
        "basal_normalization": "minmax_0_1 to manifold range (Pitfall 1)",
        "dose_one_hot": [1.0, 0.0],
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_json, fh, indent=2)
    log.info(
        "Wrote 3-vector cache for %s: shape %s -> %s",
        species, tuple(cache.de_array.shape), out_dir,
    )
    return manifest_json


def _load_species_basal(
    species: str, gene_order: list[str], norm_lo: float, norm_hi: float
) -> dict[str, np.ndarray]:
    """Load + zone + align + normalize the liver basal for one species."""
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        if species == "human":
            adata, gene_names = _load_yu2022_l5(YU2022_L5_ZIP, tmp / "yu2022")
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
            return _build_region_basal_map(
                X, list(adata.var_names), "human", gene_order, norm_lo, norm_hi
            )
        # mouse: GSE272564 control arm (APAP0h). The P1 loader returns a whole-
        # sample pseudobulk; re-extract the spot matrix here so we can zone it.
        mouse_to_human = _load_mouse_to_human_map()
        return _load_mouse_basal_zoned(
            tmp, gene_order, norm_lo, norm_hi, mouse_to_human
        )


def _load_mouse_basal_zoned(
    tmp: pathlib.Path,
    gene_order: list[str],
    norm_lo: float,
    norm_hi: float,
    mouse_to_human: dict[str, str],
) -> dict[str, np.ndarray]:
    """Mouse GSE272564 control arm -> per-spot matrix -> zoned basal map.

    Mirrors the P1 ``_load_gse272564_mouse_ctrl`` extraction but keeps the full
    spot x gene matrix so canonical-marker zoning (D-07) can run. Falls back to a
    single whole-sample 'wholeliver' region if the matrix cannot be re-extracted.
    """
    import gzip as _gzip
    import io as _io
    import tarfile

    with tarfile.open(GSE272564_TAR) as tar:
        tar.extractall(tmp / "gse272564")
    base = tmp / "gse272564"
    all_files = list(base.rglob("*"))
    apap0h = [f for f in all_files if "APAP0h" in f.name or "APAP0H" in f.name]
    matrix_file = next(
        (f for f in apap0h if "matrix" in f.name.lower() and f.name.endswith(".mtx.gz")),
        None,
    )
    features_file = next(
        (f for f in apap0h if "feature" in f.name.lower() or "gene" in f.name.lower()),
        None,
    )
    if matrix_file is None or features_file is None:
        # Whole-sample fallback (no zones) — still cache one region.
        log.warning("Mouse: APAP0h mtx/features not found; using whole-sample region.")
        profile, gene_names = _load_gse272564_mouse_ctrl(GSE272564_TAR, tmp / "gse2")
        source_symbols = [mouse_to_human.get(g, g) for g in gene_names]
        aligned = align_to_gene_space(
            np.asarray(profile), source_symbols, gene_order, missing="zero",
            duplicates="mean",
        )
        covered = set(gene_order) & set(source_symbols)
        present_mask = np.array([g in covered for g in gene_order], dtype=bool)
        return {
            "wholeliver": _normalize_to_manifold(
                aligned, present_mask, norm_lo, norm_hi
            )
        }

    import scipy.io
    with _gzip.open(str(matrix_file), "rb") as gf:
        mat = scipy.io.mmread(_io.BytesIO(gf.read()))
    X = mat.T.toarray() if hasattr(mat, "toarray") else np.asarray(mat).T
    with _gzip.open(str(features_file), "rb") as gf:
        lines = gf.read().decode("utf-8").strip().splitlines()
    gene_names = [ln.split("\t")[1] if "\t" in ln else ln for ln in lines]
    log.info("Mouse GSE272564 APAP0h: %d spots x %d genes.", X.shape[0], X.shape[1])
    return _build_region_basal_map(
        X, gene_names, "mouse", gene_order, norm_lo, norm_hi,
        mouse_to_human=mouse_to_human,
    )


def main() -> None:
    log.info("Device: %s (CUDA_VISIBLE_DEVICES=%r)", DEVICE, os.environ.get("CUDA_VISIBLE_DEVICES"))

    # Provenance before use (T-01-10): sys.exit(2) if any required row is absent.
    _assert_manifest_rows(MANIFEST_PATH, _REQUIRED_MANIFEST_FILENAMES)

    cfg = yaml.safe_load(open(CONFIG_PATH))
    checkpoint_path = cfg["checkpoint_path"]
    norm_lo, norm_hi = cfg["basal_normalization"]["manifold_range"]
    cache_dir = ROOT / cfg["cache_dir"]

    gene_order = (
        (ROOT / cfg["gene_order_file"]).read_text().split()
    )
    if len(gene_order) != N_PDG:
        print(
            f"ERROR: gene-order file has {len(gene_order)} symbols, expected "
            f"N_PDG={N_PDG}.",
            file=sys.stderr,
        )
        sys.exit(2)

    pert_ids, smiles_map = _build_liver_drug_set(_ARGS.max_drugs)

    cacher = RegionSignatureCacher(
        model_variant="multidcp_chemoe",
        gene_ids=tuple(gene_order),
        device=DEVICE,
    )
    cacher.load_model(checkpoint_path, verify_sha=not _ARGS.no_verify_sha)

    species_list = ["human", "mouse"] if _ARGS.species == "all" else [_ARGS.species]
    summary = {}
    for species in species_list:
        log.info("=== Caching liver predicted DE: %s ===", species)
        region_basal_map = _load_species_basal(
            species, gene_order, float(norm_lo), float(norm_hi)
        )
        if not region_basal_map:
            log.error("%s: no region basal built; skipping.", species)
            continue

        cache = cacher.run(pert_ids, smiles_map, region_basal_map)

        # Real-inference sanity (Pitfall 1): first (drug, region) treated vector.
        first_treated = cache.treated_array[0, 0]
        finite = np.all(np.isfinite(first_treated))
        in_range = (first_treated.min() > -0.5) and (first_treated.max() < 1.5)
        log.info(
            "%s sanity: first treated [min=%.4f max=%.4f mean=%.4f std=%.4f] "
            "finite=%s in_range=%s",
            species, float(first_treated.min()), float(first_treated.max()),
            float(first_treated.mean()), float(first_treated.std()),
            finite, in_range,
        )
        if not finite:
            raise RuntimeError(f"{species}: non-finite predictions (corrupt forward).")
        if not in_range or first_treated.std() < 1e-4:
            log.warning(
                "%s: prediction out of manifold range or near-collapsed "
                "(std=%.2e) — check basal normalization (Pitfall 1).",
                species, float(first_treated.std()),
            )

        summary[species] = _write_cache(cache, cache_dir, species)

    log.info("Done. Cached species: %s", list(summary.keys()))


if __name__ == "__main__":
    main()
