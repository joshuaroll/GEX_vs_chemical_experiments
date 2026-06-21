"""Configuration for spatial transcriptomics basal state library.

Defines organ-specific regions, gene space constants, and aggregation methods.
Pure configuration module — no file I/O, no computation.

Hard rules enforced:
  - Gene space constants match upstream MultiDCP training (N_LANDMARK=978, N_PDG=10716).
  - Region definitions are anatomically curated from literature and dataset annotations.
  - All dicts and lists are frozen (immutable for safety).
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Gene space constants (frozen, matches MultiDCP training regime)
# ---------------------------------------------------------------------------

N_LANDMARK: Final[int] = 978
"""
Number of LINCS L1000 landmark genes. This is the dimension of predicted
signatures returned by frozen MultiDCP models. All upstream training used
exactly this gene set.
"""

N_PDG: Final[int] = 10716
"""
Number of protein-coding genes in the human genome (ENSEMBL reference).
Used for imputation from landmark space to full genome where applicable.
"""

# ---------------------------------------------------------------------------
# Region definitions by organ
# ---------------------------------------------------------------------------

REGION_DEFS: Final[dict[str, dict[str, list[str]]]] = {
    "liver": {
        "major_zones": [
            "periportal",      # Portal triad region (PP)
            "pericentral",     # Central vein region (CV)
        ],
        "zonation_subzones": [
            "zone_1",          # Strict periportal (hepatocyte hypoxia < 1%)
            "zone_1_5",        # Periportal-intermediate
            "zone_2",          # Mid-zonal
            "zone_2_5",        # Zonal-intermediate
            "zone_3",          # Pericentral (hepatocyte hypoxia ≈ 5%)
        ],
    },
    "kidney": {
        "major_compartments": [
            "glomerular",      # Renal corpuscle (filtration unit)
            "proximal_tubule", # Epithelial reabsorption
            "distal_tubule",   # Ion regulation
            "vascular",        # Endothelial / peritubular
            "interstitial",    # Fibroblasts / stroma
        ],
    },
    "brain": {
        "cortical_layers": [
            "Layer1",          # Molecular/plexiform layer
            "Layer2",          # External granular layer
            "Layer3",          # External pyramidal layer
            "Layer4",          # Internal granular layer (primary sensory input)
            "Layer5",          # Internal pyramidal layer
            "Layer6",          # Multiform / polymorphic layer
        ],
        "white_matter": [
            "WM",              # White matter (unmyelinated axons + glia)
        ],
    },
    "heart": {
        "anatomical_regions": [
            "left_ventricle",     # Main pumping chamber
            "right_ventricle",    # Pulmonary circulation
            "left_atrium",        # Pulmonary input
            "right_atrium",       # Systemic input
            "interventricular_septum",  # Ventricular divider
            "conduction_system",  # Purkinje fibers / AV node
            "epicardium",         # Outer layer
            "endocardium",        # Inner layer
        ],
    },
}
"""
Anatomically curated region definitions for each organ.

Liver regions follow established hepatic zonation literature (periportal PP
hypoxia < 1%, pericentral CV hypoxia ≈ 5%; subzone definitions per histology
and metabolic gradients). Kidney compartments follow KPMP atlas structure.
Brain layers are standard cortical cytoarchitecture (Brodmann). Heart regions
follow anatomical compartments from Kanemaru et al. 2023.

Each region is a string token used as a key in spatial dataset annotations
and basal state caches.
"""

# ---------------------------------------------------------------------------
# Aggregation methods (for pooling spot/cell-level data to region averages)
# ---------------------------------------------------------------------------

AGGREGATIONS: Final[tuple[str, ...]] = (
    "mean",      # Arithmetic mean (standard basal state)
    "median",    # Median (robust to outliers)
    "sum",       # Sum (for abundance-weighted features if needed)
)
"""
Supported aggregation methods for computing region-level basal states from
spot-level or cell-level spatial data.

Usage:
  - "mean" is the default and most commonly used for basal state representation.
  - "median" is an optional robustness check (less sensitive to extreme values).
  - "sum" is rarely used but included for completeness (may be useful for
    abundance or cell-count features).
"""

# ---------------------------------------------------------------------------
# Summary metadata (for documentation)
# ---------------------------------------------------------------------------

REGION_COUNTS: Final[dict[str, int]] = {
    organ: sum(len(regions) for regions in defs.values())
    for organ, defs in REGION_DEFS.items()
}
"""Precomputed region counts per organ for reference."""
