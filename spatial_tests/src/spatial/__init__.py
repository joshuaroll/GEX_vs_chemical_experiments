"""Spatial transcriptomics basal state library.

Per-region basal-conditioned MultiDCP signatures, attention-pooled for toxicity
prediction. Provides spatial context-aware baseline expression states across
liver (periportal, pericentral, zonation subzones), kidney (glomerular,
proximal_tubule, distal_tubule, vascular, interstitial), brain (Layer1-6, WM),
and heart regions.

This library is a pure data library exposing region definitions and dataset
registries. No computational code or hardcoded file paths are included.
"""

from __future__ import annotations
