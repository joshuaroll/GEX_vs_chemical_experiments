"""Unit tests for ``src/spatial/apap_validation.py`` (WIRE-03 / Halt Gate 3).

In-memory fixtures only — no real data, no absolute system paths, no real
Visium, no real model. Analogs: ``tests/spatial/test_eda_region.py`` (AnnData + ortholog-mapped
Pearson) and ``tests/spatial/test_pseudobulk.py`` (AnnData -> per-label
pseudobulk).

RED until 02-04: ``src.spatial.apap_validation`` does not exist yet. The import
below raises ``ImportError`` so every test errors (RED). 02-04 writes the module
(assign_zones, measured_zone_de, zone_pearson, halt_gate_3_fires) and flips
these GREEN by editing source only.

Behaviors covered (WIRE-03):
  * zone assignment from canonical markers (D-07): pericentral Glul/Cyp2e1,
    periportal Sds/Cyp2f2.
  * measured DE = APAP_zone - ctrl_zone, with mouse->human ortholog alignment
    reporting n_genes_compared (intersection size); absent genes flagged via a
    present_mask, NOT zero-filled (D-08).
  * per-zone Pearson (dict pearson_r/p_value/n_genes_compared) with a <2 guard.
  * Halt Gate 3 keyed to pericentral: pearson_r < 0.3 -> fires (D-08/D-09).

Purity gate
-----------
- No absolute system paths anywhere in this file (purity gate).
- Small synthetic arrays only; the AnnData and ortholog fixtures come from
  conftest.py (zone_marker_adata, ortholog_table_small).
"""

from __future__ import annotations

import numpy as np
import pytest

# RED until 02-04: module does not exist yet -> ImportError at collection.
from src.spatial.apap_validation import (
    assign_zones,
    halt_gate_3_fires,
    measured_zone_de,
    zone_pearson,
)


# ---------------------------------------------------------------------------
# -k zone : zone assignment from canonical markers (D-07)
# ---------------------------------------------------------------------------


def test_assign_zones_from_markers(zone_marker_adata):
    """Spots high in Glul/Cyp2e1 -> pericentral; high in Sds/Cyp2f2 -> periportal."""
    labels = assign_zones(zone_marker_adata)

    # One label per spot, in spot order.
    labels = list(labels)
    assert len(labels) == zone_marker_adata.n_obs

    truth = list(zone_marker_adata.obs["zone_truth"])
    assert labels == truth, (
        f"Marker-based zone assignment must match planted zones {truth}, "
        f"got {labels}. (pericentral: Glul/Cyp2e1; periportal: Sds/Cyp2f2)"
    )


# ---------------------------------------------------------------------------
# -k measured_de : measured DE = APAP_zone - ctrl_zone + ortholog align (D-08)
# ---------------------------------------------------------------------------


def test_measured_de_apap_minus_control(ortholog_table_small):
    """measured_zone_de returns APAP_zone - ctrl_zone per zone, ortholog-aligned.

    Mouse measured vectors live in mouse symbol space; the target human space is
    a strict superset, so n_genes_compared (intersection) is < the human space
    and absent human genes are flagged via present_mask (NOT zero-filled, D-08).
    """
    mouse_genes = ["Glul", "Cyp2e1", "Sds", "Cyp2f2"]
    # Per-zone pseudobulk vectors (APAP arm and control arm), pericentral zone.
    apap_pericentral = np.array([5.0, 4.0, 1.0, 0.5], dtype=np.float64)
    ctrl_pericentral = np.array([2.0, 1.0, 1.0, 0.4], dtype=np.float64)

    # Human target space is a strict superset (HMOX1 has no mouse ortholog here).
    human_genes = ["GLUL", "CYP2E1", "SDS", "CYP2F2", "HMOX1"]

    result = measured_zone_de(
        apap_zone=apap_pericentral,
        ctrl_zone=ctrl_pericentral,
        mouse_genes=mouse_genes,
        human_genes=human_genes,
        ortholog_table=ortholog_table_small,
    )

    assert "de" in result, "measured_zone_de must return a 'de' vector"
    assert "present_mask" in result, (
        "measured_zone_de must return a 'present_mask' (D-08 flag-not-zero)"
    )
    assert "n_genes_compared" in result, (
        "measured_zone_de must report 'n_genes_compared' (intersection size)"
    )

    de = np.asarray(result["de"])
    present_mask = np.asarray(result["present_mask"], dtype=bool)

    # DE in human space length.
    assert de.shape == (len(human_genes),)
    assert present_mask.shape == (len(human_genes),)

    # 4 of 5 human genes have a mouse ortholog present -> intersection size 4.
    assert result["n_genes_compared"] == 4
    assert present_mask.sum() == 4
    # HMOX1 (no ortholog) flagged absent, NOT zero-filled.
    hmox1 = human_genes.index("HMOX1")
    assert not present_mask[hmox1], "HMOX1 must be flagged absent (D-08)"

    # Present genes: APAP - ctrl, element-wise (rule on measured DE).
    glul = human_genes.index("GLUL")
    np.testing.assert_almost_equal(de[glul], 5.0 - 2.0)


# ---------------------------------------------------------------------------
# -k pearson_gate : per-zone Pearson + Halt Gate 3 (pericentral < 0.3, D-08/D-09)
# ---------------------------------------------------------------------------


def test_zone_pearson_and_halt_gate3():
    """zone_pearson returns the dict + <2 guard; halt_gate_3 keyed to pericentral."""
    # --- zone_pearson dict + present_mask intersection ---
    pred_de = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    measured_de = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=np.float64)
    present_mask = np.array([True, True, True, True, False])

    out = zone_pearson(pred_de, measured_de, present_mask)
    assert set(out.keys()) >= {"pearson_r", "p_value", "n_genes_compared"}
    assert out["n_genes_compared"] == int(present_mask.sum()) == 4
    assert -1.0 <= out["pearson_r"] <= 1.0

    # --- <2 guard: fewer than 2 present genes raises ValueError ---
    too_few = np.array([True, False, False, False, False])
    with pytest.raises(ValueError):
        zone_pearson(pred_de, measured_de, too_few)

    # --- Halt Gate 3 keyed to pericentral (D-08/D-09) ---
    # Pericentral r < 0.3 -> gate fires (stop-and-reframe), regardless of periportal.
    assert halt_gate_3_fires({"pericentral": 0.21, "periportal": 0.6}) is True
    # Pericentral r >= 0.3 -> gate does not fire.
    assert halt_gate_3_fires({"pericentral": 0.45, "periportal": 0.6}) is False
