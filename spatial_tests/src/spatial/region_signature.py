"""Per-region predicted DE caching via a frozen MultiDCP/CheMoE model.

The PURE parts (DE math, cache assembly, manifest, cache keys) carry NO torch
or /raid dependency and are directly testable with in-memory fixtures. The
IMPURE parts (``load_model``, ``_featurize_drug``, ``_call_model``) load the
frozen MultiDCP-CheMoE backbone and run real inference; torch + the cross-
project ``sys.path`` injection live ONLY inside those methods so the pure
functions stay import-clean.

Fully implemented, fixture-testable (pure):

  1. DE computation (rule B, D-02): ``predicted_DE_region =
     predicted_treated(drug, region_basal) - predicted_control(region_basal)``
     (element-wise float32 subtraction, shape [n_genes]).
  2. Cache-key construction: a deterministic, gene-order-stable, region-order-
     stable, pert_id-order-stable string key for each (pert_id, region) pair.
  3. Manifest construction: a frozen NamedTuple capturing the region order, gene
     order, and pert_id order used for a caching run, so results arrays can
     always be reconstructed from disk unambiguously.
  4. Result containers: ``RegionDE`` (single (pert_id, region) pair) and
     ``RegionSignatureCache`` (the 3-vector cache, D-03: de_array,
     treated_array, control_array, aligned over all pert_ids × all regions).

DE convention used in this module (rule B, D-02)
------------------------------------------------
``predicted_DE_region = predicted_treated(drug) - predicted_control(region_basal)``

This is rule B (bias-corrected): the second operand is the model's OWN predicted
control output for the same region context, NOT the raw observed basal (that was
the superseded rule A). Rationale: both terms stay in the model's output
manifold, cancelling basal-reconstruction error, and predicted DE becomes the
same kind of quantity (``treated - control``) as the measured DE the APAP gate
tests against. The control is an inert/empty-drug control pass (D-02 amendment,
2026-06-24): the training vocab has no vehicle/DMSO, so ``predicted_control`` is
a forward pass with a designated inert reference SMILES in the same region basal
context (``INERT_CONTROL_SMILES``). That reference was never in training, so its
output is an extrapolation; the control vector is cached (D-03) so the choice is
auditable and re-derivable. This still diverges from the main v0.4 DE rule
(``treated - diseased``) because the spatial arm uses a HEALTHY-TISSUE basal.

Model-inference path
--------------------
``load_model`` strict-loads ``MultiDCP_CheMoE_AE`` from the D-04 row-17
checkpoint ``/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt``
(0 missing / 0 unexpected keys); ``_call_model`` runs one real forward and
returns absolute predicted treated expression over 10,716 genes. The row-18
KPGT checkpoint is collapsed/incompatible and is NOT wired (D-04 amendment /
S-B descope).

Hard rules honored
-------------------
- Real data only: real strict-load + real inference; no mocked model outputs.
- DE rule (rule B): differential expression only; raw expression as the DE
  operand is forbidden.
- N_PDG = 10716 (D-01) is the model I/O + cache space; N_LANDMARK = 978 is kept
  for the 978-landmark callers (no regression).
- Pure functions stay torch-free; torch + sys.path injection are confined to the
  impure model methods.
- Gene order, region order, and pert_id order are always explicit and
  deterministic (caller-supplied, not auto-detected).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Final, NamedTuple, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gene-space constant (canonical; matches spatial_basal.config.N_LANDMARK)
# ---------------------------------------------------------------------------

N_LANDMARK: Final[int] = 978
"""Number of LINCS L1000 landmark genes. Kept for 978-landmark callers; the
spatial-arm cache uses N_PDG (10716), not this space (D-01)."""

N_PDG: Final[int] = 10716
"""Number of genes in the MultiDCP-CheMoE PDG space (D-01). This is the frozen
backbone's native I/O dimension and the spatial cache column space. The
978->10,716 imputation is the OOD factor the project instruments, not avoids."""

# Inert/empty-drug control reference for rule-B DE (D-02 amendment, 2026-06-24).
# The PDGrapher training vocab has no vehicle/DMSO, so predicted_control is a
# forward pass with this fixed inert reference SMILES in the same region basal
# context. Methane ("C") is the minimal valid molecular graph the NeuralFinger-
# print accepts; its output is an extrapolation (never in training) and is
# cached (D-03) so the control choice stays auditable and re-derivable.
INERT_CONTROL_SMILES: Final[str] = "C"

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class RegionDE(NamedTuple):
    """Predicted differential expression for one (pert_id, region) pair.

    DE convention (rule B, D-02):
    ``de_vector = predicted_treated(drug) - predicted_control(region_basal)``.
    Both input and output are in the model's gene space (N_PDG = 10716 for the
    spatial arm; the 978-landmark space is also supported for legacy callers).

    Attributes
    ----------
    pert_id : str
        Perturbation identifier (drug/compound key, e.g. a SMILES or CID
        string). Must be the same string used to build cache keys via
        ``make_cache_key``.
    region : str
        Region label string (e.g. ``"periportal"``, ``"Layer4"``). Must match
        a key from the caller-supplied ``region_basal_map``.
    de_vector : np.ndarray
        Shape ``(n_genes,)`` float32. Element-wise
        ``predicted_treated - region_basal``. Positive values = gene
        up-regulated compared to the region basal state; negative = down.
    n_genes : int
        Length of ``de_vector``. Must equal ``N_LANDMARK`` (978) for the
        frozen-MultiDCP landmark space. Callers may use a different gene
        count (e.g. 10716 for PDG imputed space) but must document the
        deviation.
    """

    pert_id: str
    region: str
    de_vector: np.ndarray
    n_genes: int


class RegionSignatureManifest(NamedTuple):
    """Immutable metadata describing the alignment axes of a caching run.

    A ``RegionSignatureCache`` is meaningless without the manifest because
    the result arrays are indexed by (pert_id_idx, region_idx, gene_idx).
    Store the manifest alongside the .npy array on disk so reconstruction
    is unambiguous.

    Attributes
    ----------
    pert_ids : tuple[str, ...]
        Ordered sequence of perturbation IDs corresponding to axis 0 of
        ``RegionSignatureCache.de_array``. Frozen at cache-build time.
    regions : tuple[str, ...]
        Ordered sequence of region labels corresponding to axis 1 of
        ``RegionSignatureCache.de_array``. Frozen at cache-build time.
    gene_ids : tuple[str, ...]
        Ordered sequence of gene identifiers (ENTREZ IDs or HGNC symbols)
        corresponding to axis 2 of ``RegionSignatureCache.de_array``.
        Frozen at cache-build time.
    model_variant : str
        Which frozen model produced these predictions: one of
        ``"multidcp_pdg"``, ``"multidcp_chemoe"``. Used as part of the
        cache-key namespace so PDG and CheMoE caches don't collide.
    de_convention : str
        Frozen description of the DE formula used (rule B, D-02). Must be
        ``"predicted_treated(drug) - predicted_control(region_basal)"`` for all
        spatial-arm outputs. (Distinct from the superseded rule A
        ``"predicted_treated - region_basal"`` and from the main v0.4 convention
        ``"treated - diseased"``.)
    n_pert_ids : int
        ``len(pert_ids)``; stored redundantly for fast validation.
    n_regions : int
        ``len(regions)``; stored redundantly for fast validation.
    n_genes : int
        ``len(gene_ids)``; stored redundantly for fast validation.
    """

    pert_ids: tuple[str, ...]
    regions: tuple[str, ...]
    gene_ids: tuple[str, ...]
    model_variant: str
    de_convention: str
    n_pert_ids: int
    n_regions: int
    n_genes: int


class RegionSignatureCache(NamedTuple):
    """Aligned 3-vector predicted-signature arrays for all pert_ids × regions.

    D-03: persist the final DE plus the intermediate predicted_treated and
    predicted_control vectors (not DE-only), so reconstruction error is
    auditable, rule A<->B is recomputable without re-running the frozen model,
    and the APAP comparison works in either DE or absolute form.

    Attributes
    ----------
    de_array : np.ndarray
        Shape ``(n_pert_ids, n_regions, n_genes)`` float32.
        ``de_array[i, j, :]`` is the rule-B predicted DE vector
        (``treated_array[i, j] - control_array[i, j]``) for
        ``manifest.pert_ids[i]`` in region ``manifest.regions[j]``.
    treated_array : np.ndarray
        Shape ``(n_pert_ids, n_regions, n_genes)`` float32. The model's absolute
        predicted treated expression for each (drug, region).
    control_array : np.ndarray
        Shape ``(n_pert_ids, n_regions, n_genes)`` float32. The model's absolute
        predicted control expression (inert-drug pass, D-02) for each
        (drug, region) — same region basal context, control SMILES.
    manifest : RegionSignatureManifest
        Alignment metadata; axes only interpretable via this manifest.
    """

    de_array: np.ndarray
    treated_array: np.ndarray
    control_array: np.ndarray
    manifest: RegionSignatureManifest


# ---------------------------------------------------------------------------
# Cache key and manifest construction
# ---------------------------------------------------------------------------


def make_cache_key(
    pert_id: str,
    region: str,
    model_variant: str,
    gene_ids: tuple[str, ...],
) -> str:
    """Build a deterministic cache key for one (pert_id, region) pair.

    The key is a hex digest of a JSON-serialized dict containing all four
    inputs. Gene order is embedded so keys from different gene-space
    orderings never collide.

    Parameters
    ----------
    pert_id : str
        Perturbation identifier (drug key). Order-sensitive with respect to
        the outer caching loop but the key itself is pert_id-stable.
    region : str
        Region label string (e.g. ``"periportal"``).
    model_variant : str
        One of ``"multidcp_pdg"`` or ``"multidcp_chemoe"``.
    gene_ids : tuple[str, ...]
        Ordered gene identifiers. Included in the hash so a key from a
        978-gene run never matches a 10716-gene run.

    Returns
    -------
    str
        64-character lowercase hex SHA-256 digest of the canonical JSON.
        The canonical JSON is ``json.dumps(payload, sort_keys=True)`` where
        ``payload`` is ``{"pert_id": ..., "region": ...,
        "model_variant": ..., "gene_ids": [...]}``
        — deterministic across Python versions for basic string types.
    """
    payload = {
        "pert_id": pert_id,
        "region": region,
        "model_variant": model_variant,
        "gene_ids": list(gene_ids),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_manifest(
    pert_ids: list[str],
    regions: list[str],
    gene_ids: list[str],
    model_variant: str,
) -> RegionSignatureManifest:
    """Construct a ``RegionSignatureManifest`` from caller-supplied axes.

    Parameters
    ----------
    pert_ids : list[str]
        Ordered perturbation IDs. Must be non-empty and contain unique values
        (duplicates raise ``ValueError`` — ambiguous array axis).
    regions : list[str]
        Ordered region labels. Must be non-empty and unique.
    gene_ids : list[str]
        Ordered gene identifiers. Must be non-empty and unique.
    model_variant : str
        One of ``"multidcp_pdg"`` or ``"multidcp_chemoe"``.

    Returns
    -------
    RegionSignatureManifest
        Frozen NamedTuple with all alignment axes.

    Raises
    ------
    ValueError
        If any of the three ordered axes is empty or contains duplicates.
    ValueError
        If ``model_variant`` is not one of the two recognized strings.
    """
    _VALID_VARIANTS: frozenset[str] = frozenset(
        {"multidcp_pdg", "multidcp_chemoe"}
    )

    # --- Validate axes ---
    for axis_name, axis_vals in (
        ("pert_ids", pert_ids),
        ("regions", regions),
        ("gene_ids", gene_ids),
    ):
        if not axis_vals:
            raise ValueError(
                f"build_manifest: '{axis_name}' must be non-empty."
            )
        if len(axis_vals) != len(set(axis_vals)):
            dupes = [v for v in axis_vals if axis_vals.count(v) > 1]
            raise ValueError(
                f"build_manifest: '{axis_name}' has duplicate values: "
                f"{sorted(set(dupes))}. Each axis must be unique."
            )

    if model_variant not in _VALID_VARIANTS:
        raise ValueError(
            f"build_manifest: model_variant must be one of "
            f"{sorted(_VALID_VARIANTS)}, got {model_variant!r}."
        )

    return RegionSignatureManifest(
        pert_ids=tuple(pert_ids),
        regions=tuple(regions),
        gene_ids=tuple(gene_ids),
        model_variant=model_variant,
        de_convention="predicted_treated(drug) - predicted_control(region_basal)",
        n_pert_ids=len(pert_ids),
        n_regions=len(regions),
        n_genes=len(gene_ids),
    )


# ---------------------------------------------------------------------------
# DE computation (pure, testable without model)
# ---------------------------------------------------------------------------


def compute_de(
    predicted_treated: np.ndarray,
    predicted_control: np.ndarray,
) -> np.ndarray:
    """Compute per-region predicted DE (rule B, D-02): ``treated - control``.

    Both arrays must be 1-D float arrays of the same length (n_genes). The
    result is returned as float32.

    This is the canonical DE computation for the spatial arm. It subtracts the
    model's OWN predicted control output (rule B, bias-corrected, D-02), NOT the
    raw observed basal (the superseded rule A). It is invoked by
    ``assemble_cache`` and directly by tests that inject synthetic
    ``predicted_treated`` / ``predicted_control`` arrays.

    DE convention (rule B):
    ``predicted_DE_region = predicted_treated(drug) - predicted_control(region_basal)``.
    The control is an inert/empty-drug control pass in the same region context
    (``INERT_CONTROL_SMILES``); both operands live in the model's output
    manifold. This still differs from the main v0.4 DE rule
    (``treated - diseased``) because the spatial arm uses a HEALTHY-TISSUE basal.

    Parameters
    ----------
    predicted_treated : np.ndarray
        Shape ``(n_genes,)`` float. The model's absolute predicted treated
        expression for the drug in the region's cell context.
    predicted_control : np.ndarray
        Shape ``(n_genes,)`` float. The model's absolute predicted control
        expression — a forward pass with the inert reference drug in the SAME
        region basal context (rule B, D-02). NOT the raw observed basal.

    Returns
    -------
    np.ndarray
        Shape ``(n_genes,)`` float32. Element-wise
        ``predicted_treated - predicted_control``.

    Raises
    ------
    ValueError
        If either array is not 1-D, or if they have different lengths.
    """
    pt = np.asarray(predicted_treated, dtype=np.float32)
    pc = np.asarray(predicted_control, dtype=np.float32)

    if pt.ndim != 1:
        raise ValueError(
            f"compute_de: predicted_treated must be 1-D, got shape {pt.shape}."
        )
    if pc.ndim != 1:
        raise ValueError(
            f"compute_de: predicted_control must be 1-D, got shape {pc.shape}."
        )
    if pt.shape != pc.shape:
        raise ValueError(
            f"compute_de: shape mismatch — predicted_treated {pt.shape} vs "
            f"predicted_control {pc.shape}. Both must have the same n_genes."
        )

    return pt - pc


# ---------------------------------------------------------------------------
# Assemble full cache from injected predictions (pure, testable)
# ---------------------------------------------------------------------------


def assemble_cache(
    predicted_treated_map: dict[tuple[str, str], np.ndarray],
    predicted_control_map: dict[tuple[str, str], np.ndarray],
    manifest: RegionSignatureManifest,
) -> RegionSignatureCache:
    """Build the 3-vector ``RegionSignatureCache`` from predicted arrays (rule B).

    PURE: only rule-B DE subtraction (``compute_de``) and array stacking; it does
    NOT call any model. Entry point for:

      a. Tests, which inject synthetic treated/control maps.
      b. ``RegionSignatureCacher.run()``, which populates both maps by running the
         real frozen forward (treated drug + inert-control pass per D-02).

    Parameters
    ----------
    predicted_treated_map : dict[tuple[str, str], np.ndarray]
        Keys are ``(pert_id, region)`` pairs. Values are 1-D float arrays of
        shape ``(n_genes,)`` — the model's absolute predicted treated GEX for
        that drug in that region's cell context. All manifest
        ``(pert_id, region)`` combinations must be present.
    predicted_control_map : dict[tuple[str, str], np.ndarray]
        Keys are ``(pert_id, region)`` pairs. Values are 1-D float arrays of
        shape ``(n_genes,)`` — the model's absolute predicted CONTROL GEX (the
        inert-drug pass, D-02) in the SAME region basal context. The control is
        region-determined; it is keyed per (pert_id, region) for a uniform
        schema and so the cache aligns position-for-position with the treated
        and DE arrays. All manifest ``(pert_id, region)`` combinations must be
        present.
    manifest : RegionSignatureManifest
        Alignment metadata (produced by ``build_manifest``). Defines the
        iteration order for axes 0 (pert_ids) and 1 (regions).

    Returns
    -------
    RegionSignatureCache
        ``de_array``, ``treated_array``, ``control_array`` each shape
        ``(n_pert_ids, n_regions, n_genes)`` float32, with
        ``de_array == treated_array - control_array`` (rule B). ``.manifest`` is
        the supplied manifest (passed through unchanged).

    Raises
    ------
    KeyError
        If any ``(pert_id, region)`` pair in the manifest is missing from either
        ``predicted_treated_map`` or ``predicted_control_map``.
    ValueError
        If any predicted array does not match the gene count in
        ``manifest.n_genes``, or if ``compute_de`` raises a shape error.
    """
    n_p = manifest.n_pert_ids
    n_r = manifest.n_regions
    n_g = manifest.n_genes
    de_array = np.empty((n_p, n_r, n_g), dtype=np.float32)
    treated_array = np.empty((n_p, n_r, n_g), dtype=np.float32)
    control_array = np.empty((n_p, n_r, n_g), dtype=np.float32)

    for i, pert_id in enumerate(manifest.pert_ids):
        for j, region in enumerate(manifest.regions):
            key = (pert_id, region)
            if key not in predicted_treated_map:
                raise KeyError(
                    f"assemble_cache: predicted_treated_map is missing entry "
                    f"for (pert_id={pert_id!r}, region={region!r}). Ensure all "
                    f"manifest pert_id × region combinations are populated."
                )
            if key not in predicted_control_map:
                raise KeyError(
                    f"assemble_cache: predicted_control_map is missing entry "
                    f"for (pert_id={pert_id!r}, region={region!r}). Ensure all "
                    f"manifest pert_id × region combinations are populated "
                    f"(rule-B control pass, D-02)."
                )
            pred = np.asarray(predicted_treated_map[key], dtype=np.float32)
            pred_control = np.asarray(predicted_control_map[key], dtype=np.float32)
            de_vec = compute_de(pred, pred_control)
            if de_vec.shape[0] != n_g:
                raise ValueError(
                    f"assemble_cache: DE vector for ({pert_id!r}, {region!r}) "
                    f"has {de_vec.shape[0]} genes, but manifest.n_genes={n_g}."
                )
            treated_array[i, j, :] = pred
            control_array[i, j, :] = pred_control
            de_array[i, j, :] = de_vec

    log.info(
        "assemble_cache: built 3-vector cache %s (pert_ids=%d, regions=%d, "
        "genes=%d; rule B = treated - control, D-02/D-03)",
        de_array.shape,
        n_p,
        n_r,
        n_g,
    )
    return RegionSignatureCache(
        de_array=de_array,
        treated_array=treated_array,
        control_array=control_array,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# RegionSignatureCacher — the main interface
# ---------------------------------------------------------------------------


class RegionSignatureCacher:
    """Interface that produces per-region predicted DE by calling a frozen model.

    Usage pattern
    -------------
    ::

        cacher = RegionSignatureCacher(
            model_variant="multidcp_chemoe",
            gene_ids=symbols_10716,   # tuple of 10,716 HGNC symbols (N_PDG, D-01)
            device="cuda:0",          # caller sets CUDA_VISIBLE_DEVICES first
        )
        cacher.load_model(checkpoint_path)   # strict-loads row-17 best_model.pt
        cache = cacher.run(
            pert_ids=["drug_A", "drug_B"],
            smiles_map={"drug_A": "CCO", "drug_B": "c1ccccc1"},
            region_basal_map={"periportal": np.array([...]),   # 0-1 normalized
                              "pericentral": np.array([...])},
        )

    The PURE parts (``compute_de``, ``build_manifest``, ``make_cache_key``,
    ``assemble_cache``) are module-level functions, tested independently of the
    model. The IMPURE parts (``load_model``, ``_featurize_drug``,
    ``_call_model``) keep torch + the cross-project ``sys.path`` injection inside
    the methods so the pure functions stay import-clean.

    Parameters
    ----------
    model_variant : str
        One of ``"multidcp_pdg"`` or ``"multidcp_chemoe"``. Determines both the
        checkpoint loaded and the cache-key namespace. The wired backbone this
        phase is ``"multidcp_chemoe"`` (D-04 amendment; ``"multidcp_pdg"`` /
        S-B is descoped — its row-18 checkpoint is collapsed/incompatible).
    gene_ids : tuple[str, ...]
        Ordered gene identifiers for the model's output space. For the spatial
        arm this is the 10,716-symbol PDG order (N_PDG, D-01). Legacy 978-landmark
        callers are still accepted. The caller aligns the Visium basal to this
        order before passing it (and 0-1 normalizes it; Pitfall 1).
    device : str
        Torch device string (e.g. ``"cpu"`` or ``"cuda:0"``). The driver sets
        ``CUDA_VISIBLE_DEVICES`` BEFORE importing torch (Hard Rule 5), so
        ``"cuda:0"`` here resolves to the chosen visible GPU. Default ``"cpu"``.
    """

    def __init__(
        self,
        model_variant: str,
        gene_ids: tuple[str, ...],
        device: str = "cpu",
    ) -> None:
        _VALID_VARIANTS: frozenset[str] = frozenset(
            {"multidcp_pdg", "multidcp_chemoe"}
        )
        if model_variant not in _VALID_VARIANTS:
            raise ValueError(
                f"RegionSignatureCacher: model_variant must be one of "
                f"{sorted(_VALID_VARIANTS)}, got {model_variant!r}."
            )
        self.model_variant = model_variant
        self.gene_ids = tuple(gene_ids)
        self.device = device
        self._model: Optional[object] = None
        # Cross-project featurizer + gene tensor, lazily bound in load_model
        # (kept off the module top so the pure functions stay torch-free).
        self._convert_smile_to_feature = None
        self._create_mask_feature = None
        self._gene_tensor = None
        self._torch = None

    # ------------------------------------------------------------------
    # Model loading (IMPURE — real strict-load of the row-17 CheMoE backbone)
    # ------------------------------------------------------------------

    # D-04 amendment (row-17, SHA fbee15f…): the only working, tissue-basal-
    # capable frozen backbone. The row-18 KPGT checkpoint is
    # collapsed/incompatible and is NOT wired (S-B descoped).
    _MDCP_SRC = "/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src"
    _CHECKPOINT_SHA256 = (
        "fbee15faade904cacd6484832cea211ebd4b28186ca40def40af3fea27c7d2a0"
    )

    def load_model(self, checkpoint_path: str, verify_sha: bool = True) -> None:
        """Strict-load the frozen MultiDCP-CheMoE backbone (row-17 checkpoint).

        Replaces the former NotImplementedError seam with the VERIFIED strict
        load of ``MultiDCP_CheMoE_AE`` (alias ``MultiDCP_CheMoEBase``) from
        ``/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt``
        (0 missing / 0 unexpected keys). torch + the cross-project ``sys.path``
        injection are confined to this method (and ``_featurize_drug`` /
        ``_call_model``) so the pure functions stay torch-free.

        Security (T-02-03): the kpgt-vs-CheMoE mixup is the integrity risk. The
        checkpoint path is asserted to exist and (when ``verify_sha``) its SHA256
        is asserted against the MANIFEST row-17 pin ``fbee15f…``; the strict load
        must be 0/0 or this raises — there is NO silent fallback.

        Parameters
        ----------
        checkpoint_path : str
            Absolute path to the row-17 ``best_model.pt``. Must exist.
        verify_sha : bool
            If True (default), assert the file's SHA256 matches the row-17 pin.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_path`` does not exist.
        ValueError
            If the SHA256 does not match the pinned row-17 value.
        RuntimeError
            If ``load_state_dict(strict=True)`` finds any missing/unexpected key.
        """
        import hashlib as _hashlib
        import os
        import sys

        import torch

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"load_model: checkpoint not found: {checkpoint_path}. Use the "
                f"D-04 row-17 path "
                f"/raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/best_model.pt"
            )

        if verify_sha:
            h = _hashlib.sha256()
            with open(checkpoint_path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
            if digest != self._CHECKPOINT_SHA256:
                raise ValueError(
                    f"load_model: checkpoint SHA256 mismatch (T-02-03 integrity "
                    f"guard). Expected row-17 {self._CHECKPOINT_SHA256}, got "
                    f"{digest}. Refusing to load — this may be the collapsed "
                    f"row-18 KPGT checkpoint (S-B, NOT wired)."
                )

        # Cross-project import (read-only static dependency); inside the method
        # so the pure functions stay import-clean (Cross-project import landmine).
        sys.path.insert(0, os.path.join(self._MDCP_SRC, "models"))
        sys.path.insert(0, os.path.join(self._MDCP_SRC, "utils"))
        from multidcp_ae_pdg_utils import initialize_model_registry  # noqa: E402
        import multidcp_chemoe_pdg as mc  # noqa: E402
        from data_utils_pdg import (  # noqa: E402
            convert_smile_to_feature,
            create_mask_feature,
        )

        device = torch.device(self.device)
        reg = initialize_model_registry()
        reg.update(
            {
                "num_gene": N_PDG,
                "pert_idose_input_dim": 2,  # VERIFIED dose_encoder (64, 2)
                "dropout": 0.3,
                "linear_encoder_flag": False,  # checkpoint has the transformer cell encoder
            }
        )
        model = mc.MultiDCP_CheMoE_AE(device=device, model_param_registry=reg)
        # Move params to device BEFORE .double() (mirrors the upstream training
        # script; the NeuralFingerprint sub-params otherwise stay on CPU and the
        # forward errors on a cpu/cuda device mismatch).
        model.to(device)
        model = model.double()

        state = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        # strict=True: in this torch version load_state_dict raises on any
        # missing/unexpected key, so reaching the next line == 0/0 (no fallback).
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self._torch = torch
        self._model = model
        self._convert_smile_to_feature = convert_smile_to_feature
        self._create_mask_feature = create_mask_feature
        # input_gene is unused by the CheMoE architecture (gene features are
        # learned per-gene embeddings inside the model); a placeholder index
        # tensor satisfies the positional forward signature.
        self._gene_tensor = torch.arange(N_PDG, device=device)
        log.info(
            "load_model: strict-loaded MultiDCP_CheMoE_AE from %s "
            "(0 missing / 0 unexpected) on %s.",
            checkpoint_path,
            device,
        )

    # ------------------------------------------------------------------
    # Drug featurization (IMPURE — repo NeuralFingerprint graph, do not hand-roll)
    # ------------------------------------------------------------------

    def _featurize_drug(self, smiles: str):
        """Build the ``input_drug`` dict + attention ``mask`` for one SMILES.

        Uses the repo's ``convert_smile_to_feature`` / ``create_mask_feature``
        (atom=62 / bond=6) so the graph matches the frozen NeuralFingerprint
        exactly — any hand-rolled featurizer would silently produce the wrong
        drug embedding. Returns ``(input_drug, mask)`` on ``self.device`` as
        float64 (the model is ``.double()``).
        """
        if self._model is None or self._convert_smile_to_feature is None:
            raise RuntimeError(
                "_featurize_drug: model not loaded. Call load_model() first."
            )
        device = self._torch.device(self.device)
        drug = self._convert_smile_to_feature([smiles], device)
        mask = self._create_mask_feature(drug, device)
        return drug, mask

    # ------------------------------------------------------------------
    # Model inference (IMPURE — one real forward, absolute treated [N_PDG])
    # ------------------------------------------------------------------

    def _call_model(
        self,
        smiles: str,
        region_basal: np.ndarray,
    ) -> np.ndarray:
        """Run one frozen forward for (drug, region) → absolute treated GEX.

        Returns the model's RAW predicted treated expression (rule-B subtraction
        — treated minus the predicted control — happens later in ``compute_de``;
        do NOT subtract anything here). The caller passes an ALREADY 0-1
        manifold-normalized ``region_basal`` (Pitfall 1) aligned to the model's
        gene order; this method only casts to float64, adds the batch dim, and
        applies the fixed 2-dim dose one-hot (Pitfall 4).

        Parameters
        ----------
        smiles : str
            SMILES string for the drug/compound (or ``INERT_CONTROL_SMILES`` for
            the rule-B control pass).
        region_basal : np.ndarray
            Shape ``(N_PDG,)`` float, already aligned + 0-1 normalized.

        Returns
        -------
        np.ndarray
            Shape ``(N_PDG,)`` float32. Absolute predicted treated GEX.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via ``load_model``.
        """
        if self._model is None:
            raise RuntimeError(
                "_call_model: model not loaded. Call load_model() first."
            )
        torch = self._torch
        device = torch.device(self.device)

        drug, mask = self._featurize_drug(smiles)
        basal = torch.as_tensor(
            region_basal, dtype=torch.float64, device=device
        ).unsqueeze(0)  # [1, N_PDG]
        dose = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float64, device=device
        )  # fixed 2-dim one-hot (Pitfall 4, D-05 dose-agnostic)

        with torch.no_grad():
            pred, _cell_hidden = self._model(
                input_cell_gex=basal,
                input_drug=drug,
                input_gene=self._gene_tensor,
                mask=mask,
                input_pert_idose=dose,
                job_id="perturbed",
                epoch=0,
            )
        return pred.squeeze(0).float().cpu().numpy()  # [N_PDG] absolute treated

    # ------------------------------------------------------------------
    # Full caching run (real frozen forward; rule-B control pass per region)
    # ------------------------------------------------------------------

    def run(
        self,
        pert_ids: list[str],
        smiles_map: dict[str, str],
        region_basal_map: dict[str, np.ndarray],
    ) -> RegionSignatureCache:
        """Produce the 3-vector rule-B cache for all pert_ids × all regions.

        For each region, the inert-drug control pass (``INERT_CONTROL_SMILES``,
        D-02) is computed ONCE and reused for every drug in that region (control
        is region-determined, not drug-determined; the +1 forward per region is
        cheap). For each (drug, region), the treated forward is computed. Both
        maps are then handed to the pure ``assemble_cache`` which builds
        ``de_array = treated - control`` plus the cached treated/control vectors
        (D-03).

        Parameters
        ----------
        pert_ids : list[str]
            Ordered perturbation IDs. Must match keys in ``smiles_map``.
        smiles_map : dict[str, str]
            ``{pert_id: smiles}`` for every pert_id in ``pert_ids``.
        region_basal_map : dict[str, np.ndarray]
            ``{region_label: basal_array}`` for each region. Arrays must be shape
            ``(n_genes,)`` with ``n_genes == len(self.gene_ids)``, already
            aligned to the model's gene order and 0-1 manifold-normalized
            (Pitfall 1) by the caller.

        Returns
        -------
        RegionSignatureCache
            The 3-vector cache (de/treated/control) plus manifest. Axes:
            ``(n_pert_ids, n_regions, n_genes)`` float32.

        Raises
        ------
        RuntimeError
            If the model has not been loaded via ``load_model``.
        KeyError
            If ``smiles_map`` is missing a pert_id from ``pert_ids``.
        """
        if self._model is None:
            raise RuntimeError(
                "RegionSignatureCacher.run: model not loaded. Call "
                "load_model(checkpoint_path) before run()."
            )

        regions = sorted(region_basal_map.keys())  # deterministic region order
        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=list(self.gene_ids),
            model_variant=self.model_variant,
        )

        # Rule-B control pass per region (once): inert/empty-drug reference in the
        # same region basal context (D-02 amendment).
        region_control_map: dict[str, np.ndarray] = {}
        for region in regions:
            basal = region_basal_map[region]
            region_control_map[region] = self._call_model(
                INERT_CONTROL_SMILES, basal
            )

        predicted_treated_map: dict[tuple[str, str], np.ndarray] = {}
        predicted_control_map: dict[tuple[str, str], np.ndarray] = {}
        for pert_id in pert_ids:
            if pert_id not in smiles_map:
                raise KeyError(
                    f"RegionSignatureCacher.run: smiles_map is missing "
                    f"pert_id {pert_id!r}. Supply SMILES for all pert_ids."
                )
            smiles = smiles_map[pert_id]
            for region in regions:
                basal = region_basal_map[region]
                predicted_treated_map[(pert_id, region)] = self._call_model(
                    smiles, basal
                )
                # Region-determined control, replicated per (pert_id, region)
                # so the cache aligns position-for-position (D-03).
                predicted_control_map[(pert_id, region)] = region_control_map[
                    region
                ]

        return assemble_cache(
            predicted_treated_map=predicted_treated_map,
            predicted_control_map=predicted_control_map,
            manifest=manifest,
        )
