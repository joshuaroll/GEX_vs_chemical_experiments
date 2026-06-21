"""Per-region predicted DE caching via a frozen MultiDCP/CheMoE model.

Pure library — no hardcoded /raid paths, no real-data filenames, no model
checkpoint loading (that is deferred behind NotImplementedError with a clear
TODO below). Only the following are fully implemented here and are directly
testable with in-memory fixtures:

  1. DE computation: ``predicted_DE_region = predicted_treated - region_basal``
     (element-wise float subtraction, shape [n_genes]).
  2. Cache-key construction: a deterministic, gene-order-stable, region-order-
     stable, pert_id-order-stable string key for each (pert_id, region) pair.
  3. Manifest construction: a frozen dataclass capturing the region order, gene
     order, and pert_id order used for a caching run, so results arrays can
     always be reconstructed from disk unambiguously.
  4. Result containers: ``RegionDE`` (single (pert_id, region) pair) and
     ``RegionSignatureCache`` (aligned arrays over all pert_ids × all regions).

DE convention used in this module
----------------------------------
``predicted_DE_region = predicted_treated(drug, region_basal) - region_basal``

This is a DIVERGENCE from the main v0.4 DE rule (``treated - diseased``).
The spatial arm uses a HEALTHY-TISSUE basal anchor, not a diseased cell-line
anchor, because the spatial basal profiles are from healthy Visium datasets
(Yu et al. 2022 liver, Maynard et al. 2021 DLPFC, etc.). This divergence is
an open item requiring explicit professor sign-off — see
``09_spatial_decisions.md`` open item #3.

Model-inference stub
---------------------
The actual call to a frozen MultiDCP-PDG or CheMoE model is not implemented;
it raises ``NotImplementedError`` with a TODO that points to the unresolved
checkpoint paths in MANIFEST.md and the decisions in 09_spatial_decisions.md.
Callers wishing to use the cacher for real inference must:

  1. Confirm checkpoint paths in MANIFEST.md (Phase 3 rows are currently
     populated with placeholder "Phase 3" strings, not real file paths).
  2. Read 09_spatial_decisions.md open item #3 (DE-rule divergence approval)
     and open item #1 (spatial Halt Gate go/no-go decision).
  3. Subclass ``RegionSignatureCacher`` and override ``_call_model``.

Hard rules honored
-------------------
- Pure library: NO hardcoded /raid paths or real-data filenames.
- No mocking or synthetic model outputs; the real inference path raises
  NotImplementedError to prevent accidental fabrication.
- N_LANDMARK = 978 (gene space constant, matches spatial_basal.config).
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
"""Number of LINCS L1000 landmark genes. The frozen MultiDCP/CheMoE models
predict over this 978-gene space. Per-region basal vectors are also subset to
this space before being passed as cell context."""

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class RegionDE(NamedTuple):
    """Predicted differential expression for one (pert_id, region) pair.

    DE convention: ``de_vector = predicted_treated - region_basal``.
    Both input and output are in the 978-landmark-gene space.

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
        Frozen description of the DE formula used. Must be
        ``"predicted_treated - region_basal"`` for all spatial-arm outputs.
        (Distinct from main v0.4 convention ``"treated - diseased"``.)
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
    """Aligned predicted DE arrays for all pert_ids × all regions.

    Attributes
    ----------
    de_array : np.ndarray
        Shape ``(n_pert_ids, n_regions, n_genes)`` float32.
        ``de_array[i, j, :]`` is the predicted DE vector for
        ``manifest.pert_ids[i]`` in region ``manifest.regions[j]``.
    manifest : RegionSignatureManifest
        Alignment metadata; axes only interpretable via this manifest.
    """

    de_array: np.ndarray
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
        de_convention="predicted_treated - region_basal",
        n_pert_ids=len(pert_ids),
        n_regions=len(regions),
        n_genes=len(gene_ids),
    )


# ---------------------------------------------------------------------------
# DE computation (pure, testable without model)
# ---------------------------------------------------------------------------


def compute_de(
    predicted_treated: np.ndarray,
    region_basal: np.ndarray,
) -> np.ndarray:
    """Compute per-region predicted DE: ``predicted_treated - region_basal``.

    Both arrays must be 1-D float arrays of the same length (n_genes). The
    result is returned as float32.

    This function is the canonical DE computation for the spatial arm and
    is designed to be called by ``RegionSignatureCacher._compute_de_from_batch``
    or invoked directly by tests that inject synthetic ``predicted_treated``
    arrays (bypassing the model stub).

    DE convention: ``predicted_DE_region = predicted_treated - region_basal``.
    This is the HEALTHY-TISSUE convention for the spatial arm; it differs from
    the main v0.4 DE rule (``treated - diseased``) and requires professor
    sign-off (see ``09_spatial_decisions.md`` open item #3).

    Parameters
    ----------
    predicted_treated : np.ndarray
        Shape ``(n_genes,)`` float. Predicted gene expression of cells in
        ``region`` after drug perturbation. Produced by a frozen
        MultiDCP/CheMoE model with ``region_basal`` as the cell context.
    region_basal : np.ndarray
        Shape ``(n_genes,)`` float. Pseudobulked, region-averaged basal
        expression from the spatial Visium dataset (e.g., periportal
        pseudobulk from Yu et al. 2022 liver). Subset to the model's gene
        space (N_LANDMARK = 978) before passing here.

    Returns
    -------
    np.ndarray
        Shape ``(n_genes,)`` float32. Element-wise
        ``predicted_treated - region_basal``.

    Raises
    ------
    ValueError
        If either array is not 1-D, or if they have different lengths.
    """
    pt = np.asarray(predicted_treated, dtype=np.float32)
    rb = np.asarray(region_basal, dtype=np.float32)

    if pt.ndim != 1:
        raise ValueError(
            f"compute_de: predicted_treated must be 1-D, got shape {pt.shape}."
        )
    if rb.ndim != 1:
        raise ValueError(
            f"compute_de: region_basal must be 1-D, got shape {rb.shape}."
        )
    if pt.shape != rb.shape:
        raise ValueError(
            f"compute_de: shape mismatch — predicted_treated {pt.shape} vs "
            f"region_basal {rb.shape}. Both must have the same n_genes."
        )

    return pt - rb


# ---------------------------------------------------------------------------
# Assemble full cache from injected predictions (pure, testable)
# ---------------------------------------------------------------------------


def assemble_cache(
    predicted_treated_map: dict[tuple[str, str], np.ndarray],
    region_basal_map: dict[str, np.ndarray],
    manifest: RegionSignatureManifest,
) -> RegionSignatureCache:
    """Build a ``RegionSignatureCache`` from caller-supplied predicted arrays.

    This function is PURE: it only does DE subtraction and array stacking. It
    does NOT call any model. It is the entry point for:

      a. Tests, which inject synthetic ``predicted_treated_map`` arrays.
      b. A future ``RegionSignatureCacher.run()`` implementation, which will
         populate ``predicted_treated_map`` by calling the model stub (once
         the checkpoint paths are resolved).

    Parameters
    ----------
    predicted_treated_map : dict[tuple[str, str], np.ndarray]
        Keys are ``(pert_id, region)`` pairs. Values are 1-D float arrays of
        shape ``(n_genes,)`` representing the model's predicted treated GEX
        for that drug in that region's cell context.
        All ``(pert_id, region)`` combinations in the manifest must be present.
    region_basal_map : dict[str, np.ndarray]
        Keys are region labels (must cover all ``manifest.regions``). Values
        are 1-D float arrays of shape ``(n_genes,)`` — pseudobulked basal
        expression for that region.
    manifest : RegionSignatureManifest
        Alignment metadata (produced by ``build_manifest``). Defines the
        iteration order for axes 0 (pert_ids) and 1 (regions).

    Returns
    -------
    RegionSignatureCache
        ``.de_array`` shape is ``(n_pert_ids, n_regions, n_genes)`` float32.
        ``.manifest`` is the supplied manifest (passed through unchanged).

    Raises
    ------
    KeyError
        If any ``(pert_id, region)`` pair in the manifest is missing from
        ``predicted_treated_map``, or if any region is missing from
        ``region_basal_map``.
    ValueError
        If any predicted array or basal array does not match the gene count
        in ``manifest.n_genes``, or if ``compute_de`` raises a shape error.
    """
    # Validate region_basal_map completeness before the main loop.
    missing_regions = set(manifest.regions) - set(region_basal_map.keys())
    if missing_regions:
        raise KeyError(
            f"assemble_cache: region_basal_map is missing regions "
            f"{sorted(missing_regions)}. All manifest regions must be present."
        )

    n_p = manifest.n_pert_ids
    n_r = manifest.n_regions
    n_g = manifest.n_genes
    de_array = np.empty((n_p, n_r, n_g), dtype=np.float32)

    for i, pert_id in enumerate(manifest.pert_ids):
        for j, region in enumerate(manifest.regions):
            key = (pert_id, region)
            if key not in predicted_treated_map:
                raise KeyError(
                    f"assemble_cache: predicted_treated_map is missing entry "
                    f"for (pert_id={pert_id!r}, region={region!r}). Ensure all "
                    f"manifest pert_id × region combinations are populated."
                )
            pred = predicted_treated_map[key]
            basal = region_basal_map[region]
            de_vec = compute_de(pred, basal)
            if de_vec.shape[0] != n_g:
                raise ValueError(
                    f"assemble_cache: DE vector for ({pert_id!r}, {region!r}) "
                    f"has {de_vec.shape[0]} genes, but manifest.n_genes={n_g}."
                )
            de_array[i, j, :] = de_vec

    log.info(
        "assemble_cache: built DE array %s (pert_ids=%d, regions=%d, genes=%d)",
        de_array.shape,
        n_p,
        n_r,
        n_g,
    )
    return RegionSignatureCache(de_array=de_array, manifest=manifest)


# ---------------------------------------------------------------------------
# RegionSignatureCacher — the main interface
# ---------------------------------------------------------------------------


class RegionSignatureCacher:
    """Interface that produces per-region predicted DE by calling a frozen model.

    Usage pattern (once checkpoint paths are confirmed)
    ---------------------------------------------------
    ::

        cacher = RegionSignatureCacher(
            model_variant="multidcp_pdg",
            gene_ids=landmark_gene_ids,   # tuple of 978 HGNC symbols / ENTREZ IDs
        )
        cacher.load_model(checkpoint_path)   # raises NotImplementedError until
                                              # MANIFEST.md Phase 3 rows are filled
        cache = cacher.run(
            pert_ids=["drug_A", "drug_B"],
            smiles_map={"drug_A": "CCO", "drug_B": "c1ccccc1"},
            region_basal_map={"periportal": np.array([...]), "pericentral": np.array([...])},
        )

    The PURE parts (``compute_de``, ``build_manifest``, ``make_cache_key``,
    ``assemble_cache``) are available as module-level functions and are tested
    independently of the model stub.

    Parameters
    ----------
    model_variant : str
        One of ``"multidcp_pdg"`` or ``"multidcp_chemoe"``. Determines both
        which checkpoint will be loaded and which namespace is used in cache keys.
    gene_ids : tuple[str, ...]
        Ordered gene identifiers for the model's output space. Must have
        ``len(gene_ids) == N_LANDMARK`` (978) for the landmark-space models.
        Caller is responsible for aligning this with the Visium basal vectors.
    """

    def __init__(
        self,
        model_variant: str,
        gene_ids: tuple[str, ...],
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
        self._model: Optional[object] = None

    # ------------------------------------------------------------------
    # Model loading (STUB — raises NotImplementedError)
    # ------------------------------------------------------------------

    def load_model(self, checkpoint_path: str) -> None:
        """Load a frozen MultiDCP/CheMoE checkpoint from disk.

        TODO: NOT IMPLEMENTED. Checkpoint paths are unconfirmed in MANIFEST.md
        (Phase 3 rows currently show placeholder strings "Phase 3", not real
        file paths). Before implementing:

          1. Complete Phase 3 upstream training runs and fill in the
             MANIFEST.md "Trained checkpoints" table with real paths in the
             form:
             ``/raid/home/joshua/projects/GEX_vs_chemical_experiments/
               dili_downstream/trained_models/{condition}_seed{seed}/best.pt``
          2. Resolve 09_spatial_decisions.md open item #3 (DE-rule divergence
             approval for spatial arm ``treated - region_basal`` vs main v0.4
             ``treated - diseased``).
          3. Resolve 09_spatial_decisions.md open item #1 (confirm a negative
             spatial result is scientifically acceptable, sets whether the
             spatial Halt Gate is stop-and-reframe or stop-and-abandon).
          4. Implement the actual ``torch.load`` / ``model.eval()`` /
             ``model.requires_grad_(False)`` sequence using the correct class
             from ``src/models/upstream/multidcp_pdg.py`` (Condition B/E) or
             ``src/models/upstream/multidcp_pdgrapher_fusion.py`` (Condition C/F),
             both SHA-pinned at 871b8de0332045a3ad5ab3a39e689014317dadfe.

        Parameters
        ----------
        checkpoint_path : str
            Absolute path to the ``.pt`` checkpoint file. Must exist and be
            readable; the GPU used is governed by the ``--gpu`` argparse flag
            (set BEFORE ``import torch``; always leave one GPU free on the
            shared box per HARD RULE 5).

        Raises
        ------
        NotImplementedError
            Always, until the checkpoint paths are confirmed in MANIFEST.md
            and the above open items in 09_spatial_decisions.md are resolved.
        """
        raise NotImplementedError(
            "RegionSignatureCacher.load_model: checkpoint loading is not "
            "implemented. Checkpoint paths for the frozen MultiDCP/CheMoE "
            "models are unconfirmed in MANIFEST.md (Phase 3 rows are "
            "placeholders). See also 09_spatial_decisions.md open items #1 "
            "and #3 (DE-rule divergence approval, spatial Halt Gate go/no-go). "
            "Implement this method once Phase 3 upstream training is complete."
        )

    # ------------------------------------------------------------------
    # Model inference (STUB — raises NotImplementedError)
    # ------------------------------------------------------------------

    def _call_model(
        self,
        smiles: str,
        region_basal: np.ndarray,
    ) -> np.ndarray:
        """Call the frozen model for one (drug, region) pair.

        TODO: NOT IMPLEMENTED. See ``load_model`` docstring for the full
        dependency chain. This method must:

          1. Convert ``smiles`` to the model's drug-feature representation
             (NeuralFingerprint graph encoding for MultiDCP-PDG; CheMoE
             drug encoder for CheMoE variant).
          2. Supply ``region_basal`` (shape ``(n_genes,)``) as the cell-context
             vector (analogous to a cell-line basal profile). The basal vector
             must be subset to the model's gene space before passing here.
          3. Run a forward pass with ``torch.no_grad()`` and ``model.eval()``.
          4. Return the raw predicted treated GEX (shape ``(n_genes,)``) as
             a CPU float32 numpy array. Do NOT subtract the basal here —
             subtraction happens in ``compute_de`` so that step is separately
             testable.

        Parameters
        ----------
        smiles : str
            SMILES string for the drug/compound.
        region_basal : np.ndarray
            Shape ``(n_genes,)`` float32. Region pseudobulk basal expression
            in the model's gene space.

        Returns
        -------
        np.ndarray
            Shape ``(n_genes,)`` float32. Raw predicted treated GEX.

        Raises
        ------
        NotImplementedError
            Always, until the checkpoint is loaded and the model inference
            pipeline is implemented.
        """
        raise NotImplementedError(
            "RegionSignatureCacher._call_model: model inference is not "
            "implemented. Load a checkpoint with load_model() first, and see "
            "09_spatial_decisions.md open items #1 and #3."
        )

    # ------------------------------------------------------------------
    # Full caching run (delegates to stubs; pure parts are testable)
    # ------------------------------------------------------------------

    def run(
        self,
        pert_ids: list[str],
        smiles_map: dict[str, str],
        region_basal_map: dict[str, np.ndarray],
    ) -> RegionSignatureCache:
        """Produce predicted DE for all pert_ids × all regions.

        Calls ``_call_model`` for every ``(pert_id, region)`` combination,
        then delegates to ``assemble_cache`` for the pure DE-subtraction and
        array-stacking step.

        Because ``_call_model`` raises ``NotImplementedError``, this method
        also raises until the checkpoint is loaded. The pure post-processing
        (DE subtraction, manifest construction, array assembly) can be tested
        independently via ``assemble_cache``.

        Parameters
        ----------
        pert_ids : list[str]
            Ordered perturbation IDs. Must match keys in ``smiles_map``.
        smiles_map : dict[str, str]
            ``{pert_id: smiles}`` for every pert_id in ``pert_ids``.
        region_basal_map : dict[str, np.ndarray]
            ``{region_label: basal_array}`` for each region. Arrays must be
            shape ``(n_genes,)`` with ``n_genes == len(self.gene_ids)``.

        Returns
        -------
        RegionSignatureCache
            Fully aligned DE array plus manifest. Axes:
            ``(n_pert_ids, n_regions, n_genes)`` float32.

        Raises
        ------
        NotImplementedError
            Propagated from ``_call_model`` (see ``load_model`` docstring).
        KeyError
            If ``smiles_map`` is missing a pert_id from ``pert_ids``.
        """
        regions = sorted(region_basal_map.keys())  # deterministic region order
        manifest = build_manifest(
            pert_ids=pert_ids,
            regions=regions,
            gene_ids=list(self.gene_ids),
            model_variant=self.model_variant,
        )

        predicted_treated_map: dict[tuple[str, str], np.ndarray] = {}
        for pert_id in pert_ids:
            if pert_id not in smiles_map:
                raise KeyError(
                    f"RegionSignatureCacher.run: smiles_map is missing "
                    f"pert_id {pert_id!r}. Supply SMILES for all pert_ids."
                )
            smiles = smiles_map[pert_id]
            for region in regions:
                basal = region_basal_map[region]
                # This raises NotImplementedError until load_model() is done.
                predicted = self._call_model(smiles, basal)
                predicted_treated_map[(pert_id, region)] = predicted

        return assemble_cache(
            predicted_treated_map=predicted_treated_map,
            region_basal_map=region_basal_map,
            manifest=manifest,
        )
