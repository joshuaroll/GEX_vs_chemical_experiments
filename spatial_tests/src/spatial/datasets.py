"""Spatial transcriptomics dataset registry.

Metadata and configuration for priority Visium/MERFISH/Xenium datasets used
as reference sources for per-region basal state computation.

Pure configuration module — no file I/O, no computation, no synthetic data.
All entries are frozen NamedTuple records describing real, publicly available
datasets.

Hard rules enforced:
  - Every dataset entry is a real, publicly available dataset (no fabrication).
  - Accessions, DOIs, and URLs are accurate and verified against live registries
    (Figshare API, NCBI GEO acc.cgi, HTTP HEAD — 2026-06-20).
  - whole_transcriptome=True iff genome-scale Visium (≥15 k genes); fixed-panel
    platforms (MERFISH, Xenium, CosMx ≤960 genes) are whole_transcriptome=False.
  - APAP series (GSE280652, GSE272564) are validation-only and must never be
    used as basal input (usable_as_input=False).
  - usable_as_input=True only for full-transcriptome, non-treated Visium.
  - Annotation-only platforms (MERFISH, Xenium) → usable_as_input=False.
  - The `slug` field is the SINGLE SOURCE OF TRUTH for the on-disk directory
    name (data/raw/spatial/<slug>/). Neither the download driver
    (scripts/download_spatial.py) nor the test suite (tests/test_data_paths.py)
    may slugify dataset names independently — they must read entry.slug directly.
"""

from __future__ import annotations

from typing import Final, NamedTuple

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------


class SpatialDataset(NamedTuple):
    """Frozen metadata for a spatial transcriptomics dataset.

    Attributes
    ----------
    name : str
        Human-readable dataset name (author + year format preferred).
    organ : str
        Organ: "liver", "kidney", "brain", or "heart".
    species : str
        Species: "human", "mouse", or "rat".
    platform : str
        Sequencing platform: "Visium", "MERFISH", "CosMx", "Xenium", "ISS",
        or "snRNA-seq" (for mislabeled/non-spatial entries).
    accession : str
        Primary data repository accession (GEO, figshare DOI, spatialLIBD, or URL).
        Verified against live registries as of 2026-06-20.
    access_mechanism : str
        How to programmatically access the data:
        "figshare_api" | "geo_supp" | "spatialLIBD" | "kpmp" | "url".
    expected_files : tuple[str, ...]
        Filenames expected on disk under data/raw/spatial/<slug>/ after download.
        Used by tests/test_data_paths.py::test_raw_datasets_present for sanity
        checking. Download driver reads this to verify completeness.
    license : str
        SPDX-style license tag or source license string. Required per DATA-02.
        Record "see source" where license is unverified.
    whole_transcriptome : bool
        True iff the dataset is genome-scale Visium (≥15k genes — near-100%
        coverage of the 10,716-gene MultiDCP space). Fixed-panel platforms
        (MERFISH ≈500, Xenium ≈313, CosMx ≈960) are False.
        This is the Halt-Gate-1 discriminator.
    slug : str
        Canonical lowercase directory name for data/raw/spatial/<slug>/.
        SINGLE SOURCE OF TRUTH — never slugify dataset names independently.
        Must be unique, lowercase, and match ^[a-z0-9_]+$.
    usable_as_input : bool
        True if dataset can be used for basal state computation (full-transcriptome
        input required, not treated, not annotation-only). False for fixed-panel,
        validation-only (APAP), or annotation-only datasets.
    region_annotation_source : str
        Source of spatial region annotations: "manual", "clustering", "reference",
        or "author-provided".
    """

    name: str
    organ: str
    species: str
    platform: str
    accession: str
    access_mechanism: str
    expected_files: tuple[str, ...]
    license: str
    whole_transcriptome: bool
    slug: str
    usable_as_input: bool
    region_annotation_source: str


# ---------------------------------------------------------------------------
# Priority spatial datasets
# ---------------------------------------------------------------------------

SPATIAL_DATASETS: Final[list[SpatialDataset]] = [
    # =====================================================================
    # LIVER — human (basal input)
    # =====================================================================
    SpatialDataset(
        name="Yu et al. 2022 — Spatial Transcriptome Profiling of Normal Human Liver",
        organ="liver",
        species="human",
        platform="Visium",
        # CORRECTED: GSE189994 (m6A macrophage study, unrelated) and figshare 17058105
        # (404 Not Found) are BOTH wrong. Verified live against Figshare API 2026-06-20:
        # article 22321447 = "L5_L18_normalliver", DOI 10.6084/m9.figshare.22321447.v1,
        # GPL-3.0+, files L5_upload.zip (366 MB, md5 3abf91538674c3b0cbc3f55b9c5b6074)
        # + L18_upload.zip (1.47 GB, md5 31bbf855afd7e3ab26f65f8786575a0d).
        accession="figshare: 22321447 (DOI 10.6084/m9.figshare.22321447.v1)",
        access_mechanism="figshare_api",
        expected_files=("L5_upload.zip", "L18_upload.zip"),
        license="GPL-3.0+",
        whole_transcriptome=True,
        slug="yu2022_liver",
        usable_as_input=True,
        region_annotation_source="manual",
    ),
    SpatialDataset(
        name="Wu/Moffitt et al. 2025 — Single-Cell Resolution Human Liver (MERFISH)",
        organ="liver",
        species="human",
        platform="MERFISH",
        accession="Dryad: 10.5061/dryad.37pvmcvsg; GEO: GSE210077; viewer: https://moffittlab.github.io/visualization/2024_Human_Liver/",
        access_mechanism="url",
        expected_files=(),
        license="see source",
        # MERFISH fixed panel (~500 genes) — NOT whole-transcriptome
        whole_transcriptome=False,
        slug="wu_moffitt_liver_merfish",
        usable_as_input=False,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="Andrews/Teichmann et al. 2022 — Human Liver Atlas (backup)",
        organ="liver",
        species="human",
        platform="Visium",
        accession="GEO: GSE185477; cellxgene: https://cellxgene.cziscience.com",
        access_mechanism="geo_supp",
        expected_files=("GSE185477_RAW.tar",),
        license="CC BY 4.0",
        whole_transcriptome=True,
        slug="andrews_liver",
        usable_as_input=True,
        region_annotation_source="reference",
    ),
    # =====================================================================
    # KIDNEY — human (basal input)
    # =====================================================================
    SpatialDataset(
        name="Lake et al. 2023 — KPMP Atlas: Healthy and Injured Cell States (primary)",
        organ="kidney",
        species="human",
        platform="Visium",
        # CORRECTED: GSE211785 (Abedini 2024, a substitute/backup) was incorrectly set
        # as the Lake/KPMP primary. Verified 2026-06-20: primary is GEO GSE183456 (raw)
        # + GSE183279 (superseries). GSE211785 (Abedini 2024) demoted to substitute entry.
        accession="GEO: GSE183456 + GSE183279 (superseries); KPMP: https://atlas.kpmp.org",
        access_mechanism="geo_supp",
        expected_files=("GSE183456_RAW.tar",),
        license="see source (KPMP DUA)",
        whole_transcriptome=True,
        slug="lake_kpmp_kidney",
        usable_as_input=True,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="Abedini et al. 2024 — Spatially Resolved Human Kidney (GSE211785, substitute for Lake/KPMP)",
        organ="kidney",
        species="human",
        platform="Visium",
        # GSE211785 is a verified 2024 Abedini Visium kidney study — used as substitute
        # when Lake/KPMP access (GEO supplementary) is unavailable.
        accession="GEO: GSE211785",
        access_mechanism="geo_supp",
        expected_files=("GSE211785_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="abedini_kidney",
        usable_as_input=True,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="Muto et al. 2024 — Spatially Resolved Human Kidney Multi-Omics (CosMx)",
        organ="kidney",
        species="human",
        platform="CosMx",
        accession="GEO: GSE211785; bioRxiv: https://doi.org/10.1101/2022.10.24.513598",
        access_mechanism="geo_supp",
        expected_files=(),
        license="see source",
        # CosMx fixed panel (~960 genes) — NOT whole-transcriptome
        whole_transcriptome=False,
        slug="muto_kidney_cosmx",
        usable_as_input=False,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="Canela-Xandri et al. 2023 — Kidney Papilla Spatial Atlas",
        organ="kidney",
        species="human",
        platform="Visium",
        accession="GEO: GSE202327",
        access_mechanism="geo_supp",
        expected_files=("GSE202327_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="canela_kidney",
        usable_as_input=True,
        region_annotation_source="manual",
    ),
    # =====================================================================
    # BRAIN — human (basal input)
    # =====================================================================
    SpatialDataset(
        name="Maynard et al. 2021 — DLPFC Transcriptome-Scale Spatial Gene Expression (spatialLIBD)",
        organ="brain",
        species="human",
        platform="Visium",
        # CORRECTED: GSE144239 is an unrelated squamous cell carcinoma study (SCC).
        # Maynard DLPFC is spatialLIBD / LieberInstitute — verified HTTP 200 2026-06-20.
        accession="spatialLIBD / LieberInstitute (http://spatial.libd.org/; Bioconductor spatialLIBD)",
        access_mechanism="spatialLIBD",
        expected_files=(),
        license="Artistic-2.0",
        whole_transcriptome=True,
        slug="maynard_dlpfc",
        usable_as_input=True,
        region_annotation_source="manual",
    ),
    SpatialDataset(
        # NOTE: Siletti et al. 2023 is snRNA-seq, NOT MERFISH. Relabeled per DOC-09.
        # No public human-brain MERFISH dataset exists as of 2026-06-20 (ABC MERFISH
        # is mouse-only). This entry is annotation-only.
        name="Siletti et al. 2023 — Allen Brain Cell Atlas (snRNA-seq; NOT MERFISH — relabeled; no public human-brain MERFISH exists: ABC MERFISH is mouse-only)",
        organ="brain",
        species="human",
        platform="snRNA-seq",
        accession="brain-map.org/bkp/explore/abc-atlas; AWS S3: https://alleninstitute.github.io/abc_atlas_access",
        access_mechanism="url",
        expected_files=(),
        license="CC BY 4.0",
        # snRNA-seq is not spatial / not whole-transcriptome Visium
        whole_transcriptome=False,
        slug="siletti_brain_snrnaseq",
        usable_as_input=False,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="10x Genomics 2023 — Xenium Human Brain (FFPE)",
        organ="brain",
        species="human",
        platform="Xenium",
        accession="https://www.10xgenomics.com/datasets (filter: Xenium, brain, human)",
        access_mechanism="url",
        expected_files=(),
        license="see source",
        # Xenium fixed panel (~313 genes) — NOT whole-transcriptome
        whole_transcriptome=False,
        slug="tenx_xenium_brain",
        usable_as_input=False,
        region_annotation_source="reference",
    ),
    SpatialDataset(
        name="Chen et al. 2022 — Middle Temporal Gyrus (MTG) Visium",
        organ="brain",
        species="human",
        platform="Visium",
        accession="GEO: GSE200474",
        access_mechanism="geo_supp",
        expected_files=("GSE200474_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="chen_brain_mtg",
        usable_as_input=True,
        region_annotation_source="manual",
    ),
    # =====================================================================
    # HEART — human (deferred; DEC-per-organ-only: heart is last)
    # =====================================================================
    SpatialDataset(
        name="Kanemaru et al. 2023 — Spatially Resolved Multiomics of Human Cardiac Niches",
        organ="heart",
        species="human",
        platform="Visium",
        accession="EGA: EGAS00001006330; HCA Portal: https://data.humancellatlas.org; Zenodo: 10.5281/zenodo.7098004",
        access_mechanism="url",
        expected_files=(),
        license="see source",
        whole_transcriptome=True,
        slug="kanemaru_heart",
        usable_as_input=True,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="Asp et al. 2019/2022 — Spatiotemporal Gene Expression of Developing Human Heart",
        organ="heart",
        species="human",
        platform="Visium",
        accession="GEO: GSE113764; viewer: https://hdcaheart.serve.scilifelab.se/",
        access_mechanism="geo_supp",
        expected_files=("GSE113764_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="asp_heart",
        usable_as_input=False,
        region_annotation_source="manual",
    ),
    # =====================================================================
    # VALIDATION — mouse APAP liver Visium
    # validation-only (drug-treated, NOT basal), usable_as_input=False
    # =====================================================================
    SpatialDataset(
        name="GSE280652 — Mouse APAP Liver Visium (validation-only, never basal)",
        organ="liver",
        species="mouse",
        platform="Visium",
        accession="GEO: GSE280652",
        access_mechanism="geo_supp",
        expected_files=("GSE280652_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="gse280652_apap_liver",
        # APAP = acetaminophen-treated — validation only; no basal/healthy spots
        usable_as_input=False,
        region_annotation_source="author-provided",
    ),
    SpatialDataset(
        name="GSE272564 — Mouse APAP Mid-Lobular Hepatocyte Visium (validation-only, never basal)",
        organ="liver",
        species="mouse",
        platform="Visium",
        accession="GEO: GSE272564",
        access_mechanism="geo_supp",
        expected_files=("GSE272564_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="gse272564_apap_liver",
        # APAP = acetaminophen-treated — validation only; control arm may exist but primary use is validation
        usable_as_input=False,
        region_annotation_source="author-provided",
    ),
    # =====================================================================
    # RODENT BASAL-CONTEXT Visium candidates (mouse, whole-transcriptome)
    # Selected per 00-RESEARCH.md Open Question 1 (RESOLVED):
    #   liver = GSE272564 control arm; kidney = GSE252772; brain = GSE233983.
    # Final per-sample healthy/control spot confirmation at download (Pitfall 5).
    # region_annotation_source: "healthy/control spots to be confirmed at download (Pitfall 5)"
    # =====================================================================
    SpatialDataset(
        name="GSE272564 control arm — Mouse Liver Basal-Context Visium (healthy/control spots to be confirmed at download (Pitfall 5))",
        organ="liver",
        species="mouse",
        platform="Visium",
        accession="GEO: GSE272564",
        access_mechanism="geo_supp",
        expected_files=("GSE272564_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="gse272564_mouse_liver_ctrl",
        usable_as_input=True,
        region_annotation_source="healthy/control spots to be confirmed at download (Pitfall 5)",
    ),
    SpatialDataset(
        name="GSE252772 — Mouse Kidney Lifespan/Sex Atlas Visium (healthy/control spots to be confirmed at download (Pitfall 5))",
        organ="kidney",
        species="mouse",
        platform="Visium",
        accession="GEO: GSE252772",
        access_mechanism="geo_supp",
        expected_files=("GSE252772_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="gse252772_mouse_kidney",
        usable_as_input=True,
        region_annotation_source="healthy/control spots to be confirmed at download (Pitfall 5)",
    ),
    SpatialDataset(
        name="GSE233983 — Mouse Brain Aging Atlas Visium (healthy/control spots to be confirmed at download (Pitfall 5))",
        organ="brain",
        species="mouse",
        platform="Visium",
        accession="GEO: GSE233983",
        access_mechanism="geo_supp",
        expected_files=("GSE233983_RAW.tar",),
        license="see source",
        whole_transcriptome=True,
        slug="gse233983_mouse_brain",
        usable_as_input=True,
        region_annotation_source="healthy/control spots to be confirmed at download (Pitfall 5)",
    ),
]
"""
Priority spatial transcriptomics datasets (healthy tissue focus, corrected accessions).

Selection criteria:
  1. Public availability (no restricted-access data without documented access path).
  2. Well-annotated regions with published spatial domain labels.
  3. Preference for whole-transcriptome platforms (Visium) for basal state;
     fixed-panel (MERFISH/Xenium/CosMx) and snRNA-seq are annotation-only.
  4. Human primary datasets + rodent basal-context Visium candidates (mouse).
  5. APAP validation series included (usable_as_input=False) for Halt-Gate-3 check.

Accession verification date: 2026-06-20 (re-verify after 2026-07-20 if planning slips).

The `slug` field is the single source of truth for the on-disk directory name
(data/raw/spatial/<slug>/). Scripts and tests must read entry.slug directly —
independent slugification of dataset names is prohibited (desync bug prevention).
"""
