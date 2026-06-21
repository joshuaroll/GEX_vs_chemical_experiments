---
phase: 00-dataset-acquisition-manifest
reviewed: 2026-06-20T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/spatial/datasets.py
  - src/spatial/orthology.py
  - scripts/compute_sha256.py
  - scripts/download_spatial.py
  - scripts/download_labels.py
  - scripts/build_orthologs.py
  - scripts/report_coverage.py
  - tests/test_data_paths.py
findings:
  critical: 3
  warning: 9
  info: 6
  total: 18
status: issues_found
---

# Phase 0: Code Review Report

**Reviewed:** 2026-06-20
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 0 delivers a corrected dataset registry, a pure ortholog filter, five network download drivers, a SHA-256 helper, a coverage reporter, and data-path tests. The pure-library separation is mostly clean and the ortholog filter is correct. However, several defects undermine the stated hard-rule invariants:

- **The single-source-of-truth invariant is broken for Figshare.** `download_spatial.py` hardcodes `FIGSHARE_ARTICLE_ID = "22321447"` instead of reading it from the registry entry. The article ID lives in `entry.accession` but is never parsed; the driver ignores the registry's per-entry identity.
- **The KPMP non-fatal exception is dead code.** The actual Lake/KPMP entry declares `access_mechanism="geo_supp"`, not `"kpmp"`, so it routes through the hard-halting `_download_geo_supp` path. The documented "KPMP 404 → log and continue" behavior never executes, and a KPMP supplementary 404 will fire Halt Gate 1 and abort the entire run.
- **Figshare integrity check is downgraded to a non-blocking warning.** An MD5 mismatch on downloaded data logs a warning and proceeds, leaving corrupt/substituted bytes on disk that downstream phases will treat as real. This violates the "real data only / never silently substituted" rule in spirit — a mismatch should be fatal, not advisory.

Plus a latent off-by-prefix bug in the GEO URL builder, two MANIFEST/test consistency gaps, and several robustness issues in archive parsing.

## Critical Issues

### CR-01: Figshare article ID hardcoded — breaks registry single-source-of-truth

**File:** `scripts/download_spatial.py:65`, `:163`
**Issue:** `FIGSHARE_ARTICLE_ID = "22321447"` is a module-level constant. `_download_figshare_api` builds the API URL from this constant and never reads `entry.accession` (which contains `"figshare: 22321447 (DOI ...)"`). The registry is supposed to be the single source of truth for every dataset's identity, mirroring the `entry.slug` discipline. Today there is exactly one `figshare_api` entry so the value happens to match, but:
- If a second Figshare dataset is added, both will silently download from article 22321447, writing the wrong files into the second slug's directory while passing the expected-files assertion only by coincidence (or producing confusing 404s).
- The accession field for the Figshare entry is decorative — changing it has no effect on what is fetched, which is a correctness trap.

**Fix:** Parse the article ID from `entry.accession` and pass it in, exactly as `_download_geo_supp` parses GSE IDs:
```python
import re

def _extract_figshare_id(accession: str) -> str:
    m = re.search(r"figshare:\s*(\d+)", accession)
    return m.group(1) if m else ""

def _download_figshare_api(entry, dest_dir: Path) -> None:
    article_id = _extract_figshare_id(entry.accession)
    if not article_id:
        _halt(f"No figshare article id in accession: {entry.accession}",
              dataset=entry.name, slug=entry.slug)
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    ...
```
Then delete the `FIGSHARE_ARTICLE_ID` constant.

### CR-02: KPMP non-fatal exception is unreachable — Lake/KPMP 404 will abort the whole download run

**File:** `scripts/download_spatial.py:283-328` (the `_download_kpmp` dispatcher) vs `src/spatial/datasets.py:161`
**Issue:** The Lake/KPMP primary entry declares `access_mechanism="geo_supp"` (datasets.py line 161), not `"kpmp"`. The main dispatch (download_spatial.py:482-485) routes `geo_supp` to `_download_geo_supp`, which calls `_stream_download`, which calls `_halt()` on any 404. The KPMP-specific `_download_kpmp` function — and all the documented "if GEO supplementary 404s, log ToS click-through and continue" logic, plus the `kpmp_failed` tracking and the post-download KPMP assertion skip (lines 521-529) — is never exercised because no registry entry uses `access_mechanism="kpmp"`. The KPMP DUA file (`GSE183456_RAW.tar`) is gated and the no-auth GEO path commonly 404s, so the realistic outcome is: the entire spatial download run halts at the KPMP entry and writes HALT_REASON.md, even though the plan explicitly states KPMP failure must be non-fatal.

**Fix:** Set the Lake/KPMP entry's mechanism to match its handler:
```python
# src/spatial/datasets.py — Lake/KPMP entry
access_mechanism="kpmp",
```
Or, if the registry value is intentional, route by a `kpmp`-aware predicate in `main()`. Either way the actual KPMP entry must reach the non-fatal path. Add a registry-vs-dispatch test asserting every `access_mechanism` value present in `SPATIAL_DATASETS` has a corresponding branch.

### CR-03: Figshare MD5 mismatch is non-fatal — corrupt/substituted data is kept

**File:** `scripts/download_spatial.py:214-226`
**Issue:** On MD5 mismatch the code only `log.warning(...)` and proceeds; the mismatched file stays on disk and passes the post-download expected-files assertion (which only checks existence and non-zero size). This directly conflicts with the hard rule that data must never be silently substituted, and with the integrity guarantee a checksum exists to provide. A truncated, tampered, or wrong-article download will be accepted as real input by Plan 04 and all downstream phases. The module docstring claims "warn-and-record on mismatch, never silently pass," but a warning that does not stop the pipeline and does not quarantine the file *is* effectively a silent pass for any non-interactive run.

**Fix:** Treat a checksum mismatch as a Halt Gate 1 condition (or at minimum delete the bad file and re-raise):
```python
if local_md5 != meta["md5"]:
    dest_file.unlink(missing_ok=True)  # do not leave corrupt bytes on disk
    _halt(
        f"MD5 mismatch for {dest_file.name}: local={local_md5} remote={meta['md5']}",
        url=meta["download_url"], dataset=entry.name, slug=entry.slug,
    )
```

## Warnings

### WR-01: GEO supplementary URL builder mangles accessions shorter than 6 digits

**File:** `scripts/download_spatial.py:229-244`
**Issue:** `prefix = gse_id[:-3] + "nnn"` slices the last 3 characters off the *whole* string including the `GSE` prefix. For 6+ digit accessions this is correct (`GSE183456` → `GSE183nnn`). But for short accessions it corrupts the prefix: `GSE12` → `GSnnn` (verified). The correct GEO convention is to blank the last 3 digits of the numeric portion only, padding to `GSEnnn` for <1000. All current registry accessions are 6 digits, so this is latent, but it is a real off-by-prefix bug in a reusable helper.

**Fix:** Operate on the numeric portion:
```python
num_str = gse_id[3:]
if not num_str.isdigit():
    return ""
stub = num_str[:-3] if len(num_str) > 3 else ""
prefix = f"GSE{stub}nnn"
```

### WR-02: `_download_geo_supp` assumes the file lives under the FIRST GSE ID only

**File:** `scripts/download_spatial.py:253-280`
**Issue:** `primary_gse = gse_ids[0]` takes the first GSE token in the accession string. For the Lake/KPMP entry the accession is `"GEO: GSE183456 + GSE183279 (superseries); ..."`, so `gse_ids[0]` = GSE183456 (the raw series — correct here by luck of ordering). But for any entry where the superseries is listed first, or where the RAW.tar lives under a different series than the first-mentioned accession, the constructed URL 404s and halts. The expected file `GSE183456_RAW.tar` *names* its own series — the driver should derive the series directory from the expected filename, not from accession token order.

**Fix:** Parse the GSE prefix from each `expected_name` (e.g. `re.match(r"(GSE\d+)_", fname)`) and build the URL per file, falling back to the accession list only if the filename carries no GSE prefix.

### WR-03: Bare `except Exception: pass` swallows all archive-read errors, hiding real corruption

**File:** `scripts/report_coverage.py:97-98, 156-157, 176-177, 178-179, 203-209, 214-215`; `tests/...` (none)
**Issue:** Every archive reader (`_try_read_yu_zip`, `_try_read_tar_genes`, the h5ad/h5 loop in `_read_genes_for_slug`) wraps reads in bare `except Exception: pass`/`continue`. A genuinely corrupt or truncated download (the exact failure Halt Gate 1 exists to catch) is indistinguishable from "format not recognized": both surface as `genes is None`, which the report records as `N/A (no readable matrix)` and explicitly does **not** fire the gate (report_coverage.py:255-269). So a dataset that downloaded as a broken tarball silently passes the coverage gate as "N/A". This weakens the very gate the file is meant to enforce.

**Fix:** Log the exception (`log.warning("failed reading %s: %s", path, exc)`) instead of swallowing it, and distinguish "unreadable archive present" from "no archive present" so the former can be surfaced rather than treated as benign N/A.

### WR-04: Decompression-bomb / unbounded nested-archive reads on untrusted downloads

**File:** `scripts/report_coverage.py:140-177` (`_try_read_tar_genes` nested `.tar.gz` and `.zip` members read fully into memory via `f.read()` / `zip_data = f.read()`)
**Issue:** The reviewer brief flags untrusted-download handling. These readers call `.read()` on nested archive members with no size cap, then re-open them in-memory. A maliciously crafted (or merely pathological) nested archive can exhaust memory. While `extractfile`/`z.open` avoid path-traversal-to-disk (good — no `extractall`), the unbounded in-memory inflation is a robustness/DoS concern on bytes fetched over the network. The `[:5]` cap on zip members (line 161) is an ad-hoc partial guard that does not bound per-member size.

**Fix:** Cap member sizes before reading (`if member.size > MAX_MEMBER_BYTES: continue`), and prefer streaming the h5 member to a temp file over `io.BytesIO` for large matrices.

### WR-05: `_download_url` fetches the SAME URL for every expected file

**File:** `scripts/download_spatial.py:394-420`
**Issue:** `primary_url = urls[0]` and then the loop downloads `primary_url` into each `expected_name`. If a `url`-mechanism entry ever has more than one expected file, all of them get the contents of the first URL written under different names — silent data corruption. Currently all `url`-mechanism entries have empty `expected_files`, so the loop body never runs, but this is a latent multi-file bug.

**Fix:** Require a 1:1 filename→URL mapping for multi-file `url` entries, or assert `len(entry.expected_files) <= 1` for the `url` mechanism and document the restriction.

### WR-06: `build_orthologs.py` BioMart error sniff produces false positives

**File:** `scripts/build_orthologs.py:231`
**Issue:** `if body.lower().startswith("query error") or "error" in body[:100].lower():` will misclassify a perfectly valid TSV as a BioMart error whenever a legitimate gene symbol or Ensembl description containing the substring "error" appears in the first 100 characters of the response (e.g. a gene named/annotated with "...error..."). That would write HALT_REASON.md and abort on good data. The substring search over the first 100 chars of a headerless TSV is too blunt.

**Fix:** Detect BioMart errors structurally — BioMart emits errors as a line beginning with `Query ERROR:`. Match that explicitly (case-insensitive, anchored to a line start) rather than scanning for the substring "error":
```python
if re.search(r"(?im)^query error", body):
    _write_halt_reason(...)
```

### WR-07: `datetime.datetime.utcnow()` is deprecated and emits a DeprecationWarning on 3.12

**File:** `scripts/build_orthologs.py:109, 185`
**Issue:** The active interpreter is Python 3.12, where `datetime.datetime.utcnow()` is deprecated. It still works but produces a naive timestamp and a deprecation warning; the `Z` suffix on line 109 is appended to a naive ISO string, implying UTC without actually carrying tzinfo.

**Fix:** Use timezone-aware UTC:
```python
from datetime import datetime, timezone
datetime.now(timezone.utc).isoformat()                    # for the .meta / HALT timestamp
datetime.now(timezone.utc).strftime("%Y%m%d")             # for the filename stub
```

### WR-08: MANIFEST completeness test under-checks — validation anchors and rodent-context slugs are not required

**File:** `tests/test_data_paths.py:256-259`
**Issue:** `test_manifest_complete` only requires `d.slug` in MANIFEST.md when `d.usable_as_input` is True. The APAP validation anchors (`gse280652_apap_liver`, `gse272564_apap_liver`) have `usable_as_input=False` but are downloaded (per `_should_fetch`) and have SHA-bearing files on disk. They are therefore exempt from the MANIFEST check despite being real tracked data — a gap in DATA-02 provenance coverage. The download driver and the MANIFEST test disagree about which datasets are "real downloads."

**Fix:** Mirror the driver's `_should_fetch`/validation-anchor predicate in the test so every actually-downloaded dataset must appear in the MANIFEST:
```python
is_anchor = d.slug.startswith("gse280652_") or d.slug.startswith("gse272564_apap")
if d.usable_as_input or is_anchor:
    if d.slug not in text:
        missing_from_manifest.append(d.slug)
```

### WR-09: SIDER frequency file existence-check uses HEAD but the "already present" branch skips the freq fetch entirely

**File:** `scripts/download_labels.py:352-368`
**Issue:** Minor logic asymmetry: if `meddra_freq_dest` already exists the code logs "already on disk" (good), but the `meddra_all_se` required file (lines 346-350) has no SHA verification at all, unlike the liver labels which verify SHA-256 against a pinned digest. SIDER and the FDA Excel files are both "required, non-deferred" per the docstring, yet only the liver set has an integrity check. A truncated SIDER download (non-zero but partial) passes. Given the hard-rule emphasis on integrity, the asymmetry is a defect.

**Fix:** Pin and verify a SHA-256 for `meddra_all_se.tsv.gz` after download (record it in MANIFEST.md), matching the DILIst/DILIrank pattern, or document why SIDER is exempt.

## Info

### IN-01: Unused imports in `compute_sha256.py`

**File:** `scripts/compute_sha256.py:18`
**Issue:** `import os` is never used (path handling is all via `pathlib.Path`).
**Fix:** Remove the unused `os` import.

### IN-02: `compute_sha256.py` warning text says "file/directory" but the branch only triggers for neither

**File:** `scripts/compute_sha256.py:77`
**Issue:** The message `"{p} does not exist or is not a file/directory"` is fine, but a symlink-to-dir or special file falls through silently. Minor; cosmetic.
**Fix:** Optionally log skipped non-regular entries explicitly.

### IN-03: Duplicate GSE272564 download into two directories

**File:** `src/spatial/datasets.py:333-347, 355-368`
**Issue:** `gse272564_apap_liver` and `gse272564_mouse_liver_ctrl` both resolve to accession GSE272564 with identical `expected_files=("GSE272564_RAW.tar",)`. The driver downloads the same multi-GB tar twice into two slug directories. Not incorrect, but wasteful and easy to mistake for two distinct datasets.
**Fix:** Document the intentional dual-purpose use, or symlink/dedupe the shared RAW.tar.

### IN-04: `_extract_gse_ids` imports `re` inside the function

**File:** `scripts/download_spatial.py:249`
**Issue:** `import re` is local to the function. Harmless, but inconsistent with the module's top-level import style and re-imports on every call.
**Fix:** Move `import re` to module top.

### IN-05: `OrthologTable.pairs` typed as `object` defeats type checking

**File:** `src/spatial/orthology.py:86`
**Issue:** `pairs: object` (with a comment about NamedTuple compat) means static checkers cannot verify `.pairs` is a DataFrame. NamedTuple supports `pd.DataFrame` as a field annotation directly.
**Fix:** Annotate `pairs: pd.DataFrame` (it is already imported); the "NamedTuple compat" concern does not apply to runtime behavior.

### IN-06: Coverage gate uses strict `>` 0.80 in code but docstrings/MANIFEST say "> 0.80" inconsistently with the `<= 0.80` halt branch

**File:** `scripts/report_coverage.py:287-288`; `tests/test_data_paths.py:228`
**Issue:** Pass condition is `cov > 0.80` and halt is `cov <= 0.80`, so a dataset with coverage of exactly 0.80 fails the gate. The prose "Coverage > 0.80 required" matches the code, but elsewhere the threshold is described loosely as "≥15k genes / near-100%". The exact-boundary semantics (0.80 fails) should be stated once and consistently.
**Fix:** State explicitly that coverage must strictly exceed 0.80 (0.80 is a FAIL) wherever the gate is documented, so the boundary is unambiguous.

---

_Reviewed: 2026-06-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
