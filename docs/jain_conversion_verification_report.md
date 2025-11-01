# Jain Dataset Conversion – Verification Report

**Date:** 2025-11-01  
**Issue:** #2 – Jain dataset preprocessing  
**Status:** ✅ Conversion validated

---

## Executive Summary

- ✅ `scripts/convert_jain_excel_to_csv.py` reproduces the canonical Jain dataset (137 antibodies) directly from PNAS supplementary files SD01–SD03.
- ✅ `scripts/validate_jain_conversion.py` reconverts the Excel files in-memory and matches the saved CSV byte-for-byte (SHA256 `b1a6d7399260aef1a894743877a726caa248d12d948b8216822cb2a5b9bc96a3`).
- ✅ Table 1 thresholds (Jain et al. 2017) are implemented as four developability flag clusters with full audit columns.
- ✅ VH/VL sequences are sanitized to contain only IMGT-valid amino acids; no gap or whitespace artifacts remain.
- ⚠️ Flag distribution is highly imbalanced (3 antibodies classified as `non_specific` with ≥4 flags, 67 mild cases with 1–3 flags). Downstream evaluation must exclude mild cases as in Sakhnini et al. 2025.

---

## Source Files

| File | Description |
|------|-------------|
| `test_datasets/pnas.1616408114.sd01.xlsx` | Metadata (clinical status, provenance) |
| `test_datasets/pnas.1616408114.sd02.xlsx` | VH/VL protein sequences |
| `test_datasets/pnas.1616408114.sd03.xlsx` | Biophysical measurements & assays |
| `literature/markdown/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape/...` | Markdown-converted main paper (Table 1 thresholds) |
| `literature/pdf/pnas.201616408si.pdf` | Supporting information (assay grouping narratives) |

---

## Output Artifacts

| File | Purpose |
|------|---------|
| `test_datasets/jain.csv` | Cleaned dataset with flags, labels, and supporting assays |
| `scripts/convert_jain_excel_to_csv.py` | Deterministic Excel→CSV converter |
| `scripts/validate_jain_conversion.py` | Integrity + provenance validation |

### Column Overview (ordered)

```
id, heavy_seq, light_seq,
flags_total, flag_category, label,
flag_self_interaction, flag_chromatography, flag_polyreactivity, flag_stability,
source, smp, ova, bvp_elisa,
psr_smp, acsins_dlmax_nm, csi_bli_nm, cic_min,
hic_min, smac_min, sgac_sins_mM, as_slope_pct_per_day,
hek_titer_mg_per_L, fab_tm_celsius,
heavy_seq_length, light_seq_length,
... (SD01 provenance fields retained for audit)
```

- `smp` is sourced from SD03 PSR SMP scores.
- `ova` is the SD03 ELISA fold-over-background metric (aligns with Sakhnini nomenclature).
- `flags_total` (0–4) and per-cluster booleans derive from Table 1 thresholds.
- `label` uses nullable `Int64`: 0 (specific), 1 (non-specific), `<NA>` (mild; to be filtered externally).

---

## Threshold Implementation (Table 1, Jain et al. 2017)

| Cluster (`flag_*`) | Assays | Threshold | Direction |
|--------------------|--------|-----------|-----------|
| `flag_self_interaction` | PSR SMP, AC-SINS Δλ, CSI-BLI, CIC | 0.27, 11.8 nm, 0.01 RU, 10.1 min | `>` (any breach triggers flag) |
| `flag_chromatography` | HIC, SMAC, SGAC-SINS | 11.7 min, 12.8 min, 370 mM | HIC/SMAC `>`, SGAC-SINS `<` |
| `flag_polyreactivity` | ELISA, BVP | 1.9, 4.3 fold-over-background | `>` |
| `flag_stability` | Accelerated stability (SEC slope) | 0.08 % monomer loss/day | `>` |

`flags_total` = sum of cluster booleans (max 4).  
Label mapping: 0 flags → specific; 1–3 flags → mild; 4 flags → non-specific.

---

## Validation Highlights

Run: `python3 scripts/validate_jain_conversion.py`

| Check | Result |
|-------|--------|
| Row/column count | 137 rows × 36 columns |
| CSV vs regenerated DF | ✅ `pandas.testing.assert_frame_equal` |
| Flag distribution | specific 67 / mild 67 / non_specific 3 |
| Label distribution | 0→67, `<NA>`→67, 1→3 |
| Sequence audit | ✅ All VH/VL sequences strictly in `ACDEFGHIKLMNPQRSTVWYX` |
| SHA256 checksum | `b1a6d7399260aef1a894743877a726caa248d12d948b8216822cb2a5b9bc96a3` |

Notable antibodies (≥4 flags / label 1): **bimagrumab**, **briakinumab**, **cixutumumab**.

---

## Next Steps

1. Update downstream preprocessing / evaluation to ignore mild (`label` = `<NA>`) entries when performing binary classification, mirroring Sakhnini et al. 2025.
2. Regenerate fragment CSVs via `python3 preprocessing/process_jain.py` once PR is updated to the new canonical dataset.
3. Append these validation results to future PR descriptions for traceability.
