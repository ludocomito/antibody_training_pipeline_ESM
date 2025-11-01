# Jain Dataset Data Cleaning Log

**Date:** 2025-11-01  
**Issue:** #2 – Jain dataset preprocessing  
**Files Modified:**  
- `scripts/convert_jain_excel_to_csv.py` (new)  
- `scripts/validate_jain_conversion.py` (new)  
- `test_datasets/jain.csv` (regenerated canonical dataset)

---

## Key Data Quality Findings & Fixes

### 1. Missing Antibodies & Mismatched Metrics (CRITICAL)
- **Symptom:** Legacy `test_datasets/jain.csv` contained only 80 antibodies and SMP/OVA values diverged from PNAS SD03 (e.g., `abituzumab` SMP 0.126 vs 0.167).
- **Resolution:** Rebuilt the dataset from SD01–SD03 using deterministic merges on `Name` (137 antibodies retained).
- **Validation:** `scripts/validate_jain_conversion.py` confirms regenerated DataFrame matches the saved CSV (`assert_frame_equal`).

### 2. Supplementary Metadata Rows (HIGH)
- **Symptom:** SD03 includes two descriptive rows (e.g., “aArbitrarily long RT…”), inflating row count to 139.
- **Resolution:** Filtered SD03 to names present in SD02 sequence table before merging; resulting dataset size matches SD01/SD02 (137).

### 3. Sequence Sanitization (HIGH)
- **Symptom:** VH/VL strings include spaces, tabs, and hyphen alignment artifacts.
- **Resolution:** `sanitize_sequence()` uppercases and removes all characters not in `ACDEFGHIKLMNPQRSTVWYX`; sequences reduced to clean amino acid strings.
- **Outcome:** No VH/VL sequences contain invalid residues (validated via `validate_jain_conversion.py`).

### 4. Developability Flag Calculation (HIGH)
- **Symptom:** No explicit ELISA flag field in supplementary data; Sakhnini pipeline requires `>3` flags for class 1.
- **Resolution:** Implemented Table 1 thresholds from Jain et al. 2017:
  - Four clusters, each contributing 0/1 flag.
  - SGAC-SINS uses `< 370 mM` as adverse criterion; all other metrics flagged on `>` exceeding thresholds.
- **Outcome:** New columns `flag_*`, `flags_total`, `flag_category`, and nullable `label` provide clear audit trail.

### 5. Mild vs Binary Labels (MEDIUM)
- **Symptom:** Sakhnini excludes antibodies with 1–3 flags; legacy CSV flattened them into binary labels.
- **Resolution:** `label` now uses pandas nullable `Int64` (0 for specific, 1 for ≥4 flags, `<NA>` for mild). Downstream code must drop `<NA>` before binary evaluation.

### 6. Column Provenance (LOW)
- **Symptom:** Prior CSV lacked raw assay columns, limiting reproducibility.
- **Resolution:** Retained PSR/AC-SINS/CSI/CIC/HIC/SMAC/SGAC/AS columns (snake_case) alongside clinical metadata from SD01. New columns `ova` (ELISA) and `bvp_elisa` (BVP) preserve Sakhnini-compatible semantics while keeping originals (`elisa_fold`, `bvp_elisa_fold`).

---

## Outstanding Considerations

1. **Class Imbalance:** Only 3 antibodies meet the ≥4 flag criterion. Mild cases (67) should be evaluated carefully—consider reporting AUROC/PR metrics that account for high imbalance.
2. **Clinical Status Context:** Distribution indicates higher flag counts in Phase 2/3 assets; document this in analysis notebooks for traceability.
3. **Fragment Regeneration:** Re-run `preprocessing/process_jain.py` after downstream consumers adapt to the new schema.

---

## Reference Commands

```bash
# Rebuild dataset
python3 scripts/convert_jain_excel_to_csv.py --verbose

# Validate output
python3 scripts/validate_jain_conversion.py
```

Both scripts assume SD01/SD02/SD03 reside in `test_datasets/`. Checksums and summary statistics are recorded in `docs/jain_conversion_verification_report.md`.
