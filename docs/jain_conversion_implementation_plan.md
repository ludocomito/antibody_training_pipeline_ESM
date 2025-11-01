# Jain Dataset Excel-to-CSV Conversion Implementation Plan

**Issue:** #2 - Jain dataset preprocessing
**Author:** Ray (ray/learning branch)
**Date:** 2025-11-01
**Status:** Phase 1 (Excel-to-CSV conversion) – ✅ Completed, awaiting PR integration

---

## Executive Summary

### What is the Jain Dataset?
The Jain dataset (Jain et al. 2017, PNAS) contains **137 clinical-stage IgG1-formatted antibodies** evaluated for non-specificity using **ELISA with a panel of common antigens** (ssDNA, dsDNA, insulin, LPS, cardiolipin, KLH).

### Role in the Pipeline
The Jain dataset is an **EXTERNAL TEST SET ONLY** - it is NOT used for training.

**Training vs Testing:**
- **Training data:** Boughter dataset (>1000 antibodies, ELISA-based labels)
- **Test sets:**
  - **Jain (137 clinical antibodies, ELISA)** ← We are here
  - Shehata (398 antibodies, PSR assay)
  - Harvey (140K nanobodies, PSR assay)

**Why test on Jain?**
1. **External validation** of models trained on Boughter
2. **Clinical relevance**: Therapeutic antibodies at various clinical stages
3. **Same assay as training data**: ELISA (vs PSR for Shehata/Harvey)
4. **Balanced dataset**: Similar distribution to Boughter (specific/mildly non-specific/non-specific)

From Sakhnini paper (Section 2.1):
> "the most balanced dataset (i.e. Boughter one) was selected for training of ML models, while the remaining three (i.e. Jain, Shehata and Harvey, which consists exclusively of VHH sequences) were used for testing."

---

## Data Provenance Issue Discovered

### The Problem:
The existing `test_datasets/jain.csv` in the repository:
- Contains only **80 antibodies** (should be 137)
- Has different SMP values than PNAS supplementary files:
  - Example: abituzumab `smp` in `jain.csv`: 0.126
  - PNAS SD03 `PSR SMP Score`: 0.167
- 29 antibodies in `jain.csv` are NOT in PNAS files
- Unknown origin - may have been incorrectly converted or from different source

### The Solution:
Convert the **authoritative PNAS supplementary files** directly:
1. `pnas.1616408114.sd01.xlsx` - Metadata (137 antibodies)
2. `pnas.1616408114.sd02.xlsx` - VH/VL sequences (137 antibodies)
3. `pnas.1616408114.sd03.xlsx` - Biophysical measurements (139 entries, 137 match)

Create `jain_v2.csv` (or replace `jain.csv`) with correctly sourced data.

---

## Paper's Preprocessing Procedure (from Sakhnini et al. 2025)

### Labeling System (Methods Section 4.3, Line 236):

**ELISA Flag-Based Labels:**
From Section 2.2 (Line 55):
> "the Boughter dataset was first parsed into two groups: specific (0 flags) and non-specific group (>3 flags), leaving out the mildly non-specific antibodies (1-3 flags)"

**For Jain dataset** (same labeling logic):
- **Class 0 (Specific)**: 0 ELISA flags
- **Class 1 (Non-specific)**: >3 ELISA flags (≥4 flags)
- **Excluded from training**: 1-3 flags (mildly non-specific)

**ELISA Panel (6 ligands):**
- ssDNA (single-stranded DNA)
- dsDNA (double-stranded DNA)
- Insulin
- LPS (lipopolysaccharide)
- Cardiolipin
- KLH (keyhole limpet hemocyanin)

**Flag Counting:**
- Each antibody tested against all 6 ligands
- Each positive result = 1 flag
- Total flags range: 0 (perfectly specific) to 6 (highly non-specific)

### Required Output Format:

Based on existing `jain.csv` structure:
```csv
id,heavy_seq,light_seq,label,source,smp,ova
abituzumab,QVQLQQSGGE...,DIQMTQSPSS...,0,jain2017,0.166666,1.137375
```

**Column Definitions (planned for v2):**
- `id`: Antibody name (SD0X `Name`)
- `heavy_seq`: VH sequence (SD02 `VH`)
- `light_seq`: VL sequence (SD02 `VL`)
- `flags_total`: Integer count 0–4 derived from Table 1 thresholds
- `flag_category`: One of `specific`, `mild`, `non_specific`
- `label`: Binary label used by Sakhnini pipeline
  - 0 for `specific`
  - 1 for `non_specific`
  - `NaN` for `mild` (explicitly excluded downstream)
- `source`: Literal `"jain2017"`
- `smp`: PSR SMP score (SD03 `Poly-Specificity Reagent (PSR) SMP Score (0-1)`)
- `ova`: **ELISA fold-over-background** (SD03 `ELISA`) — retained to preserve prior schema (`ova` ≈ ovalbumin ligand in original flag panel)
- `bvp_elisa`: SD03 `BVP ELISA` (new column to make flag calculation auditable)
- `acsins`, `csi`, `cic`, `hic`, `smac`, `sgac_sins`, `as_slope`: optional passthrough columns so validations can be rerun without re-reading Excel (see conversion script design)
- `heavy_seq_length` / `light_seq_length`: optional QC metrics (remove if file size becomes unwieldy)

### ✅ RESOLVED: ELISA Flag Derivation

**BREAKTHROUGH (2025-11-01):** Successfully recovered the complete flag derivation methodology from Jain et al. 2017 PNAS paper (Table 1, converted to markdown at `literature/markdown/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape/`).

#### Flag Derivation Methodology (Table 1)

**Threshold Calculation:**
- Thresholds are set at the **90th percentile of APPROVED antibodies** (48 molecules) to capture the “10% worst values among approved drugs” (Lipinski-style heuristic).
- Within each assay cluster, **any** metric breaching its threshold triggers a **single flag** for that cluster.
- SGAC-SINS is the only assay where **lower** values are undesirable (`< 370 mM` raises the flag); all other assays are flagged when **greater** than the limit.

**Assay Clusters (maximum 4 flags per antibody):**

1. **Self-Interaction / Cross-Interaction**  
   | Assay | Threshold | Units | Flag condition |
   |-------|-----------|-------|----------------|
   | PSR SMP | 0.27 ± 0.06 | unitless | `>` |
   | AC-SINS | 11.8 ± 6.2 | Δλ (nm) | `>` |
   | CSI-BLI | 0.01 ± 0.02 | response units | `>` |
   | CIC | 10.1 ± 0.5 | min | `>` |
   **Cluster flag:** 1 if **any** of {PSR, AC-SINS, CSI-BLI, CIC} exceed threshold.

2. **Chromatography / Salt Stress**  
   | Assay | Threshold | Units | Flag condition |
   |-------|-----------|-------|----------------|
   | HIC | 11.7 ± 0.6 | min | `>` |
   | SMAC | 12.8 ± 1.2 | min | `>` |
   | SGAC-SINS | 370 ± 133 | mM | `<` (lower salt tolerance is bad) |
   **Cluster flag:** 1 if **any** of {HIC, SMAC} exceed or SGAC-SINS falls below threshold.

3. **Polyreactivity / Plate Binding**  
   | Assay | Threshold | Units | Flag condition |
   |-------|-----------|-------|----------------|
   | ELISA | 1.9 ± 1.0 | fold-over-background | `>` |
   | BVP ELISA | 4.3 ± 2.2 | fold-over-background | `>` |
   **Cluster flag:** 1 if ELISA > 1.9 **or** BVP > 4.3.

4. **Accelerated Stability (AS)**  
   | Assay | Threshold | Units | Flag condition |
   |-------|-----------|-------|----------------|
   | AS (SEC slope) | 0.08 ± 0.03 | % monomer loss per day | `>` |
   **Cluster flag:** 1 if AS exceeds threshold.

**Label Mapping for Sakhnini-style evaluation:**
- `flags == 0` → Specific (label = 0)
- `flags >= 4` → Non-specific (label = 1)
- `flags in {1,2,3}` → Mildly non-specific (keep in CSV with `label = 0` or NaN, but filter out before binary evaluation as Sakhnini does)

**Provenance Links:**
- Main paper markdown: `literature/markdown/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape.md`
- Supporting info PDF: `literature/pdf/pnas.201616408si.pdf`
- Threshold confirmation script snapshots stored under `docs/` (see Data Cleaning Log, forthcoming).

---

## Implementation Plan

### Phase 1: Excel-to-CSV Conversion

**Goal:** Convert PNAS supplementary files into `test_datasets/jain_v2.csv`

**Input files:**
- `test_datasets/pnas.1616408114.sd01.xlsx` (metadata)
- `test_datasets/pnas.1616408114.sd02.xlsx` (sequences)
- `test_datasets/pnas.1616408114.sd03.xlsx` (biophysical properties)

**Output file:**
- `test_datasets/jain_v2.csv` (or `jain.csv` replacement)

**Script:** `scripts/convert_jain_excel_to_csv.py`

#### Step-by-Step Process:

1. **Load all three Excel files**
   ```python
   sd01 = pd.read_excel('test_datasets/pnas.1616408114.sd01.xlsx')  # Metadata
   sd02 = pd.read_excel('test_datasets/pnas.1616408114.sd02.xlsx')  # Sequences
   sd03 = pd.read_excel('test_datasets/pnas.1616408114.sd03.xlsx')  # Properties
   ```

2. **Merge on 'Name' column**
   - SD01 ⋈ SD02 on 'Name' → Get sequences with metadata
   - Result ⋈ SD03 on 'Name' → Add biophysical properties
   - **Expected result:** 137 antibodies (SD03 has 139 entries, 2 are metadata rows)

3. **Extract required columns**
   ```python
   jain_df = pd.DataFrame({
       'id': merged['Name'],
       'heavy_seq': merged['VH'],
       'light_seq': merged['VL'],
       'label': None,  # TBD - need ELISA flag derivation logic
       'source': 'jain2017',
       'smp': merged['Poly-Specificity Reagent (PSR) SMP Score (0-1)'],
       'ova': merged['ELISA']  # Or 'BVP ELISA' or transformed version - TBD
   })
   ```

4. **Compute assay thresholds and flag counts**
   - Load Table 1 thresholds (hard-coded dict in script with provenance comment pointing to markdown path)
   - Normalize column names (strip whitespace, ensure consistent casing)
   - Apply sign corrections:
     ```python
     chromatography_flags = (
         (row['HIC Retention Time (Min)a'] > 11.7)
         or (row['SMAC Retention Time (Min)a'] > 12.8)
         or (row['SGAC-SINS AS100 ((NH4)2SO4 mM)'] < 370)
     )
     ```
   - Count boolean flags across four clusters → `flags_total`
   - Map to category (`specific` / `mild` / `non_specific`) and binary label (0 / NaN / 1)

5. **Assemble final DataFrame**
   - Merge sequences + metadata + flags
   - Insert additional derived columns (`heavy_seq_length`, etc.) for QA
   - Order columns to keep backwards compatibility: `id,heavy_seq,light_seq,label,flags_total,flag_category,source,smp,ova,bvp_elisa,...`

6. **Write CSV**
   - Save to `test_datasets/jain.csv` (overwrite existing) and optionally `jain_raw_joined.csv` for archival
   - Emit console summary: counts per flag category, threshold usage, any dropped antibodies (should be zero after filtering metadata rows)

7. **Document & log conversion**
   - Append SHA256 checksum, row counts, label distribution to `docs/jain_conversion_verification_report.md`
   - Update data cleaning log with any anomalies (missing values, sequences trimmed, etc.)
   - Run `python3 scripts/validate_jain_conversion.py` and capture output for provenance

5. **Data validation**
   - Check for missing sequences
   - Verify all sequences are valid amino acids
   - Check for duplicates
   - Validate lengths (reasonable VH/VL lengths)

6. **Save to CSV**
   ```python
   jain_df.to_csv('test_datasets/jain_v2.csv', index=False)
   ```

7. **Generate conversion report**
   - Document antibody count
   - Document any missing data
   - Document label derivation approach
   - Compare with existing `jain.csv`

### Phase 2: Validation

**Script:** `scripts/validate_jain_conversion.py`

**Validation checks:**
1. **Count validation:** Confirm 137 antibodies
2. **Sequence validation:**
   - All VH/VL sequences present
   - No invalid amino acids
   - Reasonable length distributions
3. **Merge validation:** All SD01/SD02/SD03 names matched
4. **Comparison with existing `jain.csv`:**
   - Which antibodies overlap?
   - Which antibodies are new?
   - Which antibodies are missing?
   - How do SMP values differ?

### Phase 3: Documentation

**Create:**
1. `docs/jain_conversion_verification_report.md` - Validation results
2. `docs/jain_data_cleaning_log.md` - Any manual fixes applied
3. Update `docs/jain_data_sources.md` with final resolution

---

## Open Questions / Blockers

### Critical:
1. **How to derive ELISA flags from PNAS data?**
   - PNAS SD03 has continuous ELISA scores, not discrete flags
   - Sakhnini paper requires flag-based labeling (0, 1-3, ≥4)
   - Need to check original Jain 2017 paper for flag data

2. **What is the 'ova' column?**
   - Existing `jain.csv` has 'ova' but origin unclear
   - Possible mappings: ELISA, BVP ELISA, or derived metric
   - Ranges don't match directly - may need transformation

### Non-critical:
3. Why does existing `jain.csv` have only 80 antibodies instead of 137?
4. Why do SMP values differ between `jain.csv` and PNAS files?

---

## Success Criteria

**Phase 1 Complete when:**
- ✅ `scripts/convert_jain_excel_to_csv.py` successfully runs
- ✅ Generates `test_datasets/jain_v2.csv` with 137 antibodies
- ✅ All sequences validated (no missing, no invalid amino acids)
- ✅ Conversion report documents any assumptions/limitations
- ✅ Validation script confirms data quality

**Phase 2 (Fragment Extraction) will follow** using existing `preprocessing/process_jain.py` (already implemented in PR #17).

---

## References

1. **Jain et al. 2017** - "Biophysical properties of the clinical-stage antibody landscape" PNAS 114(5):944-949. DOI: 10.1073/pnas.1616408114
2. **Sakhnini et al. 2025** - "Prediction of Antibody Non-Specificity using Protein Language Models and Biophysical Parameters"
3. **Boughter et al. 2020** - eLife 9:e61393 (original ELISA flag-based labeling approach)
