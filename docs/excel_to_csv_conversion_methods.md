# Excel to CSV Conversion Methods & Validation

**Purpose:** Document all available methods for converting supplementary Excel files (Shehata & Jain datasets) to CSV with validation.

**Last Updated:** 2025-11-01  
**Related Issues:** #3 (Shehata), #2 (Jain)

---

## Overview of Methods

### Method 1a: Python Script – Shehata (RECOMMENDED) ⭐
- **Tool:** `scripts/convert_shehata_excel_to_csv.py`
- **Pros:** Full control, validation built-in, reproducible, transparent
- **Cons:** Requires Python environment
- **Validation:** Multi-method cross-checking with `validate_shehata_conversion.py`

### Method 1b: Python Script – Jain (RECOMMENDED) ⭐
- **Tool:** `scripts/convert_jain_excel_to_csv.py`
- **Pros:** Deterministic flag derivation (Table 1 thresholds), full provenance columns, label handling consistent with Sakhnini et al.
- **Cons:** Requires Python environment
- **Validation:** `scripts/validate_jain_conversion.py` (rebuilds pipeline & checks SHA256)

### Method 2: CLI Tools
- **Tools:** `in2csv`, `ssconvert`, `xlsx2csv`
- **Pros:** Simple one-liners, no coding
- **Cons:** Less control, limited validation, may lose data fidelity

### Method 3: Excel GUI Export
- **Tool:** Microsoft Excel, LibreOffice Calc
- **Pros:** Visual inspection, familiar interface
- **Cons:** Manual process, not reproducible, error-prone

---

## Method 1a: Shehata Python Script (DETAILED)

### Installation

```bash
# Already have pandas and openpyxl from earlier
pip show pandas openpyxl

# If needed:
pip install pandas openpyxl
```

### Usage (Shehata)

```bash
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM

# Run conversion script (interactive)
python3 scripts/convert_shehata_excel_to_csv.py
```

**Interactive prompts:**
1. Shows PSR score distribution
2. Suggests threshold based on paper (7/398 non-specific)
3. Asks for confirmation or custom threshold
4. Validates sequences
5. Saves to `test_datasets/shehata.csv`

### Validation (Shehata)

```bash
# Run multi-method validation
python3 scripts/validate_shehata_conversion.py
```

**What it checks:**
- ✓ Reads Excel with pandas (openpyxl)
- ✓ Reads Excel with openpyxl directly
- ✓ Compares both methods (ensures Excel reading is correct)
- ✓ Compares CSV with original Excel (ensures conversion is correct)
- ✓ Validates sequences (VH Protein → heavy_seq, VL Protein → light_seq)
- ✓ Validates ID mapping (Clone name → id)
- ✓ Calculates checksums for integrity
- ✓ Reports summary statistics

### Output Example (Shehata)

```csv
id,heavy_seq,light_seq,label,psr_score,b_cell_subset,source
ADI-38502,EVQLLESGGGLVKPGGSLRLSCAASGFIFSDYSMNWVRQAPGKGLEWVS...,DIVMTQSPSTLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYK...,0,0.0,IgG memory,shehata2019
ADI-38501,EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVS...,DIVMTQSPATLSLSPGERATLSCRASQSISTYLAWYQQKPGQAPRLLIY...,0,0.023184,IgG memory,shehata2019
```

---

## Method 1b: Jain Python Script (DETAILED)

### Usage (Jain)

```bash
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM

# Generate canonical Jain dataset
python3 scripts/convert_jain_excel_to_csv.py --verbose
```

**What it does:**
1. Loads SD01/SD02/SD03 spreadsheets (metadata, VH/VL sequences, biophysical assays)
2. Sanitizes amino acid sequences (removes gaps/whitespace/non-standard residues)
3. Applies Jain Table 1 thresholds (four developability flag clusters)
4. Emits `test_datasets/jain.csv` with explicit `flags_total`, `flag_category`, nullable `label`, and supporting assay columns

### Validation (Jain)

```bash
python3 scripts/validate_jain_conversion.py
```

**Checks performed:**
- Re-runs the conversion pipeline in-memory and asserts equality with the CSV (`assert_frame_equal`)
- Reports flag/label distributions (specific 67 / mild 67 / non_specific 3)
- Confirms VH/VL sequences contain only valid residues (`ACDEFGHIKLMNPQRSTVWYX`)
- Prints Table 1 threshold clauses and SHA256 checksum (`b1a6d7399260aef1a894743877a726caa248d12d948b8216822cb2a5b9bc96a3`)

### Output Example (Jain)

```csv
id,heavy_seq,light_seq,flags_total,flag_category,label,flag_self_interaction,flag_chromatography,flag_polyreactivity,flag_stability,source,smp,ova,bvp_elisa,...
abituzumab,QVQLQQSGGELAKPGASVKVSCKASGYTFSSFWMHWVRQAPGQGLEWIGYINPRSGYTEYNEIFRDKATMTTDTSTSTAYMELSSLRSEDTAVYYCASFLGRGAMDYWGQGTTVTVSS,DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYYTSKIHSGVPSRFSGSGSGTDYTFTISSLQPEDIATYYCQQGNTFPYTFGQGTKVEIK,1,mild,,False,False,True,False,jain2017,0.166666,1.137375,2.720799,...
```

`label` uses pandas nullable integers: `0` for specific, `1` for ≥4 flags (non-specific), blank for mild (1–3 flags).

---

## Method 2: CLI Tools

### Option A: in2csv (csvkit)

**Install:**
```bash
pip install csvkit
```

**Usage:**
```bash
# Convert single sheet
in2csv test_datasets/mmc2.xlsx > test_datasets/mmc2_raw.csv

# Specify sheet
in2csv --sheet "Sheet1" test_datasets/mmc2.xlsx > test_datasets/mmc2_raw.csv
```

**Pros:**
- Simple one-liner
- Part of csvkit suite (useful for CSV manipulation)

**Cons:**
- No column mapping (need post-processing)
- No label conversion
- No validation

### Option B: ssconvert (Gnumeric)

**Install (macOS):**
```bash
brew install gnumeric
```

**Usage:**
```bash
ssconvert test_datasets/mmc2.xlsx test_datasets/mmc2_raw.csv
```

**Pros:**
- Fast
- Reliable for simple conversions

**Cons:**
- Requires Gnumeric installation (large dependency)
- No Python integration
- No validation

### Option C: xlsx2csv

**Install:**
```bash
pip install xlsx2csv
```

**Usage:**
```bash
xlsx2csv test_datasets/mmc2.xlsx test_datasets/mmc2_raw.csv
```

**Pros:**
- Lightweight
- Pure Python

**Cons:**
- Basic functionality
- No post-processing

### After CLI Conversion: Post-Processing Needed

```python
# Read raw CSV from CLI tool
df_raw = pd.read_csv('test_datasets/mmc2_raw.csv')

# Map to jain.csv format
df_processed = pd.DataFrame({
    'id': df_raw['Clone name'],
    'heavy_seq': df_raw['VH Protein'],
    'light_seq': df_raw['VL Protein'],
    'label': (df_raw['PSR Score'] > threshold).astype(int),
    'psr_score': df_raw['PSR Score'],
    'b_cell_subset': df_raw['B cell subset'],
    'source': 'shehata2019'
})

df_processed.to_csv('test_datasets/shehata.csv', index=False)
```

---

## Method 3: Excel GUI Export

### Steps

1. Open `test_datasets/mmc2.xlsx` in Excel or LibreOffice
2. File → Save As → CSV (UTF-8)
3. Save as `mmc2_raw.csv`
4. Run post-processing script (same as CLI method above)

**Cons:**
- Manual, error-prone
- Not reproducible
- Hard to document
- **NOT RECOMMENDED** for scientific work

---

## Validation Strategy: Multi-Layer Approach

### Layer 1: Internal Consistency
- Read Excel with multiple libraries (pandas, openpyxl)
- Compare results to ensure Excel reading is correct

### Layer 2: Conversion Accuracy
- Compare CSV output with original Excel
- Validate every sequence (VH, VL)
- Check ID mapping
- Verify label conversion logic

### Layer 3: Format Compatibility
- Compare with `jain.csv` structure
- Test loading with `data.load_local_data()`
- Ensure column names match expected format

### Layer 4: Statistical Validation
- Check row counts (should be 398-402)
- Verify label distribution (7/398 non-specific per paper)
- Validate sequence lengths (reasonable range)
- Check for missing data

### Layer 5: File Integrity
- Calculate checksums (SHA256)
- Store for future verification
- Detect any corruption or modification

---

## Comparison of Methods

| Method | Effort | Control | Validation | Reproducible | Recommended |
|--------|--------|---------|------------|--------------|-------------|
| **Python script** | Low (run script) | High | Built-in | ✓ Yes | ⭐ **YES** |
| **in2csv** | Low (one-liner) | Low | Manual | ✓ Yes | Only if Python unavailable |
| **ssconvert** | Low | Low | Manual | ✓ Yes | No (large dependency) |
| **xlsx2csv** | Low | Low | Manual | ✓ Yes | No (same as in2csv but worse) |
| **Excel GUI** | Medium | Low | None | ✗ No | ❌ **NO** |

---

## Recommended Workflow

### Step 1: Convert
```bash
cd /Users/ray/Desktop/CLARITY-DIGITAL-TWIN/antibody_training_pipeline_ESM
python3 scripts/convert_shehata_excel_to_csv.py
```

### Step 2: Validate
```bash
python3 scripts/validate_shehata_conversion.py
```

### Step 3: Compare with Jain
```bash
# Check format matches
head -n 3 test_datasets/jain.csv
head -n 3 test_datasets/shehata.csv

# Check column compatibility
python3 -c "
import pandas as pd
jain = pd.read_csv('test_datasets/jain.csv')
shehata = pd.read_csv('test_datasets/shehata.csv')
print('Jain columns:', list(jain.columns))
print('Shehata columns:', list(shehata.columns))
print('Common columns:', set(jain.columns) & set(shehata.columns))
"
```

### Step 4: Integration Test
```bash
# Test with existing data pipeline
python3 -c "
from data import load_local_data

# Load Shehata
X, y = load_local_data(
    'test_datasets/shehata.csv',
    sequence_column='heavy_seq',
    label_column='label'
)
print(f'Loaded {len(X)} sequences, {len(y)} labels')
print(f'Label distribution: {list(zip(*np.unique(y, return_counts=True)))}')
"
```

---

## Handling Potential Issues

### Issue 1: PSR Threshold Uncertainty

**Problem:** Paper says "7/398 non-specific" but doesn't specify exact threshold

**Solutions:**
1. **Percentile-based:** Use 98.24th percentile (7/398 = 1.76%)
2. **Literature search:** Check Shehata et al. 2019 original paper
3. **Conservative:** Use PSR > 0 (any polyreactivity)
4. **Ask maintainer:** Contact @ludocomito or paper authors

**Implemented:** Script calculates percentile and prompts user to confirm

### Issue 2: Missing Data

**Problem:** mmc2.xlsx has 402 rows but paper reports 398

**Solutions:**
1. Check if 4 extras are controls/outliers
2. Filter based on PSR score availability
3. Document the discrepancy
4. Keep all 402 unless paper specifies exclusion criteria

**Implemented:** Script reports row count and missing data

### Issue 3: Sequence Validation Failures

**Problem:** Invalid amino acids or corrupted sequences

**Solutions:**
1. Check against valid AA alphabet (ACDEFGHIKLMNPQRSTVWY)
2. Compare multiple Excel reading methods
3. Manual spot-check suspicious sequences
4. Report all invalid sequences for review

**Implemented:** Validation script checks AA validity

---

## External Verification (Using Multiple Agents)

### Approach 1: Use Task Tool with Multiple Agents

```python
# Launch multiple agents in parallel to verify conversion
# Agent 1: Read Excel and report statistics
# Agent 2: Read CSV and report statistics
# Agent 3: Compare and validate

# Then cross-check their consensus
```

### Approach 2: Use Different Python Environments

```bash
# Environment 1: conda with pandas 1.x
conda create -n verify1 pandas=1.5 openpyxl
conda activate verify1
python scripts/validate_shehata_conversion.py > verify1.log

# Environment 2: venv with pandas 2.x
python3 -m venv verify2
source verify2/bin/activate
pip install pandas openpyxl
python scripts/validate_shehata_conversion.py > verify2.log

# Compare logs
diff verify1.log verify2.log
```

### Approach 3: Independent Tool Verification

```bash
# Method 1: Python script
python3 scripts/convert_shehata_excel_to_csv.py

# Method 2: in2csv + manual processing
in2csv test_datasets/mmc2.xlsx > mmc2_in2csv.csv

# Method 3: R (if available)
Rscript -e "
library(readxl)
library(readr)
df <- read_excel('test_datasets/mmc2.xlsx')
write_csv(df, 'mmc2_R.csv')
"

# Compare all three CSVs
# They should have identical VH/VL sequences
```

---

## Checksums for Verification

After running conversion, store these checksums:

```bash
# Excel source file
shasum -a 256 test_datasets/mmc2.xlsx

# Generated CSV
shasum -a 256 test_datasets/shehata.csv

# Store in docs/checksums.txt for future verification
```

**Expected checksums** (after first conversion):
```
# To be filled after first successful conversion
# mmc2.xlsx: <SHA256>
# shehata.csv: <SHA256>
```

---

## Documentation Requirements

After conversion, document:

1. **Method used:** Python script / CLI tool / GUI
2. **PSR threshold:** Exact value and rationale
3. **Row count:** 398 or 402, and why
4. **Exclusions:** Any sequences excluded and reason
5. **Validation results:** Pass/fail for all checks
6. **Checksums:** SHA256 for source and output
7. **Date:** When conversion was performed
8. **Reviewer:** Who validated the conversion

**Template:** See `docs/shehata_conversion_log.md` (to be created after conversion)

---

## Next Steps

1. ✅ Scripts created (`convert_shehata_excel_to_csv.py`, `validate_shehata_conversion.py`)
2. 🔲 Run conversion interactively
3. 🔲 Review and approve PSR threshold
4. 🔲 Run validation checks
5. 🔲 Compare with jain.csv format
6. 🔲 Test with existing pipeline
7. 🔲 Document conversion in log file
8. 🔲 Commit shehata.csv to repository

---

## Conclusion

**Recommended approach:** Use Python scripts (Method 1) with multi-layer validation.

**Why:**
- ✓ Transparent and reproducible
- ✓ Built-in validation
- ✓ Follows best practices for scientific data
- ✓ Easy to review and audit
- ✓ No manual steps
- ✓ Generates comprehensive logs

**Ready to run:** `python3 scripts/convert_shehata_excel_to_csv.py`
