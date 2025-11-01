# Shehata Dataset Data Sources

## Raw Data Files

The Shehata dataset preprocessing requires the following Excel files from the paper's supplementary materials:

### Required Files

| File | Description | Source |
|------|-------------|--------|
| `mmc2.xlsx` | Shehata antibody sequences | [Sakhnini et al. 2025 Supplementary Materials](https://doi.org/10.1016/j.cell.2024.12.025) |
| `mmc3.xlsx` | Additional metadata | Same source |
| `mmc4.xlsx` | Additional metadata | Same source |
| `mmc5.xlsx` | Additional metadata | Same source |

### Download Instructions

1. Visit the paper's Cell journal page
2. Navigate to "Supplementary Materials"
3. Download `mmc2.xlsx` through `mmc5.xlsx`
4. Place them in `test_datasets/` directory

**Note:** These Excel files are NOT committed to git (see `.gitignore`). They must be downloaded manually from the paper's supplementary materials.

### Paper Reference

**Sakhnini, A., et al. (2025).** "Antibody Non-Specificity Prediction using Protein Language Models and Biophysical Features." *Cell*.

DOI: https://doi.org/10.1016/j.cell.2024.12.025

### Processed Data

After downloading the raw Excel files, run the preprocessing scripts to generate the cleaned CSV files:

```bash
# Phase 1: Convert Excel to CSV
python3 scripts/convert_shehata_excel_to_csv.py

# Phase 2: Extract antibody fragments
python3 preprocessing/process_shehata.py
```

This will generate:
- `test_datasets/shehata.csv` (Phase 1 output)
- `test_datasets/shehata/*.csv` (16 fragment-specific files)
