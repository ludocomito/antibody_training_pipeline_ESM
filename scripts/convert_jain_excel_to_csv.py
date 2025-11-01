#!/usr/bin/env python3
"""
Convert Jain dataset supplementary Excel files into a clean CSV aligned with the
Sakhnini et al. 2025 evaluation pipeline.

This script:
1. Reads SD01/SD02/SD03 spreadsheets from Jain et al. 2017 (PNAS)
2. Sanitizes VH/VL amino acid sequences
3. Applies Table 1 flag thresholds (90th percentile of approved antibodies)
4. Produces a canonical CSV with explicit flag metadata and binary labels

Usage:
    python3 scripts/convert_jain_excel_to_csv.py \
        --sd01 test_datasets/pnas.1616408114.sd01.xlsx \
        --sd02 test_datasets/pnas.1616408114.sd02.xlsx \
        --sd03 test_datasets/pnas.1616408114.sd03.xlsx \
        --output test_datasets/jain.csv

The default CLI values match the repository layout, so running without arguments
is sufficient when the Excel files are placed in `test_datasets/`.

Reference material:
- literature/markdown/jain-et-al-2017-biophysical-properties-of-the-clinical-stage-antibody-landscape/
- literature/pdf/pnas.201616408si.pdf
"""

from __future__ import annotations

import argparse
import logging
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd

LOG = logging.getLogger("convert_jain_excel_to_csv")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Valid amino acids supported by ESM (X used for unknown residues).
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")

# Mapping from verbose SD03 column names to snake_case identifiers.
COLUMN_RENAME = {
    "HEK Titer (mg/L)": "hek_titer_mg_per_L",
    "Fab Tm by DSF (°C)": "fab_tm_celsius",
    "SGAC-SINS AS100 ((NH4)2SO4 mM)": "sgac_sins_mM",
    "HIC Retention Time (Min)a": "hic_min",
    "SMAC Retention Time (Min)a": "smac_min",
    "Slope for Accelerated Stability": "as_slope_pct_per_day",
    "Poly-Specificity Reagent (PSR) SMP Score (0-1)": "psr_smp",
    "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average": "acsins_dlmax_nm",
    "CIC Retention Time (Min)": "cic_min",
    "CSI-BLI Delta Response (nm)": "csi_bli_nm",
    "ELISA": "elisa_fold",
    "BVP ELISA": "bvp_elisa_fold",
}


@dataclass(frozen=True)
class AssayThreshold:
    """Threshold definition for a single assay metric."""

    column: str
    threshold: float
    comparator: Callable[[float, float], bool]
    direction_label: str  # ">" or "<"
    original_name: str


ASSAY_CLUSTERS: Dict[str, List[AssayThreshold]] = {
    "flag_self_interaction": [
        AssayThreshold(
            "psr_smp",
            0.27,
            operator.gt,
            ">",
            "Poly-Specificity Reagent (PSR) SMP Score (0-1)",
        ),
        AssayThreshold(
            "acsins_dlmax_nm",
            11.8,
            operator.gt,
            ">",
            "Affinity-Capture Self-Interaction Nanoparticle Spectroscopy (AC-SINS) ∆λmax (nm) Average",
        ),
        AssayThreshold(
            "csi_bli_nm", 0.01, operator.gt, ">", "CSI-BLI Delta Response (nm)"
        ),
        AssayThreshold("cic_min", 10.1, operator.gt, ">", "CIC Retention Time (Min)"),
    ],
    "flag_chromatography": [
        AssayThreshold("hic_min", 11.7, operator.gt, ">", "HIC Retention Time (Min)a"),
        AssayThreshold(
            "smac_min", 12.8, operator.gt, ">", "SMAC Retention Time (Min)a"
        ),
        AssayThreshold(
            "sgac_sins_mM", 370.0, operator.lt, "<", "SGAC-SINS AS100 ((NH4)2SO4 mM)"
        ),
    ],
    "flag_polyreactivity": [
        AssayThreshold("elisa_fold", 1.9, operator.gt, ">", "ELISA"),
        AssayThreshold("bvp_elisa_fold", 4.3, operator.gt, ">", "BVP ELISA"),
    ],
    "flag_stability": [
        AssayThreshold(
            "as_slope_pct_per_day",
            0.08,
            operator.gt,
            ">",
            "Slope for Accelerated Stability",
        ),
    ],
}

# Column order for the output CSV (missing columns are appended afterwards).
PRIMARY_COLUMNS = [
    "id",
    "heavy_seq",
    "light_seq",
    "flags_total",
    "flag_category",
    "label",
    "flag_self_interaction",
    "flag_chromatography",
    "flag_polyreactivity",
    "flag_stability",
    "source",
    "smp",
    "ova",
    "bvp_elisa",
    "psr_smp",
    "acsins_dlmax_nm",
    "csi_bli_nm",
    "cic_min",
    "hic_min",
    "smac_min",
    "sgac_sins_mM",
    "as_slope_pct_per_day",
    "hek_titer_mg_per_L",
    "fab_tm_celsius",
    "heavy_seq_length",
    "light_seq_length",
]


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #


def sanitize_sequence(seq: object) -> object:
    """
    Sanitize a VH/VL amino acid sequence.

    - Uppercases characters
    - Removes whitespace, hyphens, and any characters outside VALID_AA
    - Returns pandas.NA if no valid characters remain
    """
    if pd.isna(seq):
        return pd.NA

    seq_str = "".join(ch for ch in str(seq).upper() if ch in VALID_AA)
    return seq_str or pd.NA


def load_excel_frame(
    path: Path, expected_unique: Iterable[str] | None = None
) -> pd.DataFrame:
    """Read an Excel worksheet and optionally validate set membership."""
    df = pd.read_excel(path)
    if expected_unique is not None:
        missing = set(expected_unique) - set(df["Name"].dropna())
        if missing:
            LOG.warning(
                "File %s is missing %d expected names (e.g. %s)",
                path,
                len(missing),
                list(sorted(missing))[:3],
            )
    return df


def evaluate_cluster_flags(row: pd.Series) -> Dict[str, bool]:
    """Evaluate Table 1 cluster thresholds for a single antibody."""
    flags: Dict[str, bool] = {}
    for cluster, metrics in ASSAY_CLUSTERS.items():
        triggered = False
        for metric in metrics:
            value = row.get(metric.column)
            if pd.isna(value):
                continue
            if metric.comparator(value, metric.threshold):
                triggered = True
                break
        flags[cluster] = triggered
    return flags


def assign_flag_category(total_flags: int) -> str:
    """Map numerical flag counts to categorical buckets."""
    if total_flags >= 4:
        return "non_specific"
    if total_flags == 0:
        return "specific"
    return "mild"


def ensure_output_directory(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Conversion pipeline
# --------------------------------------------------------------------------- #


def convert_jain_dataset(
    sd01_path: Path, sd02_path: Path, sd03_path: Path
) -> pd.DataFrame:
    """Convert the Jain supplementary files to a cleaned, merged DataFrame."""
    LOG.info("Reading SD01 metadata from %s", sd01_path)
    sd01 = pd.read_excel(sd01_path)
    LOG.info("  Loaded %d metadata rows", len(sd01))

    LOG.info("Reading SD02 sequences from %s", sd02_path)
    sd02 = pd.read_excel(sd02_path)
    LOG.info("  Loaded %d sequence rows", len(sd02))

    LOG.info("Reading SD03 biophysical properties from %s", sd03_path)
    sd03 = pd.read_excel(sd03_path)
    LOG.info("  Loaded %d property rows (including metadata rows)", len(sd03))

    # Restrict to antibodies that appear in the sequence table.
    valid_names = set(sd02["Name"].dropna())
    sd01 = sd01[sd01["Name"].isin(valid_names)].reset_index(drop=True)
    sd03 = sd03[sd03["Name"].isin(valid_names)].reset_index(drop=True)

    LOG.info(
        "  Retained %d property rows after filtering to valid antibody names", len(sd03)
    )

    # Rename columns for ease of use.
    sd03 = sd03.rename(columns=COLUMN_RENAME)

    # Merge metadata / sequences / properties.
    merged = (
        sd01.merge(sd02[["Name", "VH", "VL"]], on="Name", how="inner")
        .merge(sd03, on="Name", how="inner")
        .rename(columns={"Name": "id", "VH": "heavy_seq", "VL": "light_seq"})
    )

    LOG.info("Merged dataset has %d antibodies", len(merged))

    # Sanitize sequences.
    merged["heavy_seq"] = merged["heavy_seq"].apply(sanitize_sequence)
    merged["light_seq"] = merged["light_seq"].apply(sanitize_sequence)

    merged["heavy_seq_length"] = merged["heavy_seq"].str.len()
    merged["light_seq_length"] = merged["light_seq"].str.len()

    # Compute threshold-based flags.
    flag_records = merged.apply(evaluate_cluster_flags, axis=1, result_type="expand")
    for column in ASSAY_CLUSTERS:
        merged[column] = flag_records[column]

    merged["flags_total"] = merged[list(ASSAY_CLUSTERS.keys())].sum(axis=1)
    merged["flag_category"] = merged["flags_total"].apply(assign_flag_category)
    merged["label"] = (
        merged["flag_category"].map({"specific": 0, "non_specific": 1}).astype("Int64")
    )

    # Map supporting columns for historical compatibility.
    merged["source"] = "jain2017"
    merged["smp"] = merged["psr_smp"]
    merged["ova"] = merged["elisa_fold"]
    merged["bvp_elisa"] = merged["bvp_elisa_fold"]

    return merged


def prepare_output(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns and ensure consistent dtypes before export."""
    ordered_cols = [col for col in PRIMARY_COLUMNS if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    final_cols = ordered_cols + remaining_cols
    return df[final_cols]


def summarize(df: pd.DataFrame) -> None:
    """Print key summary statistics to stdout."""
    LOG.info("Summary:")
    LOG.info("  Total antibodies: %d", len(df))
    LOG.info(
        "  Flag distribution: %s",
        ", ".join(
            f"{k}={v}"
            for k, v in df["flag_category"].value_counts().sort_index().items()
        ),
    )
    label_counts = df["label"].value_counts(dropna=False)
    label_summary: List[str] = []
    for idx, val in label_counts.items():
        if pd.isna(idx):
            label_summary.append(f"NaN={val}")
        else:
            label_summary.append(f"{int(idx)}={val}")
    LOG.info("  Label counts: %s", ", ".join(label_summary))

    missing_metrics: List[str] = []
    for cluster, metrics in ASSAY_CLUSTERS.items():
        for metric in metrics:
            missing = df[metric.column].isna().sum()
            if missing:
                missing_metrics.append(f"{metric.column}({cluster})={missing}")
    if missing_metrics:
        LOG.warning("  Missing assay measurements: %s", ", ".join(missing_metrics))

    vh_lengths = df["heavy_seq_length"].dropna()
    vl_lengths = df["light_seq_length"].dropna()
    LOG.info(
        "  Sequence length (VH): min=%s, max=%s",
        int(vh_lengths.min()),
        int(vh_lengths.max()),
    )
    LOG.info(
        "  Sequence length (VL): min=%s, max=%s",
        int(vl_lengths.min()),
        int(vl_lengths.max()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Jain supplementary Excel files to CSV."
    )
    parser.add_argument(
        "--sd01",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd01.xlsx"),
        help="Path to SD01 metadata file",
    )
    parser.add_argument(
        "--sd02",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd02.xlsx"),
        help="Path to SD02 sequence file",
    )
    parser.add_argument(
        "--sd03",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd03.xlsx"),
        help="Path to SD03 biophysical measurements",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_datasets/jain.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    for path in (args.sd01, args.sd02, args.sd03):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    df = convert_jain_dataset(args.sd01, args.sd02, args.sd03)
    df = prepare_output(df)

    ensure_output_directory(args.output)
    df.to_csv(args.output, index=False)

    summarize(df)
    LOG.info("Saved cleaned Jain dataset to %s", args.output)


if __name__ == "__main__":
    main()
