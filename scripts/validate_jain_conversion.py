#!/usr/bin/env python3
"""
Validation harness for the Jain Excel->CSV conversion.

Checks performed:
1. Re-runs the conversion pipeline in-memory and compares against jain.csv
2. Verifies flag counts, label distribution, and column integrity
3. Confirms amino acid sequences contain only valid residues
4. Prints SHA256 checksum for provenance tracking
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import pandas.testing as pdt

# Make conversion helpers importable.
sys.path.append(str(Path(__file__).resolve().parent))

from convert_jain_excel_to_csv import (  # noqa: E402
    ASSAY_CLUSTERS,
    VALID_AA,
    convert_jain_dataset,
    prepare_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Jain dataset conversion output."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("test_datasets/jain.csv"),
        help="Path to the converted CSV file.",
    )
    parser.add_argument(
        "--sd01",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd01.xlsx"),
        help="Path to SD01 metadata file.",
    )
    parser.add_argument(
        "--sd02",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd02.xlsx"),
        help="Path to SD02 sequence file.",
    )
    parser.add_argument(
        "--sd03",
        type=Path,
        default=Path("test_datasets/pnas.1616408114.sd03.xlsx"),
        help="Path to SD03 property file.",
    )
    return parser.parse_args()


def checksum(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def validate_sequences(df: pd.DataFrame) -> Dict[str, int]:
    """Return counts of sequences containing invalid residues."""
    invalid_counts = {"heavy": 0, "light": 0}
    for seq in df["heavy_seq"].dropna():
        if set(seq) - VALID_AA:
            invalid_counts["heavy"] += 1
    for seq in df["light_seq"].dropna():
        if set(seq) - VALID_AA:
            invalid_counts["light"] += 1
    return invalid_counts


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    csv_df = pd.read_csv(args.csv)
    regenerated = prepare_output(convert_jain_dataset(args.sd01, args.sd02, args.sd03))

    # Align dtypes that may have changed during CSV round-trip.
    for col in ["flags_total", "label"]:
        csv_df[col] = csv_df[col].astype("Int64")
        regenerated[col] = regenerated[col].astype("Int64")
    for col in ASSAY_CLUSTERS.keys():
        csv_df[col] = csv_df[col].astype(bool)
        regenerated[col] = regenerated[col].astype(bool)

    regenerated_sorted = regenerated.sort_values("id").reset_index(drop=True)
    csv_sorted = csv_df.sort_values("id").reset_index(drop=True)

    pdt.assert_frame_equal(
        regenerated_sorted, csv_sorted, check_dtype=False, check_like=True
    )

    # High-level stats
    print("=" * 60)
    print("Jain Conversion Validation")
    print("=" * 60)
    print(f"Rows: {len(csv_df)}, Columns: {len(csv_df.columns)}")
    print(f"Flag distribution:\n{csv_df['flag_category'].value_counts().sort_index()}")
    print(
        f"Label distribution (nullable):\n{csv_df['label'].value_counts(dropna=False)}"
    )

    invalid = validate_sequences(csv_df)
    if invalid["heavy"] == 0 and invalid["light"] == 0:
        print(
            "\nSequence validation: ✅ all VH/VL sequences contain only valid amino acids"
        )
    else:
        print("\nSequence validation: ⚠ issues detected")
        print(f"  Heavy chains with invalid residues: {invalid['heavy']}")
        print(f"  Light chains with invalid residues: {invalid['light']}")

    print("\nCluster flag thresholds (Table 1)")
    for cluster, metrics in ASSAY_CLUSTERS.items():
        clauses = [f"{m.column} {m.direction_label} {m.threshold}" for m in metrics]
        print(f"  {cluster}: {' OR '.join(clauses)}")

    print("\nChecksum (SHA256):", checksum(args.csv))
    print("\nValidation complete ✅")


if __name__ == "__main__":
    main()
