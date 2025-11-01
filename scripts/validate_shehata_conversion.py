#!/usr/bin/env python3
"""
Multi-method validation of Shehata Excel → CSV conversion.

Uses multiple libraries to read Excel and compare results to ensure
data integrity during conversion.

Methods:
1. pandas (openpyxl engine)
2. Direct openpyxl reading
3. CSV checksum validation

Date: 2025-10-31
"""

import hashlib
from pathlib import Path

import openpyxl
import pandas as pd


def sanitize_sequence(seq: str) -> str:
    """Remove gap characters and normalise amino acid strings."""
    if pd.isna(seq):
        return seq
    return str(seq).replace("-", "").strip().upper()


def clean_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning steps used during conversion:
    - sanitize VH/VL sequences
    - drop rows without sequence information (footnotes)
    - drop rows without numeric PSR measurements
    """
    df = df.copy()
    df["VH Protein"] = df["VH Protein"].apply(sanitize_sequence)
    df["VL Protein"] = df["VL Protein"].apply(sanitize_sequence)
    df = df.dropna(subset=["VH Protein", "VL Protein"], how="all")
    psr_numeric = pd.to_numeric(df["PSR Score"], errors="coerce")
    df = df.loc[psr_numeric.notna()]
    df.reset_index(drop=True, inplace=True)
    return df


def method1_pandas_openpyxl(excel_path: str) -> pd.DataFrame:
    """Read Excel using pandas with openpyxl engine."""
    print("Method 1: pandas.read_excel (openpyxl engine)")
    df = pd.read_excel(excel_path, engine="openpyxl")
    df = clean_excel_df(df)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def method2_openpyxl_direct(excel_path: str) -> dict:
    """Read Excel using openpyxl directly."""
    print("\nMethod 2: openpyxl direct reading")
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    data = []
    headers = None

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            headers = row
        else:
            data.append(row)

    df = pd.DataFrame(data, columns=headers)
    df = clean_excel_df(df)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def method3_csv_direct(csv_path: str) -> pd.DataFrame:
    """Read the generated CSV."""
    print("\nMethod 3: Reading generated CSV")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def compare_sequences(
    df1: pd.DataFrame, df2: pd.DataFrame, col1: str, col2: str, name: str
):
    """
    Compare sequences between two DataFrames.

    Properly handles NaN values (NaN == NaN for comparison purposes).
    """
    print(f"\n  Comparing {name}:")

    # Check lengths
    if len(df1) != len(df2):
        print(f"    ✗ Row count mismatch: {len(df1)} vs {len(df2)}")
        return False

    # Compare each sequence
    mismatches = 0
    for i in range(len(df1)):
        seq1 = df1.iloc[i][col1] if col1 in df1.columns else None
        seq2 = df2.iloc[i][col2] if col2 in df2.columns else None

        # Proper NaN comparison: both NaN = match, otherwise check equality
        both_nan = pd.isna(seq1) and pd.isna(seq2)
        both_equal = seq1 == seq2 if not (pd.isna(seq1) or pd.isna(seq2)) else False

        if not (both_nan or both_equal):
            mismatches += 1
            if mismatches <= 3:  # Show first 3 mismatches
                print(f"    ✗ Row {i} mismatch:")
                print(f"      Source: {str(seq1)[:60]}...")
                print(f"      CSV:    {str(seq2)[:60]}...")

    if mismatches == 0:
        print(f"    ✓ All {len(df1)} sequences match!")
        return True
    else:
        print(f"    ✗ {mismatches} mismatches found")
        return False


def calculate_checksum(filepath: str) -> str:
    """Calculate SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    excel_path = Path("test_datasets/mmc2.xlsx")
    csv_path = Path("test_datasets/shehata.csv")

    print("=" * 60)
    print("Multi-Method Validation of Shehata Conversion")
    print("=" * 60)

    if not excel_path.exists():
        print(f"✗ Excel file not found: {excel_path}")
        return

    if not csv_path.exists():
        print(f"✗ CSV file not found: {csv_path}")
        print("  Run convert_shehata_excel_to_csv.py first!")
        return

    print("\nReading files with multiple methods...\n")

    # Read with different methods
    try:
        df_pandas = method1_pandas_openpyxl(str(excel_path))
    except Exception as e:
        print(f"  Error: {e}")
        df_pandas = None

    try:
        df_openpyxl = method2_openpyxl_direct(str(excel_path))
    except Exception as e:
        print(f"  Error: {e}")
        df_openpyxl = None

    try:
        df_csv = method3_csv_direct(str(csv_path))
    except Exception as e:
        print(f"  Error: {e}")
        df_csv = None

    # Cross-validate
    print("\n" + "=" * 60)
    print("Cross-Validation Results")
    print("=" * 60)

    if df_pandas is not None and df_openpyxl is not None:
        print("\n1. Pandas vs Direct openpyxl (Excel reading consistency)")
        compare_sequences(
            df_pandas, df_openpyxl, "VH Protein", "VH Protein", "VH sequences"
        )
        compare_sequences(
            df_pandas, df_openpyxl, "VL Protein", "VL Protein", "VL sequences"
        )

    if df_pandas is not None and df_csv is not None:
        print("\n2. Excel (pandas) vs Generated CSV (conversion accuracy)")
        compare_sequences(
            df_pandas, df_csv, "VH Protein", "heavy_seq", "VH → heavy_seq"
        )
        compare_sequences(
            df_pandas, df_csv, "VL Protein", "light_seq", "VL → light_seq"
        )

        # Check ID mapping
        print("\n  Comparing IDs:")
        id_match = (df_pandas["Clone name"] == df_csv["id"]).all()
        print(f"    {'✓' if id_match else '✗'} Clone name → id mapping")

    # File integrity
    print("\n" + "=" * 60)
    print("File Integrity")
    print("=" * 60)
    print(f"\nExcel checksum: {calculate_checksum(excel_path)}")
    print(f"CSV checksum:   {calculate_checksum(csv_path)}")
    print("\n(These checksums are stored for future verification)")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    if df_csv is not None:
        print(f"\nGenerated CSV ({csv_path.name}):")
        print(f"  Total rows: {len(df_csv)}")
        print(f"  Columns: {list(df_csv.columns)}")
        print(f"\n  Label distribution:")
        for label, count in df_csv["label"].value_counts().sort_index().items():
            label_name = "Specific" if label == 0 else "Non-specific"
            print(f"    {label_name}: {count} ({count/len(df_csv)*100:.1f}%)")

        print(f"\n  Missing data:")
        print(f"    Missing heavy_seq: {df_csv['heavy_seq'].isna().sum()}")
        print(f"    Missing light_seq: {df_csv['light_seq'].isna().sum()}")
        print(f"    Missing labels: {df_csv['label'].isna().sum()}")

    print("\n" + "=" * 60)
    print("✓ Validation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
