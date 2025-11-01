#!/usr/bin/env python3
"""
Shehata Dataset Fragment Extraction Script

Processes the Shehata dataset to extract all 16 antibody fragment types
using ANARCI (IMGT numbering scheme) following Sakhnini et al. 2025 methodology.

Fragments extracted:
1. VH (full heavy variable domain)
2. VL (full light variable domain)
3. H-CDR1
4. H-CDR2
5. H-CDR3
6. L-CDR1
7. L-CDR2
8. L-CDR3
9. H-CDRs (concatenated H-CDR1+2+3)
10. L-CDRs (concatenated L-CDR1+2+3)
11. H-FWRs (concatenated H-FWR1+2+3+4)
12. L-FWRs (concatenated L-FWR1+2+3+4)
13. VH+VL (paired variable domains)
14. All-CDRs (H-CDRs + L-CDRs)
15. All-FWRs (H-FWRs + L-FWRs)
16. Full (VH + VL = same as #13 for compatibility)

Date: 2025-10-31
Issue: #3 - Shehata dataset preprocessing (Phase 2)
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import riot_na
from tqdm.auto import tqdm

# Initialize ANARCI for amino acid annotation (IMGT scheme)
annotator = riot_na.create_riot_aa()


def annotate_sequence(
    seq_id: str, sequence: str, chain: str
) -> Optional[Dict[str, str]]:
    """
    Annotate a single amino acid sequence using ANARCI (IMGT).

    Args:
        seq_id: Unique identifier for the sequence
        sequence: Amino acid sequence string
        chain: 'H' for heavy or 'L' for light

    Returns:
        Dictionary with extracted fragments, or None if annotation fails
    """
    assert chain in ("H", "L"), "chain must be 'H' or 'L'"

    try:
        annotation = annotator.run_on_sequence(seq_id, sequence)

        # Extract all fragments
        fragments = {
            f"full_seq_{chain}": annotation.sequence_alignment_aa,
            f"fwr1_aa_{chain}": annotation.fwr1_aa,
            f"cdr1_aa_{chain}": annotation.cdr1_aa,
            f"fwr2_aa_{chain}": annotation.fwr2_aa,
            f"cdr2_aa_{chain}": annotation.cdr2_aa,
            f"fwr3_aa_{chain}": annotation.fwr3_aa,
            f"cdr3_aa_{chain}": annotation.cdr3_aa,
            f"fwr4_aa_{chain}": annotation.fwr4_aa,
        }

        # Create concatenated fragments
        fragments[f"cdrs_{chain}"] = "".join(
            [
                fragments[f"cdr1_aa_{chain}"],
                fragments[f"cdr2_aa_{chain}"],
                fragments[f"cdr3_aa_{chain}"],
            ]
        )

        fragments[f"fwrs_{chain}"] = "".join(
            [
                fragments[f"fwr1_aa_{chain}"],
                fragments[f"fwr2_aa_{chain}"],
                fragments[f"fwr3_aa_{chain}"],
                fragments[f"fwr4_aa_{chain}"],
            ]
        )

        return fragments

    except Exception as e:
        print(f"Warning: Failed to annotate {seq_id} ({chain}): {e}", file=sys.stderr)
        return None


def process_shehata_dataset(csv_path: str) -> pd.DataFrame:
    """
    Process Shehata CSV to extract all fragments.

    Args:
        csv_path: Path to shehata.csv

    Returns:
        DataFrame with all fragments and metadata
    """
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total antibodies: {len(df)}")
    print(f"  Annotating sequences with ANARCI (IMGT scheme)...")

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Annotating"):
        # Annotate heavy chain
        heavy_frags = annotate_sequence(f"{row['id']}_VH", row["heavy_seq"], "H")

        # Annotate light chain
        light_frags = annotate_sequence(f"{row['id']}_VL", row["light_seq"], "L")

        if heavy_frags is None or light_frags is None:
            print(f"  Skipping {row['id']} - annotation failed")
            continue

        # Combine all fragments and metadata
        result = {
            "id": row["id"],
            "label": row["label"],
            "psr_score": row["psr_score"],
            "b_cell_subset": row["b_cell_subset"],
            "source": row["source"],
        }

        result.update(heavy_frags)
        result.update(light_frags)

        # Create paired/combined fragments
        result["vh_vl"] = result["full_seq_H"] + result["full_seq_L"]
        result["all_cdrs"] = result["cdrs_H"] + result["cdrs_L"]
        result["all_fwrs"] = result["fwrs_H"] + result["fwrs_L"]

        results.append(result)

    df_annotated = pd.DataFrame(results)

    print(f"\n  Successfully annotated: {len(df_annotated)}/{len(df)} antibodies")

    return df_annotated


def create_fragment_csvs(df: pd.DataFrame, output_dir: Path):
    """
    Create separate CSV files for each fragment type.

    Following the 16-fragment methodology from Sakhnini et al. 2025.

    Args:
        df: DataFrame with all fragments
        output_dir: Directory to save fragment CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define all 16 fragment types
    fragments = {
        # 1-2: Full variable domains
        "VH_only": ("full_seq_H", "heavy_seq"),
        "VL_only": ("full_seq_L", "light_seq"),
        # 3-5: Heavy CDRs
        "H-CDR1": ("cdr1_aa_H", "h_cdr1"),
        "H-CDR2": ("cdr2_aa_H", "h_cdr2"),
        "H-CDR3": ("cdr3_aa_H", "h_cdr3"),
        # 6-8: Light CDRs
        "L-CDR1": ("cdr1_aa_L", "l_cdr1"),
        "L-CDR2": ("cdr2_aa_L", "l_cdr2"),
        "L-CDR3": ("cdr3_aa_L", "l_cdr3"),
        # 9-10: Concatenated CDRs
        "H-CDRs": ("cdrs_H", "h_cdrs"),
        "L-CDRs": ("cdrs_L", "l_cdrs"),
        # 11-12: Concatenated FWRs
        "H-FWRs": ("fwrs_H", "h_fwrs"),
        "L-FWRs": ("fwrs_L", "l_fwrs"),
        # 13: Paired variable domains
        "VH+VL": ("vh_vl", "paired_variable_domains"),
        # 14-15: All CDRs/FWRs
        "All-CDRs": ("all_cdrs", "all_cdrs"),
        "All-FWRs": ("all_fwrs", "all_fwrs"),
        # 16: Full (alias for VH+VL for compatibility)
        "Full": ("vh_vl", "full_sequence"),
    }

    print(f"\nCreating {len(fragments)} fragment-specific CSV files...")

    for fragment_name, (column_name, sequence_alias) in fragments.items():
        output_path = output_dir / f"{fragment_name}_shehata.csv"

        # Create fragment-specific CSV with standardized column names
        fragment_df = pd.DataFrame(
            {
                "id": df["id"],
                "sequence": df[column_name],
                "label": df["label"],
                "psr_score": df["psr_score"],
                "b_cell_subset": df["b_cell_subset"],
                "source": df["source"],
            }
        )

        fragment_df.to_csv(output_path, index=False)

        print(f"  ✓ {fragment_name:12s} → {output_path.name}")

    print(f"\n✓ All fragments saved to: {output_dir}/")


def main():
    """Main processing pipeline."""
    # Paths
    csv_path = Path("test_datasets/shehata.csv")
    output_dir = Path("test_datasets/shehata")

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Please run scripts/convert_shehata_excel_to_csv.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Shehata Dataset: Fragment Extraction (Phase 2)")
    print("=" * 60)
    print(f"\nInput:  {csv_path}")
    print(f"Output: {output_dir}/")
    print(f"Method: ANARCI (IMGT numbering scheme)")
    print()

    # Process dataset
    df_annotated = process_shehata_dataset(str(csv_path))

    # Create fragment CSVs
    create_fragment_csvs(df_annotated, output_dir)

    # Validation summary
    print("\n" + "=" * 60)
    print("Fragment Extraction Summary")
    print("=" * 60)

    print(f"\nAnnotated antibodies: {len(df_annotated)}")
    print(f"Label distribution:")
    for label, count in df_annotated["label"].value_counts().sort_index().items():
        label_name = "Specific" if label == 0 else "Non-specific"
        print(f"  {label_name}: {count} ({count/len(df_annotated)*100:.1f}%)")

    print(f"\nFragment files created: 16")
    print(f"Output directory: {output_dir.absolute()}")

    print("\n" + "=" * 60)
    print("✓ Phase 2 Complete!")
    print("=" * 60)

    print(f"\nNext steps:")
    print(f"  1. Test loading fragments with data.load_local_data()")
    print(f"  2. Run model inference on fragment-specific CSVs")
    print(f"  3. Compare results with paper (Sakhnini et al. 2025)")
    print(f"  4. Create PR to close Issue #3")


if __name__ == "__main__":
    main()
