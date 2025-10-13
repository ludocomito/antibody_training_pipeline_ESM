#!/usr/bin/env python
import sys
from pathlib import PosixPath
from pathlib import Path
from typing import Any

from more_itertools import batched
from tqdm.auto import tqdm
from itertools import chain
import pandas as pd
import riot_na
import Bio
from Bio.SeqIO import parse


annotator = riot_na.create_riot_nt()


def annotate(
    seqs: list[Bio.SeqRecord.SeqRecord], chain: str, dataset: str
) -> list[dict[str, Any]]:
    """Annotate antibody nucleotide sequences.

    Args:
        seqs: SeqRecord objects from fasta file.
        chain: Antibody chain. Either 'H' or 'L'.
        dataset: Name of the dataset.
    """
    assert chain in ("H", "L"), "chain must be one of ('H', 'L')"
    annotations = []
    for i, seq in enumerate(seqs):
        ant = annotator.run_on_sequence(
            seq.name if seq.name else f"seq_{i}", str(seq.seq)
        )
        annotations.append(
            {
                "dataset": dataset,
                f"seq_id_{chain}": ant.sequence_header,
                f"full_seq_{chain}": ant.sequence_alignment_aa,
                f"fwr1_aa_{chain}": ant.fwr1_aa,
                f"cdr1_aa_{chain}": ant.cdr1_aa,
                f"fwr2_aa_{chain}": ant.fwr2_aa,
                f"cdr2_aa_{chain}": ant.cdr2_aa,
                f"fwr3_aa_{chain}": ant.fwr3_aa,
                f"cdr3_aa_{chain}": ant.cdr3_aa,
                f"fwr4_aa_{chain}": ant.fwr4_aa,
            }
        )
    return annotations


def process_dataset(
    name: str, paths: tuple[PosixPath, PosixPath, PosixPath]
) -> list[dict[str, Any]]:
    """
    Collate metadata for and annotate an antibody dataset.

    Args:
        name: name of the dataset.
        paths: tuple of target (label) path, heavy chain path, light chain path

    """
    target_path, heavy_path, light_path = paths

    heavy_seqs = list(parse(heavy_path, "fasta"))
    light_seqs = list(parse(light_path, "fasta"))

    heavy_annotations = annotate(heavy_seqs, "H", name)
    light_annotations = annotate(light_seqs, "L", name)

    annotations = [h | l for h, l in zip(heavy_annotations, light_annotations)]

    targets = target_path.read_text().splitlines()
    # Some target files have headers
    if len(targets) == len(annotations) + 1:
        targets = targets[1:]
    for annotation, target in zip(annotations, targets):
        annotation["target"] = target
    return annotations


def main():
    usage = "./process_boughter.py path/to/AIMS_manuscripts/app_data/full_sequences"
    if len(sys.argv) <= 1:
        raise SystemExit(usage)
    aims_data_path = sys.argv[1]
    full_seq_files = [
        p for p in Path(aims_data_path).iterdir() if "README" not in p.name
    ]
    dataset_names = ("flu", "gut_hiv", "mouse", "nat_cntrl", "nat_hiv", "plos_hiv")
    groups = dict(zip(dataset_names, batched(sorted(full_seq_files), 3)))
    processed = {
        name: process_dataset(name, paths) for name, paths in tqdm(groups.items())
    }
    pd.DataFrame(list(chain.from_iterable(processed.values()))).to_csv(
        sys.stdout, index=False
    )


if __name__ == "__main__":
    main()
