#!/usr/bin/env python3
"""
Build a weak label CSV from strong labels for a list of embedding files.

Example:
    python scripts/build_labels_csv.py \
        --emb_list train.txt \
        --strong_root /data/anuraset/strong_labels \
        --output train.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.datasets import (
    _find_strong_label_file,
    build_label_index,
    events_to_labels,
    normalize_embedding_path,
    parse_strong_labels,
)


logger = logging.getLogger(__name__)


def read_embedding_paths(emb_list: str | Path) -> List[Path]:
    """
    Read embedding paths from a text file (one per line) or a glob pattern.
    """
    emb_list = Path(emb_list)
    if emb_list.is_file():
        with open(emb_list, "r") as f:
            return [Path(line.strip()) for line in f if line.strip()]
    pattern = str(emb_list)
    return [Path(p) for p in sorted(glob.glob(pattern, recursive=True))]


def load_times(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load start/end times without materializing embeddings."""
    try:
        data = np.load(npz_path)
        start_sec = data["start_sec"].astype(np.float32)
        end_sec = data["end_sec"].astype(np.float32)
    except KeyError as exc:
        raise KeyError(f"Missing start_sec/end_sec in {npz_path}") from exc
    return start_sec, end_sec


def build_rows(
    emb_paths: Iterable[Path],
    strong_root: Path,
    label_index: dict[str, int],
) -> pd.DataFrame:
    """Create rows with embedding path and one-hot species labels."""
    species_by_index = [sp for sp, _ in sorted(label_index.items(), key=lambda kv: kv[1])]
    rows: list[list[float | str]] = []

    for npz_path in emb_paths:
        npz_path = Path(npz_path)
        emb_path_norm = normalize_embedding_path(npz_path)

        strong_path = _find_strong_label_file(npz_path, strong_root)
        if strong_path and strong_path.exists():
            events = parse_strong_labels(strong_path)
        else:
            events = []
            logger.warning("No strong label file found for %s", npz_path)

        try:
            start_sec, end_sec = load_times(npz_path)
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Failed to read timing data from {npz_path}") from exc
        weak_labels, _ = events_to_labels(events, label_index, start_sec, end_sec)

        # species_by_index comes directly from label_index keys, so lookups are safe
        row = [emb_path_norm] + [
            float(weak_labels[label_index[sp]]) for sp in species_by_index
        ]
        rows.append(row)

    columns = ["embedding_path"] + [f"SPECIES_{sp}" for sp in species_by_index]
    return pd.DataFrame(rows, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a weak label CSV from strong labels and embedding list",
    )
    parser.add_argument(
        "--emb_list",
        required=True,
        help="Path to text file containing embedding paths (one per line) or a glob pattern",
    )
    parser.add_argument(
        "--strong_root",
        required=True,
        help="Root directory containing strong label .txt files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path (e.g., train.csv)",
    )
    parser.add_argument(
        "--species_list",
        type=str,
        default=None,
        help="Optional path to text file with species names (one per line); "
        "otherwise species are derived from the strong labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    strong_root = Path(args.strong_root)
    emb_paths = read_embedding_paths(args.emb_list)
    if not emb_paths:
        logger.error("No embedding paths found in %s", args.emb_list)
        return 1

    if args.species_list:
        species_path = Path(args.species_list)
        if not species_path.is_file():
            logger.error("Species list file not found: %s", species_path)
            return 1
        with species_path.open("r") as f:
            species = [line.strip() for line in f if line.strip()]
        label_index = build_label_index(species_list=species)
    else:
        label_index = build_label_index(strong_root=strong_root)

    df = build_rows(emb_paths, strong_root, label_index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
