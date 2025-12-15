#!/usr/bin/env python3
"""
Analyze dataset coverage for training and test sets.

The script loads training labels, test labels, and optional test predictions
from common formats (.npy, .json, .csv/.txt) and reports per-class counts,
missing classes, and imbalanced classes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".json":
        with open(path, "r") as f:
            arr = np.array(json.load(f))
    elif suffix in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type for {path}")
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def ensure_class_count(arrays: Sequence[np.ndarray]) -> int:
    n_classes = {a.shape[1] for a in arrays if a is not None}
    if not n_classes:
        raise ValueError("At least one label or prediction array is required to infer class count")
    if len(n_classes) != 1:
        raise ValueError("Label and prediction arrays must have the same number of classes")
    return n_classes.pop()


def load_class_names(path: Path | None, n_classes: int) -> list[str]:
    if path is None:
        return [f"class_{i}" for i in range(n_classes)]
    with open(path, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    if len(names) != n_classes:
        logging.warning(
            "Number of class names (%d) does not match n_classes (%d); falling back to indices",
            len(names),
            n_classes,
        )
        return [f"class_{i}" for i in range(n_classes)]
    return names


def summarize_distribution(
    labels: np.ndarray,
    class_names: Sequence[str],
    imbalance_threshold: float,
) -> dict:
    total = labels.shape[0]
    counts = labels.sum(axis=0).astype(int)
    freqs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
    missing = [class_names[i] for i, c in enumerate(counts) if c == 0]
    imbalanced = [
        class_names[i]
        for i, f in enumerate(freqs)
        if f > 0 and f < imbalance_threshold
    ]
    if missing:
        logging.warning("Missing classes (count=0): %s", missing)
    if imbalanced:
        logging.info(
            "Imbalanced classes (<%s of samples): %s",
            f"{imbalance_threshold * 100:.2f}%",
            imbalanced,
        )
    per_class = []
    for idx, (count, freq) in enumerate(zip(counts, freqs)):
        per_class.append(
            {
                "class": class_names[idx],
                "count": int(count),
                "frequency": float(freq),
            }
        )
    return {
        "total_samples": int(total),
        "per_class": per_class,
        "missing_classes": missing,
        "imbalanced_classes": imbalanced,
    }


def binarize(matrix: np.ndarray, threshold: float) -> np.ndarray:
    return (matrix >= threshold).astype(int)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze dataset coverage for training and test labels")
    parser.add_argument("--train-labels", required=True, help="Path to training labels file (.npy/.json/.csv/.txt)")
    parser.add_argument("--test-labels", required=True, help="Path to test labels file (.npy/.json/.csv/.txt)")
    parser.add_argument("--test-predictions", help="Optional path to test predictions file (.npy/.json/.csv/.txt)")
    parser.add_argument("--class-names", help="Optional text file with one class name per line")
    parser.add_argument("--imbalance-threshold", type=float, default=0.01, help="Frequency threshold to flag imbalance (default: 1%)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize inputs (default: 0.5)")
    parser.add_argument("--output", type=str, default="coverage_analysis.json", help="Output JSON path (default: coverage_analysis.json)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    train_path = Path(args.train_labels)
    test_path = Path(args.test_labels)
    pred_path = Path(args.test_predictions) if args.test_predictions else None
    class_path = Path(args.class_names) if args.class_names else None

    if not train_path.exists():
        raise FileNotFoundError(f"File not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"File not found: {test_path}")
    if pred_path and not pred_path.exists():
        raise FileNotFoundError(f"File not found: {pred_path}")

    train_labels = binarize(load_array(train_path), args.threshold)
    test_labels = binarize(load_array(test_path), args.threshold)
    test_predictions = binarize(load_array(pred_path), args.threshold) if pred_path else None

    n_classes = ensure_class_count([train_labels, test_labels, test_predictions])
    class_names = load_class_names(class_path, n_classes)

    report = {
        "class_names": class_names,
        "imbalance_threshold": args.imbalance_threshold,
        "train": {
            "y_true": summarize_distribution(train_labels, class_names, args.imbalance_threshold)
        },
        "test": {
            "y_true": summarize_distribution(test_labels, class_names, args.imbalance_threshold)
        },
    }

    if test_predictions is not None:
        report["test"]["y_pred"] = summarize_distribution(
            test_predictions, class_names, args.imbalance_threshold
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved coverage analysis to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())