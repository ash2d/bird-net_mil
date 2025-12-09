#!/usr/bin/env python3
"""
CLI script for evaluating trained MIL pooling heads on BirdNET embeddings.

Usage:
    # Evaluate a trained model
    python scripts/evaluate_mil.py --checkpoint runs/run_XXX/lme_last.pt \
        --emb_dir /data/embeddings --strong_root /data/anuraset/strong_labels

    # Evaluate on test set with detailed metrics
    python scripts/evaluate_mil.py --checkpoint runs/run_XXX/attn_best.pt \
        --emb_dir /data/embeddings --strong_root /data/anuraset/strong_labels \
        --metrics --output results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.heads import PoolingHead
from mil.datasets import EmbeddingBagDataset, collate_fn, build_label_index
from mil.train import load_checkpoint
from mil.evaluate import pointing_game


def load_species_list(path: str | Path) -> list:
    """Load species list from text file (one species per line)."""
    species = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                species.append(line)
    return species


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@torch.no_grad()
def evaluate_model(
    head: PoolingHead,
    loader: DataLoader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate model on a dataset.
    
    Args:
        head: Trained PoolingHead model.
        loader: DataLoader for evaluation.
        device: Device to run on.
        threshold: Classification threshold for binary predictions.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    head.eval()
    head.to(device)
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    for embeddings, weak_labels, _, _ in loader:
        embeddings = embeddings.to(device)
        weak_labels = weak_labels.numpy()
        
        # Get predictions
        y_clip, _ = head(embeddings)
        
        # Convert to probabilities
        if head.returns_probabilities:
            probs = y_clip.cpu().numpy()
        else:
            probs = torch.sigmoid(y_clip).cpu().numpy()
        
        # Binary predictions
        preds = (probs >= threshold).astype(int)
        
        all_preds.append(preds)
        all_probs.append(probs)
        all_labels.append(weak_labels)
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    
    # Compute metrics
    metrics = {}
    
    # Per-sample accuracy (exact match)
    metrics["sample_accuracy"] = accuracy_score(
        all_labels, all_preds
    )
    
    # Per-class metrics (micro and macro averaged)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    
    metrics["precision_micro"] = float(precision_micro)
    metrics["recall_micro"] = float(recall_micro)
    metrics["f1_micro"] = float(f1_micro)
    metrics["precision_macro"] = float(precision_macro)
    metrics["recall_macro"] = float(recall_macro)
    metrics["f1_macro"] = float(f1_macro)
    
    # AUC-ROC (macro averaged)
    try:
        auc_roc = roc_auc_score(all_labels, all_probs, average="macro")
        metrics["auc_roc_macro"] = float(auc_roc)
    except ValueError:
        # Handle case where some classes have no positive samples
        metrics["auc_roc_macro"] = None
    
    # Average Precision (macro averaged)
    try:
        avg_precision = average_precision_score(all_labels, all_probs, average="macro")
        metrics["avg_precision_macro"] = float(avg_precision)
    except ValueError:
        metrics["avg_precision_macro"] = None
    
    # Per-class accuracy
    class_accuracy = []
    for c in range(all_labels.shape[1]):
        if all_labels[:, c].sum() > 0:  # Only if class has positive samples
            acc = accuracy_score(all_labels[:, c], all_preds[:, c])
            class_accuracy.append(acc)
    
    if class_accuracy:
        metrics["class_accuracy_mean"] = float(np.mean(class_accuracy))
        metrics["class_accuracy_std"] = float(np.std(class_accuracy))
    else:
        metrics["class_accuracy_mean"] = None
        metrics["class_accuracy_std"] = None
    
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MIL pooling heads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--emb_dir", type=str, default=None,
        help="Directory containing .embeddings.npz files",
    )
    data_group.add_argument(
        "--emb_glob", type=str, default=None,
        help="Glob pattern for finding .npz files (alternative to --emb_dir)",
    )
    data_group.add_argument(
        "--strong_root", type=str, required=True,
        help="Root directory containing strong label .txt files",
    )
    data_group.add_argument(
        "--species_list", type=str, default=None,
        help="Path to text file with species names (one per line)",
    )
    data_group.add_argument(
        "--small_test", type=float, default=None,
        help="Fraction or count of test data to use (e.g., 0.1 for 10% or 100 for 100 samples)",
    )
    
    # Evaluation arguments
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for evaluation (default: 32)",
    )
    eval_group.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    eval_group.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Device for evaluation (default: auto-detect)",
    )
    eval_group.add_argument(
        "--pointing_game", action="store_true",
        help="Evaluate pointing game accuracy",
    )
    
    # Output arguments
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for results (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.emb_dir is None and args.emb_glob is None:
        parser.error("Must specify either --emb_dir or --emb_glob")
    
    # Resolve device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        head, checkpoint = load_checkpoint(args.checkpoint, device=device)
        logger.info(f"Loaded {head.pool_name} model with {head.n_classes} classes")
        
        # Build label index (must match training)
        logger.info("Building label index...")
        if args.species_list:
            species_list = load_species_list(args.species_list)
            label_index = build_label_index(species_list=species_list)
        else:
            label_index = build_label_index(strong_root=args.strong_root)
        
        n_classes = len(label_index)
        if n_classes != head.n_classes:
            logger.warning(
                f"Number of classes mismatch: checkpoint has {head.n_classes}, "
                f"but label index has {n_classes}. This may cause errors."
            )
        
        # Find NPZ files
        if args.emb_glob:
            npz_pattern = args.emb_glob
        else:
            npz_pattern = str(Path(args.emb_dir) / "**/*.embeddings.npz")
        
        if args.small_test:
            logger.info("Using small test set for quick testing")
            all_paths = sorted(glob.glob(npz_pattern, recursive=True))
            if not all_paths:
                raise FileNotFoundError(f"No .npz files found for pattern: {npz_pattern}")

            n_total = len(all_paths)
            if args.small_train <= 0:
                raise ValueError("small_train must be > 0")

            # If small_train < 1 treat as fraction, otherwise treat as count
            if args.small_train < 1:
                k = max(1, int(n_total * args.small_train))
            else:
                k = int(min(n_total, args.small_train))

            # Use deterministic sampling for consistency
            # Note: Global seed (if set) takes precedence
            if args.seed is None:
                random.seed(0)  # Default seed for small_train sampling
            selected_paths = random.sample(all_paths, k) if k < n_total else list(all_paths)

            npz_pattern = selected_paths
            logger.info(f"Using {len(selected_paths)}/{n_total} npz files for small_train ({args.small_train})")

        # Create dataset
        logger.info("Loading dataset...")
        dataset = EmbeddingBagDataset(
            npz_paths=npz_pattern,
            strong_root=args.strong_root,
            label_index=label_index,
        )
        
        logger.info(f"Dataset: {len(dataset)} samples")
        
        # Create data loader
        import os
        num_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(head, loader, device, threshold=args.threshold)
        
        # Add pointing game if requested
        if args.pointing_game:
            logger.info("Evaluating pointing game...")
            pg_acc, hits, total = pointing_game(loader, head, device)
            metrics["pointing_game_accuracy"] = float(pg_acc)
            metrics["pointing_game_hits"] = int(hits)
            metrics["pointing_game_total"] = int(total)
        
        # Add metadata
        results = {
            "checkpoint": str(args.checkpoint),
            "pool_name": head.pool_name,
            "n_classes": head.n_classes,
            "n_samples": len(dataset),
            "threshold": args.threshold,
            "metrics": metrics,
        }
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("Evaluation Results")
        logger.info("="*50)
        logger.info(f"Model: {head.pool_name}")
        logger.info(f"Samples: {len(dataset)}")
        logger.info(f"Threshold: {args.threshold}")
        logger.info("-"*50)
        
        for key, value in metrics.items():
            if value is not None:
                if isinstance(value, float):
                    logger.info(f"{key:30s}: {value:.4f}")
                else:
                    logger.info(f"{key:30s}: {value}")
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nSaved results to {output_path}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
