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
from typing import Dict, Any, Callable
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
from mil.datasets import (
    EmbeddingBagDataset,
    collate_fn,
    build_label_index,
    normalize_species_name,
    load_species_list,
)
from mil.train import load_checkpoint
from mil.evaluate import pointing_game


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
    
    logger = logging.getLogger(__name__)
    # Compute metrics
    metrics = {}
    
    # Per-sample accuracy (exact match)
    metrics["sample_accuracy"] = accuracy_score(
        all_labels, all_preds
    )
    
    # Per-class metrics (micro and macro averaged)
    positive_counts = all_labels.sum(axis=0)
    pred_positive_counts = all_preds.sum(axis=0)
    missing_positive_classes = np.where(positive_counts == 0)[0].tolist()
    zero_prediction_classes = np.where(pred_positive_counts == 0)[0].tolist()

    if missing_positive_classes:
        logger.warning(
            "Skipping metrics for classes with no positive samples in y_true: %s",
            missing_positive_classes,
        )
    if zero_prediction_classes:
        logger.warning(
            "Predictions are zero for all samples in classes: %s",
            zero_prediction_classes,
        )

    per_precision, per_recall, per_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    per_precision = per_precision.astype(float)
    per_recall = per_recall.astype(float)
    per_f1 = per_f1.astype(float)

    valid_mask = positive_counts > 0
    per_precision[~valid_mask] = np.nan
    per_recall[~valid_mask] = np.nan
    per_f1[~valid_mask] = np.nan

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="micro", zero_division=0
    )
    
    metrics["precision_micro"] = float(precision_micro)
    metrics["recall_micro"] = float(recall_micro)
    metrics["f1_micro"] = float(f1_micro)

    if valid_mask.any():
        metrics["precision_macro"] = float(np.nanmean(per_precision))
        metrics["recall_macro"] = float(np.nanmean(per_recall))
        metrics["f1_macro"] = float(np.nanmean(per_f1))
    else:
        metrics["precision_macro"] = None
        metrics["recall_macro"] = None
        metrics["f1_macro"] = None

    # AUC-ROC (macro averaged)
    def _per_class_scores(
        scorer: Callable[[np.ndarray, np.ndarray], float], require_negative: bool = False
    ) -> list[float]:
        """
        Compute per-class scores while skipping classes without required labels.
        
        Args:
            scorer: callable accepting (y_true, y_prob) and returning a scalar score.
            require_negative: if True, skip classes that lack negative samples in y_true.
        
        Returns:
            List of scores for classes that satisfy the sampling requirements.
        """
        scores = []
        for class_idx in range(all_labels.shape[1]):
            y_true_cls = all_labels[:, class_idx]
            y_prob_cls = all_probs[:, class_idx]
            pos_count = y_true_cls.sum()
            neg_count = len(y_true_cls) - pos_count
            if pos_count == 0 or (require_negative and neg_count == 0):
                continue
            try:
                scores.append(scorer(y_true_cls, y_prob_cls))
            except ValueError:
                continue
        return scores

    auc_scores = _per_class_scores(roc_auc_score, require_negative=True)
    metrics["auc_roc_macro"] = (
        float(np.mean(auc_scores)) if auc_scores else None
    )
    
    # Average Precision (macro averaged)
    avg_precision_scores = _per_class_scores(average_precision_score)
    metrics["avg_precision_macro"] = (
        float(np.mean(avg_precision_scores)) if avg_precision_scores else None
    )
    
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
    
    metrics["missing_positive_classes"] = missing_positive_classes
    metrics["zero_prediction_classes"] = zero_prediction_classes
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
        help="Glob pattern for finding .npz files, or path to text file with paths (one per line, alternative to --emb_dir)",
    )
    data_group.add_argument(
        "--strong_root", type=str, default=None,
        help="Root directory containing strong label .txt files",
    )
    data_group.add_argument(
        "--weak_csv", type=str, default=None,
        help="Path to weak label CSV file (alternative to strong labels)",
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
    
    if args.strong_root is None and args.weak_csv is None:
        parser.error("Must specify either --strong_root or --weak_csv")
    
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
        label_index = None
        checkpoint_label_index = checkpoint.get("label_index")
        if checkpoint_label_index:
            # Ensure indices are ints (JSON may convert to strings)
            label_index = {k: int(v) for k, v in checkpoint_label_index.items()}
            logger.info("Loaded label index from checkpoint metadata")
        
        species_list_index = None
        if args.species_list:
            species_list = load_species_list(args.species_list)
            species_list_index = build_label_index(species_list=species_list)
            if label_index is None:
                label_index = species_list_index
            elif set(label_index.keys()) != set(species_list_index.keys()):
                logger.warning(
                    "species_list does not match checkpoint label index; "
                    "using checkpoint mapping to preserve class order."
                )
        
        if label_index is None:
            if args.weak_csv:
                label_index = build_label_index(weak_csv=args.weak_csv)
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
            # Check if it's a file containing paths
            emb_glob_path = Path(args.emb_glob)
            if emb_glob_path.is_file():
                # Read paths from file
                logger.info(f"Reading paths from {args.emb_glob}")
                with open(emb_glob_path, 'r') as f:
                    npz_pattern = [line.strip() for line in f if line.strip()]
            else:
                # Use as glob pattern
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
            weak_csv=args.weak_csv,
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
