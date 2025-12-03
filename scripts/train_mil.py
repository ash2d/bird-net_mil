#!/usr/bin/env python3
"""
CLI script for training MIL pooling heads on BirdNET embeddings.

Usage:
    # Train all poolers with default settings
    python scripts/train_mil.py --emb_dir /data/embeddings --strong_root /data/anuraset/strong_labels

    # Train specific poolers with W&B logging
    python scripts/train_mil.py --emb_dir /data/embeddings --strong_root /data/anuraset/strong_labels \
        --poolers lme attn autopool --epochs 20 --batch_size 32 \
        --wandb --wandb_project bird-mil

    # Use custom species list
    python scripts/train_mil.py --emb_dir /data/embeddings --strong_root /data/anuraset/strong_labels \
        --species_list species.txt --poolers attn --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.heads import PoolingHead, POOLER_NAMES
from mil.datasets import EmbeddingBagDataset, collate_fn, build_label_index
from mil.train import Trainer
from mil.evaluate import pointing_game
import glob
import random


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_species_list(path: str | Path) -> List[str]:
    """Load species list from text file (one species per line)."""
    species = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                species.append(line)
    return species


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train MIL pooling heads on BirdNET embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    "--small_train", type=float, default=None,
    help="Fraction or count of training data to use (e.g., 0.1 for 10% or 100 for 100 samples)",
    )
    data_group.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of data to use for validation (default: 0.1)",
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--poolers", nargs="+", default=["lme", "attn", "autopool", "linsoft", "noisyor", "mean", "max"],
        choices=POOLER_NAMES,
        help="Poolers to train (default: all)",
    )
    train_group.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    train_group.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_group.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "cuda"],
        help="Device for training (default: auto-detect)",
    )
    train_group.add_argument(
        "--early_stop", type=int, default=0,
        help="Early stopping patience (0 to disable, default: 0)",
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--out_dir", type=str, default="runs",
        help="Output directory for checkpoints and logs (default: runs)",
    )
    output_group.add_argument(
        "--save_every", type=int, default=0,
        help="Save checkpoint every N epochs (0 to disable, default: 0)",
    )
    
    # W&B arguments
    wandb_group = parser.add_argument_group("Weights & Biases")
    wandb_group.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging",
    )
    wandb_group.add_argument(
        "--wandb_project", type=str, default="bird-mil",
        help="W&B project name (default: bird-mil)",
    )
    wandb_group.add_argument(
        "--wandb_entity", type=str, default=None,
        help="W&B entity/team name",
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--eval_pointing", action="store_true",
        help="Evaluate pointing game after training",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Set random seed if specified
    if args.seed is not None:
        logger.info(f"Setting random seed to {args.seed}")
        set_seed(args.seed)
    
    # Create run ID and output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {run_dir}")
    
    # Validate arguments
    if args.emb_dir is None and args.emb_glob is None:
        parser.error("Must specify either --emb_dir or --emb_glob")
    
    # Resolve device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Build label index
        logger.info("Building label index...")
        if args.species_list:
            species_list = load_species_list(args.species_list)
            label_index = build_label_index(species_list=species_list)
        else:
            label_index = build_label_index(strong_root=args.strong_root)
        
        n_classes = len(label_index)
        logger.info(f"Found {n_classes} species")
        
        # Find NPZ files
        if args.emb_glob:
            npz_pattern = args.emb_glob
        else:
            npz_pattern = str(Path(args.emb_dir) / "**/*.embeddings.npz")

        if args.small_train:
            logger.info("Using small training set for quick testing")
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
        
        in_dim = dataset.get_embedding_dim()
        logger.info(f"Dataset: {len(dataset)} samples, embedding dim: {in_dim}")
        
        # Split into train/val
        if args.val_split > 0:
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            logger.info(f"Train/Val split: {train_size}/{val_size}")
        else:
            train_dataset = dataset
            val_dataset = None
        
        # Create data loaders
        num_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) - 1)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
        
        # Train each pooler
        results = {}
        
        for pool_name in args.poolers:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training pooler: {pool_name.upper()}")
            logger.info(f"{'='*50}")
            
            # Create model
            head = PoolingHead(
                in_dim=in_dim,
                n_classes=n_classes,
                pool=pool_name,
            )
            
            # Create trainer
            trainer = Trainer(
                head=head,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=args.lr,
                device=device,
                out_dir=run_dir,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_run_id=run_id,
                wandb_group=run_id,
                wandb_config={
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "n_samples": len(dataset),
                    "val_split": args.val_split,
                    "seed": args.seed,
                    "run_id": run_id,
                },
            )
            
            # Train
            losses = trainer.train(
                epochs=args.epochs,
                save_every=args.save_every,
                early_stop_patience=args.early_stop,
            )
            
            # Evaluate pointing game
            if args.eval_pointing:
                pg_acc, _, _ = pointing_game(train_loader, head, device)
                results[pool_name] = {
                    "final_train_loss": losses["train"][-1],
                    "pointing_game": pg_acc,
                }
            else:
                results[pool_name] = {
                    "final_train_loss": losses["train"][-1],
                }
            
            logger.info(f"Finished {pool_name}: final loss = {losses['train'][-1]:.4f}")
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("Training Summary")
        logger.info("="*50)
        
        for pool_name, res in results.items():
            line = f"{pool_name:12s} | loss: {res['final_train_loss']:.4f}"
            if "pointing_game" in res:
                line += f" | pointing: {res['pointing_game']:.3f}"
            logger.info(line)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
