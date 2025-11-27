#!/usr/bin/env python3
"""
CLI script for visualizing attention weights from trained MIL models.

Usage:
    # Basic attention plot
    python scripts/plot_attention.py \
        --npz /data/embeddings/REC_001.embeddings.npz \
        --checkpoint runs/attn_last.pt \
        --class "Boana faber" \
        --out runs/attention_REC_001.png

    # With spectrogram overlay
    python scripts/plot_attention.py \
        --npz /data/embeddings/REC_001.embeddings.npz \
        --checkpoint runs/attn_last.pt \
        --class "Boana faber" \
        --audio /data/anuraset/wavs/SITE_A/REC_001.wav \
        --spectrogram \
        --out runs/attention_REC_001_spec.png

    # With ground truth highlighting and W&B logging
    python scripts/plot_attention.py \
        --npz /data/embeddings/REC_001.embeddings.npz \
        --checkpoint runs/attn_last.pt \
        --class "Boana faber" \
        --strong /data/anuraset/strong_labels/SITE_A/REC_001.txt \
        --wandb --wandb_project bird-mil
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.train import load_checkpoint
from mil.datasets import load_embeddings_npz, parse_strong_labels
from mil.evaluate import (
    get_clip_attention,
    plot_attention,
    compute_spectrogram,
    deletion_insertion_curves,
    plot_deletion_insertion,
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_ground_truth_spans(
    strong_path: str | Path,
    class_name: str,
) -> List[Tuple[float, float]]:
    """Get ground truth event spans for a specific class."""
    events = parse_strong_labels(strong_path)
    spans = [(start, end) for start, end, species in events if species == class_name]
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot attention weights from trained MIL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--npz", type=str, required=True,
        help="Path to .embeddings.npz file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--class", dest="class_name", type=str, required=True,
        help="Class name (species) to visualize",
    )
    
    # Output
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for attention plot (default: auto-generated)",
    )
    
    # Optional enhancements
    parser.add_argument(
        "--audio", type=str, default=None,
        help="Path to original audio file (for spectrogram)",
    )
    parser.add_argument(
        "--spectrogram", action="store_true",
        help="Include mel spectrogram in plot (requires --audio)",
    )
    parser.add_argument(
        "--strong", type=str, default=None,
        help="Path to strong label file (for ground truth shading)",
    )
    parser.add_argument(
        "--del_ins", action="store_true",
        help="Also compute deletion/insertion curves",
    )
    
    # Model/processing options
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output figure DPI (default: 150)",
    )
    
    # W&B options
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log plots to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="bird-mil",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="W&B entity/team name",
    )
    
    # Other
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if args.spectrogram and args.audio is None:
        parser.error("--spectrogram requires --audio")
    
    npz_path = Path(args.npz)
    if not npz_path.exists():
        logger.error(f"NPZ file not found: {npz_path}")
        return 1
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return 1
    
    try:
        # Load model
        logger.info(f"Loading model from {ckpt_path}")
        head, checkpoint = load_checkpoint(ckpt_path, args.device)
        
        # Load embeddings
        logger.info(f"Loading embeddings from {npz_path}")
        embeddings, start_sec, end_sec = load_embeddings_npz(npz_path)
        
        # Find class index
        # We need to get the label index from somewhere - check if it's in the dataset
        # For now, we'll try to infer from checkpoint or use a heuristic
        # This is a limitation - ideally we'd save label_index in checkpoint
        
        # Try to find species list from strong labels
        species_to_idx = None
        if args.strong:
            from mil.datasets import build_label_index
            strong_root = Path(args.strong).parent.parent  # Go up from file to root
            species_to_idx = build_label_index(strong_root=strong_root)
        
        if species_to_idx is None:
            # Fall back to using class index as class name
            logger.warning("Could not determine species index. Using class 0.")
            class_idx = 0
        elif args.class_name in species_to_idx:
            class_idx = species_to_idx[args.class_name]
        else:
            # Try to find by partial match
            matches = [sp for sp in species_to_idx.keys() if args.class_name.lower() in sp.lower()]
            if matches:
                class_idx = species_to_idx[matches[0]]
                logger.info(f"Matched class '{args.class_name}' to '{matches[0]}'")
            else:
                logger.error(f"Class '{args.class_name}' not found in label index")
                logger.info(f"Available classes: {list(species_to_idx.keys())[:10]}...")
                return 1
        
        # Get attention weights
        logger.info(f"Computing attention for class {class_idx} ({args.class_name})")
        attention = get_clip_attention(head, embeddings, class_idx, args.device)
        
        # Compute time points (center of each chunk)
        times = (start_sec + end_sec) / 2
        
        # Get ground truth spans if available
        gt_spans = None
        if args.strong and Path(args.strong).exists():
            gt_spans = get_ground_truth_spans(args.strong, args.class_name)
            if gt_spans:
                logger.info(f"Found {len(gt_spans)} ground truth events")
        
        # Compute spectrogram if requested
        spectrogram = None
        spec_times = None
        spec_freqs = None
        
        if args.spectrogram and args.audio:
            logger.info("Computing spectrogram...")
            spectrogram, spec_times, spec_freqs = compute_spectrogram(args.audio)
        
        # Generate output path
        if args.out:
            out_path = Path(args.out)
        else:
            stem = npz_path.stem.replace(".embeddings", "")
            class_clean = args.class_name.replace(" ", "_").replace("/", "-")
            out_path = Path("runs") / f"attention_{stem}_{class_clean}.png"
        
        # Plot attention
        logger.info(f"Saving attention plot to {out_path}")
        plot_attention(
            attention_weights=attention,
            times=times,
            class_name=args.class_name,
            out_path=out_path,
            ground_truth_spans=gt_spans,
            spectrogram=spectrogram,
            spec_times=spec_times,
            spec_freqs=spec_freqs,
            dpi=args.dpi,
        )
        
        # Deletion/insertion curves
        if args.del_ins:
            logger.info("Computing deletion/insertion curves...")
            fractions, del_scores, ins_scores = deletion_insertion_curves(
                head, embeddings, class_idx, args.device
            )
            
            del_ins_path = out_path.parent / f"{out_path.stem}_del_ins.png"
            plot_deletion_insertion(
                fractions, del_scores, ins_scores,
                del_ins_path, args.class_name, args.dpi
            )
        
        # W&B logging
        if args.wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb not installed, skipping W&B logging")
            else:
                try:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        name=f"attention_{npz_path.stem}_{args.class_name}",
                        job_type="visualization",
                    )
                    
                    wandb.log({
                        "attention_plot": wandb.Image(str(out_path)),
                        "class_name": args.class_name,
                        "npz_file": str(npz_path),
                    })
                    
                    if args.del_ins:
                        from mil.evaluate import compute_auc
                        wandb.log({
                            "deletion_insertion_plot": wandb.Image(str(del_ins_path)),
                            "deletion_auc": compute_auc(fractions, del_scores),
                            "insertion_auc": compute_auc(fractions, ins_scores),
                        })
                    
                    wandb.finish()
                    logger.info("Logged to W&B")
                    
                except Exception as e:
                    logger.warning(f"Failed to log to W&B: {e}")
        
        logger.info("Done!")
        return 0
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
