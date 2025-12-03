"""
Training loop for MIL models.

This module provides the training infrastructure for MIL pooling heads
with optional Weights & Biases integration for experiment tracking.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .heads import PoolingHead, PROB_SPACE_POOLERS

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_criterion(pool_name: str) -> nn.Module:
    """
    Get appropriate loss criterion for a pooler.
    
    Args:
        pool_name: Name of the pooling method.
        
    Returns:
        BCELoss for probability-space poolers, BCEWithLogitsLoss otherwise.
    """
    if pool_name in PROB_SPACE_POOLERS:
        return nn.BCELoss()
    return nn.BCEWithLogitsLoss()


def train_epoch(
    loader: DataLoader,
    head: PoolingHead,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Train for one epoch.
    
    Args:
        loader: DataLoader for training data.
        head: PoolingHead model.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        
    Returns:
        Average loss for the epoch.
    """
    head.train()
    total_loss = 0.0
    n_samples = 0
    
    for embeddings, weak_labels, _, _ in loader:

        embeddings = embeddings.to(device, non_blocking=True)
        weak_labels = weak_labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        y_clip, _ = head(embeddings)
        loss = criterion(y_clip, weak_labels)
        loss.backward()
        optimizer.step()
        
        batch_size = embeddings.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    
    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate_epoch(
    loader: DataLoader,
    head: PoolingHead,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Evaluate for one epoch.
    
    Args:
        loader: DataLoader for validation data.
        head: PoolingHead model.
        criterion: Loss function.
        device: Device to evaluate on.
        
    Returns:
        Average loss for the epoch.
    """
    head.eval()
    total_loss = 0.0
    n_samples = 0
    
    for embeddings, weak_labels, _, _ in loader:
        embeddings = embeddings.to(device)
        weak_labels = weak_labels.to(device)
        
        y_clip, _ = head(embeddings)
        loss = criterion(y_clip, weak_labels)
        
        batch_size = embeddings.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    
    return total_loss / max(n_samples, 1)


def save_loss_csv(
    losses: Dict[str, List[float]],
    out_path: Path,
) -> None:
    """Save loss values to CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        
        train_losses = losses.get("train", [])
        val_losses = losses.get("val", [])
        
        for i, train_loss in enumerate(train_losses):
            val_loss = val_losses[i] if i < len(val_losses) else ""
            writer.writerow([i + 1, train_loss, val_loss])
    
    logger.info(f"Saved loss CSV to {out_path}")


def save_loss_plot(
    losses: Dict[str, List[float]],
    out_path: Path,
    title: str = "Training Loss",
) -> None:
    """Save loss curve plot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = list(range(1, len(losses.get("train", [])) + 1))
    
    if "train" in losses:
        ax.plot(epochs, losses["train"], label="Train", marker="o", markersize=4)
    if "val" in losses and losses["val"]:
        ax.plot(epochs, losses["val"], label="Val", marker="s", markersize=4)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    
    logger.info(f"Saved loss plot to {out_path}")


class Trainer:
    """
    Trainer for MIL pooling heads.
    
    Supports training, validation, checkpointing, and W&B logging.
    """
    
    def __init__(
        self,
        head: PoolingHead,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        device: str = "cpu",
        out_dir: Path | str = "runs",
        use_wandb: bool = False,
        wandb_project: str = "bird-mil",
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            head: PoolingHead model to train.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            lr: Learning rate.
            device: Device to train on.
            out_dir: Output directory for checkpoints and logs.
            use_wandb: Whether to use Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity/team name.
            wandb_config: Additional config to log to W&B.
        """
        self.head = head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = torch.optim.Adam(head.parameters(), lr=lr)
        self.criterion = get_criterion(head.pool_name)
        
        self.losses: Dict[str, List[float]] = {"train": [], "val": []}
        
        # W&B setup
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        
        if use_wandb:
            if not WANDB_AVAILABLE:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False
            else:
                try:
                    config = {
                        "pool_name": head.pool_name,
                        "n_classes": head.n_classes,
                        "in_dim": head.in_dim,
                        "lr": lr,
                        **(wandb_config or {}),
                    }
                    
                    self.wandb_run = wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        config=config,
                        name=f"{head.pool_name}",
                        reinit=True,
                    )
                    logger.info(f"W&B run initialized: {self.wandb_run.name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize W&B: {e}")
                    self.use_wandb = False
    
    def train(
        self,
        epochs: int = 20,
        save_every: int = 0,
        early_stop_patience: int = 0,
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            epochs: Number of epochs to train.
            save_every: Save checkpoint every N epochs (0 to disable).
            early_stop_patience: Stop if val loss doesn't improve for N epochs (0 to disable).
            
        Returns:
            Dictionary of loss histories.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss = train_epoch(
                self.train_loader, self.head, self.optimizer, self.criterion, self.device
            )
            self.losses["train"].append(train_loss)
            
            # Validation
            val_loss = None
            if self.val_loader is not None:
                val_loss = evaluate_epoch(
                    self.val_loader, self.head, self.criterion, self.device
                )
                self.losses["val"].append(val_loss)
            
            # Logging
            log_msg = f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            logger.info(log_msg)
            
            # W&B logging
            if self.use_wandb and self.wandb_run and wandb is not None:
                log_dict = {"epoch": epoch, "train_loss": train_loss}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                wandb.log(log_dict)
            
            # Checkpointing
            if save_every > 0 and epoch % save_every == 0:
                self.save_checkpoint(f"{self.head.pool_name}_epoch{epoch}.pt")
            
            # Early stopping
            if early_stop_patience > 0 and val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint(f"{self.head.pool_name}_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        # Save final checkpoint and plots
        self.save_checkpoint(f"{self.head.pool_name}_last.pt")
        self.save_results()
        
        # Finish W&B run
        if self.use_wandb and self.wandb_run and wandb is not None:
            # Log final artifacts
            loss_csv = self.out_dir / f"loss_{self.head.pool_name}.csv"
            loss_png = self.out_dir / f"loss_{self.head.pool_name}.png"
            
            if loss_csv.exists():
                wandb.save(str(loss_csv))
            if loss_png.exists():
                wandb.log({"loss_curve": wandb.Image(str(loss_png))})
            
            wandb.finish()
        
        return self.losses
    
    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        path = self.out_dir / filename
        torch.save({
            "model_state_dict": self.head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pool_name": self.head.pool_name,
            "n_classes": self.head.n_classes,
            "in_dim": self.head.in_dim,
            "losses": self.losses,
        }, path)
        logger.debug(f"Saved checkpoint to {path}")
        return path
    
    def save_results(self) -> Tuple[Path, Path]:
        """Save loss CSV and plot."""
        csv_path = self.out_dir / f"loss_{self.head.pool_name}.csv"
        png_path = self.out_dir / f"loss_{self.head.pool_name}.png"
        
        save_loss_csv(self.losses, csv_path)
        save_loss_plot(
            self.losses, png_path,
            title=f"Training Loss - {self.head.pool_name.upper()}"
        )
        
        return csv_path, png_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Tuple[PoolingHead, Dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.
        
    Returns:
        Tuple of (model, checkpoint_dict).
        
    Note:
        This function loads checkpoints that contain metadata (pool_name, etc.)
        in addition to model weights. Only load checkpoints from trusted sources.
    """
    # Note: We need weights_only=False because our checkpoints contain
    # metadata like pool_name, n_classes, etc. in addition to model weights.
    # Only load checkpoints from trusted sources.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    head = PoolingHead(
        in_dim=checkpoint["in_dim"],
        n_classes=checkpoint["n_classes"],
        pool=checkpoint["pool_name"],
    )
    head.load_state_dict(checkpoint["model_state_dict"])
    head.to(device)
    head.eval()
    
    return head, checkpoint
