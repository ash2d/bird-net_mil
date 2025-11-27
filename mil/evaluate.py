"""
Evaluation utilities for MIL models.

This module provides evaluation metrics and visualization tools:
- Pointing game: Does the model attend to labeled events?
- Deletion/Insertion: Faithfulness metrics
- Attention visualization: Plot attention over time with optional spectrogram
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .heads import PoolingHead

logger = logging.getLogger(__name__)


@torch.no_grad()
def pointing_game(
    loader: DataLoader,
    head: PoolingHead,
    device: str = "cpu",
    use_attention: bool = True,
) -> Tuple[float, int, int]:
    """
    Evaluate pointing game accuracy.
    
    For each positive class in each clip, check if the most-salient second
    (by attention weight or max logit) falls within a labeled event.
    
    Args:
        loader: DataLoader yielding (embeddings, weak_labels, time_labels, times).
        head: Trained PoolingHead model.
        device: Device to run on.
        use_attention: If True and head is attention-based, use attention weights.
                      Otherwise use max per-second logit.
        
    Returns:
        Tuple of (accuracy, hits, total) where accuracy = hits/total.
    """
    head.eval()
    head.to(device)
    
    hits = 0
    total = 0
    
    for embeddings, weak_labels, time_labels, _ in loader:
        embeddings = embeddings.to(device)
        B, T, D = embeddings.shape
        C = weak_labels.shape[1]
        
        # Get per-second scores
        if use_attention and head.get_attention_weights(embeddings) is not None:
            # Use attention weights
            scores = head.get_attention_weights(embeddings)  # (B, T, C)
        else:
            # Use per-second logits
            _, z = head(embeddings)  # z: (B, T, C)
            scores = z
        
        # Find top-scoring second for each class
        t_star = scores.argmax(dim=1).cpu().numpy()  # (B, C)
        
        # Check hits
        time_labels_np = time_labels.numpy()
        weak_labels_np = weak_labels.numpy()
        
        for b in range(B):
            for c in range(C):
                # Only evaluate positive classes
                if weak_labels_np[b, c] < 0.5:
                    continue
                
                t = int(t_star[b, c])
                
                # Check if selected second overlaps with ground truth
                if t < time_labels_np.shape[1] and time_labels_np[b, t, c] > 0.5:
                    hits += 1
                total += 1
    
    accuracy = hits / max(total, 1)
    logger.info(f"Pointing game: {hits}/{total} = {accuracy:.3f}")
    
    return accuracy, hits, total


def plot_attention(
    attention_weights: np.ndarray,
    times: np.ndarray,
    class_name: str,
    out_path: Path | str,
    ground_truth_spans: Optional[List[Tuple[float, float]]] = None,
    spectrogram: Optional[np.ndarray] = None,
    spec_times: Optional[np.ndarray] = None,
    spec_freqs: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> Path:
    """
    Plot attention weights over time with optional spectrogram overlay.
    
    Args:
        attention_weights: (T,) array of attention weights for one class.
        times: (T,) array of time points (center of each second).
        class_name: Name of the class for labeling.
        out_path: Path to save the figure.
        ground_truth_spans: Optional list of (start, end) tuples for ground truth events.
        spectrogram: Optional (F, T') mel spectrogram array.
        spec_times: Time axis for spectrogram.
        spec_freqs: Frequency axis for spectrogram.
        title: Optional custom title.
        figsize: Figure size (width, height).
        dpi: Output resolution.
        
    Returns:
        Path to saved figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    if spectrogram is not None:
        fig, (ax_spec, ax_attn) = plt.subplots(
            2, 1, figsize=figsize, sharex=True,
            gridspec_kw={"height_ratios": [1, 1]}
        )
    else:
        fig, ax_attn = plt.subplots(figsize=(figsize[0], figsize[1] / 2))
        ax_spec = None
    
    # Plot spectrogram if provided
    if ax_spec is not None and spectrogram is not None:
        if spec_times is not None and spec_freqs is not None:
            extent = [spec_times[0], spec_times[-1], spec_freqs[0], spec_freqs[-1]]
            ax_spec.imshow(
                spectrogram, aspect="auto", origin="lower",
                extent=extent, cmap="magma"
            )
            ax_spec.set_ylabel("Frequency (Hz)")
        else:
            ax_spec.imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
            ax_spec.set_ylabel("Frequency bin")
        
        ax_spec.set_title(title or f"Attention for {class_name}")
        
        # Shade ground truth spans
        if ground_truth_spans:
            for start, end in ground_truth_spans:
                ax_spec.axvspan(start, end, alpha=0.3, color="green", label="GT")
    
    # Plot attention weights
    ax_attn.bar(times, attention_weights, width=times[1] - times[0] if len(times) > 1 else 1.0,
                alpha=0.7, color="steelblue", edgecolor="none")
    ax_attn.set_xlabel("Time (s)")
    ax_attn.set_ylabel("Attention")
    
    if ax_spec is None:
        ax_attn.set_title(title or f"Attention for {class_name}")
    
    # Shade ground truth spans
    if ground_truth_spans:
        for i, (start, end) in enumerate(ground_truth_spans):
            label = "Ground truth" if i == 0 else None
            ax_attn.axvspan(start, end, alpha=0.2, color="green", label=label)
        ax_attn.legend(loc="upper right")
    
    ax_attn.set_xlim(times[0] - 0.5, times[-1] + 0.5)
    ax_attn.set_ylim(0, max(attention_weights) * 1.1)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved attention plot to {out_path}")
    return out_path


def compute_spectrogram(
    audio_path: str | Path,
    sr: int = 32000,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mel spectrogram for visualization.
    
    Uses torchaudio for computation.
    
    Args:
        audio_path: Path to audio file.
        sr: Sample rate.
        n_mels: Number of mel bands.
        hop_length: Hop length in samples.
        n_fft: FFT window size.
        
    Returns:
        Tuple of (spectrogram, times, frequencies).
    """
    import torchaudio
    
    wav, orig_sr = torchaudio.load(str(audio_path))
    
    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if orig_sr != sr:
        wav = torchaudio.functional.resample(wav, orig_sr, sr)
    
    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    
    mel_spec = mel_transform(wav)  # (1, n_mels, T)
    
    # Convert to dB
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    mel_spec_db = amplitude_to_db(mel_spec)
    
    spec = mel_spec_db.squeeze(0).numpy()  # (n_mels, T)
    
    # Compute time and frequency axes
    n_frames = spec.shape[1]
    times = np.arange(n_frames) * hop_length / sr
    frequencies = np.linspace(0, sr / 2, n_mels)
    
    return spec, times, frequencies


@torch.no_grad()
def get_clip_attention(
    head: PoolingHead,
    embeddings: np.ndarray | torch.Tensor,
    class_idx: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Get attention weights for a single clip and class.
    
    Args:
        head: Trained PoolingHead with attention pooler.
        embeddings: (T, D) embeddings array.
        class_idx: Class index to get attention for.
        device: Device to run on.
        
    Returns:
        (T,) array of attention weights.
    """
    head.eval()
    head.to(device)
    
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    embeddings = embeddings.unsqueeze(0).to(device)  # (1, T, D)
    
    # Get attention weights
    attn = head.get_attention_weights(embeddings)  # (1, T, C) or None
    
    if attn is None:
        # Fall back to per-second logits
        _, z = head(embeddings)
        attn = torch.softmax(z, dim=1)  # (1, T, C)
    
    return attn[0, :, class_idx].cpu().numpy()


# ============================================================================
# Deletion/Insertion Faithfulness Metrics (Stretch Goal)
# ============================================================================

@torch.no_grad()
def deletion_insertion_curves(
    head: PoolingHead,
    embeddings: np.ndarray | torch.Tensor,
    class_idx: int,
    device: str = "cpu",
    n_steps: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deletion and insertion curves for faithfulness evaluation.
    
    Deletion: Progressively remove most important time steps.
    Insertion: Progressively add most important time steps.
    
    Args:
        head: Trained PoolingHead model.
        embeddings: (T, D) embeddings array.
        class_idx: Class index to evaluate.
        device: Device to run on.
        n_steps: Number of steps in the curve.
        
    Returns:
        Tuple of (fractions, deletion_scores, insertion_scores).
    """
    head.eval()
    head.to(device)
    
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    
    T, D = embeddings.shape
    embeddings = embeddings.to(device)
    
    # Get importance scores (attention or logits)
    emb_batch = embeddings.unsqueeze(0)  # (1, T, D)
    
    attn = head.get_attention_weights(emb_batch)
    if attn is not None:
        importance = attn[0, :, class_idx].cpu().numpy()
    else:
        _, z = head(emb_batch)
        importance = z[0, :, class_idx].cpu().numpy()
    
    # Sort by importance (descending)
    order = np.argsort(-importance)
    
    # Compute baseline score
    y_base, _ = head(emb_batch)
    if head.returns_probabilities:
        base_score = y_base[0, class_idx].item()
    else:
        base_score = torch.sigmoid(y_base[0, class_idx]).item()
    
    # Deletion curve: remove most important first
    deletion_scores = [base_score]
    mask = torch.ones(T, dtype=torch.bool, device=device)
    
    step_size = max(1, T // n_steps)
    
    for i in range(0, T, step_size):
        # Mask out the next batch of important steps
        end = min(i + step_size, T)
        for j in range(i, end):
            mask[order[j]] = False
        
        # Compute score with masked embeddings (zero out removed)
        masked_emb = embeddings.clone()
        masked_emb[~mask] = 0
        
        y, _ = head(masked_emb.unsqueeze(0))
        if head.returns_probabilities:
            score = y[0, class_idx].item()
        else:
            score = torch.sigmoid(y[0, class_idx]).item()
        
        deletion_scores.append(score)
    
    # Insertion curve: add most important first
    insertion_scores = [0.0]  # Start with zero embedding
    mask = torch.zeros(T, dtype=torch.bool, device=device)
    
    for i in range(0, T, step_size):
        # Add the next batch of important steps
        end = min(i + step_size, T)
        for j in range(i, end):
            mask[order[j]] = True
        
        # Compute score with only inserted embeddings
        masked_emb = torch.zeros_like(embeddings)
        masked_emb[mask] = embeddings[mask]
        
        y, _ = head(masked_emb.unsqueeze(0))
        if head.returns_probabilities:
            score = y[0, class_idx].item()
        else:
            score = torch.sigmoid(y[0, class_idx]).item()
        
        insertion_scores.append(score)
    
    # Create fraction axis
    fractions = np.linspace(0, 1, len(deletion_scores))
    
    return fractions, np.array(deletion_scores), np.array(insertion_scores)


def compute_auc(fractions: np.ndarray, scores: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    return float(np.trapz(scores, fractions))


def plot_deletion_insertion(
    fractions: np.ndarray,
    deletion_scores: np.ndarray,
    insertion_scores: np.ndarray,
    out_path: Path | str,
    class_name: str = "",
    dpi: int = 150,
) -> Path:
    """
    Plot deletion and insertion curves.
    
    Args:
        fractions: X-axis values (fraction of time steps modified).
        deletion_scores: Y-axis values for deletion curve.
        insertion_scores: Y-axis values for insertion curve.
        out_path: Path to save figure.
        class_name: Class name for title.
        dpi: Output resolution.
        
    Returns:
        Path to saved figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(fractions, deletion_scores, label="Deletion", marker="o", markersize=4)
    ax.plot(fractions, insertion_scores, label="Insertion", marker="s", markersize=4)
    
    del_auc = compute_auc(fractions, deletion_scores)
    ins_auc = compute_auc(fractions, insertion_scores)
    
    ax.set_xlabel("Fraction of time steps")
    ax.set_ylabel("Prediction score")
    ax.set_title(f"Deletion/Insertion Curves{' - ' + class_name if class_name else ''}")
    ax.legend(title=f"AUC: Del={del_auc:.3f}, Ins={ins_auc:.3f}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved deletion/insertion plot to {out_path}")
    return out_path
