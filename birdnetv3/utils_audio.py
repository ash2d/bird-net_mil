"""
Audio loading and preprocessing utilities for BirdNET V3.

This module provides functions to load, resample, and frame audio
for processing with the BirdNET V3 model which expects 32 kHz mono audio.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# BirdNET V3 expects 32 kHz mono audio
TARGET_SR = 32000


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        path: Path to audio file (supports wav, mp3, flac, ogg, etc.).
        target_sr: Target sample rate in Hz. Default is 32000 for BirdNET V3.
        
    Returns:
        Tuple of (waveform, sample_rate) where waveform is a 1D float32 tensor
        and sample_rate is the target sample rate.
        
    Raises:
        RuntimeError: If audio file cannot be loaded.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path}: {e}") from e
    
    # Convert to mono by averaging channels
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    
    # Return as 1D tensor
    return wav.squeeze(0), sr


def frame_audio(
    wav: torch.Tensor,
    sr: int,
    chunk_length: float,
    overlap: float = 0.0,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Split audio waveform into fixed-length chunks with optional overlap.
    
    Args:
        wav: 1D waveform tensor.
        sr: Sample rate.
        chunk_length: Length of each chunk in seconds (must be > 0).
        overlap: Overlap between consecutive chunks in seconds 
                 (must be >= 0 and < chunk_length).
        
    Returns:
        Tuple of (chunks, start_sec, end_sec) where:
        - chunks: (T, n_samples) tensor of audio chunks
        - start_sec: (T,) array of start times in seconds
        - end_sec: (T,) array of end times in seconds
        
    Raises:
        ValueError: If chunk_length or overlap values are invalid.
    """
    if chunk_length <= 0:
        raise ValueError("chunk_length must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_length:
        raise ValueError("overlap must be < chunk_length")
    
    n_per_chunk = int(round(chunk_length * sr))
    hop = int(round((chunk_length - overlap) * sr))
    
    if hop <= 0:
        raise ValueError("Invalid step size (adjust overlap/chunk_length)")
    
    n = wav.numel()
    if n == 0:
        return (
            torch.zeros((0, n_per_chunk), dtype=torch.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    
    # Calculate number of chunks
    n_chunks = 1 + max(0, math.ceil((n - n_per_chunk) / hop))
    
    # Calculate start and end positions
    starts = np.arange(n_chunks) * hop
    ends = starts + n_per_chunk
    
    # Pad waveform if needed for last chunk
    pad_needed = max(0, int(ends[-1]) - n)
    if pad_needed > 0:
        wav = torch.nn.functional.pad(wav, (0, pad_needed))
    
    # Extract chunks
    chunks = torch.stack([
        wav[int(s):int(e)] for s, e in zip(starts, ends)
    ], dim=0)
    
    # Convert to seconds (cap end times to actual audio length)
    start_sec = (starts / sr).astype(np.float32)
    end_sec = (np.minimum(ends, n) / sr).astype(np.float32)
    
    return chunks, start_sec, end_sec
