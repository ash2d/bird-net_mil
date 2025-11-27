"""
Batch embedding extraction for BirdNET V3.

This module provides functions to extract embeddings from audio files
and save them as .npz files for efficient downstream processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List

import numpy as np
import torch

from .model_loader import load_birdnet_model
from .utils_audio import load_audio, frame_audio, TARGET_SR

logger = logging.getLogger(__name__)


@torch.inference_mode()
def embed_file(
    wav_path: str | Path,
    out_path: str | Path,
    model: torch.nn.Module | None = None,
    device: str | None = None,
    model_path: str | Path | None = None,
    chunk_length: float = 1.0,
    overlap: float = 0.0,
    batch_size: int = 64,
) -> Path:
    """
    Extract embeddings from an audio file and save to .npz format.
    
    The output .npz file contains:
    - embeddings: (T, D) array of per-chunk embeddings
    - start_sec: (T,) array of chunk start times
    - end_sec: (T,) array of chunk end times
    - sr: sample rate used
    
    Args:
        wav_path: Path to input audio file.
        out_path: Path for output .npz file.
        model: Pre-loaded BirdNET model (optional, will load if None).
        device: Device to use ('cpu', 'cuda', or None for auto).
        model_path: Path to model file (only used if model is None).
        chunk_length: Length of each chunk in seconds.
        overlap: Overlap between chunks in seconds.
        batch_size: Batch size for inference.
        
    Returns:
        Path to the saved .npz file.
        
    Raises:
        FileNotFoundError: If audio file not found.
        RuntimeError: If embedding extraction fails.
    """
    wav_path = Path(wav_path)
    out_path = Path(out_path)
    
    # Load model if not provided
    if model is None:
        model, device = load_birdnet_model(model_path, device)
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and frame audio
    wav, sr = load_audio(wav_path, TARGET_SR)
    chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length, overlap)
    
    if len(chunks) == 0:
        logger.warning(f"No audio chunks to process for {wav_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            embeddings=np.zeros((0, 0), dtype=np.float32),
            start_sec=start_sec,
            end_sec=end_sec,
            sr=sr,
        )
        return out_path
    
    # Extract embeddings in batches
    embeddings_list: List[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size].to(device)
        emb, _ = model(batch)  # predictions are confidences; ignore
        embeddings_list.append(emb.detach().cpu().numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    
    # Save to .npz
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        embeddings=embeddings,
        start_sec=start_sec,
        end_sec=end_sec,
        sr=sr,
    )
    
    logger.debug(f"Saved {embeddings.shape[0]} embeddings to {out_path}")
    return out_path


def _iter_wav_files(
    wav_dir: str | Path | None = None,
    glob_pattern: str = "**/*.wav",
    list_file: str | Path | None = None,
) -> Generator[Path, None, None]:
    """
    Iterate over WAV files from directory glob or list file.
    
    Args:
        wav_dir: Directory to search for WAV files.
        glob_pattern: Glob pattern for finding WAV files.
        list_file: Path to text file with one WAV path per line.
        
    Yields:
        Path objects for each WAV file.
    """
    if list_file is not None:
        list_file = Path(list_file)
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield Path(line)
    elif wav_dir is not None:
        wav_dir = Path(wav_dir)
        yield from sorted(wav_dir.glob(glob_pattern))
    else:
        raise ValueError("Must provide either wav_dir or list_file")


def embed_directory(
    wav_dir: str | Path | None = None,
    out_dir: str | Path = "embeddings",
    glob_pattern: str = "**/*.wav",
    list_file: str | Path | None = None,
    model_path: str | Path | None = None,
    device: str | None = None,
    chunk_length: float = 1.0,
    overlap: float = 0.0,
    batch_size: int = 64,
    preserve_structure: bool = True,
) -> int:
    """
    Batch extract embeddings from multiple audio files.
    
    Args:
        wav_dir: Directory containing audio files.
        out_dir: Output directory for .npz files.
        glob_pattern: Glob pattern for finding audio files.
        list_file: Alternative: text file with paths to process.
        model_path: Path to BirdNET model file.
        device: Device to use for inference.
        chunk_length: Length of each chunk in seconds.
        overlap: Overlap between chunks in seconds.
        batch_size: Batch size for inference.
        preserve_structure: If True, preserve subdirectory structure from wav_dir.
        
    Returns:
        Number of files processed.
    """
    out_dir = Path(out_dir)
    wav_dir_path = Path(wav_dir) if wav_dir else None
    
    # Load model once
    model, device = load_birdnet_model(model_path, device)
    
    # Process files
    count = 0
    wav_files = list(_iter_wav_files(wav_dir_path, glob_pattern, list_file))
    total = len(wav_files)
    
    for i, wav_path in enumerate(wav_files):
        try:
            # Determine output path
            if preserve_structure and wav_dir_path:
                rel_path = wav_path.relative_to(wav_dir_path)
                out_path = out_dir / rel_path.with_suffix(".embeddings.npz")
            else:
                out_path = out_dir / (wav_path.stem + ".embeddings.npz")
            
            embed_file(
                wav_path=wav_path,
                out_path=out_path,
                model=model,
                device=device,
                chunk_length=chunk_length,
                overlap=overlap,
                batch_size=batch_size,
            )
            count += 1
            
            if (i + 1) % 50 == 0 or (i + 1) == total:
                logger.info(f"Processed {i + 1}/{total} files")
                
        except Exception as e:
            logger.error(f"Failed to process {wav_path}: {e}")
            continue
    
    logger.info(f"Completed: {count}/{total} files embedded successfully")
    return count
