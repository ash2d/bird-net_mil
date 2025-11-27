"""
BirdNET V3 model loading utilities.

This module provides functions to load the BirdNET V3 TorchScript model
with automatic device detection and optional model download.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Tuple

import torch

logger = logging.getLogger(__name__)

# Default model paths and URLs
DEFAULT_MODEL_PATH = "models/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt"
DEFAULT_MODEL_URL = (
    "https://zenodo.org/records/17631020/files/"
    "BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt?download=1"
)


def _download_model(url: str, dst: Path) -> bool:
    """
    Download model file from URL to destination path.
    
    Args:
        url: URL to download from.
        dst: Destination file path.
        
    Returns:
        True if download succeeded, False otherwise.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, dir=dst.parent, suffix=".pt.tmp"
        ) as tmp:
            tmp_path = Path(tmp.name)
            logger.info(f"Downloading model from {url}")
            with urllib.request.urlopen(url, timeout=300) as response:
                shutil.copyfileobj(response, tmp)
        os.replace(tmp_path, dst)
        logger.info(f"Model saved to {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()
        return False


def load_birdnet_model(
    model_path: str | Path | None = None,
    device: str | None = None,
    auto_download: bool = True,
) -> Tuple[Callable, str]:
    """
    Load BirdNET V3 TorchScript model.
    
    The model returns (embeddings, predictions) where:
    - embeddings: (B, D) tensor of per-chunk embeddings
    - predictions: (B, C) tensor of post-sigmoid confidences in [0,1]
    
    Args:
        model_path: Path to the TorchScript .pt file. If None, uses default path.
        device: Device to load model on ('cpu', 'cuda', or None for auto-detect).
        auto_download: If True, download default model if not found.
        
    Returns:
        Tuple of (model, device_str) where model is callable and device_str
        is the actual device used.
        
    Raises:
        FileNotFoundError: If model file not found and auto_download fails.
        RuntimeError: If model loading fails.
    """
    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Resolve model path
    model_path = Path(model_path) if model_path else Path(DEFAULT_MODEL_PATH)
    
    # Auto-download if needed
    if not model_path.exists():
        if auto_download and str(model_path) == DEFAULT_MODEL_PATH:
            logger.info(f"Model not found at {model_path}, attempting download...")
            if not _download_model(DEFAULT_MODEL_URL, model_path):
                raise FileNotFoundError(
                    f"Model file not found at {model_path} and download failed. "
                    f"Please download manually from {DEFAULT_MODEL_URL}"
                )
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    logger.info(f"Loading model from {model_path} on {device}")
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    logger.info(f"Model loaded successfully on {device}")
    return model, device
