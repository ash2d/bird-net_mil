"""BirdNET V3 embedding extraction utilities."""

from .model_loader import load_birdnet_model
from .utils_audio import load_audio, frame_audio
from .embed_all import embed_file, embed_directory

__all__ = [
    "load_birdnet_model",
    "load_audio",
    "frame_audio",
    "embed_file",
    "embed_directory",
]
