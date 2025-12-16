"""
Dataset classes for MIL training with BirdNET embeddings.

This module provides dataset and data loading utilities for training
MIL models on pre-extracted embeddings with strong labels from AnuraSet.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def normalize_embedding_path(path: str | Path) -> str:
    """
    Normalize embedding paths for consistent matching.
    """
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return str(Path(path))


def load_embeddings_npz(npz_path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from .npz file.
    
    Args:
        npz_path: Path to .npz file created by embed_file.
        
    Returns:
        Tuple of (embeddings, start_sec, end_sec) where:
        - embeddings: (T, D) array of embeddings
        - start_sec: (T,) array of chunk start times
        - end_sec: (T,) array of chunk end times
    """
    data = np.load(npz_path)
    embeddings = data["embeddings"].astype(np.float32)
    start_sec = data["start_sec"].astype(np.float32)
    end_sec = data["end_sec"].astype(np.float32)
    return embeddings, start_sec, end_sec


def parse_strong_labels(txt_path: str | Path) -> List[Tuple[float, float, str, str]]:
    """
    Parse AnuraSet strong label file.
    
    Format: Each line contains "<start_sec> <end_sec> <species_quality>"
    where <species_quality> is formatted as "<species>_<quality>" with quality
    being one of L (low), M (medium), or H (high).
    
    Examples:
        "0.5 2.3 Boana_faber_H" -> (0.5, 2.3, "Boana_faber", "H")
        "1.0 3.5 Dendropsophus_minutus_M" -> (1.0, 3.5, "Dendropsophus_minutus", "M")
    
    Args:
        txt_path: Path to strong label .txt file.
        
    Returns:
        List of (start_sec, end_sec, species, quality) tuples where quality
        is one of 'L', 'M', 'H' or empty string if not present.
    """
    events = []
    txt_path = Path(txt_path)
    
    if not txt_path.exists():
        return events
    
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                species_quality = parts[2]
                
                # Parse species_quality format: <species>_<quality>
                # Quality is the last character after the last underscore (L, M, or H)
                species, quality = parse_species_quality(species_quality)
                events.append((start, end, species, quality))
            except ValueError:
                logger.warning(f"Failed to parse line in {txt_path}: {line}")
                continue
    
    return events


def parse_species_quality(species_quality: str) -> Tuple[str, str]:
    """
    Parse species_quality string into species and quality components.
    
    Format: "<species>_<quality>" where quality is one of L, M, H.
    If the last part after underscore is not a valid quality indicator,
    the entire string is treated as the species name.
    
    Args:
        species_quality: Combined species and quality string (e.g., "Boana_faber_H")
        
    Returns:
        Tuple of (species, quality) where quality is 'L', 'M', 'H', or ''
        
    Examples:
        "Boana_faber_H" -> ("Boana_faber", "H")
        "Dendropsophus_minutus_M" -> ("Dendropsophus_minutus", "M")
        "Species_name" -> ("Species_name", "")
    """
    valid_qualities = {'L', 'M', 'H'}
    
    if '_' not in species_quality:
        return species_quality, ''
    
    # Split from the right to get the last part
    last_underscore_idx = species_quality.rfind('_')
    potential_quality = species_quality[last_underscore_idx + 1:]
    
    if potential_quality in valid_qualities:
        species = species_quality[:last_underscore_idx]
        return species, potential_quality
    else:
        # Not a valid quality, treat entire string as species
        return species_quality, ''


def load_weak_labels_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load weak labels from CSV file.
    
    CSV format should have columns:
    - Either:
      - 'embedding_path': Full path to embedding file, plus SPECIES_* columns
      - or legacy columns 'MONITORING_SITE' and 'AUDIO_FILE_ID' plus SPECIES_* columns
    - 'SPECIES_<species_name>': Binary columns (0 or 1) for each species
    
    Args:
    csv_path: Path to weak label CSV file.
    
    Returns:
    DataFrame with weak labels.
    """
    df = pd.read_csv(csv_path)
    
    if "embedding_path" in df.columns:
        df["embedding_path"] = df["embedding_path"].astype(str)
        paths = df["embedding_path"]
        df["embedding_path_norm"] = [normalize_embedding_path(p) for p in paths]
    else:
        # Validate legacy required columns
        if 'MONITORING_SITE' not in df.columns:
            raise ValueError("CSV must have 'MONITORING_SITE' column")
        if 'AUDIO_FILE_ID' not in df.columns:
            raise ValueError("CSV must have 'AUDIO_FILE_ID' column")
    
    return df


def extract_species_from_weak_csv(df: pd.DataFrame) -> List[str]:
    """
    Extract species names from weak label CSV columns.
    
    Args:
        df: DataFrame with weak labels.
        
    Returns:
        List of species names (sorted).
    """
    species = []
    for col in df.columns:
        if col.startswith('SPECIES_'):
            species_name = col[8:]  # Remove 'SPECIES_' prefix
            species.append(species_name)
    
    return sorted(species)


def get_weak_labels_for_recording(
    df: pd.DataFrame,
    audio_file_id: str,
    label_index: Dict[str, int],
    embedding_path: str | Path | None = None,
) -> np.ndarray:
    """
    Get weak labels for a specific recording from the weak label DataFrame.
    
    Args:
        df: DataFrame with weak labels.
        audio_file_id: Audio file ID (without path, without _<start>_<end> suffix).
        label_index: Mapping from species to class index.
        embedding_path: Optional full path to embedding for path-based CSVs.
        
    Returns:
        (C,) binary array of weak labels, or all zeros if recording not found.
    """
    C = len(label_index)
    weak_labels = np.zeros((C,), dtype=np.float32)
    
    row = None

    # Path-based lookup if available
    if embedding_path is not None and "embedding_path" in df.columns:
        path_norm = normalize_embedding_path(embedding_path)
        if "embedding_path_norm" in df.columns:
            matches = df[df["embedding_path_norm"] == path_norm]
        else:
            matches = df[df["embedding_path"].apply(normalize_embedding_path) == path_norm]
        if len(matches) > 0:
            row = matches.iloc[0]

    # Legacy lookup by AUDIO_FILE_ID
    if row is None and "AUDIO_FILE_ID" in df.columns:
        matching_rows = df[df['AUDIO_FILE_ID'] == audio_file_id]
        if len(matching_rows) > 0:
            row = matching_rows.iloc[0]

    if row is None:
        return weak_labels
    
    # Fill in species labels
    for species, idx in label_index.items():
        col_name = f'SPECIES_{species}'
        alt_col = species
        if col_name in row:
            weak_labels[idx] = float(row[col_name])
        elif alt_col in row:
            weak_labels[idx] = float(row[alt_col])
    
    return weak_labels


def build_label_index(
    strong_root: str | Path | None = None,
    species_list: List[str] | None = None,
    weak_csv: str | Path | None = None,
) -> Dict[str, int]:
    """
    Build a mapping from species names to class indices.
    
    Can build from:
    - Explicit species list
    - Strong label directory (scans all .txt files)
    - Weak label CSV file (extracts SPECIES_* columns)
    
    Args:
        strong_root: Root directory containing strong label .txt files.
        species_list: Optional explicit list of species to include.
        weak_csv: Optional path to weak label CSV file.
        
    Returns:
        Dictionary mapping species name to class index.
    """
    if species_list is not None:
        all_species = sorted(set(species_list))
    elif weak_csv is not None:
        df = load_weak_labels_csv(weak_csv)
        all_species = extract_species_from_weak_csv(df)
        logger.info(f"Found {len(all_species)} species in weak labels CSV")
    elif strong_root is not None:
        strong_root = Path(strong_root)
        all_species = set()
        
        # Scan all .txt files for species names
        for txt_path in strong_root.glob("**/*.txt"):
            events = parse_strong_labels(txt_path)
            for _, _, species, _ in events:  # Now unpacking 4 elements
                all_species.add(species)
        
        all_species = sorted(all_species)
        logger.info(f"Found {len(all_species)} species in strong labels")
    else:
        raise ValueError("Must provide either strong_root, species_list, or weak_csv")
    
    return {sp: i for i, sp in enumerate(all_species)}


def events_to_labels(
    events: List[Tuple[float, float, str, str]],
    label_index: Dict[str, int],
    start_sec: np.ndarray,
    end_sec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert strong label events to weak and per-second labels.
    
    A chunk is considered positive for a species if any event
    overlaps with the chunk's time interval.
    
    Args:
        events: List of (start, end, species, quality) tuples.
        label_index: Mapping from species to class index.
        start_sec: (T,) array of chunk start times.
        end_sec: (T,) array of chunk end times.
        
    Returns:
        Tuple of:
        - weak_labels: (C,) binary array for clip-level presence
        - time_labels: (T, C) binary array for per-second labels
    """
    T = len(start_sec)
    C = len(label_index)
    
    weak_labels = np.zeros((C,), dtype=np.float32)
    time_labels = np.zeros((T, C), dtype=np.float32)
    
    for start, end, species, quality in events:  # Now unpacking 4 elements
        if species not in label_index:
            continue
        
        c = label_index[species]
        weak_labels[c] = 1.0
        
        # Mark overlapping chunks as positive
        # Overlap condition: chunk_start < event_end AND chunk_end > event_start
        overlaps = (start_sec < end) & (end_sec > start)
        time_labels[overlaps, c] = 1.0
    
    return weak_labels, time_labels


def extract_recording_id(clip_stem: str) -> str:
    """
    Extract recording ID from clip filename stem.
    
    Handles two filename formats:
    1. New format: "INCT17_20200211_041500_7_10" -> "INCT17_20200211_041500" (removes _startSec_endSec suffix)
    2. Old format: "INCT17_20200211_041500" -> "INCT17_20200211_041500" (no change)
    
    The recording ID can contain underscores. The last two underscore-separated
    values are the clip start and end times in seconds.
    
    Args:
        clip_stem: Filename stem of the clip (e.g., "INCT17_20200211_041500_7_10")
        
    Returns:
        Recording ID without clip time suffix (e.g., "INCT17_20200211_041500")
    """
    # Try to detect if this is the new format with _start_end suffix
    # Pattern: <recording_id>_<start>_<end> where start and end are integers
    # and start < end (to distinguish from timestamp-like recording IDs)
    parts = clip_stem.rsplit('_', 2)
    
    if len(parts) == 3:
        # Check if last two parts are numeric (start and end seconds)
        try:
            end_sec = int(parts[-1])
            start_sec = int(parts[-2])
            # Validate that this looks like a time range (start < end)
            # This helps avoid misinterpreting timestamps in recording IDs
            if start_sec < end_sec:
                # This is the new format, return everything before the last two underscores
                return parts[0]
        except ValueError:
            pass
    
    # Not in new format or couldn't parse, return as-is
    return clip_stem


def extract_recording_id_from_path(npz_path: Path) -> str:
    """
    Extract recording ID from NPZ path.
    
    Removes .embeddings suffix and extracts base recording ID.
    
    Args:
        npz_path: Path to .embeddings.npz file.
        
    Returns:
        Recording ID without clip time suffix.
    """
    stem = npz_path.stem
    if stem.endswith(".embeddings"):
        stem = stem[:-11]  # Remove ".embeddings"
    return extract_recording_id(stem)


def _find_strong_label_file(
    npz_path: Path,
    strong_root: Path,
) -> Optional[Path]:
    """
    Find matching strong label file for an embedding file.
    
    Handles both old and new filename formats:
    - New format: embeddings/SITE_A/REC_000001_0_3.embeddings.npz -> strong_labels/SITE_A/REC_000001.txt
    - Old format: embeddings/SITE_A/REC_000001.embeddings.npz -> strong_labels/SITE_A/REC_000001.txt
    
    Tries several strategies:
    1. Same recording ID with .txt extension in corresponding subdirectory
    2. Direct recording ID match in any subdirectory
    
    Args:
        npz_path: Path to .embeddings.npz file.
        strong_root: Root directory of strong labels.
        
    Returns:
        Path to matching .txt file or None if not found.
    """
    # Extract recording ID (handles both old and new formats)
    recording_id = extract_recording_id_from_path(npz_path)
    
    # Try to preserve directory structure
    # e.g., embeddings/SITE_A/REC_001_0_3.embeddings.npz -> strong_labels/SITE_A/REC_001.txt
    try:
        # Check if parent directory matches a site
        site_dir = npz_path.parent.name
        candidate = strong_root / site_dir / f"{recording_id}.txt"
        if candidate.exists():
            return candidate
    except Exception:
        pass
    
    # Try direct match
    candidate = strong_root / f"{recording_id}.txt"
    if candidate.exists():
        return candidate
    
    # Search all subdirectories
    for txt_path in strong_root.glob(f"**/{recording_id}.txt"):
        return txt_path
    
    return None


class EmbeddingBagDataset(Dataset):
    """
    Dataset for MIL training on pre-extracted embeddings.
    
    Loads .embeddings.npz files and matches them with strong labels
    to produce weak (clip-level) and per-second labels.
    """
    
    def __init__(
        self,
        npz_paths: List[str | Path] | str,
        strong_root: str | Path | None = None,
        label_index: Dict[str, int] | None = None,
        species_list: List[str] | None = None,
        weak_csv: str | Path | None = None,
    ):
        """
        Initialize dataset.
        
        Args:
            npz_paths: List of paths to .embeddings.npz files, or glob pattern.
            strong_root: Root directory of strong label files (for strong labels).
            label_index: Pre-built label index. If None, will build from available sources.
            species_list: Optional species list for building label index.
            weak_csv: Optional path to weak label CSV file.
        """
        # Handle glob pattern or list of paths
        if isinstance(npz_paths, str):
            self.npz_paths = [Path(p) for p in sorted(glob.glob(npz_paths))]
        else:
            self.npz_paths = [Path(p) for p in npz_paths]
        
        if len(self.npz_paths) == 0:
            raise ValueError("No .npz files found")
        
        self.strong_root = Path(strong_root) if strong_root else None
        self.weak_csv_path = Path(weak_csv) if weak_csv else None
        
        # Load weak labels CSV if provided
        self.weak_labels_df = None
        if self.weak_csv_path:
            self.weak_labels_df = load_weak_labels_csv(self.weak_csv_path)
            logger.info(f"Loaded weak labels from {self.weak_csv_path}")
        
        # Build or use provided label index
        if label_index is not None:
            self.label_index = label_index
        elif species_list is not None:
            self.label_index = build_label_index(species_list=species_list)
        elif weak_csv is not None:
            self.label_index = build_label_index(weak_csv=weak_csv)
        elif strong_root is not None:
            self.label_index = build_label_index(strong_root=strong_root)
        else:
            raise ValueError("Must provide label_index, species_list, weak_csv, or strong_root")
        
        self.n_classes = len(self.label_index)
        
        # Build mapping from npz to strong label files (if using strong labels)
        self.strong_paths: Dict[int, Optional[Path]] = {}
        if self.strong_root:
            for i, npz_path in enumerate(self.npz_paths):
                self.strong_paths[i] = _find_strong_label_file(npz_path, self.strong_root)
        
        logger.info(
            f"Dataset initialized with {len(self.npz_paths)} files, "
            f"{self.n_classes} classes"
        )
    
    def __len__(self) -> int:
        return len(self.npz_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[np.ndarray, np.ndarray]]:
        """
        Get a single sample.
        
        Returns:
            Tuple of:
            - embeddings: (T, D) tensor
            - weak_labels: (C,) tensor of clip-level labels
            - time_labels: (T, C) tensor of per-second labels
            - times: tuple of (start_sec, end_sec) arrays
        """
        npz_path = self.npz_paths[idx]
        
        # Load embeddings
        embeddings, start_sec, end_sec = load_embeddings_npz(npz_path)
        
        # Initialize labels
        T = len(start_sec)
        C = self.n_classes
        weak_labels = np.zeros((C,), dtype=np.float32)
        time_labels = np.zeros((T, C), dtype=np.float32)
        
        # Try to load weak labels from CSV first
        if self.weak_labels_df is not None:
            # Extract recording ID from npz path
            recording_id = extract_recording_id_from_path(npz_path)
            embedding_path = normalize_embedding_path(npz_path)
            
            # Get weak labels for this recording/path
            weak_labels = get_weak_labels_for_recording(
                self.weak_labels_df,
                recording_id,
                self.label_index,
                embedding_path=embedding_path,
            )
            # For weak labels, we don't have time-level labels, keep them as zeros
        else:
            # Fall back to strong labels
            strong_path = self.strong_paths.get(idx)
            if strong_path and strong_path.exists():
                events = parse_strong_labels(strong_path)
            else:
                events = []
            
            weak_labels, time_labels = events_to_labels(
                events, self.label_index, start_sec, end_sec
            )
        
        return (
            torch.from_numpy(embeddings),
            torch.from_numpy(weak_labels),
            torch.from_numpy(time_labels),
            (start_sec, end_sec),
        )
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension from first sample."""
        embeddings, _, _ = load_embeddings_npz(self.npz_paths[0])
        return embeddings.shape[1]
    
    def get_index_to_species(self) -> Dict[int, str]:
        """Get reverse mapping from class index to species name."""
        return {i: sp for sp, i in self.label_index.items()}


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[np.ndarray, np.ndarray]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Collate function that pads variable-length sequences.
    
    Args:
        batch: List of (embeddings, weak_labels, time_labels, times) tuples.
        
    Returns:
        Tuple of:
        - embeddings: (B, T_max, D) padded tensor
        - weak_labels: (B, C) tensor
        - time_labels: (B, T_max, C) padded tensor
        - times: list of (start_sec, end_sec) tuples
    """
    embeddings_list, weak_list, time_list, times_list = zip(*batch)
    
    B = len(batch)
    T_max = max(e.shape[0] for e in embeddings_list)
    D = embeddings_list[0].shape[1]
    C = weak_list[0].shape[0]
    
    # Pad embeddings and time labels
    embeddings_padded = torch.zeros(B, T_max, D, dtype=torch.float32)
    time_labels_padded = torch.zeros(B, T_max, C, dtype=torch.float32)
    
    for i, (emb, time_lab) in enumerate(zip(embeddings_list, time_list)):
        T = emb.shape[0]
        embeddings_padded[i, :T] = emb
        time_labels_padded[i, :T] = time_lab
    
    weak_labels = torch.stack(weak_list, dim=0)
    
    return embeddings_padded, weak_labels, time_labels_padded, list(times_list)
