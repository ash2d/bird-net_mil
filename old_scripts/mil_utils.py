# mil_utils.py
from __future__ import annotations
import numpy as np
from pathlib import Path

def load_embeddings_npz(npz_path: str | Path):
    Z = np.load(npz_path)
    E = Z["embeddings"].astype(np.float32)   # (T,D)
    start_sec = Z["start_sec"].astype(np.float32)  # (T,)
    end_sec = Z["end_sec"].astype(np.float32)      # (T,)
    return E, start_sec, end_sec

def parse_anuraset_strong(txt_path: str | Path):
    """
    strong_labels/<site>/<file>.txt lines look like: start_sec end_sec species [quality]
    We'll parse first 3 fields; ignore quality if present.
    """
    events = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) < 3: continue
            s, e = float(parts[0]), float(parts[1])
            species = parts[2]
            events.append((s, e, species))
    return events

def build_label_index(all_species: list[str]) -> dict[str,int]:
    all_species = sorted(set(all_species))
    return {sp:i for i,sp in enumerate(all_species)}

def events_to_weak_and_time(events, label_index: dict[str,int],
                            start_sec: np.ndarray, end_sec: np.ndarray):
    """
    From strong events -> (clip-level weak labels, and optional per-second targets).
    A chunk is positive for a species if any event overlaps [start,end).
    """
    T = len(start_sec); C = len(label_index)
    weak = np.zeros((C,), dtype=np.float32)
    time_targets = np.zeros((T, C), dtype=np.float32)
    for (s,e,sp) in events:
        if sp not in label_index: continue
        c = label_index[sp]
        weak[c] = 1.0
        # mark any chunk that overlaps the event
        overlap = (start_sec < e) & (end_sec > s)
        time_targets[overlap, c] = 1.0
    return weak, time_targets  # (C,), (T,C)
