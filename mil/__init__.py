"""Multiple Instance Learning (MIL) module for BirdNET embeddings."""

from .heads import (
    PoolingHead,
    MeanPool,
    MaxPool,
    LMEPool,
    AutoPool,
    AttentionMIL,
    LinearSoftmaxPool,
    NoisyORPool,
    POOLER_NAMES,
    PROB_SPACE_POOLERS,
)
from .datasets import (
    EmbeddingBagDataset,
    collate_fn,
    build_label_index,
)

__all__ = [
    "PoolingHead",
    "MeanPool",
    "MaxPool",
    "LMEPool",
    "AutoPool",
    "AttentionMIL",
    "LinearSoftmaxPool",
    "NoisyORPool",
    "POOLER_NAMES",
    "PROB_SPACE_POOLERS",
    "EmbeddingBagDataset",
    "collate_fn",
    "build_label_index",
]
