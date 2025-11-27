"""
MIL Pooling Heads for BirdNET embeddings.

This module implements various Multiple Instance Learning (MIL) pooling operators
that aggregate per-second embeddings/logits into clip-level predictions.

Poolers are divided into two categories:
1. Logit-space poolers (mean, max, lme): return logits, use BCEWithLogitsLoss
2. Probability-space poolers (autopool, attn, linsoft, noisyor): return probabilities, use BCELoss

To add a new pooler:
1. Create a new class inheriting from nn.Module
2. Implement forward(z) where z is (B, T, C) logits
3. Add to POOLER_NAMES list
4. If it returns probabilities, add to PROB_SPACE_POOLERS set
5. Handle initialization in PoolingHead.__init__ if needed
"""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# List of all available pooler names
POOLER_NAMES = ["mean", "max", "lme", "autopool", "attn", "linsoft", "noisyor"]

# Poolers that return probabilities (use BCELoss)
# Others return logits (use BCEWithLogitsLoss)
PROB_SPACE_POOLERS = {"autopool", "attn", "linsoft", "noisyor"}


class MeanPool(nn.Module):
    """Simple mean pooling over time dimension."""
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pool by averaging logits over time.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) averaged logits
        """
        return z.mean(dim=1)


class MaxPool(nn.Module):
    """Simple max pooling over time dimension."""
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Pool by taking maximum logits over time.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) max-pooled logits
        """
        return z.max(dim=1).values


class LMEPool(nn.Module):
    """
    Log-Mean-Exp pooling with learnable temperature.
    
    This is a smooth approximation to max pooling that is differentiable.
    As alpha -> infinity, it approaches max pooling.
    As alpha -> 0, it approaches mean pooling.
    
    The implementation uses logsumexp for numerical stability.
    """
    
    def __init__(self, alpha_init: float = 5.0):
        """
        Initialize LME pooler.
        
        Args:
            alpha_init: Initial temperature value (>= 0.001).
        """
        super().__init__()
        # Store log(alpha) to ensure alpha stays positive
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply log-mean-exp pooling.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) LME-pooled logits
        """
        alpha = torch.clamp(self.log_alpha.exp(), min=1e-3)
        
        # Numerically stable logsumexp: log(mean(exp(alpha*z))) / alpha
        # = (logsumexp(alpha*z) - log(T)) / alpha
        T = z.size(1)
        scaled = alpha * z  # (B, T, C)
        
        # Use max for stability: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        m = scaled.amax(dim=1, keepdim=True)  # (B, 1, C)
        lse = torch.log(torch.exp(scaled - m).mean(dim=1)) + m.squeeze(1)  # (B, C)
        
        return lse / alpha


class AutoPool(nn.Module):
    """
    Adaptive soft-max pooling (McFee et al., 2018).
    
    Computes attention weights from logits using a learnable per-class
    sharpness parameter, then combines probabilities.
    
    Returns probabilities, use BCELoss.
    """
    
    def __init__(self, n_classes: int, alpha_init: float = 5.0):
        """
        Initialize AutoPool.
        
        Args:
            n_classes: Number of output classes.
            alpha_init: Initial sharpness value per class.
        """
        super().__init__()
        # One sharpness parameter per class
        self.alpha = nn.Parameter(torch.full((n_classes,), alpha_init))
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive pooling.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) pooled probabilities
        """
        alpha = torch.clamp(self.alpha, min=1e-3).view(1, 1, -1)  # (1, 1, C)
        
        # Attention weights from logits
        weights = F.softmax(alpha * z, dim=1)  # (B, T, C)
        
        # Weighted average of probabilities
        probs = torch.sigmoid(z)  # (B, T, C)
        return (weights * probs).sum(dim=1)  # (B, C)


class AttentionMIL(nn.Module):
    """
    Attention-based MIL pooling (Ilse et al., 2018).
    
    Uses a learned attention mechanism to weight instances.
    Computes class-wise attention over the time series of per-class logits.
    
    Returns probabilities, use BCELoss.
    """
    
    def __init__(self, n_classes: int, hidden: int = 128):
        """
        Initialize Attention MIL.
        
        Args:
            n_classes: Number of output classes.
            hidden: Hidden dimension for attention network.
        """
        super().__init__()
        self.n_classes = n_classes
        self.hidden = hidden
        
        # Attention network: processes each class's logit time series
        self.U = nn.Linear(1, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply attention-based pooling.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) pooled probabilities
        """
        B, T, C = z.shape
        
        # Reshape for class-wise attention: (B*C, T, 1)
        zc = z.transpose(1, 2).contiguous().view(B * C, T, 1)
        
        # Compute attention scores
        A = self.v(torch.tanh(self.U(zc))).squeeze(-1)  # (B*C, T)
        A = F.softmax(A, dim=1).view(B, C, T)  # (B, C, T)
        
        # Weighted sum of probabilities
        probs = torch.sigmoid(z)  # (B, T, C)
        
        # A: (B, C, T), need to weight probs (B, T, C)
        # Transpose A to (B, T, C) for element-wise multiplication
        y = (A.transpose(1, 2) * probs).sum(dim=1)  # (B, C)
        
        return y
    
    def get_attention_weights(self, z: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for visualization.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, T, C) attention weights (sum to 1 over T for each class)
        """
        B, T, C = z.shape
        zc = z.transpose(1, 2).contiguous().view(B * C, T, 1)
        A = self.v(torch.tanh(self.U(zc))).squeeze(-1)  # (B*C, T)
        A = F.softmax(A, dim=1).view(B, C, T)  # (B, C, T)
        return A.transpose(1, 2)  # (B, T, C)


class LinearSoftmaxPool(nn.Module):
    """
    Linear-Softmax pooling (Wang & Metze, 2017).
    
    A self-weighted pooling where each instance's weight is proportional
    to its probability. Used in sound event detection.
    
    Formula: y = sum(p^2) / sum(p)
    
    Returns probabilities, use BCELoss.
    """
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply linear-softmax pooling.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) pooled probabilities
        """
        probs = torch.sigmoid(z)  # (B, T, C)
        num = (probs * probs).sum(dim=1)  # (B, C)
        den = torch.clamp(probs.sum(dim=1), min=1e-6)  # (B, C)
        return num / den


class NoisyORPool(nn.Module):
    """
    Noisy-OR pooling.
    
    Models the probability that at least one instance is positive.
    P(positive) = 1 - product(1 - p_t)
    
    This is equivalent to computing:
    y = 1 - exp(sum(log(1 - p + eps)))
    
    Returns probabilities, use BCELoss.
    """
    
    def __init__(self, eps: float = 1e-7):
        """
        Initialize Noisy-OR pooler.
        
        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply Noisy-OR pooling.
        
        Args:
            z: (B, T, C) logits tensor
            
        Returns:
            (B, C) pooled probabilities
        """
        probs = torch.sigmoid(z)  # (B, T, C)
        
        # P(any positive) = 1 - prod(1 - p)
        # Use log for numerical stability: 1 - exp(sum(log(1 - p)))
        log_neg = torch.log(1.0 - probs + self.eps)  # (B, T, C)
        return 1.0 - torch.exp(log_neg.sum(dim=1))  # (B, C)


class PoolingHead(nn.Module):
    """
    Complete MIL head that maps embeddings to clip-level predictions.
    
    Architecture:
    1. Linear projection: (B, T, D) -> (B, T, C) per-second logits
    2. Pooling: (B, T, C) -> (B, C) clip-level output
    
    For logit-space poolers (mean, max, lme): output is logits, use BCEWithLogitsLoss
    For prob-space poolers (autopool, attn, linsoft, noisyor): output is probabilities, use BCELoss
    """
    
    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        pool: Literal["lme", "mean", "max", "autopool", "attn", "linsoft", "noisyor"] = "lme",
        attention_hidden: int = 128,
        lme_alpha: float = 5.0,
        autopool_alpha: float = 5.0,
    ):
        """
        Initialize pooling head.
        
        Args:
            in_dim: Input embedding dimension (D).
            n_classes: Number of output classes (C).
            pool: Pooling method to use.
            attention_hidden: Hidden dimension for attention MIL.
            lme_alpha: Initial temperature for LME pooling.
            autopool_alpha: Initial sharpness for AutoPool.
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.pool_name = pool
        
        # Linear projection to per-second logits
        self.proj = nn.Linear(in_dim, n_classes)
        
        # Initialize pooler
        if pool == "mean":
            self.pooler = MeanPool()
        elif pool == "max":
            self.pooler = MaxPool()
        elif pool == "lme":
            self.pooler = LMEPool(alpha_init=lme_alpha)
        elif pool == "autopool":
            self.pooler = AutoPool(n_classes, alpha_init=autopool_alpha)
        elif pool == "attn":
            self.pooler = AttentionMIL(n_classes, hidden=attention_hidden)
        elif pool == "linsoft":
            self.pooler = LinearSoftmaxPool()
        elif pool == "noisyor":
            self.pooler = NoisyORPool()
        else:
            raise ValueError(f"Unknown pooler: {pool}. Available: {POOLER_NAMES}")
    
    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the head.
        
        Args:
            emb: (B, T, D) embeddings tensor
            
        Returns:
            Tuple of:
            - y_clip: (B, C) clip-level output (logits or probabilities)
            - z_per_sec: (B, T, C) per-second logits (for analysis/visualization)
        """
        z = self.proj(emb)  # (B, T, C) per-second logits
        y = self.pooler(z)  # (B, C) pooled output
        return y, z
    
    def get_attention_weights(self, emb: torch.Tensor) -> torch.Tensor | None:
        """
        Extract attention weights for visualization (only for attention pooler).
        
        Args:
            emb: (B, T, D) embeddings tensor
            
        Returns:
            (B, T, C) attention weights or None if pooler doesn't support it.
        """
        if not isinstance(self.pooler, AttentionMIL):
            return None
        
        z = self.proj(emb)  # (B, T, C)
        return self.pooler.get_attention_weights(z)
    
    @property
    def returns_probabilities(self) -> bool:
        """Whether this head returns probabilities (True) or logits (False)."""
        return self.pool_name in PROB_SPACE_POOLERS
