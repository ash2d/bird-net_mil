"""Tests for MIL heads and training."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.heads import (
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


class TestPoolers:
    """Test individual pooler modules."""
    
    @pytest.fixture
    def sample_logits(self):
        """Create sample logits tensor (B=2, T=5, C=3)."""
        torch.manual_seed(42)
        return torch.randn(2, 5, 3)
    
    def test_mean_pool_shape(self, sample_logits):
        """Test MeanPool output shape."""
        pool = MeanPool()
        out = pool(sample_logits)
        assert out.shape == (2, 3)  # (B, C)
    
    def test_max_pool_shape(self, sample_logits):
        """Test MaxPool output shape."""
        pool = MaxPool()
        out = pool(sample_logits)
        assert out.shape == (2, 3)
    
    def test_lme_pool_shape(self, sample_logits):
        """Test LMEPool output shape."""
        pool = LMEPool(alpha_init=5.0)
        out = pool(sample_logits)
        assert out.shape == (2, 3)
    
    def test_autopool_shape(self, sample_logits):
        """Test AutoPool output shape."""
        pool = AutoPool(n_classes=3)
        out = pool(sample_logits)
        assert out.shape == (2, 3)
        # Output should be probabilities in [0, 1]
        assert (out >= 0).all() and (out <= 1).all()
    
    def test_attention_mil_shape(self, sample_logits):
        """Test AttentionMIL output shape."""
        pool = AttentionMIL(n_classes=3, hidden=64)
        out = pool(sample_logits)
        assert out.shape == (2, 3)
        # Output should be probabilities in [0, 1]
        assert (out >= 0).all() and (out <= 1).all()
    
    def test_attention_weights_shape(self, sample_logits):
        """Test AttentionMIL attention weights extraction."""
        pool = AttentionMIL(n_classes=3, hidden=64)
        weights = pool.get_attention_weights(sample_logits)
        assert weights.shape == (2, 5, 3)  # (B, T, C)
        # Weights should sum to 1 over T for each (B, C)
        sums = weights.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2, 3), atol=1e-5)
    
    def test_linear_softmax_pool_shape(self, sample_logits):
        """Test LinearSoftmaxPool output shape."""
        pool = LinearSoftmaxPool()
        out = pool(sample_logits)
        assert out.shape == (2, 3)
        # Output should be probabilities in [0, 1]
        assert (out >= 0).all() and (out <= 1).all()
    
    def test_noisy_or_pool_shape(self, sample_logits):
        """Test NoisyORPool output shape."""
        pool = NoisyORPool()
        out = pool(sample_logits)
        assert out.shape == (2, 3)
        # Output should be probabilities in [0, 1]
        assert (out >= 0).all() and (out <= 1).all()
    
    def test_lme_temperature_learnable(self, sample_logits):
        """Test that LME temperature is learnable."""
        pool = LMEPool(alpha_init=5.0)
        initial_alpha = pool.log_alpha.exp().item()
        
        # Run optimization step
        out = pool(sample_logits)
        target = torch.ones_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        
        # Gradient should exist
        assert pool.log_alpha.grad is not None
    
    def test_autopool_alpha_learnable(self, sample_logits):
        """Test that AutoPool alpha is learnable."""
        pool = AutoPool(n_classes=3)
        
        out = pool(sample_logits)
        target = torch.ones_like(out) * 0.5
        loss = ((out - target) ** 2).mean()
        loss.backward()
        
        assert pool.alpha.grad is not None


class TestPoolingHead:
    """Test the complete PoolingHead module."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings tensor (B=2, T=5, D=128)."""
        torch.manual_seed(42)
        return torch.randn(2, 5, 128)
    
    @pytest.mark.parametrize("pool_name", POOLER_NAMES)
    def test_head_shape(self, sample_embeddings, pool_name):
        """Test PoolingHead output shapes for all poolers."""
        head = PoolingHead(in_dim=128, n_classes=10, pool=pool_name)
        y_clip, z_per_sec = head(sample_embeddings)
        
        assert y_clip.shape == (2, 10)  # (B, C)
        assert z_per_sec.shape == (2, 5, 10)  # (B, T, C)
    
    @pytest.mark.parametrize("pool_name", POOLER_NAMES)
    def test_head_backward(self, sample_embeddings, pool_name):
        """Test that all poolers support backpropagation."""
        head = PoolingHead(in_dim=128, n_classes=10, pool=pool_name)
        y_clip, _ = head(sample_embeddings)
        
        target = torch.ones(2, 10) * 0.5
        loss = ((y_clip - target) ** 2).mean()
        loss.backward()
        
        # Check that projection layer has gradients
        assert head.proj.weight.grad is not None
    
    @pytest.mark.parametrize("pool_name", list(PROB_SPACE_POOLERS))
    def test_prob_space_poolers_range(self, sample_embeddings, pool_name):
        """Test that probability-space poolers output values in [0, 1]."""
        head = PoolingHead(in_dim=128, n_classes=10, pool=pool_name)
        y_clip, _ = head(sample_embeddings)
        
        assert (y_clip >= 0).all(), f"{pool_name} output has values < 0"
        assert (y_clip <= 1).all(), f"{pool_name} output has values > 1"
    
    def test_returns_probabilities_property(self):
        """Test returns_probabilities property."""
        for pool_name in POOLER_NAMES:
            head = PoolingHead(in_dim=128, n_classes=10, pool=pool_name)
            expected = pool_name in PROB_SPACE_POOLERS
            assert head.returns_probabilities == expected
    
    def test_attention_weights_extraction(self, sample_embeddings):
        """Test attention weights extraction for attention head."""
        head = PoolingHead(in_dim=128, n_classes=10, pool="attn")
        weights = head.get_attention_weights(sample_embeddings)
        
        assert weights is not None
        assert weights.shape == (2, 5, 10)  # (B, T, C)
    
    def test_non_attention_head_returns_none(self, sample_embeddings):
        """Test that non-attention heads return None for attention weights."""
        head = PoolingHead(in_dim=128, n_classes=10, pool="mean")
        weights = head.get_attention_weights(sample_embeddings)
        
        assert weights is None


class TestIntegration:
    """Integration tests with real-like data."""
    
    def test_forward_with_npz_like_data(self, tmp_path):
        """Test forward pass with data shaped like loaded NPZ."""
        # Simulate typical BirdNET embedding dimensions
        B, T, D = 4, 3, 1024  # 3-second clips with 1-second chunks
        C = 42  # AnuraSet has ~42 species
        
        embeddings = torch.randn(B, T, D)
        
        for pool_name in POOLER_NAMES:
            head = PoolingHead(in_dim=D, n_classes=C, pool=pool_name)
            y_clip, z_per_sec = head(embeddings)
            
            assert y_clip.shape == (B, C), f"Failed for {pool_name}"
            assert z_per_sec.shape == (B, T, C), f"Failed for {pool_name}"
    
    def test_variable_length_sequences(self):
        """Test that model handles variable length sequences."""
        D, C = 1024, 10
        head = PoolingHead(in_dim=D, n_classes=C, pool="attn")
        
        # Different sequence lengths
        for T in [1, 3, 5, 10, 30]:
            emb = torch.randn(1, T, D)
            y, z = head(emb)
            assert y.shape == (1, C)
            assert z.shape == (1, T, C)
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        head = PoolingHead(in_dim=128, n_classes=10, pool="lme")
        emb = torch.randn(1, 5, 128)
        y, z = head(emb)
        assert y.shape == (1, 10)


class TestNumericalStability:
    """Test numerical stability of poolers."""
    
    def test_lme_large_values(self):
        """Test LME stability with large input values."""
        pool = LMEPool(alpha_init=10.0)
        z = torch.randn(2, 100, 10) * 10  # Large values
        out = pool(z)
        
        assert not torch.isnan(out).any(), "LME produced NaN for large inputs"
        assert not torch.isinf(out).any(), "LME produced Inf for large inputs"
    
    def test_noisy_or_extreme_probs(self):
        """Test Noisy-OR with extreme probability values."""
        pool = NoisyORPool()
        
        # Very negative logits (probs near 0)
        z_neg = torch.full((2, 5, 3), -10.0)
        out_neg = pool(z_neg)
        assert not torch.isnan(out_neg).any()
        
        # Very positive logits (probs near 1)
        z_pos = torch.full((2, 5, 3), 10.0)
        out_pos = pool(z_pos)
        assert not torch.isnan(out_pos).any()
    
    def test_linear_softmax_all_zeros(self):
        """Test LinearSoftmax with all-zero probabilities."""
        pool = LinearSoftmaxPool()
        z = torch.full((2, 5, 3), -100.0)  # All probs near 0
        out = pool(z)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
