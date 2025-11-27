"""Tests for embedding comparison utility."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from birdnetv3.utils_audio import load_audio, frame_audio


class TestAudioUtils:
    """Test audio loading and framing utilities."""
    
    def test_frame_audio_basic(self):
        """Test basic audio framing."""
        # Create 3-second audio at 32kHz
        sr = 32000
        duration = 3.0
        wav = torch.randn(int(sr * duration))
        
        # Frame with 1-second chunks, no overlap
        chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length=1.0, overlap=0.0)
        
        assert chunks.shape[0] == 3  # 3 chunks
        assert chunks.shape[1] == sr  # Each chunk is 1 second
        
        np.testing.assert_array_almost_equal(start_sec, [0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(end_sec, [1.0, 2.0, 3.0])
    
    def test_frame_audio_with_overlap(self):
        """Test audio framing with overlap."""
        sr = 32000
        wav = torch.randn(int(sr * 3.0))
        
        # Frame with 1-second chunks, 0.5 second overlap
        chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length=1.0, overlap=0.5)
        
        # With 50% overlap, expect more chunks
        assert chunks.shape[0] == 5  # 0-1, 0.5-1.5, 1-2, 1.5-2.5, 2-3
    
    def test_frame_audio_short(self):
        """Test framing audio shorter than chunk length."""
        sr = 32000
        wav = torch.randn(int(sr * 0.5))  # 0.5 second audio
        
        chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length=1.0, overlap=0.0)
        
        assert chunks.shape[0] == 1
        assert chunks.shape[1] == sr  # Padded to 1 second
    
    def test_frame_audio_empty(self):
        """Test framing empty audio."""
        wav = torch.tensor([])
        chunks, start_sec, end_sec = frame_audio(wav, 32000, chunk_length=1.0, overlap=0.0)
        
        assert chunks.shape[0] == 0
    
    def test_frame_audio_invalid_params(self):
        """Test error handling for invalid parameters."""
        wav = torch.randn(32000)
        
        with pytest.raises(ValueError):
            frame_audio(wav, 32000, chunk_length=0.0, overlap=0.0)
        
        with pytest.raises(ValueError):
            frame_audio(wav, 32000, chunk_length=1.0, overlap=-0.1)
        
        with pytest.raises(ValueError):
            frame_audio(wav, 32000, chunk_length=1.0, overlap=1.0)


class TestEmbeddingComparison:
    """Test embedding comparison utilities."""
    
    def test_npz_format(self, tmp_path):
        """Test that NPZ files have expected format."""
        # Create mock embeddings
        T, D = 3, 1024
        embeddings = np.random.randn(T, D).astype(np.float32)
        start_sec = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        end_sec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sr = 32000
        
        # Save
        npz_path = tmp_path / "test.embeddings.npz"
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            start_sec=start_sec,
            end_sec=end_sec,
            sr=sr,
        )
        
        # Load and verify
        data = np.load(npz_path)
        assert "embeddings" in data
        assert "start_sec" in data
        assert "end_sec" in data
        assert "sr" in data
        
        np.testing.assert_array_almost_equal(data["embeddings"], embeddings)
        np.testing.assert_array_almost_equal(data["start_sec"], start_sec)
        np.testing.assert_array_almost_equal(data["end_sec"], end_sec)
        assert data["sr"] == sr
    
    def test_embedding_tolerance(self):
        """Test that embeddings match within expected tolerance."""
        # Simulate embeddings from two different sources
        # (e.g., binary NPZ vs text CSV)
        D = 1024
        
        # Original embedding
        original = np.random.randn(D).astype(np.float32)
        
        # Simulated CSV round-trip (6 decimal places)
        csv_strings = [f"{v:.6f}" for v in original]
        from_csv = np.array([float(s) for s in csv_strings], dtype=np.float32)
        
        # Should match within expected tolerance
        max_diff = np.max(np.abs(original - from_csv))
        
        # CSV formatting introduces ~1e-6 error
        assert max_diff < 1e-5, f"Max diff {max_diff} exceeds tolerance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
