"""Tests for train/test split script."""

import pytest
import sys
from pathlib import Path
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the functions from the script
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from create_train_test_split import (
    extract_site_from_path,
    group_embeddings_by_site,
    find_smallest_site,
    split_site_chronologically,
    create_train_test_split,
)


class TestExtractSiteFromPath:
    """Test site extraction from path."""
    
    def test_extract_site_from_path(self):
        """Test extracting site name from embedding path."""
        path = Path("/data/embeddings/SITE_A/REC_001.embeddings.npz")
        site = extract_site_from_path(path)
        assert site == "SITE_A"
    
    def test_extract_site_nested_path(self):
        """Test extracting site from nested path."""
        path = Path("/data/embeddings/SITE_B/subdir/REC_002.embeddings.npz")
        site = extract_site_from_path(path)
        assert site == "subdir"


class TestGroupEmbeddingsBySite:
    """Test grouping embeddings by site."""
    
    def test_group_embeddings_by_site(self):
        """Test grouping multiple embeddings by site."""
        paths = [
            Path("/data/embeddings/SITE_A/REC_001.embeddings.npz"),
            Path("/data/embeddings/SITE_A/REC_002.embeddings.npz"),
            Path("/data/embeddings/SITE_B/REC_003.embeddings.npz"),
            Path("/data/embeddings/SITE_C/REC_004.embeddings.npz"),
        ]
        
        groups = group_embeddings_by_site(paths)
        
        assert len(groups) == 3
        assert "SITE_A" in groups
        assert "SITE_B" in groups
        assert "SITE_C" in groups
        assert len(groups["SITE_A"]) == 2
        assert len(groups["SITE_B"]) == 1
        assert len(groups["SITE_C"]) == 1


class TestFindSmallestSite:
    """Test finding site with fewest recordings."""
    
    def test_find_smallest_site(self):
        """Test identifying site with minimum recordings."""
        site_groups = {
            "SITE_A": [Path(f"rec_{i}.npz") for i in range(10)],
            "SITE_B": [Path(f"rec_{i}.npz") for i in range(3)],
            "SITE_C": [Path(f"rec_{i}.npz") for i in range(7)],
        }
        
        smallest = find_smallest_site(site_groups)
        assert smallest == "SITE_B"


class TestSplitSiteChronologically:
    """Test chronological splitting of site data."""
    
    def test_split_site_default_fraction(self):
        """Test splitting with default 5% test fraction."""
        paths = [Path(f"REC_{i:03d}.embeddings.npz") for i in range(100)]
        
        train, test = split_site_chronologically(paths, test_fraction=0.05)
        
        # Should have 95 training and 5 test samples
        assert len(train) == 95
        assert len(test) == 5
        
        # Test set should be the last 5
        assert test[-1] == Path("REC_099.embeddings.npz")
        assert test[0] == Path("REC_095.embeddings.npz")
    
    def test_split_site_custom_fraction(self):
        """Test splitting with custom test fraction."""
        paths = [Path(f"REC_{i:03d}.embeddings.npz") for i in range(100)]
        
        train, test = split_site_chronologically(paths, test_fraction=0.1)
        
        assert len(train) == 90
        assert len(test) == 10
    
    def test_split_site_small_dataset(self):
        """Test splitting with very small dataset (ensures at least 1 test sample)."""
        paths = [Path("REC_001.embeddings.npz")]
        
        train, test = split_site_chronologically(paths, test_fraction=0.05)
        
        # Even with 1 file, should have 1 test sample
        assert len(train) == 0
        assert len(test) == 1


class TestCreateTrainTestSplit:
    """Test full train/test split creation."""
    
    def test_create_train_test_split(self, tmp_path):
        """Test creating train/test split with multiple sites."""
        # Create mock embedding directory structure
        emb_dir = tmp_path / "embeddings"
        
        # Create SITE_A with 10 files
        site_a = emb_dir / "SITE_A"
        site_a.mkdir(parents=True)
        for i in range(10):
            npz_file = site_a / f"REC_A_{i:03d}.embeddings.npz"
            # Create minimal npz file
            np.savez(npz_file, embeddings=np.zeros((5, 10)), start_sec=np.arange(5), end_sec=np.arange(1, 6))
        
        # Create SITE_B with 3 files (smallest site)
        site_b = emb_dir / "SITE_B"
        site_b.mkdir(parents=True)
        for i in range(3):
            npz_file = site_b / f"REC_B_{i:03d}.embeddings.npz"
            np.savez(npz_file, embeddings=np.zeros((5, 10)), start_sec=np.arange(5), end_sec=np.arange(1, 6))
        
        # Create SITE_C with 8 files
        site_c = emb_dir / "SITE_C"
        site_c.mkdir(parents=True)
        for i in range(8):
            npz_file = site_c / f"REC_C_{i:03d}.embeddings.npz"
            np.savez(npz_file, embeddings=np.zeros((5, 10)), start_sec=np.arange(5), end_sec=np.arange(1, 6))
        
        # Run the split
        train_paths, test_paths = create_train_test_split(emb_dir, test_fraction=0.05)
        
        # SITE_B (3 files) should all be in test
        # SITE_A (10 files) should have 9 train, 1 test (last 5% = 0.5 rounded up to 1)
        # SITE_C (8 files) should have 7 train, 1 test
        # Total: 16 train, 5 test
        assert len(test_paths) == 5  # 3 from SITE_B + 1 from SITE_A + 1 from SITE_C
        assert len(train_paths) == 16  # 9 from SITE_A + 7 from SITE_C
        
        # Verify all SITE_B files are in test
        site_b_test_paths = [p for p in test_paths if "SITE_B" in str(p)]
        assert len(site_b_test_paths) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
