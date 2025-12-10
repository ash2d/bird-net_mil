"""Integration test for train/test split and weak label workflow."""

import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mil.datasets import EmbeddingBagDataset, build_label_index


def test_end_to_end_workflow():
    """Test the complete workflow with train/test split and weak labels."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create mock embedding directory structure
        emb_dir = tmpdir / "embeddings"
        
        # Create SITE_A with 10 files
        site_a = emb_dir / "SITE_A"
        site_a.mkdir(parents=True)
        for i in range(10):
            npz_file = site_a / f"REC_A_{i:03d}.embeddings.npz"
            np.savez(
                npz_file,
                embeddings=np.random.randn(5, 128).astype(np.float32),
                start_sec=np.arange(5, dtype=np.float32),
                end_sec=np.arange(1, 6, dtype=np.float32)
            )
        
        # Create SITE_B with 3 files (smallest site)
        site_b = emb_dir / "SITE_B"
        site_b.mkdir(parents=True)
        for i in range(3):
            npz_file = site_b / f"REC_B_{i:03d}.embeddings.npz"
            np.savez(
                npz_file,
                embeddings=np.random.randn(5, 128).astype(np.float32),
                start_sec=np.arange(5, dtype=np.float32),
                end_sec=np.arange(1, 6, dtype=np.float32)
            )
        
        # Create SITE_C with 8 files
        site_c = emb_dir / "SITE_C"
        site_c.mkdir(parents=True)
        for i in range(8):
            npz_file = site_c / f"REC_C_{i:03d}.embeddings.npz"
            np.savez(
                npz_file,
                embeddings=np.random.randn(5, 128).astype(np.float32),
                start_sec=np.arange(5, dtype=np.float32),
                end_sec=np.arange(1, 6, dtype=np.float32)
            )
        
        # Create weak labels CSV
        weak_labels_csv = tmpdir / "weak_labels.csv"
        rows = []
        
        # Add labels for SITE_A recordings
        for i in range(10):
            rows.append({
                'MONITORING_SITE': 'SITE_A',
                'AUDIO_FILE_ID': f'REC_A_{i:03d}',
                'SPECIES_Species_A': 1 if i % 2 == 0 else 0,
                'SPECIES_Species_B': 1 if i % 3 == 0 else 0,
            })
        
        # Add labels for SITE_B recordings
        for i in range(3):
            rows.append({
                'MONITORING_SITE': 'SITE_B',
                'AUDIO_FILE_ID': f'REC_B_{i:03d}',
                'SPECIES_Species_A': 0,
                'SPECIES_Species_B': 1,
            })
        
        # Add labels for SITE_C recordings
        for i in range(8):
            rows.append({
                'MONITORING_SITE': 'SITE_C',
                'AUDIO_FILE_ID': f'REC_C_{i:03d}',
                'SPECIES_Species_A': 1,
                'SPECIES_Species_B': 1 if i % 2 == 0 else 0,
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(weak_labels_csv, index=False)
        
        # Run train/test split script
        train_out = tmpdir / "train.txt"
        test_out = tmpdir / "test.txt"
        
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_train_test_split.py",
                "--emb_dir", str(emb_dir),
                "--train_out", str(train_out),
                "--test_out", str(test_out),
                "--test_fraction", "0.05",
            ],
            capture_output=True,
            text=True,
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert train_out.exists(), "train.txt not created"
        assert test_out.exists(), "test.txt not created"
        
        # Verify split files
        with open(train_out, 'r') as f:
            train_paths = [line.strip() for line in f if line.strip()]
        
        with open(test_out, 'r') as f:
            test_paths = [line.strip() for line in f if line.strip()]
        
        # Should have all 3 SITE_B files in test, plus 1 from SITE_A and 1 from SITE_C
        assert len(test_paths) == 5
        assert len(train_paths) == 16
        
        # Count files from each site in test set
        site_b_test = sum(1 for p in test_paths if 'SITE_B' in p)
        assert site_b_test == 3, "All SITE_B files should be in test set"
        
        # Test loading dataset with weak labels
        label_index = build_label_index(weak_csv=weak_labels_csv)
        assert len(label_index) == 2  # Species_A and Species_B
        
        # Create dataset from train paths
        train_dataset = EmbeddingBagDataset(
            npz_paths=train_paths,
            weak_csv=weak_labels_csv,
            label_index=label_index,
        )
        
        assert len(train_dataset) == 16
        
        # Test loading a sample
        embeddings, weak_labels, time_labels, times = train_dataset[0]
        
        assert embeddings.shape[1] == 128  # Embedding dimension
        assert weak_labels.shape[0] == 2   # Number of species
        assert time_labels.shape[1] == 2   # Number of species
        
        # Time labels should be all zeros for weak labels
        assert time_labels.sum() == 0
        
        # Create dataset from test paths
        test_dataset = EmbeddingBagDataset(
            npz_paths=test_paths,
            weak_csv=weak_labels_csv,
            label_index=label_index,
        )
        
        assert len(test_dataset) == 5
        
        print("✓ End-to-end workflow test passed!")


if __name__ == "__main__":
    test_end_to_end_workflow()
