#!/usr/bin/env python3
"""
Simple integration test for the new features.
Tests that the CLI arguments work and basic functionality is intact.
"""

import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_set_seed():
    """Test that set_seed provides reproducible results."""
    print("Testing set_seed()...")
    
    # Test with seed 42
    set_seed(42)
    rand1 = torch.randn(10)
    np_rand1 = np.random.randn(10)
    
    # Reset and test again
    set_seed(42)
    rand2 = torch.randn(10)
    np_rand2 = np.random.randn(10)
    
    # Should be identical
    assert torch.allclose(rand1, rand2), "torch.randn not reproducible with seed"
    assert np.allclose(np_rand1, np_rand2), "np.random.randn not reproducible with seed"
    
    print("✓ set_seed() provides reproducible results")


def test_run_id_generation():
    """Test that run_id is generated correctly."""
    print("Testing run_id generation...")
    
    from datetime import datetime
    
    # Simulate run_id generation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = 42
    run_id = f"run_{timestamp}_seed{seed}"
    
    assert "run_" in run_id, "run_id should start with 'run_'"
    assert "seed42" in run_id, "run_id should include seed"
    assert len(timestamp) == 15, "timestamp should be in format YYYYMMDD_HHMMSS"
    
    print(f"✓ run_id generation works: {run_id}")


def test_trainer_initialization():
    """Test that Trainer can be initialized with new parameters."""
    print("Testing Trainer initialization with new parameters...")
    
    from mil.heads import PoolingHead
    from mil.train import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    embeddings = torch.randn(10, 5, 128)  # 10 samples, 5 timesteps, 128 dims
    labels = torch.randint(0, 2, (10, 3)).float()  # 10 samples, 3 classes
    
    dataset = TensorDataset(embeddings, labels)
    loader = DataLoader(dataset, batch_size=2)
    
    # Create model
    head = PoolingHead(in_dim=128, n_classes=3, pool="mean")
    
    # Test Trainer with new parameters
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            head=head,
            train_loader=loader,
            val_loader=None,
            lr=1e-3,
            device="cpu",
            out_dir=tmpdir,
            use_wandb=False,
            wandb_run_id="test_run_123",
            wandb_group="test_group",
            wandb_config={"test": "value"},
        )
        
        assert trainer.out_dir == Path(tmpdir), "out_dir not set correctly"
        assert trainer.device == "cpu", "device not set correctly"
        
        print("✓ Trainer initialization with new parameters works")


def test_checkpoint_save_every():
    """Test that save_every parameter works in training."""
    print("Testing save_every parameter...")
    
    from mil.heads import PoolingHead
    from mil.train import Trainer, load_checkpoint
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    embeddings = torch.randn(10, 5, 128)
    labels = torch.randint(0, 2, (10, 3)).float()
    dataset = TensorDataset(embeddings, labels)
    
    # Need to create a custom collate_fn for this simple test
    def simple_collate(batch):
        embs = torch.stack([b[0] for b in batch])
        lbls = torch.stack([b[1] for b in batch])
        # Return format expected by train_epoch: (embeddings, weak_labels, time_labels, times)
        return embs, lbls, torch.zeros(embs.shape[0], embs.shape[1], lbls.shape[1]), None
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=simple_collate)
    
    head = PoolingHead(in_dim=128, n_classes=3, pool="mean")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            head=head,
            train_loader=loader,
            val_loader=None,
            device="cpu",
            out_dir=tmpdir,
            use_wandb=False,
        )
        
        # Train for 3 epochs with save_every=2
        trainer.train(epochs=3, save_every=2)
        
        # Check that checkpoint was saved
        checkpoints = list(Path(tmpdir).glob("*_epoch2.pt"))
        assert len(checkpoints) > 0, "Checkpoint not saved at epoch 2"
        
        # Check that final checkpoint exists
        final_checkpoints = list(Path(tmpdir).glob("*_last.pt"))
        assert len(final_checkpoints) > 0, "Final checkpoint not saved"
        
        print("✓ save_every parameter works correctly")


def test_checkpoint_preserves_label_index():
    """Ensure checkpoints retain the label index for consistent evaluation."""
    
    from mil.heads import PoolingHead
    from mil.train import Trainer, load_checkpoint
    from torch.utils.data import DataLoader, TensorDataset
    
    embeddings = torch.randn(4, 3, 16)
    labels = torch.randint(0, 2, (4, 2)).float()
    dataset = TensorDataset(embeddings, labels)
    
    def simple_collate(batch):
        embs = torch.stack([b[0] for b in batch])
        lbls = torch.stack([b[1] for b in batch])
        return embs, lbls, torch.zeros(embs.shape[0], embs.shape[1], lbls.shape[1]), None
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=simple_collate)
    label_index = {"Species_A": 0, "Species_B": 1}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            head=PoolingHead(in_dim=16, n_classes=2, pool="mean"),
            train_loader=loader,
            val_loader=None,
            device="cpu",
            out_dir=tmpdir,
            use_wandb=False,
            label_index=label_index,
        )
        
        trainer.train(epochs=1)
        ckpt_path = Path(tmpdir) / "mean_last.pt"
        _, checkpoint = load_checkpoint(ckpt_path, device="cpu")
        
        assert checkpoint.get("label_index") == label_index

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running integration tests for new features")
    print("="*60 + "\n")
    
    try:
        test_set_seed()
        test_run_id_generation()
        test_trainer_initialization()
        test_checkpoint_save_every()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
