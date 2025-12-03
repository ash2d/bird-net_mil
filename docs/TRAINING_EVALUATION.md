# Training and Evaluation Guide

## Overview of Changes

This document describes the new features added to the bird-net_mil training and evaluation pipeline:

1. **Evaluation Script**: Comprehensive model evaluation with multiple metrics
2. **Seed Support**: Reproducible training runs with random seeds
3. **Organized Output Structure**: Run-based folder organization for better experiment tracking
4. **Checkpoint Saving**: Periodic checkpoint saving during training
5. **Improved WandB Integration**: Better organization of experiments in Weights & Biases

## Training Models

### Basic Training

Train all pooling heads with default settings:

```bash
python scripts/train_mil.py \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels
```

### Reproducible Training with Seed

To ensure reproducibility, use the `--seed` argument:

```bash
python scripts/train_mil.py \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels \
    --poolers lme attn autopool \
    --epochs 20 \
    --seed 42
```

This will:
- Set random seeds for PyTorch, NumPy, and Python's random module
- Create a run directory named `run_YYYYMMDD_HHMMSS_seed42`
- Ensure deterministic training (when possible)

### Checkpoint Saving

Save model checkpoints every N epochs:

```bash
python scripts/train_mil.py \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels \
    --poolers attn \
    --epochs 50 \
    --save_every 5 \
    --out_dir ./experiments
```

This creates checkpoints:
- `experiments/run_YYYYMMDD_HHMMSS/attn_epoch5.pt`
- `experiments/run_YYYYMMDD_HHMMSS/attn_epoch10.pt`
- ...
- `experiments/run_YYYYMMDD_HHMMSS/attn_last.pt` (final checkpoint)
- `experiments/run_YYYYMMDD_HHMMSS/attn_best.pt` (best validation loss, if using early stopping)

### Output Directory Structure

Each training run creates a unique directory:

```
runs/
├── run_20231203_143022_seed42/
│   ├── lme_last.pt
│   ├── lme_epoch5.pt
│   ├── lme_epoch10.pt
│   ├── loss_lme.csv
│   ├── loss_lme.png
│   ├── attn_last.pt
│   ├── loss_attn.csv
│   └── loss_attn.png
└── run_20231203_150815/
    └── ...
```

Benefits:
- Each run is self-contained
- Easy to compare different runs
- No risk of overwriting previous experiments
- Run ID includes timestamp and seed for easy identification

### Weights & Biases Integration

When using WandB, all pooling heads from the same training run are grouped together:

```bash
python scripts/train_mil.py \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels \
    --poolers lme attn autopool \
    --wandb \
    --wandb_project bird-mil \
    --seed 42
```

In WandB:
- All runs are tagged with the `run_id` (e.g., `run_20231203_143022_seed42`)
- Runs are grouped by `run_id` for easy comparison
- You can filter by tag to see all poolers from a specific training session

## Evaluating Models

### Basic Evaluation

Evaluate a trained model on a test set:

```bash
python scripts/evaluate_mil.py \
    --checkpoint runs/run_20231203_143022_seed42/attn_last.pt \
    --emb_dir /data/test_embeddings \
    --strong_root /data/anuraset/strong_labels
```

### Detailed Metrics

The evaluation script computes:

- **Sample Accuracy**: Exact match accuracy (all classes correct)
- **Precision/Recall/F1**: Both micro and macro averaged
  - Micro: Aggregate contributions of all classes
  - Macro: Average of per-class metrics
- **AUC-ROC**: Area under ROC curve (macro averaged)
- **Average Precision**: Area under precision-recall curve (macro averaged)
- **Per-Class Accuracy**: Mean and std of accuracy across classes

Example output:

```
==================================================
Evaluation Results
==================================================
Model: attn
Samples: 1000
Threshold: 0.5
--------------------------------------------------
sample_accuracy               : 0.7850
precision_micro               : 0.8234
recall_micro                  : 0.7912
f1_micro                      : 0.8070
precision_macro               : 0.8156
recall_macro                  : 0.7845
f1_macro                      : 0.7998
auc_roc_macro                 : 0.9123
avg_precision_macro           : 0.8567
class_accuracy_mean           : 0.9345
class_accuracy_std            : 0.0521
```

### Pointing Game Evaluation

Evaluate whether the model attends to ground-truth events:

```bash
python scripts/evaluate_mil.py \
    --checkpoint runs/run_20231203_143022_seed42/attn_last.pt \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels \
    --pointing_game
```

### Save Results to File

Save evaluation results as JSON:

```bash
python scripts/evaluate_mil.py \
    --checkpoint runs/run_20231203_143022_seed42/attn_best.pt \
    --emb_dir /data/test_embeddings \
    --strong_root /data/anuraset/strong_labels \
    --output results/attn_evaluation.json
```

## Complete Workflow Example

Here's a complete workflow from training to evaluation:

```bash
# 1. Train models with reproducible seed
python scripts/train_mil.py \
    --emb_dir /data/train_embeddings \
    --strong_root /data/anuraset/strong_labels \
    --poolers lme attn autopool \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --val_split 0.1 \
    --save_every 10 \
    --seed 42 \
    --wandb \
    --wandb_project bird-mil \
    --out_dir ./experiments

# 2. Evaluate best checkpoint on test set
python scripts/evaluate_mil.py \
    --checkpoint ./experiments/run_20231203_143022_seed42/attn_best.pt \
    --emb_dir /data/test_embeddings \
    --strong_root /data/anuraset/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output ./results/attn_test_results.json

# 3. View results
cat ./results/attn_test_results.json
```

## Command-Line Arguments

### train_mil.py

**Data Arguments:**
- `--emb_dir`: Directory containing .embeddings.npz files
- `--emb_glob`: Glob pattern for .npz files (alternative to --emb_dir)
- `--strong_root`: Root directory with strong label .txt files
- `--species_list`: Path to species list file
- `--small_train`: Fraction or count of training data to use
- `--val_split`: Fraction for validation (default: 0.1)

**Training Arguments:**
- `--poolers`: Poolers to train (default: all)
- `--epochs`: Number of epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-3)
- `--device`: Device (cpu/cuda, default: auto)
- `--early_stop`: Early stopping patience (default: 0)

**Output Arguments:**
- `--out_dir`: Output directory (default: runs)
- `--save_every`: Save checkpoint every N epochs (default: 0)

**WandB Arguments:**
- `--wandb`: Enable W&B logging
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity/team

**Other:**
- `--seed`: Random seed for reproducibility
- `--verbose`: Verbose logging
- `--eval_pointing`: Evaluate pointing game after training

### evaluate_mil.py

**Model Arguments:**
- `--checkpoint`: Path to checkpoint .pt file (required)

**Data Arguments:**
- `--emb_dir`: Directory with .embeddings.npz files
- `--emb_glob`: Glob pattern for .npz files
- `--strong_root`: Root directory with strong labels (required)
- `--species_list`: Path to species list file

**Evaluation Arguments:**
- `--batch_size`: Batch size (default: 32)
- `--threshold`: Classification threshold (default: 0.5)
- `--device`: Device (cpu/cuda, default: auto)
- `--pointing_game`: Evaluate pointing game

**Output:**
- `--output`: Output JSON file for results
- `--verbose`: Verbose logging

## Notes

- **Reproducibility**: Using `--seed` enables deterministic training, but note that some operations may still be non-deterministic on GPU
- **WandB Organization**: All poolers from one `train_mil.py` invocation share the same run_id and are grouped together
- **Checkpoint Format**: Checkpoints include model weights, optimizer state, and training metadata
- **Evaluation Metrics**: Use macro-averaged metrics for imbalanced datasets, micro-averaged for overall performance
