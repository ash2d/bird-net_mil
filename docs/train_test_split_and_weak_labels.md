# Train/Test Split and Weak Label Support

This document describes the new features for creating train/test splits and using weak labels.

## Features

### 1. Train/Test Split Script

The `scripts/create_train_test_split.py` script creates a train/test split of embedding files based on the following strategy:

- **Test set includes:**
  - All recordings from the site with the fewest recordings (held-out site)
  - The last 5% of recordings from all other sites (held-out time periods)

- **Train set includes:**
  - All remaining recordings

This ensures testing on both held-out sites and held-out time periods.

#### Usage

```bash
python scripts/create_train_test_split.py \
    --emb_dir /path/to/embeddings \
    --train_out train.txt \
    --test_out test.txt \
    --test_fraction 0.05
```

**Arguments:**
- `--emb_dir`: Directory containing .embeddings.npz files organized by site
- `--train_out`: Output file for training paths (default: train.txt)
- `--test_out`: Output file for test paths (default: test.txt)
- `--test_fraction`: Fraction of recordings from non-test sites to use for testing (default: 0.05 = 5%)

**Output:**
- Two text files containing paths to embeddings (one path per line)
- These can be passed to training/evaluation scripts via `--emb_glob`

### 2. Weak Label Support

The dataset code now supports weak labels from CSV files in addition to strong labels from text files.

#### Weak Label CSV Format

The CSV must have the following columns:
- `MONITORING_SITE`: Site/location identifier
- `AUDIO_FILE_ID`: Audio filename without path or `_<start>_<end>` suffix
- `SPECIES_<species_name>`: Binary columns (0 or 1) for each species

**Example:**
```csv
MONITORING_SITE,AUDIO_FILE_ID,SPECIES_Boana_faber,SPECIES_Dendropsophus_minutus
SITE_A,REC_001,1,0
SITE_A,REC_002,0,1
SITE_B,REC_003,1,1
```

#### Using Weak Labels with Training/Evaluation

**Training with weak labels:**
```bash
python scripts/train_mil.py \
    --emb_glob train.txt \
    --weak_csv /path/to/weak_labels.csv \
    --poolers attn \
    --epochs 20
```

**Evaluation with weak labels:**
```bash
python scripts/evaluate_mil.py \
    --checkpoint runs/run_XXX/attn_best.pt \
    --emb_glob test.txt \
    --weak_csv /path/to/weak_labels.csv
```

**Note:** When using weak labels, time-level labels are not available (set to zeros). Only clip-level (weak) labels are used.

### 3. Reading Paths from File

Both `train_mil.py` and `evaluate_mil.py` now support reading embedding paths from a text file via the `--emb_glob` argument.

**Usage:**
```bash
# Read from text file
python scripts/train_mil.py --emb_glob train.txt --weak_csv labels.csv

# Or use as glob pattern (original behavior)
python scripts/train_mil.py --emb_glob "/data/embeddings/**/*.npz" --weak_csv labels.csv
```

The script automatically detects whether `--emb_glob` is a file or a pattern.

## Complete Workflow Example

```bash
# 1. Create train/test split
python scripts/create_train_test_split.py \
    --emb_dir /data/embeddings \
    --train_out train.txt \
    --test_out test.txt

# 2. Train model on training set with weak labels
python scripts/train_mil.py \
    --emb_glob train.txt \
    --weak_csv /data/weak_labels.csv \
    --poolers attn \
    --epochs 50 \
    --out_dir ./experiments

# 3. Evaluate on test set
python scripts/evaluate_mil.py \
    --checkpoint ./experiments/run_XXX/attn_best.pt \
    --emb_glob test.txt \
    --weak_csv /data/weak_labels.csv \
    --output test_results.json
```

## API Changes

### `mil.datasets.EmbeddingBagDataset`

**New parameter:**
- `weak_csv` (str | Path | None): Path to weak label CSV file

**Example:**
```python
from mil.datasets import EmbeddingBagDataset, build_label_index

# Build label index from weak CSV
label_index = build_label_index(weak_csv="weak_labels.csv")

# Create dataset with weak labels
dataset = EmbeddingBagDataset(
    npz_paths=["path1.npz", "path2.npz"],
    weak_csv="weak_labels.csv",
    label_index=label_index,
)
```

### `mil.datasets.build_label_index`

**New parameter:**
- `weak_csv` (str | Path | None): Path to weak label CSV file to extract species from

**Example:**
```python
# Build from weak CSV
label_index = build_label_index(weak_csv="weak_labels.csv")

# Build from strong labels (original)
label_index = build_label_index(strong_root="/path/to/labels")

# Build from explicit list (original)
label_index = build_label_index(species_list=["Species_A", "Species_B"])
```

## Testing

Run the test suite to verify the implementation:

```bash
# Test weak label functions
python -m pytest tests/test_datasets.py::TestWeakLabels -v

# Test train/test split
python -m pytest tests/test_train_test_split.py -v

# Run integration test
python tests/test_integration.py

# Run all tests
python -m pytest tests/ -v
```
