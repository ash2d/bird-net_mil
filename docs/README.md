# birdnet-V3.0-dev + MIL Pooling Heads

CLI to analyze audio with BirdNET+ V3.0 developer preview models, export per-chunk detections, and train/evaluate Multiple Instance Learning (MIL) pooling heads for bioacoustic classification.

This repository provides tools for:
1. **BirdNET V3 analysis**: Run species detection on audio files
2. **Embedding extraction**: Export per-second embeddings to `.npz` files
3. **MIL training**: Train various pooling heads (LME, Attention, AutoPool, etc.) on embeddings
4. **Evaluation**: Pointing game metrics, attention visualization, deletion/insertion curves

## Usage

Upon first run, the default model and labels will be automatically downloaded to the `models/` directory. You can download them manually from [Zenodo](https://zenodo.org/record/17571190).

Run the analysis with:

```bash
python analyze.py /path/to/audio.wav
```

### Options
- `--model` Path to model file (default: models/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt)
- `--labels` Path to labels CSV (default: models/BirdNET+_V3.0-preview2_EUNA_1K_Labels.csv)
- `--chunk_length` Chunk length in seconds (default: 3.0)
- `--overlap` Chunk overlap in seconds (default: 0.0)
- `--device` cpu|cuda (default: auto)
- `--min-conf` Minimum confidence threshold for exporting detections (default: 0.15)
- `--out-csv` Output CSV path (default: <audio>.results.csv)
- `--export-embeddings` Export per-chunk embeddings as additional column in results CSV

### Output
- Per-chunk CSV with columns: `name,start_sec,end_sec,confidence,label` and optionally `embeddings`
- One row per (chunk, label) with confidence ≥ `--min-conf`
- Multiple rows per chunk if multiple labels exceed threshold


## Examples
```bash
# Minimal (uses defaults where available)
python analyze.py example/soundscape.wav

# Specify model, chunk length, min confidence, and output CSV with embeddings
python analyze.py example/soundscape.wav --chunk_length 2.0 --min-conf 0.2 --out-csv results.csv --export-embeddings

# Specify model and run on CUDA-enabled GPU
python analyze.py example/soundscape.wav --model models/BirdNET+_V3.0-preview1_EUNA_1K_FP32.pt --device cuda
```

**Note:** The minimal model call will return embeddings and predictions for all chunks and needs to look like this:

```python
embeddings, predictions = model(input)
```

## MIL Pooling Heads

This repository includes a complete pipeline for training Multiple Instance Learning (MIL) pooling heads on BirdNET embeddings using strong labels from AnuraSet.

### Project Structure

```
bird-net_mil/
├── birdnetv3/               # BirdNET V3 embedding extraction
│   ├── model_loader.py      # Load TorchScript model
│   ├── utils_audio.py       # Audio loading/framing utilities
│   └── embed_all.py         # Batch embedding export
├── mil/                     # MIL training and evaluation
│   ├── heads.py             # Pooling heads (LME, Attention, AutoPool, etc.)
│   ├── datasets.py          # Dataset for embeddings + strong labels
│   ├── train.py             # Training loop with W&B integration
│   └── evaluate.py          # Pointing game, attention visualization
├── scripts/                 # CLI entry points
│   ├── export_embeddings.py # Export embeddings from audio
│   ├── train_mil.py         # Train MIL pooling heads
│   └── plot_attention.py    # Visualize attention weights
├── configs/                 # Configuration files
│   └── default.yaml         # Default training config
├── tests/                   # Unit tests
├── runs/                    # Output directory for results
└── pyproject.toml           # Project configuration
```

### Dataset Layout (AnuraSet)

The MIL pipeline expects data organized as follows. Each 1-minute recording is split into overlapping 3-second clips (sliding by 1 second), and strong labels are provided per recording.

```
/data/
├── anuraset/
│   ├── wavs/                 # Preprocessed 3-s clips by site
│   │   ├── SITE_A/
│   │   │   ├── INCT17_20200211_041500_0_3.wav    # Recording INCT17_20200211_041500, seconds 0-3
│   │   │   ├── INCT17_20200211_041500_1_4.wav    # Recording INCT17_20200211_041500, seconds 1-4
│   │   │   ├── INCT17_20200211_041500_2_5.wav    # Recording INCT17_20200211_041500, seconds 2-5
│   │   │   ├── ...
│   │   │   └── INCT17_20200211_042500_0_3.wav
│   │   └── SITE_B/...
│   ├── metadata.csv          # Class labels (multi-label)
│   └── strong_labels/        # Per-event annotations (one file per recording)
│       ├── SITE_A/
│       │   ├── INCT17_20200211_041500.txt  # Lines: "<start_sec> <end_sec> <species_quality>"
│       │   └── INCT17_20200211_042500.txt
│       └── SITE_B/...
└── embeddings/               # Output from embedding extraction
    ├── SITE_A/
    │   ├── INCT17_20200211_041500_0_3.embeddings.npz
    │   ├── INCT17_20200211_041500_1_4.embeddings.npz
    │   └── ...
    └── SITE_B/...
```

#### Clip Filename Format

Each 3-second clip filename follows the pattern: `<recording_id>_<start_sec>_<end_sec>.wav`

The recording ID can contain underscores (e.g., `INCT17_20200211_041500`). The last two underscore-separated values are always the start and end seconds.

- `INCT17_20200211_041500_0_3.wav` - Recording INCT17_20200211_041500, from second 0 to second 3
- `INCT17_20200211_041500_1_4.wav` - Recording INCT17_20200211_041500, from second 1 to second 4
- `INCT17_20200211_041500_7_10.wav` - Recording INCT17_20200211_041500, from second 7 to second 10

#### Strong Label Format

Strong labels are per-recording annotation files with one event per line:

```
<start_sec> <end_sec> <species_quality>
```

Where `<species_quality>` is the species name followed by an underscore and an audio quality indicator (L=low, M=medium, H=high).

**Example** (`INCT17_20200211_041500.txt`):
```
0.5 2.3 Boana_faber_H
1.2 4.5 Dendropsophus_minutus_M
15.0 18.0 Scinax_fuscovarius_L
```

The code automatically:
- Extracts the recording ID from clip filenames (e.g., `INCT17_20200211_041500_7_10` → `INCT17_20200211_041500`)
- Matches clips to their corresponding strong label file
- Parses species names by removing the quality suffix (e.g., `Boana_faber_H` → `Boana_faber`)
- Assigns labels to clips based on time overlap between clip boundaries and event boundaries

### Available Pooling Heads

| Pooler | Description | Output | Loss |
|--------|-------------|--------|------|
| `mean` | Simple mean over time | Logits | BCEWithLogitsLoss |
| `max` | Max pooling over time | Logits | BCEWithLogitsLoss |
| `lme` | Log-mean-exp with learnable temperature | Logits | BCEWithLogitsLoss |
| `autopool` | Adaptive softmax pooling (McFee et al., 2018) | Probabilities | BCELoss |
| `attn` | Attention-based MIL (Ilse et al., 2018) | Probabilities | BCELoss |
| `linsoft` | Linear-softmax pooling (Wang & Metze, 2017) | Probabilities | BCELoss |
| `noisyor` | Noisy-OR pooling | Probabilities | BCELoss |

### Quick Start: MIL Training

```bash
# 1) Export embeddings for all WAVs (1-s chunks)
python scripts/export_embeddings.py \
    --wav_dir /data/anuraset/wavs \
    --out_dir /data/embeddings \
    --chunk_length 1.0 \
    --overlap 0.0

# 2) Train several poolers and log to W&B
python scripts/train_mil.py \
    --emb_dir /data/embeddings \
    --strong_root /data/anuraset/strong_labels \
    --poolers lme attn autopool linsoft noisyor mean max \
    --epochs 20 \
    --batch_size 32 \
    --out_dir ./runs \
    --wandb \
    --wandb_project bird-mil

# 3) Plot attention for a specific file & class
python scripts/plot_attention.py \
    --npz /data/embeddings/SITE_A/REC_000001.embeddings.npz \
    --checkpoint runs/attn_last.pt \
    --class "Boana faber" \
    --out runs/attention_REC_000001_Boana_faber.png \
    --wandb
```

### Install for MIL Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Or install with W&B support
pip install -e ".[wandb]"

# Or install everything
pip install -e ".[all]"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mil_heads.py -v

# Run with coverage
pytest tests/ --cov=mil --cov=birdnetv3
```

### Python API Example

```python
import torch
from mil.heads import PoolingHead

# Create a pooling head
head = PoolingHead(
    in_dim=1024,    # BirdNET embedding dimension
    n_classes=42,   # Number of species
    pool="attn"     # Attention-based pooling
)

# Forward pass
embeddings = torch.randn(4, 3, 1024)  # (batch, time, features)
y_clip, z_per_sec = head(embeddings)

print(f"Clip predictions: {y_clip.shape}")      # (4, 42)
print(f"Per-second logits: {z_per_sec.shape}")  # (4, 3, 42)

# Get attention weights for visualization
attention = head.get_attention_weights(embeddings)  # (4, 3, 42)
```

### W&B Integration

When running with `--wandb`, the training script will log:
- Per-epoch loss curves
- Final loss plots as images
- Model checkpoints
- Training configuration

![W&B Dashboard Placeholder](img/wandb-placeholder.png)

### Adding New Pooling Heads

To add a custom pooler:

1. Create a new class in `mil/heads.py`:
```python
class MyCustomPool(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, C) logits
        # Return: (B, C) pooled output
        ...
```

2. Add to `POOLER_NAMES` list
3. If it returns probabilities, add to `PROB_SPACE_POOLERS`
4. Handle initialization in `PoolingHead.__init__`

---
