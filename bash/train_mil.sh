#!/bin/bash
#SBATCH --job-name=train_mil_heads
#SBATCH --output=logs/train_mil_heads%j/o.out
#SBATCH --error=logs/train_mil_heads%j/e.err
#SBATCH --time=20:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=gpuhost015

# Navigate to your project directory
cd /home/users/dash/guppies/embeddings/wildlife-tools

# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd birdnet-V3.0-dev
# Run your Python script with desired arguments
srun uv run scripts/train_mil.py     --emb_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings     --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels     --poolers lme attn linsoft  mean max     --epochs 20     --batch_size 32     --out_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs     --wandb     --wandb_project bird-mil
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_24_27.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_26_29.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_39_42.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_41_44.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_44_47.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_47_50.embeddings.npz     --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_REC_000001_Boana_faber.png     --wandb