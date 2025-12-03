#!/bin/bash
#SBATCH --job-name=train_mil_heads
#SBATCH --output=logs/train_mil_heads%j/o.out
#SBATCH --error=logs/train_mil_heads%j/e.err
#SBATCH --time=1:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
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
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_24_27.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_24_27.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_24_27.png         --spectrogram    --wandb --wandb_project bird-mil-scratch
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_26_29.embeddings.npz  --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_26_29.wav   --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_26_29.png           --spectrogram  --wandb --wandb_project bird-mil-scratch
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_39_42.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_39_42.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_39_42.png          --spectrogram   --wandb --wandb_project bird-mil-scratch
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_41_44.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_41_44.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_41_44.png           --spectrogram  --wandb --wandb_project bird-mil-scratch
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_44_47.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_44_47.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_44_47.png           --spectrogram  --wandb --wandb_project bird-mil-scratch
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_47_50.embeddings.npz    --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_47_50.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_47_50.png          --spectrogram   --wandb --wandb_project bird-mil-scratch