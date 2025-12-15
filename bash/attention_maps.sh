#!/bin/bash
#SBATCH --job-name=attention_maps
#SBATCH --output=logs/attention_maps%j/o.out
#SBATCH --error=logs/attention_maps%j/e.err
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
export run_dir=/gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/run_20251215_113543_seed42

cd birdnet-V3.0-dev
# Run your Python script with desired arguments
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_24_27.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_24_27.wav  --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_24_27.png         --spectrogram    --wandb --wandb_project bird-mil-scratc-train-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_26_29.embeddings.npz  --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_26_29.wav   --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_26_29.png           --spectrogram  --wandb --wandb_project bird-mil-scratc-train-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_39_42.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_39_42.wav  --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_39_42.png          --spectrogram   --wandb --wandb_project bird-mil-scratc-train-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_41_44.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_41_44.wav  --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_41_44.png           --spectrogram  --wandb --wandb_project bird-mil-scratc-train-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_44_47.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_44_47.wav  --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_44_47.png           --spectrogram  --wandb --wandb_project bird-mil-scratc-train-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_47_50.embeddings.npz    --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_47_50.wav  --checkpoint $run_dir/attn_last.pt     --class "BOAFAB"     --out $run_dir/attention/INCT4_20191223_033000_47_50.png          --spectrogram   --wandb --wandb_project bird-mil-scratc-train-0_1