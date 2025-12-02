#!/bin/bash
#SBATCH --job-name=embed_all
#SBATCH --output=logs/embed_all%j/o.out
#SBATCH --error=logs/embed_all%j/e.err
#SBATCH --time=10:00:00
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

mkdir -p logs
# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd birdnet-V3.0-dev
# Run your Python script with desired arguments
srun uv run scripts/export_embeddings.py --wav_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio --out_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings --chunk_length 1.0 --overlap 0.0
