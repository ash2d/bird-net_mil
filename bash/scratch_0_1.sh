#!/bin/bash
#SBATCH --job-name=scratch_0_1
#SBATCH --output=logs/scratch_0_1%j/o.out
#SBATCH --error=logs/scratch_0_1%j/e.err
#SBATCH --time=20:00:00
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --exclude=gpuhost015

# Stage data to scratch
export JOB_SCRATCH=/work/scratch-pw4/$USER/emb_0_1
# mkdir -p "$JOB_SCRATCH/emb" "$JOB_SCRATCH/tmp"
export TMPDIR="$JOB_SCRATCH/tmp"

# mkdir -p logs
# rsync -av --progress /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/ "$JOB_SCRATCH/emb/"

# Navigate to your project directory
cd /home/users/dash/guppies/embeddings/wildlife-tools

# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd birdnet-V3.0-dev
# change test train file lists to different scratch version

srun uv run scripts/scratch_train_test.py --input_txt /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1.txt --replacement "$JOB_SCRATCH/emb"
