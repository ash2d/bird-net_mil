#!/bin/bash
#SBATCH --job-name=train_test
#SBATCH --output=logs/train_test%j/o.out
#SBATCH --error=logs/train_test%j/e.err
#SBATCH --time=1:00:00
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
srun uv run scripts/create_train_test_split.py --emb_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1 --train_out /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/train_0_1.txt --test_out /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1.txt