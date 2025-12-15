#!/bin/bash
#SBATCH --job-name=train_mil_heads_scratch_0.1
#SBATCH --output=logs/train_mil_heads_scratch_0_1_%j/o.out
#SBATCH --error=logs/train_mil_heads_scratch_0_1_%j/e.err
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

# # Stage data to scratch
# export JOB_SCRATCH=/work/scratch-pw4/$USER/emb_0_1
# # mkdir -p "$JOB_SCRATCH/emb" "$JOB_SCRATCH/tmp"
# export TMPDIR="$JOB_SCRATCH/tmp"
# rsync -a /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/ "$JOB_SCRATCH/emb/"

mkdir -p logs

# Make Slurm CPU count visible to the loader
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
# Navigate to your project directory
cd /home/users/dash/guppies/embeddings/wildlife-tools

# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd birdnet-V3.0-dev
# Run your Python script with desired arguments

# srun uv run scripts/scratch_train_test.py --input_txt /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/train_0_1.txt --replacement "$JOB_SCRATCH/emb"

srun uv run scripts/train_mil.py     --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/train_0_1_scratch.txt     --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels    \
 --poolers lme attn linsoft mean max     --epochs 50     --batch_size 32  --seed 42   --out_dir /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs    --wandb     --wandb_project bird-mil-scratc-train-0_1 --species_list /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/species_list.txt

# srun uv run scripts/plot_attention.py      --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_24_27.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_24_27.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_24_27.png         --spectrogram    --wandb --wandb_project bird-mil-scratch-0_1
# srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_26_29.embeddings.npz  --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_26_29.wav   --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_26_29.png           --spectrogram  --wandb --wandb_project bird-mil-scratch-0_1
# srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_39_42.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_39_42.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_39_42.png          --spectrogram   --wandb --wandb_project bird-mil-scratch-0_1
# srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_41_44.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_41_44.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_41_44.png           --spectrogram  --wandb --wandb_project bird-mil-scratch-0_1
# srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_44_47.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_44_47.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_44_47.png           --spectrogram  --wandb --wandb_project bird-mil-scratch-0_1
# srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings/INCT4/INCT4_20191223_033000_47_50.embeddings.npz    --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_47_50.wav  --checkpoint /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/attn_last.pt     --class "Boana faber"     --out runs/attention_INCT4_20191223_033000_47_50.png          --spectrogram   --wandb --wandb_project bird-mil-scratch-0_1