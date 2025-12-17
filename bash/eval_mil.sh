#!/bin/bash
#SBATCH --job-name=eval_mil_heads
#SBATCH --output=logs/eval_mil_heads%j/o.out
#SBATCH --error=logs/eval_mil_heads%j/e.err
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


# Make Slurm CPU count visible to the loader
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-8}
# Navigate to your project directory
cd /home/users/dash/guppies/embeddings/wildlife-tools

# Activate the virtual environment
source /home/users/dash/guppies/embeddings/wildlife-tools/.venv/bin/activate

cd birdnet-V3.0-dev
mkdir -p logs

export gws_run_dir=/gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/runs/run_20251210_165545_seed42

srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_24_27.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_24_27.wav  --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_24_27.png   --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt     --spectrogram   # --wandb --wandb_project bird-mil-scratch-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_26_29.embeddings.npz  --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_26_29.wav   --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_26_29.png     --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt      --spectrogram  #--wandb --wandb_project bird-mil-scratch-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_39_42.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_39_42.wav  --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_39_42.png     --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt      --spectrogram  # --wandb --wandb_project bird-mil-scratch-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_41_44.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_41_44.wav  --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_41_44.png     --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt      --spectrogram # --wandb --wandb_project bird-mil-scratch-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_44_47.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_44_47.wav  --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_44_47.png     --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt      --spectrogram # --wandb --wandb_project bird-mil-scratch-0_1
srun uv run scripts/plot_attention.py     --npz /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/embeddings_0_1/INCT4/INCT4_20191223_033000_47_50.embeddings.npz   --audio /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/anuraset/audio/INCT4/INCT4_20191223_033000_47_50.wav  --checkpoint $gws_run_dir/attn_last.pt     --class "BOAFAB"     --out $gws_run_dir/attention_INCT4_20191223_033000_47_50.png    --strong  /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels/INCT4/INCT4_20191223_033000.txt      --spectrogram  # --wandb --wandb_project bird-mil-scratch-0_1

srun uv run scripts/evaluate_mil.py \
    --checkpoint $gws_run_dir/attn_last.pt \
    --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1_scratch.txt \
    --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output $gws_run_dir/tests/attn_test_results.json

srun uv run scripts/evaluate_mil.py \
    --checkpoint $gws_run_dir/linsoft_last.pt \
    --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1_scratch.txt \
    --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output $gws_run_dir/tests/linsoft_test_results.json

srun uv run scripts/evaluate_mil.py \
    --checkpoint $gws_run_dir/lme_last.pt \
    --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1_scratch.txt \
    --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output $gws_run_dir/tests/lme_test_results.json

srun uv run scripts/evaluate_mil.py \
    --checkpoint $gws_run_dir/max_last.pt \
    --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1_scratch.txt \
    --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output $gws_run_dir/tests/max_test_results.json

srun uv run scripts/evaluate_mil.py \
    --checkpoint $gws_run_dir/mean_last.pt \
    --emb_glob /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/train_test/test_0_1_scratch.txt \
    --strong_root /gws/nopw/j04/iecdt/dash/birdnet-V3.0-dev/data/AnuraSet/strong_labels \
    --batch_size 64 \
    --pointing_game \
    --output $gws_run_dir/tests/mean_test_results.json
