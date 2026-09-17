#!/bin/bash
#SBATCH --job-name=yeast_37C_pos_marg_beaver_dam
#SBATCH --output=logs/quaileggs_%A_%a.out
#SBATCH --error=logs/quaileggs_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --time=4:00:00
#SBATCH --array=0-9

# --- Logic to divide 100 indices (0-99) into 10 runs ---
# Task 0 gets 0-9, Task 1 gets 10-19, ..., Task 9 gets 90-99
START_IDX=$((SLURM_ARRAY_TASK_ID * 10))
END_IDX=$((START_IDX + 9))

# Generate the space-separated list of indices (e.g., "0 1 2 3 4 5 6 7 8 9")
INDICES=$(seq -s " " $START_IDX $END_IDX)

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing indices: $INDICES"

module load conda
conda activate gpy5


python yeast_get_pos_marginals_batched-cli.py \
    -i $INDICES \
    -k 2 \
    -model_name GP_model_k=8_top40percent.model \
    -run_name quaileggs \
    --device cuda:0