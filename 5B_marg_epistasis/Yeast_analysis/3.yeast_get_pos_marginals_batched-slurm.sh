#!/bin/bash
#SBATCH --job-name=yeast_pos_marg
#SBATCH --output=logs/yeast_pos_marg_%A_%a.out
#SBATCH --error=logs/yeast_pos_marg_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --time=8:00:00
#SBATCH --array=0-99%10

set -euo pipefail

# One posterior beta sample is processed by each array task. Limiting the array
# to ten concurrent tasks avoids requesting all 100 GPUs at once.
environment="${1:-}"
dose="${2:-}"
case "$environment:$dose" in
    37C:)
        top_percent=40
        dataset_name=37C
        ;;
    FLU:CON|FLU:QMIC|FLU:HMIC|FLU:FMIC|PUL:CON|PUL:HMIC|PUL:FMIC|PUL:DMIC)
        top_percent=60
        dataset_name="${environment}_${dose}"
        ;;
    *)
        echo "Usage: sbatch $0 [37C | FLU {CON,QMIC,HMIC,FMIC} | PUL {CON,HMIC,FMIC,DMIC}]" >&2
        exit 2
        ;;
esac
if (( $# > 2 )); then
    echo "Expected at most two arguments: environment and dose" >&2
    exit 2
fi

# Slurm runs a copy of this script, so locate the source via the submit directory.
for base_dir in "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    [[ -n "$base_dir" ]] || continue
    for candidate in "$base_dir" "$base_dir/Yeast_analysis" "$base_dir/5B_marg_epistasis/Yeast_analysis"; do
        if [[ -f "$candidate/3.yeast_get_pos_marginals_batched-cli.py" ]]; then
            analysis_dir="$candidate"
            break 2
        fi
    done
done
if [[ -z "${analysis_dir:-}" ]]; then
    echo "Submit from BB-QTL_HT, 5B_marg_epistasis, or Yeast_analysis." >&2
    exit 1
fi
cd "$analysis_dir"

task_id="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array job}"
if (( task_id < 0 || task_id > 99 )); then
    echo "SLURM_ARRAY_TASK_ID must be between 0 and 99; received $task_id" >&2
    exit 2
fi

module load conda
conda activate gpy5

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "No CUDA GPU available")'

checkpoint_dir="../model_checkpoints/${dataset_name}_r2_threshold=None_MAF_threshold=0_top${top_percent}_percent"
model_path="$checkpoint_dir/GP_model_k=8_top${top_percent}percent.model"
out_path="../results/yeast_analysis/$dataset_name/posterior_marginals"

if [[ ! -s "$model_path" ]]; then
    echo "Required checkpoint is missing or empty: $model_path" >&2
    exit 1
fi
mkdir -p "$out_path"

echo "Array job ${SLURM_ARRAY_JOB_ID:-unknown}, task $task_id"
echo "Dataset: $dataset_name; posterior sample: $task_id"

python 3.yeast_get_pos_marginals_batched-cli.py \
    --indices "$task_id" \
    --order 2 \
    --model-path "$model_path" \
    --out-path "$out_path" \
    --run-name "$dataset_name" \
    --device cuda:0
