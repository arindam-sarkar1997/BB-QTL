#!/bin/bash
#SBATCH --job-name=yeast_gp
#SBATCH --output=yeast_gp_%j.out
#SBATCH --error=yeast_gp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00

set -euo pipefail

# Examples: sbatch 0_run_analyses_slurm.sh FLU FMIC
#           sbatch 0_run_analyses_slurm.sh PUL DMIC
environment="${1:-37C}"
dose="${2:-}"
case "$environment:$dose" in
    37C:) top_percent=40; dataset_name=37C ;;
    FLU:CON|FLU:QMIC|FLU:HMIC|FLU:FMIC|PUL:CON|PUL:HMIC|PUL:FMIC|PUL:DMIC)
        top_percent=60
        dataset_name="${environment}_${dose}"
        ;;
    *) echo "Usage: sbatch $0 [37C | FLU {CON,QMIC,HMIC,FMIC} | PUL {CON,HMIC,FMIC,DMIC}]" >&2; exit 2 ;;
esac
if (( $# > 2 )); then
    echo "Expected at most two arguments: environment and dose" >&2
    exit 2
fi

# Slurm runs a copy of this script, so locate the source via the submit directory.
for base_dir in "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    [[ -n "$base_dir" ]] || continue
    for candidate in "$base_dir" "$base_dir/Yeast_analysis" "$base_dir/5B_marg_epistasis/Yeast_analysis"; do
        if [[ -f "$candidate/1.Fit_GP_model.py" ]]; then
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

module load conda
conda activate gpy5

python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "No CUDA GPU available")'

checkpoint_path="../model_checkpoints/${dataset_name}_r2_threshold=None_MAF_threshold=0_top${top_percent}_percent"
model_name="GP_model_k=8_top${top_percent}percent.model"
if [[ "$environment" == "37C" ]]; then
    out_path="../results/yeast_analysis/5_19"
    train_args=(--env "$environment")
else
    out_path="../results/yeast_analysis/$dataset_name"
    train_args=(--env "$environment" --dose "$dose")
fi

python 1.Fit_GP_model.py "${train_args[@]}"
if [[ ! -s "$checkpoint_path/$model_name" ]]; then
    echo "Training did not produce $checkpoint_path/$model_name" >&2
    exit 1
fi

for k in 1 2 3; do
    python 2.yeast_marginals_cli.py \
        -k "$k" \
        -model_name "$model_name" \
        --env "$dataset_name" \
        --out_path "$out_path" \
        --checkpoint_path "$checkpoint_path" \
        --device cuda:0
done
