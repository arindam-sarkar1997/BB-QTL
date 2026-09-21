#!/bin/bash
#SBATCH --job-name=yeast_37C_sugarglider
#SBATCH --output=yeast_37C_sugarglider_%j.out
#SBATCH --error=yeast_37C_sugarglider_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --time=12:00:00

set -euo pipefail

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

checkpoint_path="../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0_top40_percent"
model_name="GP_model_k=8_top40percent.model"
out_path="../results/yeast_analysis/5_19"

python 1.Fit_GP_model.py
if [[ ! -s "$checkpoint_path/$model_name" ]]; then
    echo "Training did not produce $checkpoint_path/$model_name" >&2
    exit 1
fi

for k in 1 2 3; do
    python 2.yeast_marginals_cli.py \
        -k "$k" \
        -model_name "$model_name" \
        --env 37C \
        --out_path "$out_path" \
        --checkpoint_path "$checkpoint_path" \
        --device cuda:0
done
