#!/bin/bash
#SBATCH --job-name=yeast_37C_sugarglider
#SBATCH --output=logs/yeast_37C_sugarglider_%j.out
#SBATCH --error=logs/yeast_37C_sugarglider_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --time=12:00:00

module load conda
conda activate gpy5


python 1_Fit_GP_model.py 

python 2_yeast_marginals_cli.py -k 1 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0/ --device cuda:0

python 2_yeast_marginals_cli.py -k 2 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0/ --device cuda:0

python 2_yeast_marginals_cli.py -k 3 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0/ --device cuda:0