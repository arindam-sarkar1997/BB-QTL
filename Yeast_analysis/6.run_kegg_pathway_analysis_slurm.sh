#!/bin/bash
#SBATCH --job-name=yeast_37C_blackdiamond_k=2
#SBATCH --output=logs/yeast_37C_blackdiamond_k=2%j.out
#SBATCH --error=logs/yeast_37C_blackdiamond_k=2%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=hpg-b200
#SBATCH --gpus=b200:1
#SBATCH --time=15:00:00

module load conda
conda activate gpy5

python 6.run_kegg_pathway_analysis.py
