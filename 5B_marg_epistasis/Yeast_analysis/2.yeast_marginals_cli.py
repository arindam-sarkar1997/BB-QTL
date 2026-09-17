import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Set CUDA path as per original script
os.environ['CUDA_PATH'] = '/apps/compilers/cuda/13.2.1' 

import vcme
from vcme.utils import yeast_data, YeastMetadata
from vcme.marginals import MargEpistasis
from epikVC.models import GPModel, make_GP_model, load_GP_model

def main():
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run yeast marginal epistasis analysis.")
    parser.add_argument("-k", type=int, required=True, help="Number of interactions/k-mer size")
    parser.add_argument("-model_name", type=str, required=True, help="name of saved model")
    parser.add_argument("--env", type=str, required=True, help="Phenotype environment (e.g., '37C')")
    parser.add_argument("--out_path", type=str, default="../results/yeast_analysis/", help="Output directory")
    parser.add_argument("--checkpoint_path", type=str, default='../model_checkpoints/', help="Model checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    
    args = parser.parse_args()

    # --- Configuration ---
    k = args.k
    env = args.env
    out_path = args.out_path
    output_device = args.device
    checkpoint_path = args.checkpoint_path

    # Create the directory if it doesn't exist
    Path(out_path).mkdir(parents=True, exist_ok=True)    
    
    geno_path = "/orange/juannanzhou/MarginalEpistasis/data/"
    pheno_path = "/orange/juannanzhou/dryad_data/"
    pheno_name = '37C'
    
    
    GP = load_GP_model(checkpoint_path + args.model_name)
    train_x, train_y = GP.genos, GP.y
    log_lda = GP.get_lda()
    A, L = 2, int(train_x.shape[1]/2)
    print(f'L = {L}')
    
    # --- Marginal Epistasis Calculation ---
    log_lda = GP.get_lda()
    alpha = GP.get_alpha()
    
    # Loci candidates from original script
    loci_candidate = list(range(L))

    marg = MargEpistasis(A, L, train_x, train_y, alpha, log_lda, chunk_size=10**10, beta_samples=None)
    
    if k == 1:
        site_marg_df = marg.get_additive_vc(loci_candidate)

    else:        
        site_marg_df, site_marg_df_percent = marg.get_VC_positions(k, loci_candidate)

    # --- Save Results ---
    output_file = os.path.join(out_path, f"yeast_marginals_k={k}_env={env}.csv")
    site_marg_df.to_csv(output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()