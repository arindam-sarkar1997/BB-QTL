import os
import argparse
import torch
import numpy as np
import pandas as pd

# Set environment before other imports
os.environ['CUDA_PATH'] = '/apps/compilers/cuda/13.2.1'

import vcme
from vcme.functions import prepare_proteingym_data
from vcme.utils import yeast_data, clear_gpu_mem, reload_obj, get_pathway_info
from vcme.utils import tensor_scatter, read_marginal_results, get_gene_interval
from epik.utils import encode_seqs, split_training_test
import epikVC
from epikVC.models import GPModel, make_GP_model, load_GP_model
from vcme.marginals import MargEpistasis
from vcme.kernels import SiteMarginalKernel, EpKernel, seq_at_p, PathwayMarginalKernel

def main():
    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run yeast marginal epistasis analysis.")
    # Changed -i to accept a list of integers
    parser.add_argument("-i", type=int, nargs='+', required=True, help="List of indices for the beta samples")
    parser.add_argument("-k", type=int, required=True, help="The k-mer or order parameter")
    parser.add_argument("-model_name", type=str, required=True, help="name of saved model")
    parser.add_argument("-run_name", type=str, required=True, help="name for the run")    
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)")

    args = parser.parse_args()
    indices = args.i  # This is now a list
    k = args.k
    output_device = args.device

    print(f"Running samples i={indices}, k={k} on {output_device}")
    print(f"Number of available GPUs = {torch.cuda.device_count()}")

    # --- Paths ---
    checkpoint_path = '../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0_top40_percent/'
    geno_path = "/orange/juannanzhou/MarginalEpistasis/data/"
    pheno_path = "/orange/juannanzhou/dryad_data/"
    out_path = '../results/yeast_analysis/' 
    pheno_name = '37C'
    
    # Load data once outside the loop for efficiency
    data = yeast_data(geno_path, pheno_path, pheno_name, output_device, prune_snps=True, r2_threshold=0.995, maf_threshold=0)
    pruned_loci = data.loci_pruned
    
    GP = load_GP_model(checkpoint_path + args.model_name)
    train_x, train_y = GP.genos, GP.y
    log_lda = GP.get_lda()
    A, L = 2, int(train_x.shape[1]/2)
    print(f'L = {L}')
    loci_candidate = list(range(L))
    
    # Prepend alpha to beta_samples
    beta_samples = torch.cat((GP.alpha.unsqueeze(0), GP.beta_samples), dim=0)
    
    # Ensure output directory exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # --- Loop through the provided indices ---
    for i in indices:
        print(f"--- Processing sample index {i} ---")
        
        try:
            beta = beta_samples[i]
            
            marg = MargEpistasis(A, L, train_x, train_y, beta, log_lda, chunk_size=10**10, beta_samples=None)
            site_marg_df, site_marg_df_percent = marg.get_VC_positions(k, loci_candidate)    

            site_marg_df.index = pruned_loci
            site_marg_df_percent.index = pruned_loci
            
            # --- Save Results for index i ---
            output_file = os.path.join(out_path, f"{args.run_name}_k={k}_sample_{i}.csv")
            site_marg_df.to_csv(output_file)

            output_file_pct = os.path.join(out_path, f"{args.run_name}_k={k}_sample_{i}_percent.csv")
            site_marg_df_percent.to_csv(output_file_pct)
            
            print(f"Results for sample {i} saved successfully.")
            
        except IndexError:
            print(f"Error: Index {i} is out of bounds for beta_samples.")
        
    print("All requested indices processed.")

if __name__ == "__main__":
    main()