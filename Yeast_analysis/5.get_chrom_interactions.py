import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import importlib
from tqdm import tqdm

import os
os.environ['CUDA_PATH'] = '/apps/compilers/cuda/13.2.1' 

import torch
print(f"number of available GPUs = {torch.cuda.device_count()}")
output_device = 'cuda:0'

import vcme
from vcme.functions import prepare_proteingym_data
from vcme.utils import yeast_data, clear_gpu_mem, reload_obj, get_pathway_info
from vcme.utils import tensor_scatter, read_marginal_results, get_gene_interval
from vcme.utils import YeastMetadata

from epik.utils import encode_seqs, split_training_test

import epikVC
from epikVC.models import GPModel, make_GP_model, load_GP_model

fig_path = 'figures/'
result_path = 'results/'



wd = '/blue/juannanzhou/juannanzhou/vcme_project/'
checkpoint_path = wd + 'model_checkpoints/37C_r2_threshold=None_MAF_threshold=0/'
geno_path = "/orange/juannanzhou/MarginalEpistasis/data/"
pheno_path = "/orange/juannanzhou/dryad_data/"
pheno_name = '37C'
model_name = 'GP_model_k=8.model'

data = yeast_data(geno_path, pheno_path, pheno_name, output_device, prune_snps=False, r2_threshold=None, maf_threshold=0, top_percent=25)

snp_info_path = wd + "data/yeast/snp_info_final.csv"
pathway_info_path = wd + "data/yeast/all_pathway_gene_tables.tsv"
genomic_meta_data = YeastMetadata(snp_info_path, pathway_info_path)
snp_info = genomic_meta_data.snp_info
gene_info = genomic_meta_data.gene_info
gene_info = gene_info.set_index('gene_standard_name')
genes = snp_info.gene_standard_name.unique()

gene_snps = {}

for g in genes:
    snps = snp_info.index[snp_info.gene_standard_name == g]
    gene_snps[g] = snps
    
    
GP = load_GP_model(checkpoint_path + model_name)
train_x, train_y = GP.genos, GP.y
A, L = 2, int(train_x.shape[1]/2)

alpha = GP.get_alpha()
log_lda = GP.get_lda()

from vcme.marginals import MargEpistasis
marg = MargEpistasis(A, L, train_x, train_y, alpha, log_lda, chunk_size=10**10, beta_samples=None)


ss_2 = marg.get_SS_k(2) # pairwise VC

from itertools import combinations_with_replacement

chrs = [np.where(snp_info.chrom == i)[0] for i in range(1, 17)]
chr_pairs_no = list(combinations_with_replacement(range(16), 2))

results = []

for a, b in tqdm(chr_pairs_no):
    test_stat = marg.get_gxg_interaction(chrs[a], chrs[b])
    results.append(test_stat)

# Counts assume self-pairs exclude a SNP interacting with itself
# and count each unordered SNP pair once.
n_snp_pairs = [
    len(chrs[a]) * len(chrs[b])
    if a != b
    else len(chrs[a]) * (len(chrs[a]) - 1) // 2
    for a, b in chr_pairs_no
]

chr_interaction_df = pd.DataFrame({
    "chromosome_a": [a for a, b in chr_pairs_no],
    "chromosome_b": [b for a, b in chr_pairs_no],
    "SS": np.asarray(results),
    "VC": np.asarray(results)/ss_2, 
    "n_snp_pairs": n_snp_pairs,
})



chr_interaction_df.to_csv(result_path + 'chrom_interactions.csv')