import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Total memory: {total / 1024**3:.2f} GiB")
    print(f"  Free memory:  {free / 1024**3:.2f} GiB")
    print(f"  Used memory:  {(total - free) / 1024**3:.2f} GiB")
    print(f"  PyTorch allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GiB")
    print(f"  PyTorch reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f} GiB")

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

result_path = wd + '/results/yeast_analysis/8_31/'
marginals_add = read_marginal_results(result_path + f"yeast_marginals_k=1_env={pheno_name}.csv")

GP = load_GP_model(checkpoint_path + model_name)
train_x, train_y = GP.genos, GP.y
A, L = 2, int(train_x.shape[1]/2)
alpha = GP.get_alpha()
log_lda = GP.get_lda()

del GP
import gc
gc.collect()
torch.cuda.empty_cache()

from vcme.marginals import MargEpistasis
marg = MargEpistasis(A, L, train_x, train_y, alpha, log_lda, chunk_size=10**10, beta_samples=None)

import ld_matched_gene_set_sampler
import importlib
importlib.reload(ld_matched_gene_set_sampler)
from ld_matched_gene_set_sampler import LDMatchedGeneSetSampler

sampler = LDMatchedGeneSetSampler(
    data=data,
    snp_info=snp_info,
    gene_standard_name=False,
    seed=42,
)

pathway_genes = genomic_meta_data.pathway_genes

import pathway_analysis_ld
importlib.reload(pathway_analysis_ld)
from pathway_analysis_ld import LDMatchedPathwayAnalysis

analysis = LDMatchedPathwayAnalysis(
    marg=marg,
    gene_sets=pathway_genes,
    sampler=sampler,
    marginals_add=marginals_add,
    min_genes=3,
)

# Download pathway names/descriptions from SGD.
analysis.download_pathway_descriptions()


# Observed epistatic and additive statistics.
df_marginals = analysis.calculate_pathway_marginals()


# Evaluate BOTH statistics using the SAME sampled null pathways.
df_results = analysis.sample_pathway_null_distributions(
    n_samples=500,
    top_n=None,
)

df_results.to_csv("pathway_results_q_values_n=500.csv")