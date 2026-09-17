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
ss_2 = marg.get_SS_k(2)

def get_mean_gxg_vc(gene1, gene2):
    snps1, snps2 = gene_snps[gene1], gene_snps[gene2]
    n = len(snps1) * len(snps2)
    mean_vc = marg.get_gxg_interaction(snps1, snps2)/n
    return mean_vc

genes = snp_info.gene_standard_name.unique()

genes_hub = ['MKT1', 'MIP1', 'HAL9', 'LMO1', 'VPS70', 'BUL2', 'VPS36', 'RSF1', 'FLO5', 'CIM1']

out = {}

for gene_focal in genes_hub:
    vc_list = {}
    for gene2  in tqdm(genes):
        vc_list[gene2] = get_mean_gxg_vc(gene_focal, gene2)
    out[gene_focal] = vc_list

df_hub_interactions = pd.DataFrame(out)

df_hub_interactions /= ss_2

df_hub_interactions.to_csv(result_path + 'df_hub_interactions.tsv')


# Sample random gene pairs without constructing all possible pairs.
rng = random.Random(42)
n_pairs = 1_0000

eligible_genes = [
    g for g in pd.unique(genes)
    if pd.notna(g) and len(gene_snps[g]) > 0
]

n_genes = len(eligible_genes)
max_pairs = n_genes * (n_genes - 1) // 2

if n_pairs > max_pairs:
    raise ValueError(
        f"Requested {n_pairs:,} pairs, but only {max_pairs:,} are available."
    )

sampled_pairs = []
seen = set()

while len(sampled_pairs) < n_pairs:
    i, j = sorted(rng.sample(range(n_genes), 2))
    if (i, j) not in seen:
        seen.add((i, j))
        sampled_pairs.append((eligible_genes[i], eligible_genes[j]))

# Convert scalar tensors, including CUDA tensors, to Python numbers.
def to_scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu().item()
    return float(value)

normalizer = to_scalar(ss_2)
if not np.isfinite(normalizer) or normalizer <= 0:
    raise ValueError("ss_2 must be finite and positive.")

records = []

with torch.no_grad():
    for gene1, gene2 in tqdm(sampled_pairs, desc="Random gene pairs"):
        strength = to_scalar(get_mean_gxg_vc(gene1, gene2)) / normalizer
        records.append({
            "gene1": gene1,
            "gene2": gene2,
            "mean_VC": strength,
        })

df_random_interactions = pd.DataFrame(records)

df_random_interactions.to_csv(
    os.path.join(result_path, "random_gene_interactions.tsv"),
    sep="\t",
    index=False,
)
