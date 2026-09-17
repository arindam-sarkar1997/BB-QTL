pheno_name = '37C'
prune_snps = False
r2_threshold = None
MAF_threshold = 0
top_percent = 40

from pathlib import Path
fig_path = '../figures/'
result_path = '../results/yeast_analysis/'

checkpoint_path = f'../model_checkpoints/{pheno_name}_r2_threshold={r2_threshold}_MAF_threshold={MAF_threshold}_top{top_percent}_percent/'
Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

geno_path = "/orange/juannanzhou/MarginalEpistasis/data/"
pheno_path = "/orange/juannanzhou/dryad_data/"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sys
import importlib
import gc


import os
os.environ['CUDA_PATH'] = '/apps/compilers/cuda/13.2.1' 

import torch
print(f"number of available GPUs = {torch.cuda.device_count()}")
output_device = 'cuda:0'

import vcme
from vcme.functions import prepare_proteingym_data
from vcme.utils import yeast_data, clear_gpu_mem, reload_obj, get_pathway_info
from vcme.utils import tensor_scatter, read_marginal_results, get_gene_interval

from epik.utils import encode_seqs, split_training_test

import epikVC
from epikVC.models import GPModel

gc.collect()
torch.cuda.empty_cache()

data = yeast_data(geno_path, pheno_path, pheno_name, output_device, prune_snps=prune_snps, r2_threshold=r2_threshold, top_percent=top_percent, maf_threshold=MAF_threshold)

A, L = data.A, data.L

print(f"L = {L}")

train_x, train_y, test_x, test_y, train_y_var = data.train_test_split(seed=666)
print(f'train_x shape = {train_x.shape}')



from epikVC.models import GPModel, make_GP_model, load_GP_model
n_devices = 1
d_max = L
device = 'cuda:0'
noise_variance = None

# Epistatic model
k_max = 8

## custom noise

GP = make_GP_model(train_x, train_y, A, L, d_max, k_max, log_lda=None, noise_variance=noise_variance, device='cuda:0')

GP.fit_model(n_steps=200, learning_rate=0.1, mll=True)
GP.fit_model(n_steps=40, learning_rate=0.1, mll=False)

GP.get_alpha()
GP.draw_pos_y(num_samples=100)
GP.get_pos_beta()

GP.save_checkpoint(checkpoint_path + f'GP_model_k={k_max}_top{top_percent}percent.model')
del GP
gc.collect()
torch.cuda.empty_cache()


# Additive model
k_max = 1
## custom noise
GP = make_GP_model(train_x, train_y, A, L, d_max, k_max, log_lda=None, noise_variance=noise_variance, device='cuda:0')

GP.fit_model(n_steps=100, learning_rate=0.1, mll=True)

GP.save_checkpoint(checkpoint_path + f'GP_model_k={k_max}_top{top_percent}percent.model')