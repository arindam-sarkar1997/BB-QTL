"""Inputs for GP training on the FLU and PUL fitness experiments."""

from pathlib import Path

import numpy as np
import pandas as pd


GENO_PATH = Path("/orange/juannanzhou/MarginalEpistasis/data/geno_top_60.npy")
IDS_PATH = Path("/orange/juannanzhou/MarginalEpistasis/data/index_top_60.tsv")
SNP_INFO_PATH = Path(
    "/blue/juannanzhou/arindam.sarkar/BB-QTL_HT/4_qtl_inference/data/snp_info_complete.tsv"
)
PHENO_PATH = Path(
    "/blue/juannanzhou/arindam.sarkar/BB-QTL_HT/3_phenotype_inference/results/Joint_inference/Filtered"
)
DRUG_CONDITIONS = {
    "FLU": ("CON", "QMIC", "HMIC", "FMIC"),
    "PUL": ("CON", "HMIC", "FMIC", "DMIC"),
}


def load_drug_inputs(
    environment,
    dose,
    geno_path=GENO_PATH,
    ids_path=IDS_PATH,
    snp_info_path=SNP_INFO_PATH,
    pheno_path=PHENO_PATH,
):
    """Return genotype, fitness, fitness variance, and aligned IDs for one dose.

    As in the QTL notebooks, retain only IDs with both fitness columns in
    every dose of the chosen drug. Genotype rows remain in index_top_60 order.
    """
    if environment not in DRUG_CONDITIONS:
        raise ValueError(f"Unsupported drug environment: {environment}")
    conditions = DRUG_CONDITIONS[environment]
    if dose not in conditions:
        raise ValueError(f"Dose {dose!r} is not available for {environment}: {conditions}")

    geno = np.load(geno_path, mmap_mode="r")
    ids = np.loadtxt(ids_path, delimiter="\t", dtype=np.int64, ndmin=1)
    if geno.ndim != 2 or len(ids) != len(geno):
        raise ValueError("Genotype matrix rows must match index_top_60 IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Genotype IDs must be unique")
    snp_count = len(pd.read_csv(snp_info_path, sep="\t", usecols=["Index"]))
    if snp_count != geno.shape[1]:
        raise ValueError("SNP metadata rows must match genotype matrix columns")

    fitness_tables = []
    for condition in conditions:
        path = Path(pheno_path) / environment / condition / "fitness_ALLreps_with_ids.tsv"
        df = pd.read_csv(path, sep="\t", usecols=["ID", "s", "stderr(s)"])
        if df["ID"].duplicated().any():
            raise ValueError(f"Duplicate fitness IDs in {path}")
        fitness_tables.append(df.set_index("ID")[["s", "stderr(s)"]])

    fitness = pd.concat(fitness_tables, axis=1, keys=conditions).dropna()
    row_positions = fitness.index.get_indexer(ids)
    present = row_positions >= 0
    if not present.any():
        raise ValueError(f"No genotype IDs have complete {environment} fitness data")

    selected = fitness[dose].iloc[row_positions[present]]
    y = selected["s"].to_numpy(dtype=np.float32)
    stderr = selected["stderr(s)"].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all() or not np.isfinite(stderr).all() or (stderr < 0).any():
        raise ValueError(f"Invalid fitness or standard errors for {environment}/{dose}")

    y_var = stderr**2
    if not np.isfinite(y_var).all():
        raise ValueError(f"Invalid fitness variance for {environment}/{dose}")
    return np.asarray(geno[present]), y, y_var, ids[present]


def make_drug_yeast_data(environment, dose, output_device, prune_snps=False,
                         r2_threshold=None, maf_threshold=0):
    """Adapt drug inputs to the vcme yeast_data training interface."""
    import torch
    import torch.nn.functional as F
    from vcme.utils import yeast_data

    geno, y, y_var, ids = load_drug_inputs(environment, dose)
    if geno.dtype.kind not in "biu" or geno.min() < 0 or geno.max() > 1:
        raise ValueError("Drug genotype matrix must contain only 0 and 1")

    data = yeast_data.__new__(yeast_data)
    data.output_device = output_device
    data.A = 2
    data.ids = ids
    data.y = torch.as_tensor(y, device=output_device)
    data.y_var = torch.as_tensor(y_var, device=output_device)
    data.geno_t = torch.as_tensor(geno, dtype=torch.long, device=output_device)
    data.L = data.geno_t.shape[1]

    if prune_snps or r2_threshold is not None:
        data.prune_snps(
            r2_threshold=1.0 if r2_threshold is None else r2_threshold,
            maf_threshold=maf_threshold,
        )
    else:
        data.X = F.one_hot(data.geno_t, num_classes=2).reshape(len(data.geno_t), -1).float()
    data.X_id = torch.arange(len(data.X), device=output_device).unsqueeze(1)
    return data
