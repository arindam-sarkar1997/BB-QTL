# Yeast GP inputs

The `37C` run uses the existing `vcme.utils.yeast_data` loader with the
`geno_top_40.npy` and `pheno_data_37C.txt.gz` files.

FLU and PUL use `geno_top_60.npy` and `index_top_60.tsv` from
`/orange/juannanzhou/MarginalEpistasis/data/`. The SNP metadata is
`/blue/juannanzhou/arindam.sarkar/BB-QTL_HT/4_qtl_inference/data/snp_info_complete.tsv`.
Fitness comes from
`/blue/juannanzhou/arindam.sarkar/BB-QTL_HT/3_phenotype_inference/results/Joint_inference/Filtered/{FLU,PUL}/{dose}/fitness_ALLreps_with_ids.tsv`.
The loader joins fitness to genotype rows by `ID`, retains IDs with complete
`s` and `stderr(s)` in all four doses as in the QTL notebooks, and trains one
GP on the selected dose. `stderr(s)` is squared to form the fitness variance.
The current GP fitting call uses `s` as the response and retains its existing
noise model; it does not pass this variance as fixed observation noise.

Submit a run from `Yeast_analysis` with, for example:

```bash
sbatch 0_run_analyses_slurm.sh FLU FMIC
sbatch 0_run_analyses_slurm.sh PUL DMIC
sbatch 0_run_analyses_slurm.sh 37C
```

FLU doses are `CON`, `QMIC`, `HMIC`, and `FMIC`. PUL doses are `CON`, `HMIC`,
`FMIC`, and `DMIC`. Checkpoints and marginal output directories include the
drug and dose so separate runs do not overwrite one another.
