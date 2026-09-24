import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    import torch
except Exception:
    torch = None

plt.rcParams.update({
    'font.size': 20,
    'font.family': 'Nimbus Sans'})

yeast_chr_lengths = [
    230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643,
    439888, 745751, 666816, 1078177, 924431, 784333, 1091291, 948066
]
yeast_chr_size = np.array(yeast_chr_lengths).sum()

chr_labels_arabic = [str(i) for i in range(1, 17)]
chr_offsets = np.cumsum([0] + yeast_chr_lengths[:-1])

class yeast_geno_data:
    def __init__(self, geno_data_path, ids_path, snp_info_path,
                 gene_info_path, gpu_device='cuda:0'):

        self.geno = np.load(geno_data_path)

        # --- ADD THIS BLOCK ---
        if gpu_device.startswith("cuda") and (not torch.cuda.is_available()):
            gpu_device = "cpu"
        # ----------------------
        print(f"[yeast_geno_data] Using device: {gpu_device}")
        self.geno_t = torch.tensor(self.geno).to(gpu_device)

        self.ids = np.loadtxt(ids_path, delimiter='\t', dtype='int')
        self.snp_info = pd.read_csv(snp_info_path, delimiter='\t', index_col=0)
        self.gene_info = pd.read_csv(gene_info_path)
        self.L = self.geno.shape[1]

    def sub_sample_snps(self, r2_threshold):
        lim  = self.L
        loci = [0]
        i = 0
        while i < lim:
            j = i + 1
            while (j < lim) and (torch.corrcoef(
                self.geno_t[:, [i, j]].t())[0, 1]**2 >= r2_threshold):
                j += 1
            if j < lim:
                i = j
                loci.append(i)
            else: break
        self.loci_chosen = loci
        self.abs_pos = {locus: get_abs_pos(*list(self.snp_info.loc[locus][['Chromosome', 'Position (bp)']])) for locus in self.loci_chosen}

    def get_geno(self, snps):
        return self.geno[:, snps]

    # def get_gene_info(self, gene_name):
    #     return self.gene_info[self.gene_info.Standard_name == gene_name]

class yeast_fitness_data():
    def __init__(self, treatment, conditions, in_path):
        self.treatment = treatment
        self.conditions = conditions
        self.in_path = in_path


    def load_condition(self, condition, fitness):
        dfs = [pd.read_csv(self.in_path / f'fitness/{condition}_{rep}.tsv').set_index('Number') for rep in [1, 2, 3]]

        df_reps = pd.concat(dfs, axis=1, join='outer')
        df_reps['Idx'] = df_reps.index

        # df_fit = df_reps[fitness].dropna()
        df_fit = df_reps[fitness].copy()
        df_fit = df_fit
        df_fit['mean_log_fit'] = df_fit.mean(1)
        df_fit['std_log_fit'] = df_fit.std(1)

        return df_fit

    def load_all_conditions(self, fitness, sub=None):
        df_fits = {}
        for condition in self.conditions:
            df_fits[condition] = self.load_condition(condition, fitness)

        self.df_fits_all = pd.concat(list(df_fits.values()), axis=1, join='outer')
        self.df_fits = self.df_fits_all[['mean_log_fit', 'std_log_fit']]
        if len(sub) == 0:
            pass
        else: self.df_fits = self.df_fits[sub]

    def load_processed_fitness(self):
        self.df_fits = pd.read_csv(self.in_path)

    def get_fit_data(self, ids, dropna=True):

        if dropna:
            data = self.df_fits.loc[ids].dropna()
        else: data = self.df_fits.loc[ids]

        return data


def calc_LOD(y, x):
    """
    Function for calculating Log10 LOD score
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # Keep only rows where both x and y are not NaN
    keep = ~(np.isnan(y) | np.isnan(x))
    y = y[keep]
    x = x[keep]

    n = y.shape[0]
    if n < 3:
        raise ValueError("Not enough non-missing data points to compute LOD.")

    # Null model: y ~ 1
    X0 = np.ones((n, 1))
    model0 = sm.OLS(y, X0).fit()
    rss0 = np.sum(model0.resid ** 2)

    # Single-marker model: y ~ x
    X1 = sm.add_constant(x)
    model1 = sm.OLS(y, X1).fit()
    rss1 = np.sum(model1.resid ** 2)

    lod = (n / 2.0) * np.log10(rss0 / rss1)
    return lod

import statsmodels.formula.api as smf
import statsmodels.api as sm
class yeast_data():
    def __init__(self, geno_data, fitness_data):
        self.geno_data = geno_data
        self.fitness_data = fitness_data

        self.ids = set(self.geno_data.ids).intersection(list(self.fitness_data.df_fits.index))
        self.ids = list(self.ids)
        self.ids.sort()
        self.ids_geno_pos = np.where(np.isin(self.geno_data.ids, self.ids))[0]

        self.geno = self.geno_data.geno[self.ids_geno_pos]
        self.fitness = self.fitness_data.df_fits.loc[self.ids]

        self.cat_to_con = dict(enumerate(self.fitness_data.conditions))
        self.con_to_cat = {v:k for k, v in self.cat_to_con.items()}
        self.con_to_num = {'C': 0, 'Q': .25, 'H': .5, 'F': 1, 'D': 2}
        self.num_to_con = {v:k for k, v in self.con_to_num.items()}


    def calc_snp_effect(self, locus):
        mean_fit_BY = self.fitness[self.geno[:, locus] == 0]['mean_log_fit'].mean(0)
        mean_fit_RM = self.fitness[self.geno[:, locus] == 1]['mean_log_fit'].mean(0)
        diff = mean_fit_BY - mean_fit_RM
        diff.index = self.fitness_data.conditions
        return diff

    def get_LOD(self, locus, permute=False):
        fit_BY = self.fitness[self.geno[:, locus] == 0]['mean_log_fit']
        fit_RM = self.fitness[self.geno[:, locus] == 1]['mean_log_fit']
        y_conditions = np.concatenate([fit_BY, fit_RM])
        if permute:
            y_conditions = np.random.permutation(y_conditions)
        x = np.concatenate([0. * np.ones(len(fit_BY)), np.ones(len(fit_RM))])
        lods = np.array([calc_LOD(y_conditions[:, i], x) for i in range(y_conditions.shape[1])])
        return lods


    def get_main_effects(self, loci):
        main_effects = {}
        for locus in loci:
            main_effects[locus] = self.calc_snp_effect(locus)
        return pd.DataFrame(main_effects).T

    def get_LODs(self, loci, permute=False):
        LODs = {}
        for locus in loci:
            LODs[locus] = self.get_LOD(locus, permute)
        return pd.DataFrame(LODs).T

    def stack_data(self, conditions_test):
        self.conditions_test = conditions_test
        conditions_test = pd.Series(conditions_test)

        col_idx = conditions_test.map(self.con_to_cat).tolist()
        self.fitness_stacked = np.array(self.fitness['mean_log_fit'].iloc[:, col_idx]).reshape(-1)
        self.std_stacked = np.array(self.fitness['std_log_fit'].iloc[:, col_idx]).reshape(-1)
        conditions_test_ = conditions_test.map(self.con_to_num).tolist()
        self.cons = np.array([conditions_test_ for _ in range(len(self.geno))]).flatten()

    def calc_GxE_pvalue(self, locus):

        X = self.geno[:, locus]
        X_ = np.repeat(X, len(self.conditions_test)).flatten()

        df_data = pd.DataFrame({
            "fitness": self.fitness_stacked,
            "condition": self.cons,
            "genotype": X_,
            "sigma": self.std_stacked
        })

        df_data = df_data.dropna()
        df_data = df_data[df_data.sigma < df_data.sigma.quantile(.2)]
        model_wls = smf.wls("fitness ~ genotype + condition + genotype * condition", data=df_data).fit()
        # pvalue = model_wls.pvalues['genotype:condition']

        anova_results = sm.stats.anova_lm(model_wls, typ=2)
        pvalue = anova_results.loc['genotype:condition']['PR(>F)']
        coef = model_wls.params['genotype:condition']

        return pvalue, coef


# yeast_chr_lengths = [
#     230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643,
#     439888, 745751, 666816, 1078177, 924431, 784333, 1091291, 948066
# ]
# yeast_chr_size = np.array(yeast_chr_lengths).sum()

# chr_labels_arabic = [str(i) for i in range(1, 17)]
# chr_offsets = np.cumsum([0] + yeast_chr_lengths[:-1])

def get_abs_pos(chrom, chrom_pos):
    return (chrom_pos - 1) + dict(enumerate(chr_offsets))[chrom - 1]

