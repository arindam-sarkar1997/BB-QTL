import pandas as pd
import matplotlib.pyplot as plt

df_chrom_interactions = pd.read_csv('results/chrom_interactions.csv') # This is used to make the circos plot
df_chrom_interactions['mean_VC'] = df_chrom_interactions.VC/df_chrom_interactions.n_snp_pairs
df_chrom_interactions.VC *= 100

v_min = df_chrom_interactions.VC.min()
v_max = df_chrom_interactions.VC.max()

from yeast_analysis.gene_circos import plot_chromosome_circos

plt.rcParams["font.family"] = "Nimbus Sans"
fig, ax, links = plot_chromosome_circos(
    df_chrom_interactions,
    strength_col="VC",
    chromosome_base=0,
    gradient=True,
    show_colorbar=True,
    output_path = "figures/chromosome_circos.pdf",
    colorbar_label='% total\npairwise\nvariance',    
    vmin=v_min * 22,
    vmax=v_max * 1.,
    ring_width=0.16,
    figsize=(6, 6),    
    label_fontsize=15,    
    font_family="Nimbus Sans",
    font_weight="normal",       # Or "bold"
    colorbar_tick_fontsize=15,
)
