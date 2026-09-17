import os
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yeast_analysis.gene_circos as gene_circos


fig_path = 'figures/'
result_path = 'results/'


# 1. Load both tables.
df_hub = pd.read_csv(
    os.path.join(result_path, "df_hub_interactions.tsv"),
    sep=None,
    engine="python",
    index_col=0,
)


df_random = pd.read_csv(
    os.path.join(result_path, "random_gene_interactions.tsv"),
    sep=None,
    engine="python",
)

# 2. Convert hub columns into gene1, gene2, mean_VC.
# gene1 = hub; gene2 = interacting partner.
df_hub_long = (
    df_hub.rename_axis("gene2")
    .reset_index()
    .melt(
        id_vars="gene2",
        var_name="gene1",
        value_name="mean_VC",
    )[["gene1", "gene2", "mean_VC"]]
)

# 3. Combine tables.
df_genexgene = pd.concat(
    [
        df_hub_long,
        df_random[["gene1", "gene2", "mean_VC"]],
    ],
    ignore_index=True,
)

df_genexgene["mean_VC"] = pd.to_numeric(
    df_genexgene["mean_VC"], errors="raise"
)

df_genexgene = df_genexgene.dropna(
    subset=["gene1", "gene2", "mean_VC"]
)
df_genexgene = df_genexgene.loc[
    np.isfinite(df_genexgene["mean_VC"])
    & df_genexgene["gene1"].ne(df_genexgene["gene2"])
].copy()

# Treat A–B and B–A as the same pair.
# Keep the first occurrence, preferring the hub-table estimate.
pair_keys = [
    tuple(sorted((a, b)))
    for a, b in zip(df_genexgene["gene1"], df_genexgene["gene2"])
]
df_genexgene["_pair"] = pair_keys
df_genexgene = (
    df_genexgene.drop_duplicates("_pair", keep="first")
    .drop(columns="_pair")
    .reset_index(drop=True)
)

if df_genexgene.empty:
    raise ValueError("No valid gene pairs remain.")

# 4. Convert normalized values to percentages exactly once.
df_genexgene["mean_VC"] *= 100

vc_max = df_genexgene["mean_VC"].max()
vc_min = df_genexgene["mean_VC"].min()
plot_vmin = vc_min * 1

if not 0 <= plot_vmin < vc_max:
    raise ValueError(
        f"Invalid color range: vmin={plot_vmin:g}, vmax={vc_max:g}. "
        "Set plot_vmin = 0 or choose a smaller lower threshold."
    )

# 5. Plot all combined gene pairs.
os.makedirs(fig_path, exist_ok=True)



fig, ax, links = gene_circos.plot_gene_pair_circos(
    "gene_info.csv",
    df_genexgene,
    strength_col="mean_VC",
    top_n=None,
    ribbon_color="black",
    gradient=True,
    vmax=vc_max,
    vmin=plot_vmin,
    output_path=os.path.join(fig_path, "gene_circos.png"),
    colorbar_label="Mean SNP-pair VC\n(% total pairwise variance)",
    ring_width=0.16,
    figsize=(6, 6),    
    label_fontsize=15,    
    font_family="Nimbus Sans",
    font_weight="normal",       # Or "bold"
    colorbar_tick_fontsize=15,
    show_colorbar=False,
)

plt.show()