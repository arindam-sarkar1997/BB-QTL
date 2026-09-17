import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Imported to control tick placement
import numpy as np

# --- Global Font Settings ---
plt.rcParams.update({
    'font.size': 18,               
    'axes.labelsize': 18,          
    'axes.titlesize': 18,          
    'xtick.labelsize': 18,         
    'ytick.labelsize': 18,         
    'legend.fontsize': 18,         
    'font.family': 'sans-serif',   
    'font.sans-serif': ['Nimbus Sans', 'Helvetica', 'Arial'] 
})

# --- Data & Bins Setup (Multiplied by 100) ---
# Scale the base data by 100
scaled_null = results_gxg_null * 100
scaled_hub = top_gene_interactions.mean_VC * 100

# Re-calculate bins based on the scaled data
bin_width = (.05 * 10**-8) * 100
bins = np.arange(min(scaled_hub), max(scaled_hub) + .1 * bin_width, bin_width)

# Create a 2-row, 1-column subplot structure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

# --- Top Panel: Null Distribution ---
ax1.hist(scaled_null, density=True, log=True, bins=bins, 
         color='gray', edgecolor='none', alpha=0.7, 
         label='Random genes')
ax1.set_ylabel('Density')

# Force y-axis to only show whole number powers of 10
ax1.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0]))
ax1.yaxis.set_minor_locator(ticker.NullLocator()) # Removes any tiny sub-ticks

# --- Bottom Panel: Epistatic Hubs ---
ax2.hist(scaled_hub, density=True, log=True, bins=bins, 
         color='orange', edgecolor='none', alpha=0.7, 
         label='Epistatic hub genes')
ax2.set_ylabel('Density')
ax2.set_xlabel('Mean % pairwise epistasis')

# Force y-axis to only show whole number powers of 10
ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0]))
ax2.yaxis.set_minor_locator(ticker.NullLocator()) # Removes any tiny sub-ticks

# --- Extract Visual Handles Automatically ---
handle1, _ = ax1.get_legend_handles_labels()
handle2, _ = ax2.get_legend_handles_labels()

# --- Single Unified Figure Legend ---
fig.legend(handles=[handle1[0], handle2[0]], 
           labels=['Random genes', 'Epistatic hub genes'], 
           loc='lower center',        
           bbox_to_anchor=(0.55, 0.9), 
           ncol=2,                      
           frameon=False,
           columnspacing=.4,          
           handletextpad=0.1)

# Clean up layout 
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(fig_path + 'Gene_interaction_strength_distribution.pdf')
plt.show()