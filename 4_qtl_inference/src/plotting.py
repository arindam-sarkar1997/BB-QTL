from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

def make_man(abs_pos, y, ylab, df_lab=None, chrom=None, chrom_range=None, constrained_layout=False):

    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'Nimbus Sans'  # options: 'serif', 'sans-serif', 'monospace', etc.
    })

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=constrained_layout)

    ax.bar(
        abs_pos,
        y,
        width=20000,
        color='black'
    )
    ax.margins(x=0.05, y=0.2)

    if df_lab is None:
        pass
    else:
        for _, row in df_lab.iterrows():
            name = row.iloc[2]
            x, y = row.iloc[:2]
            if y > 0:
                y_offset = 5
            else:
                y_offset = -7
            ax.annotate(
                name,
                xy=(x, y),  # arrow tip
                xytext=(0, y_offset),                  # offset in points above the bar
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10,
                color='red',
                # arrowprops=dict(arrowstyle="->", color="red")
            )
    # Draw vertical lines for chromosome boundaries
    for offset in chr_offsets[1:]:
        ax.axvline(offset, color='lightgray', lw=2, zorder=0)

    # Set x‐ticks at chromosome midpoints
    midpoints = [offset + length/2 for offset, length in zip(chr_offsets, yeast_chr_lengths)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(chr_labels_arabic, fontsize=12, fontweight='bold')

    if chrom is None:
        ax.set_xlim(0, yeast_chr_size)

    else:
        if chrom_range is None:
            ax.set_xlim(chr_offsets[chrom], chr_offsets[chrom + 1])
        else:
            ax.set_xlim(chr_offsets[chrom] + chrom_range[0], chr_offsets[chrom] + chrom_range[1])

    ax.set_xlabel('Chromosome', fontsize=18)
    ax.set_ylabel(f'{ylab}', fontsize=18)
    ax.margins(x=0)

    return fig, ax


def add_labels(ax, name, x, y, color):

    df_lab = pd.DataFrame({'name': name, 'x': x, 'y': y})
    for _, row in df_lab.iterrows():
        name = row.iloc[0]
        x, y = row.iloc[1:]
        if y > 0:
            y_offset = 5
        else:
            y_offset = -7
        ax.annotate(
            name,
            xy=(x, y),  # arrow tip
            xytext=(0, y_offset),                  # offset in points above the bar
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=14,
            color=color,
            # arrowprops=dict(arrowstyle="->", color="red")
        )
    return ax