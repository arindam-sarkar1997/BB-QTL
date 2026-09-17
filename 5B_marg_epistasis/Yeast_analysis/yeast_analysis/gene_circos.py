"""Hub-centered chromosome plots using pandas and matplotlib."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgb
from matplotlib.patches import Wedge, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


# Shared defaults for chromosome-, gene-pair-, and hub-centered Circos plots.
CHROMOSOME_RING_OUTER_RADIUS = 1.11
CHROMOSOME_RING_WIDTH = .18


def _ring_geometry(ring_width):
    """Validate ring width and return its inner and center radii."""
    if (isinstance(ring_width, (bool, np.bool_))
            or not isinstance(ring_width, (int, float, np.number))
            or not np.isfinite(ring_width)
            or not 0 < ring_width < CHROMOSOME_RING_OUTER_RADIUS):
        raise ValueError(
            'ring_width must be finite and between 0 and the outer ring radius')
    inner_radius = CHROMOSOME_RING_OUTER_RADIUS - float(ring_width)
    center_radius = CHROMOSOME_RING_OUTER_RADIUS - float(ring_width) / 2
    return inner_radius, center_radius


def _style_pair_circos(fig, ax, font_family, font_weight):
    """Shared layout and typography for chromosome and gene-pair plots."""
    from matplotlib.text import Text
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=.08, right=.82, bottom=.08, top=.88)
    ax.set(xlim=(-1.28, 1.28), ylim=(-1.28, 1.28), aspect='equal')
    for text in fig.findobj(match=Text):
        text.set_fontfamily(font_family if font_family is not None else plt.rcParams['font.family'])
        text.set_fontweight(font_weight)


def plot_chromosome_circos(
    interactions, *, strength_col="VC",
    chromosome_base=0, top_n=None, min_strength=None, hard_threshold=None,
    chromosome_lengths=None, vmin=0.0, vmax=None, cmap="Blues",
    gap_degrees=3.0, figsize=(10, 10), output_path=None, dpi=300,
    ring_width=CHROMOSOME_RING_WIDTH,
    label_fontsize=14, ribbon_color=None, gradient=True,
    show_colorbar=True, title=None, colorbar_label="Interaction\nstrength",
    colorbar_tick_fontsize=None, font_family=None, font_weight='normal',
):
    """Plot filled chromosome-to-chromosome ribbons; return fig, ax, links.

    Input is a pandas DataFrame with chromosome_a, chromosome_b and
    strength_col. chromosome_base=0 interprets 0..15 as displayed 1..16;
    use 1 for one-based input. Default lengths are the 16 yeast chromosome
    lengths used in utils.py. Custom lengths map displayed chromosome numbers
    (one-based integers) to lengths. All supplied chromosomes are displayed.

    Each unordered pair is drawn once; duplicate/reversed pairs raise rather
    than being silently summed. Self-pairs are omitted. top_n selects pairs
    globally; hard_threshold overrides top_n/min_strength, as in gene circos.
    Default top_n=None retains all positive pairs. vmax defaults to the maximum
    non-self score before filtering. VC is total interaction variance;
    mean_VC is variance per SNP pair. Select explicitly for your interpretation.

    Full chromosome arcs anchor filled ribbons, including overlaps between
    pairs: width is chromosome extent, NOT interaction strength. Stronger
    ribbons are drawn last. Color encodes strength using a white-to-blue
    gradient; custom ribbon_color sets the endpoint instead. gradient=False
    uses the endpoint color throughout and suppresses the strength colorbar.
    Ribbons stop at the inner ring edge; chromosome sectors remain white.
    Zero-strength pairs are omitted on a white background. The gradient starts
    at pure white at vmin (default zero), smoothly blending into the palette.
    colorbar_label sets the title above the colorbar; use an empty string to
    hide the label. Newlines are supported.
    colorbar_tick_fontsize sets colorbar tick and scientific-offset text size
    in points; None preserves Matplotlib defaults. Applied before export.
    ring_width controls chromosome-ring thickness in radial plot units.
    """
    if chromosome_base not in (0, 1):
        raise ValueError('chromosome_base must be 0 or 1')
    if not isinstance(gradient, (bool, np.bool_)):
        raise TypeError('gradient must be boolean')
    if hard_threshold is None and top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
            raise ValueError('top_n must be a positive integer or None')
    for name, value in [('hard_threshold', hard_threshold), ('min_strength', min_strength)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(gap_degrees) or gap_degrees < 0:
        raise ValueError('gap_degrees must be finite and nonnegative')
    if not np.isfinite(label_fontsize) or label_fontsize <= 0:
        raise ValueError('label_fontsize must be finite and positive')
    ring_inner_radius, ring_center_radius = _ring_geometry(ring_width)
    if colorbar_tick_fontsize is not None and (
            not np.isfinite(colorbar_tick_fontsize) or colorbar_tick_fontsize <= 0):
        raise ValueError('colorbar_tick_fontsize must be finite and positive')
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError('interactions must be a pandas DataFrame')
    table = interactions.copy()
    required = ['chromosome_a', 'chromosome_b', strength_col]
    if set(required) - set(table.columns):
        raise ValueError(f'Required columns: {required}')
    links = table[required].copy()
    for col in required:
        links[col] = pd.to_numeric(links[col], errors='raise')
    if not np.isfinite(links.to_numpy()).all():
        raise ValueError('Chromosomes and strengths must be finite')
    for col in required[:2]:
        if (links[col] != np.floor(links[col])).any():
            raise ValueError('Chromosome IDs must be integers')
        links[col] = links[col].astype(int) + 1 - chromosome_base
    links = links.rename(columns={strength_col: 'strength'})
    if (links.strength < 0).any():
        raise ValueError('Strengths must be nonnegative')
    links[['chromosome_a', 'chromosome_b']] = np.sort(
        links[['chromosome_a', 'chromosome_b']].to_numpy(), axis=1)
    links = links[links.chromosome_a != links.chromosome_b]
    if links.duplicated(['chromosome_a', 'chromosome_b']).any():
        raise ValueError('Duplicate unordered chromosome pairs')
    default_lengths = [230218, 813184, 316620, 1531933, 576874, 270161,
                       1090940, 562643, 439888, 745751, 666816, 1078177,
                       924431, 784333, 1091291, 948066]
    lengths = (dict(enumerate(default_lengths, 1)) if chromosome_lengths is None
               else dict(chromosome_lengths))
    if not lengths or any(not np.isfinite(v) or v <= 0 for v in lengths.values()):
        raise ValueError('Chromosome lengths must be finite and positive')
    if (set(links.chromosome_a) | set(links.chromosome_b)) - set(lengths):
        raise ValueError('chromosome_lengths must cover all input chromosomes')
    chromosomes = sorted(lengths)
    available = 360 - gap_degrees * len(chromosomes)
    if available <= 0:
        raise ValueError('Chromosome gaps leave no space for arcs')
    upper = float(vmax) if vmax is not None else max(
        float(links.strength.max()) if len(links) else 0, np.finfo(float).tiny)
    if not np.isfinite(vmin) or vmin < 0 or not np.isfinite(upper) or upper <= vmin:
        raise ValueError('Color limits must be finite with 0 <= vmin < vmax')
    norm = Normalize(vmin=vmin, vmax=upper, clip=True)
    if ribbon_color is None:
        base = plt.get_cmap(cmap)
        colors = base(np.linspace(0, 1, 256))
        # Fade out the palette's near-white tint smoothly at its lower end.
        fade = np.linspace(1, 0, 256)[:, None]
        colors[:, :3] += (1 - colors[0, :3]) * fade
        colors[:, :3] = np.clip(colors[:, :3], 0, 1)
        colors[:, 3] = 1
        color_map = LinearSegmentedColormap.from_list('chromosome_strength', colors)
    else:
        color_map = LinearSegmentedColormap.from_list('chromosome_strength', ['white', to_rgb(ribbon_color)])
    links = links[links.strength > 0].sort_values('strength', ascending=False, kind='stable')
    if hard_threshold is not None:
        links = links[links.strength > hard_threshold]
    else:
        if min_strength is not None:
            links = links[links.strength >= min_strength]
        if top_n is not None:
            links = links.head(top_n)
    links = links.reset_index(drop=True)
    spans = {c: available * lengths[c] / sum(lengths.values()) for c in chromosomes}
    starts, cursor = {}, -spans[chromosomes[0]]/2
    for c in chromosomes:
        starts[c] = cursor
        cursor += spans[c] + gap_degrees
    def arc(c):
        angles = np.deg2rad(np.linspace(starts[c], starts[c]+spans[c], 40))
        return ring_inner_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    fig.subplots_adjust(left=.05, right=.82 if show_colorbar and gradient else .95,
                        bottom=.05, top=.93)
    for row in links.iloc[::-1].itertuples():
        a, b = arc(row.chromosome_a), arc(row.chromosome_b)
        vertices = list(a) + [a[-1]*.25, b[0]*.25, b[0]] + list(b[1:])
        codes = [MplPath.MOVETO] + [MplPath.LINETO]*(len(a)-1)
        codes += [MplPath.CURVE4]*3 + [MplPath.LINETO]*(len(b)-1)
        vertices += [b[-1]*.25, a[0]*.25, a[0], a[0]]
        codes += [MplPath.CURVE4]*3 + [MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(vertices, codes), edgecolor='none',
                               facecolor=color_map(norm(row.strength) if gradient else 1.0)))
    for c in chromosomes:
        ax.add_patch(Wedge((0, 0), CHROMOSOME_RING_OUTER_RADIUS,
                           starts[c], starts[c]+spans[c], width=ring_width,
                           facecolor='white', edgecolor='black', linewidth=1.6, zorder=3))
        middle = starts[c] + spans[c]/2
        rotation = (middle - 90 + 90) % 180 - 90
        theta = np.deg2rad(middle)
        ax.text(ring_center_radius*np.cos(theta), ring_center_radius*np.sin(theta),
                str(c), ha='center', va='center',
                fontsize=label_fontsize, rotation=0, zorder=4)
    ax.set(xlim=(-1.18, 1.18), ylim=(-1.18, 1.18), aspect='equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    if gradient and show_colorbar:
        cax = fig.add_axes([.8, .43, .025, .16])
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=color_map), cax=cax)
        cb.ax.set_title(colorbar_label, fontsize=label_fontsize, loc='left', pad=10)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()
        if colorbar_tick_fontsize is not None:
            cb.ax.tick_params(axis='y', labelsize=colorbar_tick_fontsize)
            cb.ax.yaxis.get_offset_text().set_fontsize(colorbar_tick_fontsize)
        cb.solids.set_rasterized(False)
    if not len(links):
        ax.text(0, 0, 'No interactions pass the display filters', ha='center')
    _style_pair_circos(fig, ax, font_family, font_weight)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
    return fig, ax, links


def _assign_ld_peaks(links, ld_sampler=None, ld_cache_path=None, threshold=0.8,
                     hub_list=None, gene_meta=None, gene_focal=None):
    """Connected components of same-chromosome, strict r2 > threshold edges.

    Components are transitive: not every pair must exceed the threshold.
    Group only displayed partners; retain every original interaction line.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("ld_threshold must be between 0 and 1")
    if ld_sampler is not None and ld_cache_path is not None:
        raise ValueError("Supply ld_sampler or ld_cache_path, not both")
    result = links.copy()
    result['ld_peak'] = None
    result['is_peak'] = False
    result['ld_available'] = False
    result['label_gene'] = None
    if isinstance(hub_list, str):
        raise ValueError('hub_list must be a sequence of gene names, not a string')
    preferred = list(dict.fromkeys(hub_list if hub_list is not None else []))
    preferred = [g for g in preferred if g != gene_focal]
    if preferred and ld_sampler is None and ld_cache_path is None:
        raise ValueError('hub_list requires ld_sampler or ld_cache_path')
    if preferred and gene_meta is None:
        raise ValueError('hub_list requires gene metadata')
    if preferred:
        unavailable = [g for g in preferred if g not in gene_meta.index]
        if unavailable:
            warnings.warn(f'Preferred hubs without unique coordinates: {unavailable}', stacklevel=2)
        preferred = [g for g in preferred if g in gene_meta.index]
    if ld_sampler is None and ld_cache_path is None:
        return result
    if ld_cache_path is not None:
        import torch
        payload = torch.load(ld_cache_path, map_location='cpu', weights_only=True)
        names = payload['genes']
        matrix = payload['gene_ld_matrix']
        if len(set(names)) != len(names) or tuple(matrix.shape) != (len(names), len(names)):
            raise ValueError("Invalid LD cache gene names or matrix shape")
        lookup = {g: i for i, g in enumerate(names)}
        def get_ld(a, b):
            return float(matrix[lookup[a], lookup[b]])
    else:
        get_ld = getattr(ld_sampler, 'gene_ld', None)
        if not callable(get_ld):
            raise TypeError(
                "This sampler does not expose gene_ld(gene_a, gene_b). "
                "Use a compatible sampler or ld_cache_path from save_ld(). "
                f"Sampler class: {type(ld_sampler).__module__}.{type(ld_sampler).__name__}"
            )
        # Use the public LD method; do not require a private indexing layout.
        # Query diagonal entries to determine which plotted genes are supported.
        lookup = {}
        for gene in dict.fromkeys(list(result.index) + preferred):
            try:
                value = float(get_ld(gene, gene))
            except KeyError:
                continue
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid gene r2 for {gene}: {value}")
            lookup[gene] = True
    missing_hubs = [g for g in preferred if g not in lookup]
    if missing_hubs:
        warnings.warn(f'Preferred hubs without LD: {missing_hubs}', stacklevel=2)
    genes = list(result.index)
    parent = list(range(len(genes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    missing = [g for g in genes if g not in lookup]
    if missing:
        warnings.warn(f"{len(missing)} plotted partners lack LD; treating them as singleton label groups", stacklevel=2)
    for i, a in enumerate(genes):
        if a not in lookup:
            continue
        result.loc[a, 'ld_available'] = True
        for j in range(i):
            b = genes[j]
            if b not in lookup or result.loc[a, 'chrom'] != result.loc[b, 'chrom']:
                continue
            value = get_ld(a, b)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid gene r2 for {a}, {b}: {value}")
            if value > threshold:
                parent[find(i)] = find(j)
    groups = {}
    for i, gene in enumerate(genes):
        groups.setdefault(find(i), []).append(gene)
    for members in groups.values():
        peak = min(members, key=lambda g: (-result.loc[g, 'strength'], str(g)))
        result.loc[members, 'ld_peak'] = peak
        result.loc[peak, 'is_peak'] = True
        label = peak
        for hub in preferred:
            if hub not in lookup or peak not in lookup:
                continue
            if gene_meta.loc[hub, 'chrom'] != result.loc[peak, 'chrom']:
                continue
            value = float(get_ld(peak, hub))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid gene r2 for {peak}, {hub}: {value}')
            if value > threshold:
                label = hub
                break
        result.loc[members, 'label_gene'] = label
    return result


def plot_gene_pair_circos(
    gene_info, interactions, *, strength_col="mean_VC",
    top_n=None, min_strength=None, hard_threshold=None, chromosome_lengths=None,
    vmin=0.0, vmax=None, cmap="Blues", gap_degrees=3.0, figsize=(10, 10),
    output_path=None, dpi=300, ribbon_color="#FFC04C", gradient=True,
    ring_width=CHROMOSOME_RING_WIDTH,
    label_fontsize=14, show_colorbar=True, colorbar_label="Interaction\nstrength",
    colorbar_tick_fontsize=None, font_family=None, font_weight='normal', title=None,
):
    """Draw gene1/gene2 interaction pairs without gene labels.

    interactions must be a loaded pandas DataFrame. gene_info accepts a
    DataFrame or comma/tab-delimited path. Metadata uses
    gene_standard_name, chrom, gene_chr_start and gene_chr_end. Chromosomes
    accept numeric, chr-prefixed or Roman identifiers. Same-chromosome ORFs
    sharing a name are merged to min(start)..max(end), as in plot_gene_circos.
    Unmapped and multi-chromosome names are omitted with a warning.

    Each unordered pair is drawn once; duplicate/reversed pairs retain their
    maximum score. Self-pairs and zero scores are excluded. top_n is GLOBAL,
    defaults to all pairs, and is overridden by hard_threshold. vmax defaults
    to the largest mapped pair before display filtering. Curves match the
    current gene plot: midpoint anchors, .35 control-point radius multiplier,
    fixed 1.2-point width, and a pure-white gradient start at vmin.
    Only chromosome numbers are labeled; no hub or LD-label selection occurs.
    ring_width controls chromosome-ring thickness in radial plot units.
    Returns fig, ax, links (gene1, gene2, strength), strongest first.
    """
    def read(value):
        return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value, sep=None, engine='python')
    def chrom_id(value):
        s = str(value).strip()
        if s.lower().startswith('chr'):
            s = s[3:]
        romans = 'I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI'.split()
        if s.upper() in romans:
            return str(romans.index(s.upper())+1)
        try:
            return str(int(float(s))) if float(s).is_integer() else s
        except ValueError:
            return s
    if hard_threshold is None and top_n is not None:
        if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
            raise ValueError('top_n must be a positive integer or None')
    for value in (min_strength, hard_threshold):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError('Thresholds must be finite and nonnegative')
    if not isinstance(gradient, (bool, np.bool_)):
        raise TypeError('gradient must be boolean')
    if not np.isfinite(gap_degrees) or gap_degrees < 0:
        raise ValueError('gap_degrees must be finite and nonnegative')
    ring_inner_radius, ring_center_radius = _ring_geometry(ring_width)
    for size in (label_fontsize, colorbar_tick_fontsize):
        if size is not None and (not np.isfinite(size) or size <= 0):
            raise ValueError('Font sizes must be finite and positive')
    meta = read(gene_info)[['gene_standard_name', 'chrom', 'gene_chr_start', 'gene_chr_end']].dropna()
    meta['chrom'] = meta.chrom.map(chrom_id)
    meta['gene_standard_name'] = meta.gene_standard_name.astype(str)
    for col in ['gene_chr_start', 'gene_chr_end']:
        meta[col] = pd.to_numeric(meta[col], errors='raise')
    coords = meta[['gene_chr_start', 'gene_chr_end']].to_numpy()
    if not np.isfinite(coords).all() or (coords < 0).any() or (coords[:, 1] < coords[:, 0]).any():
        raise ValueError('Invalid gene coordinates')
    inferred = meta.groupby('chrom').gene_chr_end.max().to_dict()
    counts = meta.groupby('gene_standard_name').chrom.nunique()
    meta = meta[meta.gene_standard_name.isin(counts[counts == 1].index)]
    meta = meta.groupby('gene_standard_name').agg(chrom=('chrom', 'first'),
        gene_chr_start=('gene_chr_start', 'min'), gene_chr_end=('gene_chr_end', 'max'))
    if not isinstance(interactions, pd.DataFrame):
        raise TypeError('interactions must be a pandas DataFrame')
    links = interactions[['gene1', 'gene2', strength_col]].copy().rename(columns={strength_col: 'strength'})
    if links[['gene1', 'gene2']].isna().any().any():
        raise ValueError('Missing gene names in interaction table')
    links[['gene1', 'gene2']] = links[['gene1', 'gene2']].astype(str)
    links['strength'] = pd.to_numeric(links.strength, errors='raise')
    if not np.isfinite(links.strength).all() or (links.strength < 0).any():
        raise ValueError('Strengths must be finite and nonnegative')
    valid = links.gene1.isin(meta.index) & links.gene2.isin(meta.index)
    if (~valid).any():
        warnings.warn(f'Omitting {int((~valid).sum())} rows with unmapped or multi-chromosome genes', stacklevel=2)
    links = links[valid & (links.gene1 != links.gene2)].copy()
    links[['gene1', 'gene2']] = np.sort(links[['gene1', 'gene2']].to_numpy(), axis=1)
    links = links.groupby(['gene1', 'gene2'], as_index=False, sort=False).strength.max()
    upper = float(vmax) if vmax is not None else max(float(links.strength.max()) if len(links) else 0, np.finfo(float).tiny)
    if not np.isfinite(vmin) or not np.isfinite(upper) or upper <= vmin:
        raise ValueError('Color limits must be finite with vmax > vmin')
    links = links[links.strength > 0].sort_values('strength', ascending=False, kind='stable')
    if hard_threshold is not None:
        links = links[links.strength > hard_threshold]
    else:
        if min_strength is not None:
            links = links[links.strength >= min_strength]
        if top_n is not None:
            links = links.head(top_n)
    links = links.reset_index(drop=True)
    lengths = inferred if chromosome_lengths is None else {chrom_id(k): float(v) for k,v in chromosome_lengths.items()}
    if not lengths or set(inferred)-set(lengths) or any(not np.isfinite(v) or v <= 0 for v in lengths.values()):
        raise ValueError('Positive chromosome lengths must cover the metadata')
    if any(lengths[c] < end for c,end in inferred.items()):
        raise ValueError('Chromosome length shorter than annotated gene end')
    chromosomes = sorted(lengths, key=lambda c: (0,int(c)) if c.isdigit() else (1,c))
    available = 360-gap_degrees*len(chromosomes)
    if available <= 0:
        raise ValueError('Chromosome gaps leave no space for arcs')
    spans = {c: available*lengths[c]/sum(lengths.values()) for c in chromosomes}
    starts, cursor = {}, -spans[chromosomes[0]]/2
    for c in chromosomes:
        starts[c] = cursor
        cursor += spans[c]+gap_degrees
    anchors = {}
    for gene,row in meta.iterrows():
        theta = np.deg2rad(starts[row.chrom]+spans[row.chrom]*(row.gene_chr_start+row.gene_chr_end)/2/lengths[row.chrom])
        anchors[gene] = ring_inner_radius*np.array([np.cos(theta), np.sin(theta)])
    norm = Normalize(vmin=vmin, vmax=upper, clip=True)
    if ribbon_color is None:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
        colors[:, :3] += (1 - colors[0, :3]) * np.linspace(1, 0, 256)[:, None]
        colors[:, :3] = np.clip(colors[:, :3], 0, 1)
        colors[:, 3] = 1
        color_map = LinearSegmentedColormap.from_list('pair_strength', colors)
    else:
        strong = np.array(to_rgb(ribbon_color))
        color_map = LinearSegmentedColormap.from_list('pair_strength', ['white', strong])
    fig,ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=.08, right=.82, bottom=.08, top=.88)
    for row in links.iloc[::-1].itertuples():
        a,b = anchors[row.gene1], anchors[row.gene2]
        path = MplPath([a, a*.35, b*.35, b], [MplPath.MOVETO]+[MplPath.CURVE4]*3)
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=color_map(norm(row.strength) if gradient else 1.0),
                               linewidth=1.2, alpha=1.0 if ribbon_color is not None else .8))
    for c in chromosomes:
        ax.add_patch(Wedge(
            (0, 0), CHROMOSOME_RING_OUTER_RADIUS,
            starts[c], starts[c] + spans[c], width=ring_width, facecolor='white',
            edgecolor='black', linewidth=1.6))
        theta = np.deg2rad(starts[c]+spans[c]/2)
        ax.text(ring_center_radius*np.cos(theta),
                ring_center_radius*np.sin(theta), c,
                ha='center', va='center', fontsize=label_fontsize)
    ax.set(xlim=(-1.28,1.28),ylim=(-1.28,1.28),aspect='equal')
    ax.axis('off')
    if show_colorbar and gradient:
        cb = fig.colorbar(ScalarMappable(norm=norm,cmap=color_map),cax=fig.add_axes([.8,.43,.025,.16]))
        cb.ax.set_title(colorbar_label,fontsize=label_fontsize,loc='left',pad=10)
        cb.formatter.set_powerlimits((-2,2)); cb.update_ticks()
        cb.solids.set_rasterized(False)
        if colorbar_tick_fontsize is not None:
            cb.ax.tick_params(labelsize=colorbar_tick_fontsize)
            cb.ax.yaxis.get_offset_text().set_fontsize(colorbar_tick_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    _style_pair_circos(fig, ax, font_family, font_weight)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=dpi,bbox_inches='tight')
    return fig,ax,links


def plot_gene_circos(
    gene_info, interactions, gene_focal, *, top_n=100, min_strength=None,
    hard_threshold=None,
    chromosome_lengths=None, vmin=0.0, vmax=None, cmap="Blues",
    gap_degrees=3.0, figsize=(10, 10), output_path=None, dpi=300,
    ring_width=CHROMOSOME_RING_WIDTH,
    ld_sampler=None, ld_cache_path=None, ld_threshold=0.8, label_fontsize=14,
    hub_list=None, ribbon_color="#FFC04C", hub_colors=(None, "#FFC04C"),
    gradient=True, nonhub_label_color=None, nonhub_label_fontstyle='normal',
    font_family=None, font_weight='normal', colorbar_tick_fontsize=None,
    show_colorbar=True, colorbar_label=None, title=None,
    gene_focal_fontweight=None, gene_focal_color=None, hub_color=None,
):
    """Plot one or two hubs' nonnegative interaction strengths around chromosomes.

    Parameters
    ----------
    gene_info : path or pandas.DataFrame
        Columns: chrom, gene_chr_start, gene_chr_end, gene_standard_name.
        ORFs sharing a standard name on the same chromosome are combined into
        one spanning interval (minimum start, maximum end), including any gaps.
        Names spanning multiple chromosomes are omitted with a warning;
        a focal gene spanning multiple chromosomes raises ValueError.
    interactions : path or pandas.DataFrame
        Gene names as row index and focal hubs as columns. File input accepts
        comma- or tab-separated tables, regardless of filename extension.
    gene_focal : str or sequence of two str
        One hub, or two hub columns and standard gene names. Selection and LD
        grouping are performed independently for each hub. Two-hub output
        includes a source_hub column; partner names may occur twice.
    hub_colors : pair of matplotlib colors
        Two-hub ribbon colors, in hub order. None uses cmap (default Blues);
        the second defaults to gold #FFC04C. Ignored for single-hub calls.
    top_n : positive int or None
        Number of top-ranked links after mapping and min_strength filtering
        (default 100). Ignored when hard_threshold is supplied. None keeps all
        links passing min_strength. Self interactions are always excluded.
    min_strength : float or None
        Inclusive score cutoff for the top_n selection. This is a display
        filter, not a statistical significance threshold.
    hard_threshold : float or None
        When supplied, plot every uniquely mapped partner with strength strictly
        above this value, ignoring top_n and min_strength. None uses the original
        top_n/min_strength selection.
    chromosome_lengths : mapping or None
        Chromosome identifier to length in the same units as gene coordinates.
        If omitted, lengths are inferred from maximum annotated gene ends.
    ring_width : float
        Chromosome-ring thickness in radial plot units. Hub marker bars span
        this full thickness.
    vmin, vmax : float
        Color limits. Default vmax is the selected hub's largest mapped score,
        before display filtering. Set the same vmax for comparable hub plots.
    ld_sampler : LDMatchedGeneSetSampler or None
        Existing sampler providing gene_ld(a, b). Supplying
        this or ld_cache_path enables peak partner labels. Without either,
        only the hub is labeled, preserving the original behavior.
    ld_cache_path : path or None
        Cache produced by LDMatchedGeneSetSampler.save_ld, from the same
        genotype dataset. Loads on CPU without requiring the genotype tensor.
        Do not supply together with ld_sampler.
    ld_threshold : float
        Strict gene-level r2 cutoff (default .8). Connected components are
        formed among displayed partners on each chromosome. Label the highest
        scoring member of each component (alphabetical tie break). Transitive
        chains are allowed. Missing LD yields warned singleton groups.
    hub_list : sequence of str or None
        Prefer the first listed hub on the peak's chromosome with direct gene
        r2 > ld_threshold to the true peak. A preferred hub need not pass the
        interaction display filter. Requires LD and unique metadata coordinates.
        The focal hub is excluded. Labels anchor at the preferred gene's actual
        location; repeated labels are drawn once. Interaction lines and scores
        are unchanged. ld_peak retains the true peak; label_gene records the label.
    ribbon_color : matplotlib color or None
        Strongest color of a pale-to-color strength gradient (default '#FFC04C',
        RGB 255, 192, 76). Strength controls both color and line width.
        None uses cmap instead. Used for single-hub calls.
    gradient : bool
        True (default) colors ribbons by strength and shows colorbars.
        False uses each hub's strongest color for every ribbon and shows a
        hub legend instead. Ribbon width still encodes strength using vmin/vmax.
    label_fontsize : float
        Font size for partner and hub labels when LD labeling is enabled.
        Also controls chromosome labels; default 14 matches the other circos plots.
    font_family, font_weight : str or None, str
        Shared circos typography. None uses the current font family;
        font_weight defaults to normal for all labels, including focal hubs.
    colorbar_tick_fontsize : float or None
        Tick and scientific-offset size in points. None uses current defaults.
    show_colorbar : bool
        Show strength colorbars in gradient mode or hub legend in solid mode.
    colorbar_label : str or None
        Override colorbar titles. None retains automatic hub-specific titles.
    title : str or None
        Optional plot title.
    nonhub_label_color : matplotlib color or None
        Color for displayed partner names absent from hub_list and gene_focal.
        None preserves the default text color. Hub labels retain their existing
        styling. Partner labels still require ld_sampler or ld_cache_path.
    gene_focal_fontweight : str, numeric weight, or None
        Override font_weight for focal gene names only. None inherits it.
        Applies to both focal hubs in a two-hub plot.
    nonhub_label_fontstyle : {'normal', 'italic', 'oblique'}
        Font style for displayed gene names absent from hub_list and gene_focal.
    gene_focal_color : matplotlib color or None
        Override focal gene name colors only. None preserves existing colors.
    hub_color : matplotlib color or None
        Set both focal gene labels and their radial bars to this color.
        Overrides gene_focal_color when supplied. Applies to both focal hubs
        in a two-hub plot. None preserves existing label colors and black bars.
    output_path : path or None
        Optional image/PDF output; extension determines format.

    Returns
    -------
    fig, ax, links : matplotlib Figure, Axes, pandas.DataFrame
        links contains the plotted genes, coordinates, and strengths, sorted
        strongest first, plus ld_peak, is_peak, and ld_available columns.
        The figure remains open for further customization.
    """
    def read_table(value, indexed=False):
        if isinstance(value, pd.DataFrame):
            return value.copy()
        return pd.read_csv(value, sep=None, engine="python",
                           index_col=0 if indexed else None)

    hubs = [gene_focal] if isinstance(gene_focal, str) else list(gene_focal)
    if colorbar_tick_fontsize is not None and (
            not np.isfinite(colorbar_tick_fontsize) or colorbar_tick_fontsize <= 0):
        raise ValueError('colorbar_tick_fontsize must be finite and positive')
    if nonhub_label_color is not None:
        # Validate before metadata processing or LD calculations.
        from matplotlib.colors import to_rgba
        to_rgba(nonhub_label_color)
    if nonhub_label_fontstyle not in ('normal', 'italic', 'oblique'):
        raise ValueError("nonhub_label_fontstyle must be 'normal', 'italic', or 'oblique'")
    if gene_focal_color is not None:
        from matplotlib.colors import to_rgba
        to_rgba(gene_focal_color)
    if hub_color is not None:
        from matplotlib.colors import to_rgba
        to_rgba(hub_color)
    if gene_focal_fontweight is not None:
        from matplotlib.font_manager import FontProperties
        FontProperties(weight=gene_focal_fontweight)
    if not isinstance(gradient, (bool, np.bool_)):
        raise TypeError("gradient must be a boolean")
    if len(hubs) not in (1, 2) or any(not isinstance(h, str) for h in hubs) or len(set(hubs)) != len(hubs):
        raise ValueError("gene_focal must be one gene name or two distinct gene names")
    if len(hubs) == 2 and len(hub_colors) != 2:
        raise ValueError("hub_colors must contain two colors")

    def chrom_id(value):
        s = str(value).strip()
        if s.lower().startswith("chr"):
            s = s[3:]
        try:
            n = float(s)
            if n.is_integer():
                return str(int(n))
        except ValueError:
            pass
        romans = ['I','II','III','IV','V','VI','VII','VIII','IX','X',
                  'XI','XII','XIII','XIV','XV','XVI']
        return str(romans.index(s.upper()) + 1) if s.upper() in romans else s

    if hard_threshold is None and top_n is not None and (isinstance(top_n, bool) or
                             not isinstance(top_n, (int, np.integer)) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None")
    if not np.isfinite(gap_degrees) or gap_degrees < 0:
        raise ValueError("gap_degrees must be finite and nonnegative")
    ring_inner_radius, ring_center_radius = _ring_geometry(ring_width)
    if hard_threshold is None and min_strength is not None and (not np.isfinite(min_strength) or min_strength < 0):
        raise ValueError("min_strength must be finite and nonnegative")

    if hard_threshold is not None and (not np.isfinite(hard_threshold) or hard_threshold < 0):
        raise ValueError("hard_threshold must be finite and nonnegative")

    meta = read_table(gene_info)
    required = ['chrom', 'gene_chr_start', 'gene_chr_end', 'gene_standard_name']
    missing = set(required) - set(meta.columns)
    if missing:
        raise ValueError(f"Missing gene metadata columns: {sorted(missing)}")
    meta = meta[required].dropna().copy()
    meta['chrom'] = meta['chrom'].map(chrom_id)
    meta['gene_standard_name'] = meta['gene_standard_name'].astype(str)
    for col in ['gene_chr_start', 'gene_chr_end']:
        meta[col] = pd.to_numeric(meta[col], errors='raise')
    coords = meta[['gene_chr_start', 'gene_chr_end']].to_numpy()
    if (not np.isfinite(coords).all() or (coords < 0).any()
            or (meta.gene_chr_end < meta.gene_chr_start).any()):
        raise ValueError("Gene coordinates must be finite, nonnegative, and start <= end")
    inferred = meta.groupby('chrom').gene_chr_end.max().to_dict()
    meta = meta.drop_duplicates()
    chromosome_counts = meta.groupby('gene_standard_name').chrom.nunique()
    ambiguous = set(chromosome_counts[chromosome_counts > 1].index)
    if any(h in ambiguous for h in hubs):
        raise ValueError(f"Multiple chromosomes for focal gene {gene_focal}")
    if ambiguous:
        warnings.warn(f"Omitting {len(ambiguous)} gene names spanning multiple chromosomes", stacklevel=2)
        meta = meta[~meta.gene_standard_name.isin(ambiguous)]
    meta = meta.groupby('gene_standard_name', sort=False).agg(
        chrom=('chrom', 'first'),
        gene_chr_start=('gene_chr_start', 'min'),
        gene_chr_end=('gene_chr_end', 'max'),
    )
    if any(h not in meta.index for h in hubs):
        raise ValueError(f"Focal gene {gene_focal} absent from gene metadata")

    matrix = read_table(interactions, indexed=True)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("Interaction table must have unique row and column names")
    if any(h not in matrix.columns for h in hubs):
        raise ValueError(f"Hub {gene_focal} absent from interaction columns")
    def select_links(hub):
        scores = pd.to_numeric(matrix[hub], errors='raise').drop(index=hub, errors='ignore')
        if not np.isfinite(scores.to_numpy()).all() or (scores < 0).any():
            raise ValueError("Interaction scores must be finite and nonnegative")
        unmapped = scores.index.difference(meta.index)
        if len(unmapped):
            warnings.warn(f"Omitting {len(unmapped)} partners without unique coordinates for {hub}", stacklevel=2)
        selected = meta.join(scores.rename('strength'), how='inner')
        maximum = selected.strength.max() if len(selected) else 0.0
        selected = selected[selected.strength > 0].sort_values('strength', ascending=False, kind='stable')
        if hard_threshold is not None:
            selected = selected[selected.strength > hard_threshold]
        else:
            if min_strength is not None:
                selected = selected[selected.strength >= min_strength]
            if top_n is not None:
                selected = selected.head(top_n)
        selected = _assign_ld_peaks(selected, ld_sampler, ld_cache_path, ld_threshold,
                                   hub_list=hub_list, gene_meta=meta, gene_focal=hub)
        selected['source_hub'] = hub
        return selected, maximum

    selections = [select_links(h) for h in hubs]
    links = pd.concat([s[0] for s in selections]).sort_values('strength', ascending=False, kind='stable')
    auto_max = max(s[1] for s in selections)
    if not np.isfinite(label_fontsize) or label_fontsize <= 0:
        raise ValueError("label_fontsize must be finite and positive")
    label_partners = ld_sampler is not None or ld_cache_path is not None

    lengths = inferred if chromosome_lengths is None else {
        chrom_id(k): float(v) for k, v in chromosome_lengths.items()}
    if set(inferred) - set(lengths):
        raise ValueError("chromosome_lengths must cover every annotated chromosome")
    if any(not np.isfinite(v) or v <= 0 for v in lengths.values()):
        raise ValueError("Chromosome lengths must be finite and positive")
    if any(lengths[c] < end for c, end in inferred.items()):
        raise ValueError("A chromosome length is shorter than an annotated gene end")
    chromosomes = sorted(lengths, key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    available = 360 - gap_degrees * len(chromosomes)
    if available <= 0:
        raise ValueError("Chromosome gaps leave no space for chromosome arcs")
    total = sum(lengths.values())
    spans = {c: available * lengths[c] / total for c in chromosomes}
    starts = {}
    cursor = -spans[chromosomes[0]] / 2
    for c in chromosomes:
        starts[c] = cursor
        cursor += spans[c] + gap_degrees

    def angle(row):
        midpoint = (row.gene_chr_start + row.gene_chr_end) / 2
        return np.deg2rad(starts[row.chrom] + spans[row.chrom] * midpoint / lengths[row.chrom])

    def point(theta, radius):
        return np.array([np.cos(theta), np.sin(theta)]) * radius

    upper = float(vmax) if vmax is not None else max(float(auto_max), np.finfo(float).tiny)
    if not np.isfinite(vmin) or not np.isfinite(upper) or upper <= vmin:
        raise ValueError("Color limits must be finite with vmax > vmin")
    norm = Normalize(vmin=vmin, vmax=upper, clip=True)
    color_maps = {}
    for hub, color in zip(hubs, hub_colors if len(hubs) == 2 else [ribbon_color]):
        if color is None:
            color_maps[hub] = plt.get_cmap(cmap)
        else:
            strong = np.array(to_rgb(color))
            color_maps[hub] = LinearSegmentedColormap.from_list(
                'ribbon_' + hub, [.95 * np.ones(3) + .05 * strong, strong])
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=.08, right=.82, bottom=.08, top=.88)
    hub_angles = {h: angle(meta.loc[h]) for h in hubs}
    for _, row in links.iloc[::-1].iterrows():
        origin = point(hub_angles[row.source_hub], ring_inner_radius)
        color_map = color_maps[row.source_hub]
        destination = point(angle(row), ring_inner_radius)
        path = MplPath([origin, origin * .35, destination * .35, destination],
                       [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor='none', edgecolor=color_map(norm(row.strength) if gradient else 1.0),
                               # linewidth=.35 + 1.8 * norm(row.strength),
                               linewidth=1.2,
                               alpha=1.0 if ribbon_color is not None else .8))
    for c in chromosomes:
        ax.add_patch(Wedge((0, 0), CHROMOSOME_RING_OUTER_RADIUS,
                           starts[c], starts[c]+spans[c], width=ring_width,
                           facecolor='white',
                           edgecolor='black', linewidth=1.6))
        middle = np.deg2rad(starts[c] + spans[c]/2)
        ax.text(*point(middle, ring_center_radius), c,
                ha='center', va='center', fontsize=label_fontsize)
    for hub, hub_angle in hub_angles.items():
        ax.plot(*np.array([
            point(hub_angle, ring_inner_radius),
            point(hub_angle, CHROMOSOME_RING_OUTER_RADIUS),
        ]).T,
                color=hub_color if hub_color is not None else 'black',
                linewidth=1.6, alpha=1.0,
                solid_capstyle='butt', zorder=5)
    if label_partners:
        known_hubs = set(hubs) | set(hub_list if hub_list is not None else [])
        # Keep isolated labels near their genomic positions. Only labels moved
        # vertically to resolve crowding receive leader lines.
        label_genes = list(dict.fromkeys(links.loc[links.is_peak, 'label_gene']))
        # Focal names never participate in label displacement or get leaders.
        for hub, hub_angle in hub_angles.items():
            side = 1 if np.cos(hub_angle) >= 0 else -1
            ax.text(*point(hub_angle, 1.17), hub,
                    ha='left' if side > 0 else 'right', va='center',
                    fontsize=label_fontsize)
        labels = [
            (gene, angle(meta.loc[gene])) for gene in label_genes if gene not in hubs]
        for side in [-1, 1]:
            entries = [(gene, theta, point(theta, 1.12)) for gene, theta in labels
                       if (1 if np.cos(theta) >= 0 else -1) == side]
            entries.sort(key=lambda item: item[2][1])
            if not entries:
                continue
            gap = min(.09, 2.3 / max(1, len(entries)-1))
            ys = [float(item[2][1]) for item in entries]
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i-1] + gap)
            if ys[-1] > 1.18:
                ys = [y - (ys[-1] - 1.18) for y in ys]
            for (gene, theta, anchor), y in zip(entries, ys):
                displaced = not np.isclose(y, anchor[1], atol=1e-6, rtol=0)
                # Follow the ring instead of routing every displaced label to
                # a distant fixed x-column, especially near top and bottom.
                label_x = side * (np.sqrt(max(0, 1.17**2 - y**2)) + .02)
                label_position = (label_x, y) if displaced else point(theta, 1.17)
                connector = dict(arrowstyle='-', color='0.5', lw=.6)
                annotation = ax.annotate(gene, xy=point(theta, 1.1), xytext=label_position,
                            ha='left' if side > 0 else 'right', va='center',
                            fontsize=label_fontsize,
                            color=(nonhub_label_color if gene not in known_hubs
                                   and nonhub_label_color is not None
                                   else plt.rcParams['text.color']),
                            fontstyle=(nonhub_label_fontstyle if gene not in known_hubs
                                       else 'normal'),
                            fontweight='bold' if gene in hubs else 'normal',
                            arrowprops=connector,
                            annotation_clip=False)
                annotation.arrow_patch.set_visible(displaced)
    else:
        for hub, hub_angle in hub_angles.items():
            ax.text(*point(hub_angle, 1.17), hub, ha='center', va='center', fontsize=label_fontsize,
                    color=color_maps[hub](1.0))
    ax.set(xlim=(-1.28, 1.28), ylim=(-1.28, 1.28), aspect='equal')
    if label_partners:
        ax.set_xlim(-1.65, 1.65)
    ax.axis('off')
    # fig.suptitle(f"Gene Interactions ({gene_focal} Hub)", fontsize=18, fontweight='bold', y=.96)
    if gradient and show_colorbar:
        for i, hub in enumerate(hubs):
            color_ax = fig.add_axes([.8, .43 if len(hubs) == 1 else .62 - .30*i, .025, .16])
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=color_maps[hub]), cax=color_ax)
            cb.ax.set_title(colorbar_label if colorbar_label is not None else
                            ('Interaction\nstrength' if len(hubs) == 1 else hub + '\nstrength'),
                            fontsize=label_fontsize, loc='left', pad=10)
            cb.formatter.set_powerlimits((-2, 2))
            cb.update_ticks()
            cb.solids.set_rasterized(False)
            if colorbar_tick_fontsize is not None:
                cb.ax.tick_params(axis='y', labelsize=colorbar_tick_fontsize)
                cb.ax.yaxis.get_offset_text().set_fontsize(colorbar_tick_fontsize)
    elif show_colorbar:
        fig.legend(handles=[Line2D([0], [0], color=color_maps[h](1.0), lw=2, label=h)
                            for h in hubs], loc='center left', bbox_to_anchor=(.8, .5),
                   frameon=False, fontsize=label_fontsize)
    if not len(links):
        ax.text(0, 0, 'No interactions pass the display filters', ha='center', fontsize=label_fontsize)
    if title:
        ax.set_title(title, fontsize=label_fontsize)
    _style_pair_circos(fig, ax, font_family, font_weight)
    for label in ax.texts:
        if label.get_text() in hubs:
            if gene_focal_fontweight is not None:
                label.set_fontweight(gene_focal_fontweight)
            if hub_color is not None:
                label.set_color(hub_color)
            elif gene_focal_color is not None:
                label.set_color(gene_focal_color)
    if label_partners:
        # Measure final font/layout bounds, keeping focal labels fixed. Test
        # nearest vertical offsets first so connectors remain short.
        from matplotlib.text import Annotation, Text
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fixed = [t for t in ax.texts if t.get_text() in hubs]
        occupied = [Text.get_window_extent(t, renderer).expanded(1.06, 1.2)
                    for t in fixed]
        partners = [t for t in ax.texts if isinstance(t, Annotation)
                    and t.get_text() not in hubs]
        for label in partners:
            original = np.asarray(label.get_position(), dtype=float)
            pixel_position = ax.transData.transform(original)
            height = Text.get_window_extent(label, renderer).height
            step = height + 4
            # Move away from the circle first when both directions are free.
            direction = 1 if original[1] >= 0 else -1
            for attempt in range(2 * len(ax.texts) + 3):
                offset = 0 if attempt == 0 else (
                    direction * ((attempt + 1)//2) * step * (1 if attempt % 2 else -1))
                candidate = ax.transData.inverted().transform(pixel_position + [0, offset])
                if np.linalg.norm(candidate) < 1.14:
                    continue
                label.set_position(candidate)
                bounds = Text.get_window_extent(label, renderer).expanded(1.06, 1.2)
                if not any(bounds.overlaps(other) for other in occupied):
                    break
            if offset != 0:
                label.arrow_patch.set_visible(True)
            occupied.append(bounds)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
    return fig, ax, links
