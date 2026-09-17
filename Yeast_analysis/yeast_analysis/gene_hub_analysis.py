import torch

def linked_gene_pair_epistasis(
    gene_a, gene_b, genotypes, fitness, gene_snps, sampler, gene_info,
    *, ld_threshold=0.8, keep_gene_a=False,
):
    """Find direct cis-LD neighbors and their maximum squared SNP epistasis.

    ``gene_info`` is a DataFrame with ``chrom`` and ``gene_standard_name``
    (the latter may instead be its index). Neighbors must share the anchor's
    chromosome and have sampler.gene_ld(anchor, gene) > ld_threshold.
    Anchors themselves are included. This uses the sampler's gene-level r2,
    NOT physical distance or transitive connected components.
    With keep_gene_a=True, the first side contains only gene_a; the second
    side still includes gene_b and its linked genes.

    ``gene_snps`` must contain zero-based locus positions matching the columns
    of the supplied (N, 2L) one-hot tensor, after any filtering/reordering.
    All distinct unordered SNP pairs are evaluated with
    epistatic_coefficient_onehot. Pairs missing any of the four genotype
    classes are skipped and counted. Other input errors raise.

    Returns (results, linked_a, linked_b), all DataFrames. Results retain the
    max_epsilon_squared, the signed coefficient at that maximum (max_epsilon),
    its SNP indices, and counts. Undefined maxima are NaN. Ties retain the
    first SNP pair in sorted index order.
    This is descriptive screening, not a significance test; larger genes
    have more opportunities to attain extreme coefficients.
    """
    import math
    import operator
    import pandas as pd
    from itertools import product

    if not math.isfinite(ld_threshold) or not 0 <= ld_threshold <= 1:
        raise ValueError('ld_threshold must be between 0 and 1')
    if genotypes.ndim != 2 or genotypes.shape[1] % 2:
        raise ValueError('genotypes must have shape (N, 2L)')
    fitness = fitness.reshape(-1).to(genotypes.device)
    if len(fitness) != len(genotypes) or not torch.isfinite(fitness).all():
        raise ValueError('fitness must contain one finite value per sample')
    meta = gene_info.reset_index() if 'gene_standard_name' not in gene_info.columns else gene_info
    chroms = meta.groupby('gene_standard_name')['chrom'].agg(
        lambda values: set(values.dropna().astype(str))
    )
    support = set(sampler.genes) & set(gene_snps) & set(chroms.index)
    for anchor in (gene_a, gene_b):
        if anchor not in support or len(chroms[anchor]) != 1:
            raise ValueError(f'{anchor}: missing data or ambiguous chromosome')

    def neighbors(anchor):
        records = []
        for gene in sorted(support):
            if chroms[gene] != chroms[anchor]:
                continue
            r2 = 1.0 if gene == anchor else float(sampler.gene_ld(anchor, gene))
            if not math.isfinite(r2) or not 0 <= r2 <= 1:
                raise ValueError(f'Invalid LD for {anchor}, {gene}: {r2}')
            if gene == anchor or r2 > ld_threshold:
                records.append({'gene': gene, 'r2_to_anchor': r2})
        return pd.DataFrame(records)

    linked_a = (pd.DataFrame([{'gene': gene_a, 'r2_to_anchor': 1.0}])
                if keep_gene_a else neighbors(gene_a))
    linked_b = neighbors(gene_b)
    loci = {}
    for gene in set(linked_a.gene) | set(linked_b.gene):
        indices = sorted(set(operator.index(i) for i in gene_snps[gene]))
        if any(i < 0 or i >= genotypes.shape[1] // 2 for i in indices):
            raise IndexError(f'SNP index out of range for {gene}')
        loci[gene] = indices

    records = []
    with torch.no_grad():
        for a, b in product(linked_a.gene, linked_b.gene):
            best, best_i, best_j = None, None, None
            attempted = valid = 0
            seen = set()
            for i, j in product(loci[a], loci[b]):
                key = (min(i, j), max(i, j))
                if i == j or key in seen:
                    continue
                seen.add(key)
                attempted += 1
                try:
                    epsilon = float(epistatic_coefficient_onehot(
                        genotypes, fitness, i, j
                    ).item())
                except ValueError as exc:
                    if str(exc).startswith('No observations for class'):
                        continue
                    raise
                if not math.isfinite(epsilon):
                    raise ValueError(f'Nonfinite coefficient for SNPs {i}, {j}')
                valid += 1
                if best is None or epsilon ** 2 > best ** 2:
                    best, best_i, best_j = epsilon, i, j
            records.append(dict(
                gene1=a, gene2=b, max_epsilon=best,
                max_epsilon_squared=None if best is None else best ** 2,
                snp_i=best_i, snp_j=best_j, n_snp_pairs=attempted,
                n_valid_pairs=valid, n_missing_class_pairs=attempted-valid,
            ))
    results = pd.DataFrame(records)
    results['max_epsilon'] = pd.to_numeric(results['max_epsilon'])
    results['max_epsilon_squared'] = pd.to_numeric(results['max_epsilon_squared'])
    results[['snp_i', 'snp_j']] = results[['snp_i', 'snp_j']].astype('Int64')
    return results.sort_values('max_epsilon_squared', ascending=False).reset_index(drop=True), linked_a, linked_b

def linked_gene_pair_subset_interaction(
    gene_a, gene_b, marg, gene_snps, sampler, gene_info,
    *, ld_threshold=0.8, keep_gene_a=False,
):
    """Maximize marg.get_subset_interaction([i, j]) for linked gene pairs.

    Returns (results, linked_a, linked_b). As in linked_gene_pair_epistasis,
    neighbors are direct same-chromosome gene-level r2 > ld_threshold,
    including anchors, with no transitive chaining. keep_gene_a=True fixes
    the first side to gene_a only. gene_info is a DataFrame with chrom and
    gene_standard_name (as a column or index).

    gene_snps contains zero-based locus indices in marg's SNP ordering.
    Each distinct unordered SNP pair is evaluated once per gene pair;
    i == j is excluded. The scalar returned by marg is maximized as-is,
    without squaring, SNP-count averaging, or variance normalization.
    Errors and nonfinite values raise rather than silently dropping pairs.
    Empty pairs have a NaN maximum and missing SNP indices. Ties retain the
    first pair in sorted SNP order. Maxima are descriptive, not p-values.
    """
    import math
    import operator
    import pandas as pd
    from itertools import product

    if not math.isfinite(ld_threshold) or not 0 <= ld_threshold <= 1:
        raise ValueError('ld_threshold must be between 0 and 1')
    meta = gene_info.reset_index() if 'gene_standard_name' not in gene_info.columns else gene_info
    chroms = meta.groupby('gene_standard_name')['chrom'].agg(
        lambda values: set(values.dropna().astype(str))
    )
    support = set(sampler.genes) & set(gene_snps) & set(chroms.index)
    for anchor in (gene_a, gene_b):
        if anchor not in support or len(chroms[anchor]) != 1:
            raise ValueError(f'{anchor}: missing data or ambiguous chromosome')

    def neighbors(anchor):
        records = []
        for gene in sorted(support):
            if chroms[gene] != chroms[anchor]:
                continue
            r2 = 1.0 if gene == anchor else float(sampler.gene_ld(anchor, gene))
            if not math.isfinite(r2) or not 0 <= r2 <= 1:
                raise ValueError(f'Invalid LD for {anchor}, {gene}: {r2}')
            if gene == anchor or r2 > ld_threshold:
                records.append({'gene': gene, 'r2_to_anchor': r2})
        return pd.DataFrame(records)

    linked_a = (pd.DataFrame([{'gene': gene_a, 'r2_to_anchor': 1.0}])
                if keep_gene_a else neighbors(gene_a))
    linked_b = neighbors(gene_b)
    loci = {}
    for gene in set(linked_a.gene) | set(linked_b.gene):
        indices = sorted(set(operator.index(i) for i in gene_snps[gene]))
        if any(i < 0 for i in indices):
            raise IndexError(f'Negative SNP index for {gene}')
        loci[gene] = indices

    records = []
    with torch.no_grad():
        for a, b in product(linked_a.gene, linked_b.gene):
            best, best_i, best_j = None, None, None
            seen = set()
            for i, j in product(loci[a], loci[b]):
                key = (min(i, j), max(i, j))
                if i == j or key in seen:
                    continue
                seen.add(key)
                value = marg.get_subset_interaction([i, j])
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite interaction for SNPs {i}, {j}')
                if best is None or value > best:
                    best, best_i, best_j = value, i, j
            records.append(dict(
                gene1=a, gene2=b, max_interaction=best,
                snp_i=best_i, snp_j=best_j, n_snp_pairs=len(seen),
            ))
    results = pd.DataFrame(records)
    results['max_interaction'] = pd.to_numeric(results['max_interaction'])
    results[['snp_i', 'snp_j']] = results[['snp_i', 'snp_j']].astype('Int64')
    return results.sort_values('max_interaction', ascending=False).reset_index(drop=True), linked_a, linked_b


def epistatic_coefficient_onehot(
    genotypes: torch.Tensor,
    fitness: torch.Tensor,
    i: int,
    j: int,
) -> torch.Tensor:
    """
    Compute epsilon_ij = f11 - f10 - f01 + f00.

    Parameters
    ----------
    genotypes : torch.Tensor, shape (N, 2L)
        One-hot-encoded binary genotypes. Columns for each locus are
        ordered as [allele_0, allele_1].
    fitness : torch.Tensor, shape (N,)
        One fitness value per genotype.
    i, j : int
        Zero-based locus indices in [0, L - 1].

    Returns
    -------
    torch.Tensor
        Scalar epistatic coefficient.
    """
    if genotypes.ndim != 2:
        raise ValueError("genotypes must have shape (N, 2L)")

    N, two_L = genotypes.shape

    if two_L % 2 != 0:
        raise ValueError(
            "The number of genotype columns must be even"
        )

    L = two_L // 2
    fitness = fitness.reshape(-1)

    if fitness.shape[0] != N:
        raise ValueError("fitness must contain one value per genotype")

    if i == j:
        raise ValueError("i and j must be different loci")

    if not (0 <= i < L and 0 <= j < L):
        raise IndexError(f"i and j must be in [0, {L - 1}]")

    # Shape: (N, L, 2)
    G = genotypes.reshape(N, L, 2)

    # Validate only the two loci being analyzed.
    selected = G[:, [i, j], :]
    valid = (
        torch.isclose(
            selected.sum(dim=-1),
            torch.ones(
                (N, 2),
                device=genotypes.device,
                dtype=genotypes.dtype,
            ),
        ).all()
        and ((selected == 0) | (selected == 1)).all()
    )

    if not valid.item():
        raise ValueError(
            "Each locus must contain exactly one active allele: "
            "[1, 0] or [0, 1]"
        )

    # allele[:, 0] is locus i; allele[:, 1] is locus j.
    allele = selected.argmax(dim=-1)

    means = {}

    for allele_i in (0, 1):
        for allele_j in (0, 1):
            mask = (
                (allele[:, 0] == allele_i)
                & (allele[:, 1] == allele_j)
            )

            count = mask.sum()

            if count.item() == 0:
                raise ValueError(
                    f"No observations for class "
                    f"({allele_i}, {allele_j}) at loci ({i}, {j})"
                )

            means[allele_i, allele_j] = fitness[mask].float().mean()

    return (
        means[1, 1]
        - means[1, 0]
        - means[0, 1]
        + means[0, 0]
    )



import matplotlib.pyplot as plt


def _style_epistatic_axes(ax, fontsize):
    """Apply the plot-wide Nimbus Sans typography."""
    text_objects = [
        ax.title, ax.xaxis.label, ax.yaxis.label,
        ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text(),
        *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts,
    ]
    legend = ax.get_legend()
    if legend is not None:
        text_objects.extend(legend.get_texts())
        text_objects.append(legend.get_title())
    for text_object in text_objects:
        text_object.set_fontfamily('Nimbus Sans')
        text_object.set_fontsize(fontsize)
    ax.tick_params(axis='both', which='both', labelsize=fontsize)


def plot_epistatic_parallelogram_onehot(
    genotypes, fitness, i, j, *, ax=None, figsize=(5, 4),
    locus_i_label=None, locus_j_label=None, ylabel='Mean fitness',
    colors=('#5177A1', '#B22222'), linewidth=2, markersize=7,
    show_counts=False, title=None, output_path=None, dpi=300,
    show_genotype_labels=False,
    bootstrap_errors=False, n_bootstrap=2000, confidence_level=0.95,
    bootstrap_seed=42, errorbar_capsize=4, errorbar_alpha=1.0,
    fontsize=10,
):
    """Plot four genotype-class means and return fig, ax, epsilon_ij.

    Inputs and epsilon match epistatic_coefficient_onehot: genotypes has
    shape (N, 2L), with [allele_0, allele_1] per locus, and i/j are zero-based
    SNP indices, not one-hot column indices. Both tensors must share a device.

    X is mutation count relative to 00 (labeled WT): 00 at 0, 10 and 01 at 1,
    and 11 at 2. Four solid edges connect exactly the one-mutation neighbors.
    Blue edges change locus i; red edges change locus j. A hollow marker at
    x=2 indicates f10 + f01 - f00; its difference from observed f11 is epsilon.
    WT is a reference label for allele 0 at both loci, not inferred ancestry.
    Counts are raw class sizes;
    these are unadjusted means over the sampled genetic backgrounds, not
    causal effects or model-adjusted estimates.

    bootstrap_errors=True adds percentile bootstrap confidence intervals for
    the means of all four classes (00, 10, 01 and 11). Resampling is with replacement within each
    class, assuming independent rows; correlated replicates require a separate
    cluster bootstrap. n_bootstrap and confidence_level control the intervals,
    and bootstrap_seed makes them reproducible. Single-observation classes
    produce a zero-width interval and a warning. The additive expectation and
    epsilon do not receive uncertainty intervals. The return tuple is unchanged;
    ax.bootstrap_intervals maps (allele_i, allele_j) to (lower, upper) bounds.
    Point genotype labels and sample counts are hidden by default; enable
    show_genotype_labels and/or show_counts to display them independently.
    fontsize controls all plot text, which uses Nimbus Sans. errorbar_alpha
    controls the opacity of bootstrap error bars and their caps.
    """
    import numpy as np
    import warnings
    if (isinstance(fontsize, bool) or not isinstance(fontsize, (int, float, np.number))
            or not np.isfinite(fontsize) or fontsize <= 0):
        raise ValueError('fontsize must be a positive finite number')
    if (isinstance(errorbar_alpha, bool)
            or not isinstance(errorbar_alpha, (int, float, np.number))
            or not np.isfinite(errorbar_alpha)
            or not 0 <= errorbar_alpha <= 1):
        raise ValueError('errorbar_alpha must be between 0 and 1')
    if bootstrap_errors:
        if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 2:
            raise ValueError('n_bootstrap must be an integer >= 2')
        if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise ValueError('confidence_level must lie strictly between 0 and 1')
        if not np.isfinite(errorbar_capsize) or errorbar_capsize < 0:
            raise ValueError('errorbar_capsize must be finite and nonnegative')
    if genotypes.device != fitness.device:
        raise ValueError('genotypes and fitness must be on the same device')
    if not torch.isfinite(fitness).all().item():
        raise ValueError('fitness must contain only finite values')
    epsilon = epistatic_coefficient_onehot(genotypes, fitness, i=i, j=j)
    alleles = genotypes.reshape(genotypes.shape[0], -1, 2)[:, [i, j]].argmax(-1)
    values = fitness.reshape(-1)
    means, counts, intervals = {}, {}, {}
    rng = np.random.default_rng(bootstrap_seed) if bootstrap_errors else None
    for a in (0, 1):
        for b in (0, 1):
            mask = (alleles[:, 0] == a) & (alleles[:, 1] == b)
            means[a, b] = values[mask].float().mean().detach().cpu().item()
            counts[a, b] = int(mask.sum().item())
            if bootstrap_errors:
                sample = values[mask].detach().cpu().double().numpy()
                n = len(sample)
                if n == 1:
                    warnings.warn(f'Class {(a, b)} has one observation; bootstrap cannot estimate its uncertainty.',
                                  stacklevel=2)
                bootstrap_means = np.empty(n_bootstrap)
                # Bound temporary index arrays for large classes.
                batch_size = max(1, min(256, 1000000 // n))
                for start in range(0, n_bootstrap, batch_size):
                    stop = min(start + batch_size, n_bootstrap)
                    indices = rng.integers(0, n, size=(stop-start, n))
                    bootstrap_means[start:stop] = sample[indices].mean(axis=1)
                tail = (1-confidence_level)/2
                intervals[a, b] = tuple(np.quantile(bootstrap_means, [tail, 1-tail]))
    if len(colors) != 2:
        raise ValueError('colors must contain two colors')
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.bootstrap_intervals = intervals
    ilabel = locus_i_label if locus_i_label is not None else f'SNP {i}'
    jlabel = locus_j_label if locus_j_label is not None else f'SNP {j}'
    edges = [((0, 0), (1, 0), colors[0], ilabel),
             ((0, 1), (1, 1), colors[0], None),
             ((0, 0), (0, 1), colors[1], jlabel),
             ((1, 0), (1, 1), colors[1], None)]
    for start, end, color, label in edges:
        ax.plot([sum(start), sum(end)], [means[start], means[end]],
                color=color, linewidth=linewidth, label=label, zorder=2)
    for genotype, mean in means.items():
        ax.plot(sum(genotype), mean, 'o', color='black', markersize=markersize, zorder=3)
    for genotype, (lower, upper) in intervals.items():
        # Center the errorbar on its interval to preserve exact percentile
        # endpoints even if a highly skewed interval excludes the sample mean.
        ax.errorbar(sum(genotype), (lower+upper)/2, yerr=(upper-lower)/2,
                    fmt='none', ecolor='black', elinewidth=1.2,
                    capsize=errorbar_capsize, alpha=errorbar_alpha, zorder=2.5)
    expected = means[1, 0] + means[0, 1] - means[0, 0]
    for single, color in [((1, 0), colors[1]), ((0, 1), colors[0])]:
        ax.plot([1, 2], [means[single], expected], '--', color=color,
                linewidth=linewidth, alpha=.6, zorder=1)
    ax.plot(2, expected, marker='o', markerfacecolor='white', markeredgecolor='0.4',
            markersize=markersize, zorder=4)
    eps_value = epsilon.detach().cpu().item()
    ax.text(.98, .10, rf'$\epsilon={eps_value:.3f}$',
            transform=ax.transAxes, ha='right', va='bottom')
    for (a, b), mean in means.items():
        parts = []
        if show_genotype_labels:
            parts.append(f'{a}{b}')
        if show_counts:
            parts.append(f'n={counts[a,b]}')
        if not parts:
            continue
        label = ' '.join(parts)
        offset = 10 if (a, b) != (0, 1) else -16
        ax.annotate(label, (a+b, mean), xytext=(0, offset),
                    textcoords='offset points', ha='center', fontsize=fontsize)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['0\nWT', '1\nSingle mutants', '2\nDouble mutant'])
    ax.set_xlabel('Number of mutations')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-.25, 2.8)
    ax.set_box_aspect(1)
    ax.margins(y=.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        frameon=False,
        prop={'family': 'Nimbus Sans', 'size': fontsize},
        loc='upper left',
        bbox_to_anchor=(0, 1),
        handlelength=1.25,
        handletextpad=0.45,
        labelspacing=0.2,
        borderaxespad=0,
    )
    if title is not None:
        ax.set_title(title)
    _style_epistatic_axes(ax, fontsize)
    if output_path is not None:
        from pathlib import Path
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches='tight')
    return fig, ax, epsilon

def _gene_parental_onehot(genotypes, fitness, i, j, snp_info,
                         snp_index_col='Index'):
    """Collapse complete parental haplotypes into two binary gene loci."""
    import pandas as pd
    import numpy as np
    import warnings

    info = snp_info.copy() if isinstance(snp_info, pd.DataFrame) else pd.read_csv(snp_info)
    if genotypes.ndim != 2 or genotypes.shape[1] % 2:
        raise ValueError('genotypes must have shape (N, 2L)')
    if fitness.device != genotypes.device or fitness.numel() != genotypes.shape[0]:
        raise ValueError('fitness must have N values on the genotype device')
    if not torch.isfinite(fitness).all().item():
        raise ValueError('fitness must contain only finite values')
    if 'gene_standard_name' not in info:
        raise ValueError('snp_info requires gene_standard_name')
    if i == j:
        raise ValueError('i and j must be different genes')
    if snp_index_col is None:
        if len(info) != genotypes.shape[1] // 2:
            raise ValueError('Row-position mapping requires one metadata row per SNP')
        indices = np.arange(len(info))
    else:
        if snp_index_col not in info:
            raise ValueError(f'Missing SNP index column {snp_index_col!r}; use None for row positions')
        indices = pd.to_numeric(info[snp_index_col], errors='raise').to_numpy()

    states, labels, snp_sets = [], [], []
    G = genotypes.reshape(genotypes.shape[0], -1, 2)
    for gene in (i, j):
        # Accept either standard names or systematic ORF names.
        match = info['gene_standard_name'].eq(gene)
        if 'gene_name' in info:
            match = match | info['gene_name'].eq(gene)
        if not match.any():
            raise ValueError(f'No SNPs mapped to gene {gene!r}')
        selected_indices = indices[match.to_numpy()]
        if (not np.isfinite(selected_indices).all()
                or not (selected_indices == np.floor(selected_indices)).all()
                or (selected_indices < 0).any()
                or (selected_indices >= G.shape[1]).any()):
            raise ValueError(f'Invalid zero-based SNP indices for {gene!r}')
        snps = sorted(set(selected_indices.astype(int).tolist()))
        snp_sets.append(snps)
        names = info.loc[match, 'gene_standard_name'].dropna().astype(str)
        names = sorted({n.strip() for n in names if n.strip()})
        if len(names) > 1:
            raise ValueError(f'Ambiguous standard gene name for {gene!r}: {names}')
        labels.append(names[0] if names else str(gene))
        selected = G[:, snps, :]
        if not (((selected == 0) | (selected == 1)).all()
                & (selected.sum(-1) == 1).all()).item():
            raise ValueError(f'Invalid binary one-hot calls for {gene!r}')
        alleles = selected.argmax(-1)
        state = torch.full((G.shape[0],), -1, dtype=torch.long, device=G.device)
        state[(alleles == 0).all(dim=1)] = 0
        state[(alleles == 1).all(dim=1)] = 1
        states.append(state)
    if set(snp_sets[0]) & set(snp_sets[1]):
        raise ValueError('Genes share SNP indices; independent parental-allele classes are not defined')
    keep = (states[0] >= 0) & (states[1] >= 0)
    excluded = int((~keep).sum().item())
    if excluded:
        warnings.warn(f'Excluded {excluded} samples with mixed within-gene haplotypes',
                      stacklevel=3)
    classes = torch.stack(states, dim=1)[keep]
    counts = {(a, b): int(((classes[:, 0] == a) & (classes[:, 1] == b)).sum().item())
              for a in (0, 1) for b in (0, 1)}
    if any(n == 0 for n in counts.values()):
        raise ValueError(f'Missing parental-allele classes for {i}, {j}; counts={counts}; excluded={excluded}')
    collapsed = torch.nn.functional.one_hot(classes, num_classes=2).reshape(-1, 4)
    details = dict(gene_names=tuple(labels), snp_indices=tuple(snp_sets),
                   counts=counts, n_excluded=excluded, n_retained=int(keep.sum().item()))
    return collapsed, fitness.reshape(-1)[keep], details


def epistatic_coefficient_gene_onehot(
    genotypes, fitness, i, j, snp_info='snp_info.csv', *,
    snp_index_col='Index', return_details=False,
):
    """Observed parental-gene contrast f11 - f10 - f01 + f00.

    i/j are standard or systematic gene names. Gene allele 0 means ALL its
    SNPs are 0; allele 1 means ALL are 1. Mixed haplotypes are excluded.
    Other genes are unrestricted. No genotypes or fitness values are edited
    or predicted; this is an unadjusted contrast across observed backgrounds.
    Allele coding must consistently represent the same parents across SNPs.

    snp_info is a CSV path or DataFrame. snp_index_col contains zero-based
    SNP indices (not one-hot columns), aligned with genotypes. Default 'Index'
    matches snp_info.csv. Use None only for metadata in full genotype SNP order.
    Returns a scalar tensor, or (tensor, details) if return_details=True.
    """
    G, y, details = _gene_parental_onehot(
        genotypes, fitness, i, j, snp_info, snp_index_col)
    epsilon = epistatic_coefficient_onehot(G, y, 0, 1)
    return (epsilon, details) if return_details else epsilon


def plot_epistatic_parallelogram_gene_onehot(
    genotypes, fitness, i, j, snp_info='snp_info.csv', *,
    snp_index_col='Index', ylim=None, yticks=None, null_p_dist=None, fontsize=10,
    errorbar_alpha=1.0, fig_path=None, **plot_kwargs,
):
    """Parental-gene version of plot_epistatic_parallelogram_onehot.

    Accepts its plotting/bootstrap options. Legend defaults to standard gene
    names with an RM superscript. Assumes allele 0 is BY and allele 1 is RM.
    Mixed haplotypes are excluded before class means or bootstrapping.
    ylim=(lower, upper) sets the y-axis limits; None keeps automatic limits.
    yticks supplies exact manual y-tick positions; None uses sparse automatic
    ticks.
    If null_p_dist is provided, the plot also reports the two-sided empirical
    permutation p-value (1 + sum(abs(null) >= abs(epsilon))) / (1 + n_null).
    fontsize applies to all text; errorbar_alpha controls bootstrap error-bar
    opacity. All text uses Nimbus Sans. If fig_path is a directory, the final
    figure is saved there as epistatic_parallelogram_<gene_i>_<gene_j>.pdf.
    Returns fig, ax, epsilon; ax.gene_allele_details reports SNPs and counts.
    When calculated, ax.permutation_p_value contains the permutation p-value.
    """
    import numpy as np

    G, y, details = _gene_parental_onehot(
        genotypes, fitness, i, j, snp_info, snp_index_col)
    plot_kwargs.setdefault('locus_i_label', details['gene_names'][0] + r'$^{\mathrm{RM}}$')
    plot_kwargs.setdefault('locus_j_label', details['gene_names'][1] + r'$^{\mathrm{RM}}$')
    if null_p_dist is not None:
        if torch.is_tensor(null_p_dist):
            null_values = null_p_dist.detach().cpu().numpy()
        else:
            null_values = np.asarray(null_p_dist)
        try:
            null_values = null_values.astype(float, copy=False).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError('null_p_dist must contain numeric values') from exc
        if null_values.size == 0:
            raise ValueError('null_p_dist must contain at least one value')
        if not np.isfinite(null_values).all():
            raise ValueError('null_p_dist must contain only finite values')
    if yticks is not None:
        try:
            ytick_values = np.asarray(yticks, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError('yticks must contain numeric values') from exc
        if ytick_values.size == 0:
            raise ValueError('yticks must contain at least one value')
        if not np.isfinite(ytick_values).all():
            raise ValueError('yticks must contain only finite values')
    # Save only after updating labels so exported and displayed plots agree.
    output_path = plot_kwargs.pop('output_path', None)
    fig, ax, epsilon = plot_epistatic_parallelogram_onehot(
        G, y, 0, 1, fontsize=fontsize, errorbar_alpha=errorbar_alpha,
        **plot_kwargs)
    ax.set_xticklabels(['BY', 'Single\nmutants', 'Double\nmutant'])
    ax.set_xlabel('')
    if ylim is not None:
        ax.set_ylim(*ylim)
    from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter
    if yticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    else:
        ax.yaxis.set_major_locator(FixedLocator(ytick_values))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    if null_p_dist is not None:
        observed = abs(float(epsilon.detach().cpu().item()))
        permutation_p_value = (
            1 + np.count_nonzero(np.abs(null_values) >= observed)
        ) / (null_values.size + 1)
        if permutation_p_value < .001:
            permutation_p_label = r'$p<0.001$'
        else:
            permutation_p_label = rf'$p={permutation_p_value:.3f}$'
        ax.text(.98, .02, permutation_p_label,
                transform=ax.transAxes, ha='right', va='bottom')
        ax.permutation_p_value = permutation_p_value
    ax.gene_allele_details = details
    _style_epistatic_axes(ax, fontsize)
    if output_path is not None:
        from pathlib import Path
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=plot_kwargs.get('dpi', 300), bbox_inches='tight')
    if fig_path is not None:
        import re
        from pathlib import Path

        output_directory = Path(fig_path)
        output_directory.mkdir(parents=True, exist_ok=True)

        def filename_gene(gene):
            cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(gene)).strip('._')
            return cleaned or 'gene'

        gene_i_name, gene_j_name = map(filename_gene, details['gene_names'])
        pdf_output = output_directory / (
            f'epistatic_parallelogram_{gene_i_name}_{gene_j_name}.pdf')
        fig.savefig(pdf_output, format='pdf', bbox_inches='tight')
        ax.figure_pdf_path = pdf_output
    return fig, ax, epsilon


def upper_triangle_scatter_matrix(df, figsize=None, s=10, alpha=0.6):
    """
    Scatter plots for all pairs of dataframe columns.
    Only the upper triangular part is shown.
    Axes are labeled with dataframe column names.
    """
    n = df.shape[1]

    if figsize is None:
        figsize = (1 * n, 1 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize)

    if n == 1:
        axes = [[axes]]

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]

            if j > i:
                ax.scatter(
                    df.iloc[:, j],
                    df.iloc[:, i],
                    s=s,
                    alpha=alpha
                )

                ax.set_xlabel(df.columns[j])
                ax.set_ylabel(df.columns[i])

            else:
                ax.axis("off")

    plt.tight_layout()
    return fig, axes


import torch

def sample_epistatic_coefficients(
    genotypes: torch.Tensor,
    fitness: torch.Tensor,
    n_samples: int = 10_000,
    seed: int = 42,
    *,
    snp_info='snp_info.csv',
    snp_index_col='Index',
):
    """Sample random gene pairs and calculate epistatic coefficients.

    Genes are the unique nonempty standard names in snp_info. Distinct gene
    pairs are sampled uniformly with replacement, then evaluated with
    epistatic_coefficient_gene_onehot. Consequently, the returned null may
    contain fewer than n_samples values when sampled pairs lack one or more
    parental-allele classes. snp_info and snp_index_col have the same meaning
    as in epistatic_coefficient_gene_onehot.

    Returns
    -------
    null_epistatic_coefficients : torch.Tensor
        Epistatic coefficients for successful pairs.
    valid_pairs : list[tuple[str, str]]
        Gene pairs corresponding to ``epsilons``.
    failed_pairs : list[tuple[str, str, str]]
        Failed gene pairs and their error messages.
    """
    import numpy as np
    import pandas as pd
    import warnings

    if genotypes.ndim != 2:
        raise ValueError("genotypes must have shape (N, 2L)")
    if genotypes.shape[1] % 2 != 0:
        raise ValueError("genotypes must have an even number of columns")
    if fitness.device != genotypes.device or fitness.numel() != genotypes.shape[0]:
        raise ValueError('fitness must have N values on the genotype device')
    if not torch.isfinite(fitness).all().item():
        raise ValueError('fitness must contain only finite values')
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)):
        raise ValueError("n_samples must be a nonnegative integer")
    if n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")

    info = snp_info.copy() if isinstance(snp_info, pd.DataFrame) else pd.read_csv(snp_info)
    if 'gene_standard_name' not in info:
        raise ValueError('snp_info requires gene_standard_name')

    # Give row-position metadata explicit SNP indices before reducing it to
    # pair-specific frames. This keeps repeated gene calculations inexpensive.
    effective_index_col = snp_index_col
    if snp_index_col is None:
        if len(info) != genotypes.shape[1] // 2:
            raise ValueError('Row-position mapping requires one metadata row per SNP')
        effective_index_col = '__sample_epistasis_snp_index__'
        while effective_index_col in info:
            effective_index_col += '_'
        info[effective_index_col] = np.arange(len(info))
    elif snp_index_col not in info:
        raise ValueError(
            f'Missing SNP index column {snp_index_col!r}; use None for row positions'
        )

    gene_names = info['gene_standard_name']
    has_name = gene_names.notna() & gene_names.astype(str).str.strip().ne('')
    info = info.loc[has_name].copy().reset_index(drop=True)
    info['gene_standard_name'] = info['gene_standard_name'].astype(str).str.strip()
    grouped_rows = info.groupby('gene_standard_name', sort=True).indices
    genes = sorted(grouped_rows)
    if len(genes) < 2:
        raise ValueError('snp_info must contain at least two named genes')

    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    i_samples = torch.randint(
        low=0,
        high=len(genes),
        size=(n_samples,),
        generator=generator,
    )
    j_samples = torch.randint(
        low=0,
        high=len(genes) - 1,
        size=(n_samples,),
        generator=generator,
    )
    j_samples += j_samples >= i_samples

    epsilon_values = []
    valid_pairs = []
    failed_pairs = []
    for i_index, j_index in zip(i_samples.tolist(), j_samples.tolist()):
        # Canonicalize order because gene epistasis is symmetric.
        i_index, j_index = sorted((i_index, j_index))
        gene_i, gene_j = genes[i_index], genes[j_index]
        pair_rows = np.concatenate((grouped_rows[gene_i], grouped_rows[gene_j]))
        pair_info = info.iloc[pair_rows]
        try:
            # Mixed haplotypes are expected for random multi-SNP genes; the
            # coefficient function still excludes them before calculation.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message=r'Excluded .* mixed within-gene haplotypes')
                epsilon = epistatic_coefficient_gene_onehot(
                    genotypes, fitness, gene_i, gene_j, pair_info,
                    snp_index_col=effective_index_col,
                )
            epsilon_values.append(epsilon)
            valid_pairs.append((gene_i, gene_j))
        except (ValueError, IndexError) as error:
            failed_pairs.append((gene_i, gene_j, str(error)))

    if epsilon_values:
        null_epistatic_coefficients = torch.stack(epsilon_values)
    else:
        null_epistatic_coefficients = torch.empty(
            0, device=fitness.device, dtype=torch.float32)

    return null_epistatic_coefficients, valid_pairs, failed_pairs



from collections import OrderedDict
import math
import random

import torch


class LDMatchedGeneSetSampler:
    """
    Sample null gene sets matched on:

    1. Number of SNPs per gene
    2. Pairwise LD distribution within the gene set
    3. Per-gene LD burden/topology

    Gene-level LD is calculated as r^2 between representative
    genotype vectors constructed from each gene's SNPs.
    """

    def __init__(
        self,
        data,
        gene_snps,
        seed=42,
        max_cached_genes=512,
        eps=1e-8,
    ):
        """
        Parameters
        ----------
        data
            Object containing data.geno_t with shape (N, L).

        gene_snps : dict
            Mapping from gene name to SNP-column indices.

        seed : int
            Random seed.

        max_cached_genes : int
            Maximum number of representative gene vectors retained
            in memory.

        eps : float
            Numerical tolerance.
        """
        self.genotypes = data.geno_t
        self.gene_snps = {
            gene: tuple(int(i) for i in indices)
            for gene, indices in gene_snps.items()
            if len(indices) > 0
        }

        if self.genotypes.ndim != 2:
            raise ValueError(
                "data.geno_t must have shape (N, L)"
            )

        self.N, self.L = self.genotypes.shape
        self.device = self.genotypes.device
        self.eps = eps

        self.rng = random.Random(seed)
        self.max_cached_genes = max_cached_genes
        self._vector_cache = OrderedDict()

        self.genes = []

        for gene, indices in self.gene_snps.items():
            if min(indices) < 0 or max(indices) >= self.L:
                raise IndexError(
                    f"SNP indices for {gene!r} are outside "
                    f"[0, {self.L - 1}]"
                )

            self.genes.append(gene)

        self.n_snps = {
            gene: len(self.gene_snps[gene])
            for gene in self.genes
        }

    @torch.no_grad()
    def _gene_vector(self, gene):
        """
        Construct one standardized genotype vector for a gene.

        SNPs are standardized and oriented relative to the first
        polymorphic SNP before averaging. Orientation prevents SNPs
        encoded with opposite reference alleles from cancelling.
        """
        if gene in self._vector_cache:
            vector = self._vector_cache.pop(gene)
            self._vector_cache[gene] = vector
            return vector

        if gene not in self.gene_snps:
            raise KeyError(
                f"{gene!r} is not present in gene_snps"
            )

        indices = torch.as_tensor(
            self.gene_snps[gene],
            device=self.device,
            dtype=torch.long,
        )

        X = self.genotypes[:, indices].float()

        means = X.mean(dim=0, keepdim=True)
        standard_deviations = X.std(
            dim=0,
            correction=1,
            keepdim=True,
        )

        # Exclude monomorphic SNPs.
        polymorphic = (
            standard_deviations.squeeze(0) > self.eps
        )

        X = X[:, polymorphic]
        means = means[:, polymorphic]
        standard_deviations = (
            standard_deviations[:, polymorphic]
        )

        if X.shape[1] == 0:
            raise ValueError(
                f"{gene!r} has no polymorphic SNPs"
            )

        Z = (X - means) / standard_deviations

        # Orient all SNPs relative to the first polymorphic SNP.
        anchor = Z[:, 0]

        correlations_with_anchor = (
            Z.T @ anchor
        ) / (self.N - 1)

        signs = torch.where(
            correlations_with_anchor >= 0,
            1.0,
            -1.0,
        )

        vector = (Z * signs).mean(dim=1)

        vector = vector - vector.mean()
        vector_std = vector.std(correction=1)

        if vector_std <= self.eps:
            raise ValueError(
                f"Could not construct a variable genotype "
                f"vector for {gene!r}"
            )

        vector = vector / vector_std

        self._vector_cache[gene] = vector

        while (
            len(self._vector_cache)
            > self.max_cached_genes
        ):
            self._vector_cache.popitem(last=False)

        return vector

    @torch.no_grad()
    def gene_ld(self, gene_a, gene_b):
        """
        Return gene-level LD, defined as squared Pearson
        correlation between representative gene vectors.
        """
        vector_a = self._gene_vector(gene_a)
        vector_b = self._gene_vector(gene_b)

        correlation = (
            torch.dot(vector_a, vector_b)
            / (self.N - 1)
        )

        return correlation.square().clamp(0, 1).item()

    @torch.no_grad()
    def ld_matrix(self, genes):
        """
        Calculate the gene-level LD matrix for a gene collection.
        """
        genes = list(genes)

        vectors = torch.stack(
            [self._gene_vector(gene) for gene in genes],
            dim=1,
        )

        correlations = (
            vectors.T @ vectors
        ) / (self.N - 1)

        return correlations.square().clamp(0, 1)

    @staticmethod
    def _upper_triangle(matrix):
        k = matrix.shape[0]

        if k < 2:
            return torch.empty(
                0,
                device=matrix.device,
                dtype=matrix.dtype,
            )

        rows, columns = torch.triu_indices(
            k,
            k,
            offset=1,
            device=matrix.device,
        )

        return matrix[rows, columns]

    @torch.no_grad()
    def _matching_loss(
        self,
        target_genes,
        target_ld,
        null_genes,
        null_ld,
        size_weight,
        ld_distribution_weight,
        ld_topology_weight,
    ):
        """
        Calculate discrepancy between observed and null sets.
        """
        target_sizes = torch.tensor(
            [
                math.log1p(self.n_snps[gene])
                for gene in target_genes
            ],
            device=self.device,
            dtype=torch.float32,
        ).sort().values

        null_sizes = torch.tensor(
            [
                math.log1p(self.n_snps[gene])
                for gene in null_genes
            ],
            device=self.device,
            dtype=torch.float32,
        ).sort().values

        size_loss = torch.mean(
            torch.abs(target_sizes - null_sizes)
        )

        if len(target_genes) > 1:
            target_pairwise = self._upper_triangle(
                target_ld
            ).sort().values

            null_pairwise = self._upper_triangle(
                null_ld
            ).sort().values

            ld_distribution_loss = torch.mean(
                (target_pairwise - null_pairwise).square()
            )

            # Mean LD of every gene with the other genes.
            target_degree = (
                (target_ld.sum(dim=1) - 1)
                / (len(target_genes) - 1)
            ).sort().values

            null_degree = (
                (null_ld.sum(dim=1) - 1)
                / (len(null_genes) - 1)
            ).sort().values

            ld_topology_loss = torch.mean(
                (target_degree - null_degree).square()
            )
        else:
            ld_distribution_loss = torch.tensor(
                0.0,
                device=self.device,
            )
            ld_topology_loss = torch.tensor(
                0.0,
                device=self.device,
            )

        total = (
            size_weight * size_loss
            + ld_distribution_weight
            * ld_distribution_loss
            + ld_topology_weight
            * ld_topology_loss
        )

        return {
            "total": total.item(),
            "snp_count": size_loss.item(),
            "ld_distribution":
                ld_distribution_loss.item(),
            "ld_topology": ld_topology_loss.item(),
        }

    def _size_matched_candidates(
        self,
        target_gene,
        excluded,
        maximum_candidates,
        expansion_factor=4,
    ):
        """
        Create a randomized pool of genes close in SNP count.
        """
        target_size = math.log1p(
            self.n_snps[target_gene]
        )

        available = [
            gene
            for gene in self.genes
            if gene not in excluded
        ]

        if not available:
            raise RuntimeError(
                "No unused candidate genes remain"
            )

        available.sort(
            key=lambda gene: abs(
                math.log1p(self.n_snps[gene])
                - target_size
            )
        )

        # Draw from a broader nearest-neighbor pool so that
        # repeated runs generate different null sets.
        broad_pool_size = min(
            len(available),
            maximum_candidates * expansion_factor,
        )

        broad_pool = available[:broad_pool_size]

        if len(broad_pool) <= maximum_candidates:
            return broad_pool

        return self.rng.sample(
            broad_pool,
            maximum_candidates,
        )

    @torch.no_grad()
    def _construct_null_set(
        self,
        target_genes,
        target_ld,
        maximum_candidates,
        choice_top_k,
        size_weight,
        ld_weight,
        excluded,
    ):
        """
        Sequentially construct one candidate null set.
        """
        k = len(target_genes)

        if k == 1:
            candidates = self._size_matched_candidates(
                target_genes[0],
                excluded,
                maximum_candidates,
            )

            return [self.rng.choice(candidates)]

        # Match the most LD-connected genes first.
        target_ld_burden = (
            target_ld.sum(dim=1) - 1
        ) / (k - 1)

        order = torch.argsort(
            target_ld_burden,
            descending=True,
        ).tolist()

        ordered_targets = [
            target_genes[index]
            for index in order
        ]

        ordered_target_ld = target_ld[
            order
        ][:, order]

        selected = []
        selected_set = set()

        for position, target_gene in enumerate(
            ordered_targets
        ):
            unavailable = (
                set(excluded)
                | selected_set
            )

            candidates = self._size_matched_candidates(
                target_gene,
                unavailable,
                maximum_candidates,
            )

            candidate_vectors = []
            usable_candidates = []

            for candidate in candidates:
                try:
                    candidate_vectors.append(
                        self._gene_vector(candidate)
                    )
                    usable_candidates.append(candidate)
                except ValueError:
                    continue

            if not usable_candidates:
                raise RuntimeError(
                    f"No usable candidates for {target_gene}"
                )

            candidate_vectors = torch.stack(
                candidate_vectors,
                dim=1,
            )

            target_size = math.log1p(
                self.n_snps[target_gene]
            )

            candidate_sizes = torch.tensor(
                [
                    math.log1p(
                        self.n_snps[candidate]
                    )
                    for candidate in usable_candidates
                ],
                device=self.device,
            )

            size_loss = torch.abs(
                candidate_sizes - target_size
            )

            if position == 0:
                ld_loss = torch.zeros_like(size_loss)
            else:
                selected_vectors = torch.stack(
                    [
                        self._gene_vector(gene)
                        for gene in selected
                    ],
                    dim=1,
                )

                candidate_ld = (
                    candidate_vectors.T
                    @ selected_vectors
                    / (self.N - 1)
                ).square()

                desired_ld = ordered_target_ld[
                    position, :position
                ].unsqueeze(0)

                ld_loss = torch.mean(
                    (candidate_ld - desired_ld).square(),
                    dim=1,
                )

            total_loss = (
                size_weight * size_loss
                + ld_weight * ld_loss
            )

            number_to_consider = min(
                choice_top_k,
                len(usable_candidates),
            )

            best_indices = torch.topk(
                total_loss,
                k=number_to_consider,
                largest=False,
            ).indices.tolist()

            chosen_index = self.rng.choice(best_indices)
            chosen_gene = usable_candidates[chosen_index]

            selected.append(chosen_gene)
            selected_set.add(chosen_gene)

        return selected

    @torch.no_grad()
    def sample(
        self,
        gene_set,
        n_restarts=25,
        maximum_candidates=64,
        choice_top_k=5,
        size_weight=1.0,
        ld_weight=10.0,
        ld_distribution_weight=10.0,
        ld_topology_weight=5.0,
        exclude_input=True,
        return_diagnostics=False,
    ):
        """
        Return one SNP-count- and LD-matched null gene set.

        Parameters
        ----------
        gene_set : collection of str
            Observed pathway genes.

        n_restarts : int
            Number of independently constructed candidate sets.
            The best-matching set is returned.

        maximum_candidates : int
            Maximum candidate genes evaluated at each step.

        choice_top_k : int
            Randomly select among this many best candidates,
            providing variation between restarts.

        exclude_input : bool
            Prevent observed pathway genes from appearing in the
            null set.

        return_diagnostics : bool
            Return matching diagnostics in addition to the set.

        Returns
        -------
        set[str]
            One matched null gene set.

        If return_diagnostics=True, returns:
            (null_gene_set, diagnostics)
        """
        target_genes = list(dict.fromkeys(gene_set))

        if not target_genes:
            raise ValueError("gene_set cannot be empty")

        missing = [
            gene
            for gene in target_genes
            if gene not in self.gene_snps
        ]

        if missing:
            raise KeyError(
                "Genes missing from gene_snps: "
                + ", ".join(missing)
            )

        # Validate target genes and calculate observed LD.
        target_ld = self.ld_matrix(target_genes)

        excluded = (
            set(target_genes)
            if exclude_input
            else set()
        )

        best_genes = None
        best_diagnostics = None

        for _ in range(n_restarts):
            try:
                null_genes = self._construct_null_set(
                    target_genes=target_genes,
                    target_ld=target_ld,
                    maximum_candidates=maximum_candidates,
                    choice_top_k=choice_top_k,
                    size_weight=size_weight,
                    ld_weight=ld_weight,
                    excluded=excluded,
                )

                null_ld = self.ld_matrix(null_genes)

                diagnostics = self._matching_loss(
                    target_genes=target_genes,
                    target_ld=target_ld,
                    null_genes=null_genes,
                    null_ld=null_ld,
                    size_weight=size_weight,
                    ld_distribution_weight=(
                        ld_distribution_weight
                    ),
                    ld_topology_weight=(
                        ld_topology_weight
                    ),
                )

                if (
                    best_diagnostics is None
                    or diagnostics["total"]
                    < best_diagnostics["total"]
                ):
                    best_genes = null_genes
                    best_diagnostics = diagnostics

            except RuntimeError:
                continue

        if best_genes is None:
            raise RuntimeError(
                "Unable to construct a matched null gene set. "
                "Increase maximum_candidates or allow input genes."
            )

        null_gene_set = set(best_genes)

        if return_diagnostics:
            best_diagnostics.update({
                "n_genes": len(null_gene_set),
                "target_snp_counts": sorted(
                    self.n_snps[g]
                    for g in target_genes
                ),
                "null_snp_counts": sorted(
                    self.n_snps[g]
                    for g in null_gene_set
                ),
                "target_mean_ld": (
                    self._upper_triangle(
                        target_ld
                    ).mean().item()
                    if len(target_genes) > 1
                    else 0.0
                ),
                "null_mean_ld": (
                    self._upper_triangle(
                        self.ld_matrix(best_genes)
                    ).mean().item()
                    if len(best_genes) > 1
                    else 0.0
                ),
            })

            return null_gene_set, best_diagnostics

        return null_gene_set
    
    
    
import numpy as np
import pandas as pd



import gzip
import io
import requests
from collections import defaultdict


def download_yeast_go_terms(
    exclude_not=True,
    evidence_codes=None,
):
    """
    Download current yeast GO annotations and return GO-to-gene mappings.

    Parameters
    ----------
    exclude_not : bool
        Exclude annotations carrying the NOT qualifier.

    evidence_codes : collection of str or None
        Keep only selected evidence codes. For example:
        {"IDA", "IMP", "IGI", "IPI"}.
        None retains all evidence codes, including IEA.

    Returns
    -------
    go_to_genes : dict[str, set[str]]
        GO ID -> standard yeast gene names.

    go_metadata : dict[str, dict]
        GO ID -> ontology aspect and evidence information.
    """
    url = (
        "https://current.geneontology.org/"
        "annotations/sgd.gaf.gz"
    )

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    go_to_genes = defaultdict(set)
    go_metadata = defaultdict(
        lambda: {
            "aspect": None,
            "evidence_codes": set(),
        }
    )

    with gzip.open(
        io.BytesIO(response.content),
        mode="rt",
        encoding="utf-8",
    ) as handle:

        for line in handle:
            if line.startswith("!"):
                continue

            fields = line.rstrip("\n").split("\t")

            if len(fields) < 15:
                continue

            gene_name = fields[2]
            qualifier = fields[3]
            go_id = fields[4]
            evidence = fields[6]
            aspect = fields[8]

            qualifiers = set(qualifier.split("|"))

            if exclude_not and "NOT" in qualifiers:
                continue

            if (
                evidence_codes is not None
                and evidence not in evidence_codes
            ):
                continue

            go_to_genes[go_id].add(gene_name)
            go_metadata[go_id]["aspect"] = {
                "P": "biological_process",
                "F": "molecular_function",
                "C": "cellular_component",
            }.get(aspect, aspect)

            go_metadata[go_id][
                "evidence_codes"
            ].add(evidence)

    return dict(go_to_genes), dict(go_metadata)




import pandas as pd


def go_enrichment_score(
    gene_values,
    go_genes,
    top_k=100,
    ascending=False,
):
    """
    Calculate GO enrichment among the top-k ranked genes.

    Parameters
    ----------
    gene_values : pd.Series
        Continuous values indexed by unique gene names.

    go_genes : collection of str
        Genes belonging to the GO term.

    top_k : int
        Number of top-ranked genes to examine.

    ascending : bool
        False: larger values rank higher.
        True: smaller values rank higher.

    Returns
    -------
    dict
        Enrichment ratio and contributing genes.
    """
    gene_values = pd.Series(gene_values).dropna()

    if not gene_values.index.is_unique:
        raise ValueError(
            "gene_values must have unique gene indices"
        )

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # Restrict the GO term to genes represented in gene_values.
    go_genes_present = set(go_genes).intersection(
        gene_values.index
    )

    if not go_genes_present:
        raise ValueError(
            "None of the GO genes occur in gene_values"
        )

    ranked_values = gene_values.sort_values(
        ascending=ascending
    )

    # Handle top_k larger than the available gene universe.
    actual_k = min(top_k, len(ranked_values))
    top_genes = list(ranked_values.index[:actual_k])

    top_go_genes = [
        gene
        for gene in top_genes
        if gene in go_genes_present
    ]

    n_total = len(ranked_values)
    n_go_total = len(go_genes_present)
    n_go_top = len(top_go_genes)

    observed_fraction = n_go_top / actual_k
    background_fraction = n_go_total / n_total

    enrichment_ratio = (
        observed_fraction / background_fraction
    )

    return {
        "enrichment_ratio": enrichment_ratio,
        "n_genes_total": n_total,
        "n_go_genes_total": n_go_total,
        "top_k": actual_k,
        "n_go_genes_top_k": n_go_top,
        "observed_fraction": observed_fraction,
        "background_fraction": background_fraction,
        "go_genes_present": sorted(go_genes_present),
        "top_go_genes": top_go_genes,
    }




# def go_enrichment_score(
#     gene_values: pd.Series,
#     go_genes,
#     weight: float = 1.0,
# ):
#     """
#     Calculate a GSEA-style enrichment score for one GO gene set.

#     Parameters
#     ----------
#     gene_values : pd.Series
#         Continuous values indexed by gene names.

#     go_genes : collection of str
#         Genes annotated to the GO term.

#     weight : float
#         GSEA weighting exponent:
#         0 = unweighted
#         1 = standard weighted GSEA

#     Returns
#     -------
#     result : dict
#         Contains the enrichment score, overlapping GO genes,
#         ranked overlapping genes, and running enrichment score.
#     """
#     if not isinstance(gene_values, pd.Series):
#         gene_values = pd.Series(gene_values)

#     # Remove missing values.
#     gene_values = gene_values.dropna()

#     if not gene_values.index.is_unique:
#         raise ValueError(
#             "gene_values must have unique gene indices"
#         )

#     # Keep only GO genes represented in gene_values.
#     go_genes_present = set(go_genes).intersection(
#         gene_values.index
#     )

#     if not go_genes_present:
#         raise ValueError(
#             "None of the GO genes occur in gene_values"
#         )

#     # Rank genes from highest to lowest value.
#     ranked_values = gene_values.sort_values(
#         ascending=False
#     )

#     ranked_genes = ranked_values.index.to_numpy()
#     values = ranked_values.to_numpy(dtype=float)

#     is_hit = np.isin(
#         ranked_genes,
#         list(go_genes_present),
#     )

#     number_of_genes = len(ranked_genes)
#     number_of_hits = is_hit.sum()
#     number_of_misses = number_of_genes - number_of_hits

#     # Weight GO genes by their absolute ranked value.
#     hit_weights = np.abs(values) ** weight
#     hit_weights[~is_hit] = 0.0

#     if hit_weights.sum() == 0:
#         # Uniform weighting if every hit has value zero.
#         hit_weights[is_hit] = 1.0

#     hit_increment = hit_weights / hit_weights.sum()

#     if number_of_misses > 0:
#         miss_decrement = (~is_hit) / number_of_misses
#     else:
#         miss_decrement = np.zeros(number_of_genes)

#     running_score = np.cumsum(
#         hit_increment - miss_decrement
#     )

#     maximum = running_score.max()
#     minimum = running_score.min()

#     # Retain the deviation furthest from zero.
#     if abs(maximum) >= abs(minimum):
#         enrichment_score = maximum
#         peak_index = running_score.argmax()
#     else:
#         enrichment_score = minimum
#         peak_index = running_score.argmin()

#     ranked_go_genes = [
#         gene
#         for gene in ranked_genes
#         if gene in go_genes_present
#     ]

#     return {
#         "enrichment_score": float(enrichment_score),
#         "go_genes_present": go_genes_present,
#         "ranked_go_genes": ranked_go_genes,
#         "n_go_genes_present": len(go_genes_present),
#         "n_go_genes_total": len(set(go_genes)),
#         "peak_index": int(peak_index),
#         "running_score": running_score,
#         "ranked_genes": ranked_genes,
#     }    
