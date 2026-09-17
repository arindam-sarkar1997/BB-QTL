"""Hub-locus enrichment with an exact group-size-matched permutation null."""
import numpy as np
import pandas as pd
from math import comb


def _hypergeom_tail(observed, population, successes, draws):
    denominator = comb(population, draws)
    return sum(comb(successes, x) * comb(population-successes, draws-x)
               for x in range(max(observed, 0, draws-(population-successes)),
                              min(successes, draws)+1)) / denominator


def hub_ld_enrichment(interactions, gene_info, hubs, *, sampler=None,
                      gene_groups=None, ld_threshold=0.5, top_k=20,
                      n_permutations=10000, seed=0):
    """Return (summary, selected_loci, gene_to_group).

    interactions: gene-indexed DataFrame with one column per focal hub.
    gene_info: CSV path or DataFrame with chrom and gene_standard_name.
    hubs: hub names used both as focal genes and to label hub-containing loci.
    Supply exactly one of sampler (exposing genes and gene_ld) or gene_groups
    (Series/dict mapping gene names to globally defined LD-group identifiers).

    Sampler groups are same-chromosome connected components with r2 strictly
    above ld_threshold. Transitive chaining is allowed. Unsupported genes,
    genes with ambiguous chromosomes, and rows with nonfinite scores for any
    hub are excluded from the common background. Missing focal hubs raise.
    Group size means eligible gene count, not physical length or SNP count.

    Each hub's entire group is excluded. Remaining loci are ranked by maximum
    gene score; ties use deterministic group order. Top-k includes zero scores
    if necessary (no significance/positivity filter). The null samples without
    replacement within exact gene-count strata, matching the selected size
    histogram. All eligible loci, including observed top loci, remain in the
    null pool. Empirical p-values use (1 + exceedances)/(1 + permutations).
    BH adjustment is across the supplied hubs. This is a hub-locus enrichment
    test, not a connectivity-adjusted test or a causal gene assignment.
    """
    hubs = list(hubs)
    if not hubs or len(set(hubs)) != len(hubs):
        raise ValueError('hubs must be a nonempty list of distinct names')
    for name, value in [('top_k', top_k), ('n_permutations', n_permutations)]:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if (sampler is None) == (gene_groups is None):
        raise ValueError('Supply exactly one of sampler or gene_groups')
    if not np.isfinite(ld_threshold) or not 0 <= ld_threshold <= 1:
        raise ValueError('ld_threshold must be between zero and one')
    if not interactions.index.is_unique or not interactions.columns.is_unique:
        raise ValueError('Interaction row and column names must be unique')
    scores = interactions.loc[:, hubs].apply(pd.to_numeric, errors='raise')
    scores = scores.loc[np.isfinite(scores.to_numpy()).all(axis=1)].copy()
    if (scores.to_numpy() < 0).any():
        raise ValueError('Expected nonnegative interaction strengths')
    meta = gene_info.copy() if isinstance(gene_info, pd.DataFrame) else pd.read_csv(gene_info)
    meta = meta[['gene_standard_name', 'chrom']].dropna().copy()
    def chromosome(value):
        s = str(value).strip()
        if s.lower().startswith('chr'):
            s = s[3:]
        romans = 'I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI'.split()
        if s.upper() in romans:
            return str(romans.index(s.upper()) + 1)
        try:
            return str(int(s)) if float(s).is_integer() else s
        except ValueError:
            try:
                return str(int(float(s))) if float(s).is_integer() else s
            except ValueError:
                return s
    meta['chrom'] = meta.chrom.map(chromosome)
    counts = meta.groupby('gene_standard_name').chrom.nunique()
    chrom = meta.drop_duplicates('gene_standard_name').set_index('gene_standard_name').chrom
    chrom = chrom.loc[counts[counts == 1].index]
    support = set(sampler.genes) if sampler is not None else set(pd.Series(gene_groups).dropna().index)
    genes = sorted(set(scores.index) & set(chrom.index) & support)
    missing = set(hubs) - set(genes)
    if missing:
        raise ValueError(f'Hubs unavailable in the common background: {sorted(missing)}')
    scores = scores.loc[genes]

    if sampler is not None:
        parent = list(range(len(genes)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for indices in pd.Series(range(len(genes)), index=genes).groupby(chrom.reindex(genes)).apply(list):
            for offset, i in enumerate(indices):
                for j in indices[:offset]:
                    value = float(sampler.gene_ld(genes[i], genes[j]))
                    if not np.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f'Invalid LD for {genes[i]}, {genes[j]}')
                    if value > ld_threshold:
                        parent[root(i)] = root(j)
        mapping = pd.Series([root(i) for i in range(len(genes))], index=genes)
    else:
        mapping = pd.Series(gene_groups)
        if not mapping.index.is_unique:
            raise ValueError('gene_groups must have unique gene names')
        mapping = mapping.reindex(genes)
    # Stable integer identifiers simplify grouping independent of caller labels.
    mapping = pd.Series(pd.factorize(mapping, sort=False)[0], index=genes, name='ld_group')
    if chrom.reindex(genes).groupby(mapping).nunique().gt(1).any():
        raise ValueError('An LD group spans multiple chromosomes')
    sizes = mapping.value_counts().sort_index()
    hub_groups = set(mapping.loc[hubs])
    rng = np.random.default_rng(seed)
    summaries, selected_tables = [], []
    for hub in hubs:
        locus = scores[hub].groupby(mapping).max().to_frame('strength')
        locus['n_genes'] = sizes
        locus['has_hub'] = locus.index.isin(hub_groups)
        locus = locus.drop(mapping[hub]).sort_values('strength', ascending=False, kind='stable')
        if len(locus) < top_k:
            raise ValueError(f'{hub}: only {len(locus)} eligible loci, fewer than top_k={top_k}')
        selected = locus.head(top_k).copy()
        observed = int(selected.has_hub.sum())
        null = np.zeros(n_permutations, dtype=int)
        expected = 0.0
        # A hypergeometric draw exactly reproduces the count from uniform
        # sampling of loci without replacement within each size stratum.
        for size, number in selected.n_genes.value_counts().items():
            pool = locus[locus.n_genes == size]
            successes = int(pool.has_hub.sum())
            expected += number * successes / len(pool)
            null += rng.hypergeometric(successes, len(pool)-successes,
                                       int(number), size=n_permutations)
        summaries.append(dict(hub=hub, eligible_genes=len(genes)-int(sizes[mapping[hub]]),
                              excluded_genes=len(interactions)-len(genes),
                              eligible_loci=len(locus), hub_loci=int(locus.has_hub.sum()),
                              selected_loci=top_k, observed_hub_loci=observed,
                              expected_matched=expected,
                              fold_enrichment=observed/expected if expected else np.nan,
                              p_matched=(1+int((null >= observed).sum()))/(n_permutations+1),
                              null_sd=float(null.std()),
                              p_hypergeom=_hypergeom_tail(observed, len(locus), int(locus.has_hub.sum()), top_k)))
        selected['hub'] = hub
        selected['genes'] = [mapping.index[mapping == g].tolist() for g in selected.index]
        selected['hub_genes'] = [[h for h in hubs if mapping[h] == g] for g in selected.index]
        selected['peak_gene'] = [scores.loc[mapping == g, hub].idxmax() for g in selected.index]
        selected_tables.append(selected.reset_index())
    summary = pd.DataFrame(summaries)
    order = np.argsort(summary.p_matched.to_numpy())
    adjusted = np.minimum.accumulate((summary.p_matched.to_numpy()[order] * len(hubs)
                                     / np.arange(1, len(hubs)+1))[::-1])[::-1]
    q = np.empty(len(hubs)); q[order] = np.minimum(adjusted, 1)
    summary['q_matched_bh'] = q
    return summary, pd.concat(selected_tables, ignore_index=True), mapping
