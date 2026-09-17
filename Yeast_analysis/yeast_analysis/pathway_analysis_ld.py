"""Pathway enrichment using a precomputed LDMatchedGeneSetSampler.

The null is an approximate, optimization-based matched gene-set distribution,
not an exact permutation test. Inspect diagnostics before interpreting p-values.
"""
import copy
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def _scalar(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().item()
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Interaction statistics must be finite scalars')
    return value


def _bh(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1)
    return out


class LDMatchedPathwayAnalysis:
    """Reuse sampler LD and gene_snps; no independent gene-name inference.

    sampler must be an LDMatchedGeneSetSampler built with SNP positions and
    ordering identical to marg. Genes unsupported by its LD matrix are removed
    from BOTH observed and null sets and reported. Gene names are deduplicated.
    Pathways with fewer than min_genes unique supported genes are discarded
    during initialization (default: 3). Discarded counts are recorded in
    discarded_pathways. Pairwise testing still requires at least two genes.
    Set download_descriptions=False to disable automatic SGD requests across
    both analysis steps. Explicit per-method settings override this default.
    """
    def __init__(self, marg, gene_sets, sampler, seed=42, descriptions=None,
                 marginals_add=None, min_genes=3, download_descriptions=True):
        if isinstance(min_genes, bool) or not isinstance(min_genes, int) or min_genes < 1:
            raise ValueError('min_genes must be a positive integer')
        self.min_genes = min_genes
        self.marg = marg
        self.sampler = sampler
        self.seed = seed
        self.descriptions = dict(descriptions or {})
        self.download_descriptions = download_descriptions
        self.marginals_add = None
        if marginals_add is not None:
            if not isinstance(marginals_add, pd.DataFrame) or 'effect' not in marginals_add:
                raise ValueError('marginals_add must be a SNP-indexed DataFrame with effect')
            if not marginals_add.index.is_unique:
                raise ValueError('marginals_add must have unique SNP index labels')
            self.marginals_add = marginals_add[['effect']].copy()
            self.marginals_add['effect'] = pd.to_numeric(self.marginals_add.effect, errors='raise')
            # Validate the entire sampling universe, not just observed pathways.
            indices = sorted({i for g in sampler.genes for i in sampler.gene_snps[g]})
            if not set(indices).issubset(self.marginals_add.index):
                raise ValueError('marginals_add is missing SNP indices in the sampler universe')
            if not np.isfinite(self.marginals_add.loc[indices, 'effect']).all():
                raise ValueError('Additive effects must be finite for every eligible SNP')
        self.pathways = []
        self.discarded_pathways = {}
        support = set(sampler.genes)
        self.genes = {}
        self.excluded = {}
        for pathway in sorted(gene_sets):
            genes = sorted(set(gene_sets[pathway]))
            supported = [g for g in genes if g in support]
            if len(supported) < min_genes:
                self.discarded_pathways[pathway] = len(supported)
                continue
            self.pathways.append(pathway)
            self.genes[pathway] = supported
            self.excluded[pathway] = [g for g in genes if g not in support]
        self.results = None
        self.pathway_marginals = None
        self.null_samples = pd.DataFrame()

    def download_pathway_descriptions(self, *, timeout=30, use_tqdm=True):
        """Fetch pathway names from SGD YeastPathways page titles.

        Existing descriptions are reused. Failures warn and remain missing,
        rather than treating an error page as a description. Returns mapping.
        This is an explicit network step; construction never downloads data.
        """
        from urllib.request import urlopen
        from urllib.parse import urlencode
        from html.parser import HTMLParser
        import warnings

        class TitleParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.inside = False
                self.parts = []
            def handle_starttag(self, tag, attrs):
                if tag.lower() == 'title':
                    self.inside = True
            def handle_endtag(self, tag):
                if tag.lower() == 'title':
                    self.inside = False
            def handle_data(self, data):
                if self.inside:
                    self.parts.append(data)

        for pathway in tqdm(self.pathways, disable=not use_tqdm, desc='SGD descriptions'):
            if self.descriptions.get(pathway):
                continue
            url = 'https://pathway.yeastgenome.org/YEAST/NEW-IMAGE?' + urlencode(
                {'object': pathway, 'type': 'PATHWAY'})
            try:
                with urlopen(url, timeout=timeout) as response:
                    parser = TitleParser()
                    parser.feed(response.read().decode('utf-8'))
                title = ' '.join(''.join(parser.parts).split())
                prefix = 'Saccharomyces cerevisiae '
                if not title.startswith(prefix) or title == prefix.strip():
                    raise ValueError(f'Unexpected SGD page title: {title!r}')
                self.descriptions[pathway] = title[len(prefix):]
            except Exception as exc:
                warnings.warn(f'Could not retrieve description for {pathway}: {exc}')
        for table in (self.pathway_marginals, self.results):
            if table is not None:
                table['description'] = table.index.map(self.descriptions)
        return dict(self.descriptions)

    def _additive(self, genes):
        """Sum/mean raw effect over unique SNPs; no extra normalization."""
        snps = sorted({i for g in genes for i in self.sampler.gene_snps[g]})
        effects = self.marginals_add.loc[snps, 'effect']
        return dict(additive_sum=float(effects.sum()),
                    additive_mean=float(effects.mean()))

    def _statistic(self, genes):
        snps = [list(self.sampler.gene_snps[g]) for g in sorted(genes)]
        with torch.no_grad():
            return _scalar(self.marg.get_pathway_interaction(snps))

    def calculate_pathway_marginals(self, *, use_tqdm=True, download_descriptions=None):
        """Calculate observed SS once, without sampling random pathways.

        Returns pathways ranked by descending mean_percent_SS; ties use pathway order.
        Recalculation clears any previous null samples and enrichment results.
        Pathways with fewer than two supported genes have undefined SS.
        n_valid_snps counts unique SNP positions in the supported genes.
        mean_percent_SS = percent_SS / n_valid_snps**2 (NaN for zero SNPs).
        download_descriptions=None inherits the constructor setting. Set False
        for offline use; supplied names are retained and missing names stay empty.
        """
        if download_descriptions is None:
            download_descriptions = self.download_descriptions
        if download_descriptions and any(not self.descriptions.get(p) for p in self.pathways):
            self.download_pathway_descriptions(use_tqdm=use_tqdm)
        with torch.no_grad():
            total = _scalar(self.marg.get_SS_k(2))
        if total <= 0:
            raise ValueError('Total pairwise SS must be positive')
        records = []
        for pathway in tqdm(self.pathways, disable=not use_tqdm):
            genes = self.genes[pathway]
            n_valid_snps = len({
                snp for gene in genes for snp in self.sampler.gene_snps[gene]
            })
            ss = self._statistic(genes) if len(genes) >= 2 else np.nan
            records.append(dict(
                pathway=pathway, n_genes=len(genes),
                excluded_genes=self.excluded[pathway],
                description=self.descriptions.get(pathway),
                status='not_sampled' if len(genes) >= 2 else 'insufficient_genes',
                SS=ss, percent_SS=100*ss/total,
                n_valid_snps=n_valid_snps,
                mean_percent_SS=(100*ss/total/n_valid_snps**2
                                 if n_valid_snps else np.nan),
            ))
            if self.marginals_add is not None:
                records[-1].update(self._additive(genes))
        self.pathway_marginals = pd.DataFrame(records, columns=[
            'pathway', 'n_genes', 'excluded_genes', 'description',
            'status', 'SS', 'percent_SS', 'n_valid_snps', 'mean_percent_SS',
        ] + (['additive_sum', 'additive_mean'] if self.marginals_add is not None else [])
        ).set_index('pathway').sort_values('mean_percent_SS', ascending=False, kind='stable')
        self.SS_total = total
        self.results = None
        self.null_samples = pd.DataFrame()
        return self.pathway_marginals.copy()

    def sample_pathway_null_distributions(self, n_samples=200, *, top_n=None,
                                         use_tqdm=True, sample_kwargs=None,
                                         download_descriptions=None):
        """Return raw SS, percent SS, null mean/SD, enrichment, p and BH q.

        sample_kwargs are forwarded to sampler.sample, e.g. n_restarts,
        maximum_candidates, choice_top_k and LD/size loss weights.
        exclude_input defaults to True (exclude the entire observed pathway).
        Each draw has distinct genes; independent draws may repeat gene sets.
        p=(1+count(null>=observed))/(n_samples+1). BH uses tested pathways only.
        Repeated runs reset a private RNG without modifying the supplied sampler.
        LD tensors are shared read-only, not recomputed or duplicated.
        Requires calculate_pathway_marginals() first; observed SS is reused.
        top_n=None tests all eligible pathways; otherwise tests the top n by mean_percent_SS.
        Unselected pathways remain in results with NaN p/q values. Sampling
        replaces previous null results. BH covers sampled pathways only;
        selecting on observed mean_percent_SS makes these exploratory, not selection-adjusted
        or full-family FDR-controlled results.
        download_descriptions=None inherits the constructor setting; False
        disables automatic downloads without removing supplied descriptions.
        """
        if self.pathway_marginals is None:
            raise ValueError('Run calculate_pathway_marginals() first')
        if download_descriptions is None:
            download_descriptions = self.download_descriptions
        if download_descriptions and any(not self.descriptions.get(p) for p in self.pathways):
            self.download_pathway_descriptions(use_tqdm=use_tqdm)
        self.pathway_marginals['description'] = self.pathway_marginals.index.map(
            self.descriptions)
        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1
        ):
            raise ValueError('top_n must be a positive integer or None')
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError('n_samples must be a positive integer')
        kwargs = dict(sample_kwargs or {})
        if {'gene_set', 'return_diagnostics'} & kwargs.keys():
            raise ValueError('Do not override gene_set or return_diagnostics')
        kwargs.setdefault('exclude_input', True)
        sampler = copy.copy(self.sampler)
        sampler.rng = random.Random(self.seed)
        total = self.SS_total
        eligible = self.pathway_marginals.loc[self.pathway_marginals.n_genes >= 2]
        ranked = eligible.sort_values('mean_percent_SS', ascending=False, kind='stable')
        selected = list(ranked.head(top_n).index if top_n is not None else ranked.index)
        records = self.pathway_marginals.drop(index=selected).reset_index().to_dict('records')
        draws = []
        for pathway in tqdm(selected, disable=not use_tqdm, desc='Sampling pathways'):
            genes = self.genes[pathway]
            row = dict(self.pathway_marginals.loc[pathway], pathway=pathway)
            observed = row['SS']
            values = []
            for draw in range(n_samples):
                null, diagnostics = sampler.sample(
                    genes, return_diagnostics=True, **kwargs
                )
                if len(null) != len(genes):
                    raise ValueError('Sampler returned incorrect null gene count')
                value = self._statistic(null)
                values.append(value)
                draws.append(dict(pathway=pathway, draw=draw, SS=value,
                                  genes=sorted(null), **diagnostics))
                if self.marginals_add is not None:
                    draws[-1].update(self._additive(null))
            values = np.asarray(values)
            mean = values.mean()
            row.update(status='tested', SS=observed, percent_SS=100*observed/total,
                       null_mean=mean, null_sd=values.std(ddof=0),
                       fold_enrichment=observed/mean if mean != 0 else np.nan,
                       p_value=(1+np.count_nonzero(values >= observed))/(n_samples+1),
                       n_samples=n_samples,
                       n_unique_null_sets=len({tuple(d['genes']) for d in draws[-n_samples:]}))
            if self.marginals_add is not None:
                for metric in ('additive_sum', 'additive_mean'):
                    null_values = np.array([d[metric] for d in draws[-n_samples:]])
                    row[metric + '_null_mean'] = null_values.mean()
                    row[metric + '_null_sd'] = null_values.std(ddof=0)
                    row[metric + '_p_value'] = (
                        1 + np.count_nonzero(null_values >= row[metric])
                    ) / (n_samples + 1)
            records.append(row)
        df = pd.DataFrame(records).reindex(columns=[
            'pathway', 'status', 'n_genes', 'excluded_genes', 'description',
            'SS', 'percent_SS', 'n_valid_snps', 'mean_percent_SS',
            'null_mean', 'null_sd', 'fold_enrichment',
            'p_value', 'n_samples', 'n_unique_null_sets',
        ] + ([metric + suffix for metric in ('additive_sum', 'additive_mean')
              for suffix in ('', '_null_mean', '_null_sd', '_p_value')]
             if self.marginals_add is not None else [])).set_index('pathway')
        df['q_value_bh'] = np.nan
        tested = df.status.eq('tested')
        if tested.any():
            df.loc[tested, 'q_value_bh'] = _bh(df.loc[tested, 'p_value'])
        if self.marginals_add is not None:
            for metric in ('additive_sum', 'additive_mean'):
                df[metric + '_q_value_bh'] = np.nan
                if tested.any():
                    df.loc[tested, metric + '_q_value_bh'] = _bh(
                        df.loc[tested, metric + '_p_value'])
        self.results = df.sort_values('p_value', kind='stable')
        self.null_samples = pd.DataFrame(draws)
        return self.results

    def run_epistatic_enrichment(self, n_samples=200, *, top_n=None,
                                 use_tqdm=True, sample_kwargs=None,
                                 download_descriptions=None):
        """Run both steps; None inherits the constructor's download setting."""
        self.calculate_pathway_marginals(
            use_tqdm=use_tqdm, download_descriptions=download_descriptions)
        return self.sample_pathway_null_distributions(
            n_samples=n_samples, top_n=top_n, use_tqdm=use_tqdm,
            sample_kwargs=sample_kwargs,
            download_descriptions=download_descriptions,
        )

    def between_pathway_interaction(self, pathway_a, pathway_b):
        """Mean cross-pathway SNP interaction, excluding self/duplicate pairs.

        Uses marg.get_subset_interaction and handles overlapping pathways.
        Unlike enrichment SS, this is averaged by distinct SNP-pair count.
        """
        a = {i for g in self.genes[pathway_a] for i in self.sampler.gene_snps[g]}
        b = {i for g in self.genes[pathway_b] for i in self.sampler.gene_snps[g]}
        seen, total = set(), 0.0
        with torch.no_grad():
            for i in sorted(a):
                for j in sorted(b):
                    key = tuple(sorted((i, j)))
                    if i == j or key in seen:
                        continue
                    seen.add(key)
                    total += _scalar(self.marg.get_subset_interaction(list(key)))
        return total / len(seen) if seen else np.nan
