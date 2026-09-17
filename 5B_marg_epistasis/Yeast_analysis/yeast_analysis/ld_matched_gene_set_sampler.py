"""LD- and SNP-count-matched null gene-set sampling."""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import os
import random
from collections.abc import Collection, Iterable, Mapping
from typing import Any

import torch
from tqdm.auto import tqdm


_SAMPLING_WORKER: Any = None
_SAMPLING_WORKER_KWARGS: dict[str, Any] | None = None


def _initialize_sampling_worker(
    sampler: Any,
    sample_kwargs: dict[str, Any],
) -> None:
    """Install one sampler in each process and prevent CPU oversubscription."""
    global _SAMPLING_WORKER, _SAMPLING_WORKER_KWARGS
    _SAMPLING_WORKER = sampler
    _SAMPLING_WORKER_KWARGS = sample_kwargs
    torch.set_num_threads(1)


def _sample_one_in_worker(seed: int) -> Any:
    """Generate one reproducible null set inside a worker process."""
    if _SAMPLING_WORKER is None or _SAMPLING_WORKER_KWARGS is None:
        raise RuntimeError("Sampling worker was not initialized")
    _SAMPLING_WORKER.rng = random.Random(seed)
    return _SAMPLING_WORKER.sample(**_SAMPLING_WORKER_KWARGS)


class LDMatchedGeneSetSampler:
    """Sample null gene sets matched on SNP count and internal LD.

    Gene-level LD is defined as squared Pearson correlation between
    representative gene-genotype vectors. Each representative vector is
    constructed by standardizing a gene's polymorphic SNPs, aligning their
    allele orientations, averaging them, and standardizing the result.

    The full gene-by-gene LD matrix is calculated once during construction or
    loaded from a cache. Subsequent sampling uses matrix indexing only.
    """

    def __init__(
        self,
        data: Any,
        gene_snps: Mapping[str, Iterable[int]] | None = None,
        seed: int = 42,
        ld_block_size: int = 512,
        ld_storage_dtype: torch.dtype = torch.float16,
        ld_storage_device: str | torch.device = "cpu",
        ld_cache_path: str | os.PathLike | None = None,
        eps: float = 1e-8,
        verbose: bool = True,
        *,
        snp_info: Any = None,
        gene_standard_name: bool = False,
    ) -> None:
        """Initialize the sampler and calculate or load gene-level LD.

        Parameters
        ----------
        data
            Object with a ``geno_t`` tensor of shape ``(N, L)``.
        gene_snps
            Mapping from gene names to SNP-column indices in ``data.geno_t``.
            Supply exactly one of gene_snps or snp_info.
        snp_info
            DataFrame or CSV path with one row per genotype SNP column, in
            exactly the same order. Row positions (not index labels) become
            SNP indices. Missing/blank gene names are skipped. Repeated names
            are combined into one gene, including names at multiple loci.
        gene_standard_name
            False selects the 'gene_name' column (default).
            True selects the 'gene_standard_name' column.
        seed
            Random seed used for null-set sampling.
        ld_block_size
            Number of LD-matrix rows calculated at a time.
        ld_storage_dtype
            Data type used to store the precomputed LD matrix.
        ld_storage_device
            Device on which the finished LD matrix is stored.
        ld_cache_path
            Optional path at which the LD matrix is saved or reloaded.
        eps
            Numerical tolerance used to identify variable SNPs.
        verbose
            Print precomputation progress when True.
        """
        if not hasattr(data, "geno_t"):
            raise AttributeError("data must contain a geno_t tensor")

        self.genotypes = data.geno_t
        if not isinstance(self.genotypes, torch.Tensor):
            raise TypeError("data.geno_t must be a torch.Tensor")
        if self.genotypes.ndim != 2:
            raise ValueError("data.geno_t must have shape (N, L)")
        if ld_block_size <= 0:
            raise ValueError("ld_block_size must be positive")

        self.N, self.L = self.genotypes.shape
        if (gene_snps is None) == (snp_info is None):
            raise ValueError("Supply exactly one of gene_snps or snp_info")
        if not isinstance(gene_standard_name, bool):
            raise TypeError("gene_standard_name must be a bool")
        gene_col = "gene_standard_name" if gene_standard_name else "gene_name"
        self.gene_col = gene_col if snp_info is not None else None
        if snp_info is not None:
            import pandas as pd

            if isinstance(snp_info, (str, os.PathLike)):
                metadata = pd.read_csv(snp_info)
            elif isinstance(snp_info, pd.DataFrame):
                metadata = snp_info
            else:
                raise TypeError("snp_info must be a DataFrame or CSV path")
            if gene_col not in metadata.columns:
                raise ValueError(f"snp_info is missing column {gene_col!r}")
            if len(metadata) != self.L:
                raise ValueError(
                    f"snp_info has {len(metadata)} rows but genotypes have "
                    f"{self.L} SNP columns; filter and order them identically"
                )
            gene_snps = {}
            for position, name in enumerate(metadata[gene_col]):
                if pd.isna(name):
                    continue
                name = str(name).strip()
                if name:
                    gene_snps.setdefault(name, []).append(position)
        self.compute_device = self.genotypes.device
        self.ld_storage_device = torch.device(ld_storage_device)
        self.ld_storage_dtype = ld_storage_dtype
        self.ld_block_size = ld_block_size
        self.ld_cache_path = (
            os.fspath(ld_cache_path) if ld_cache_path is not None else None
        )
        self.eps = eps
        self.verbose = verbose
        self.rng = random.Random(seed)

        self.gene_snps: dict[str, tuple[int, ...]] = {}
        for gene, indices_iterable in gene_snps.items():
            indices = tuple(int(index) for index in indices_iterable)
            if not indices:
                continue
            if min(indices) < 0 or max(indices) >= self.L:
                raise IndexError(
                    f"SNP indices for {gene!r} fall outside [0, {self.L - 1}]"
                )
            self.gene_snps[gene] = indices

        if not self.gene_snps:
            raise ValueError("gene_snps contains no genes with SNP indices")

        self.n_snps = {
            gene: len(indices) for gene, indices in self.gene_snps.items()
        }

        if self.ld_cache_path and os.path.exists(self.ld_cache_path):
            self._load_ld(self.ld_cache_path)
        else:
            self._precompute_gene_ld()
            if self.ld_cache_path:
                self.save_ld(self.ld_cache_path)

    @torch.no_grad()
    def _make_gene_vector(self, gene: str) -> torch.Tensor:
        """Construct one standardized representative vector for a gene."""
        indices = torch.as_tensor(
            self.gene_snps[gene],
            dtype=torch.long,
            device=self.compute_device,
        )
        genotypes = self.genotypes[:, indices].float()
        means = genotypes.mean(dim=0, keepdim=True)
        standard_deviations = genotypes.std(
            dim=0, correction=1, keepdim=True
        )
        polymorphic = standard_deviations.squeeze(0) > self.eps

        if not polymorphic.any().item():
            raise ValueError(f"{gene!r} has no polymorphic SNPs")

        standardized = (
            genotypes[:, polymorphic] - means[:, polymorphic]
        ) / standard_deviations[:, polymorphic]

        anchor = standardized[:, 0]
        correlations = standardized.T @ anchor / (self.N - 1)
        signs = torch.where(correlations >= 0, 1.0, -1.0)
        vector = (standardized * signs).mean(dim=1)
        vector = vector - vector.mean()
        vector_standard_deviation = vector.std(correction=1)

        if vector_standard_deviation <= self.eps:
            raise ValueError(
                f"Could not construct a variable genotype vector for {gene!r}"
            )
        return vector / vector_standard_deviation

    @torch.no_grad()
    def _precompute_gene_ld(self) -> None:
        """Precompute and store the complete gene-by-gene LD matrix."""
        valid_genes: list[str] = []
        vectors: list[torch.Tensor] = []
        total = len(self.gene_snps)

        for index, gene in enumerate(self.gene_snps, start=1):
            try:
                vectors.append(self._make_gene_vector(gene))
                valid_genes.append(gene)
            except ValueError:
                continue

            if self.verbose and index % 500 == 0:
                print(f"Constructed gene vectors: {index}/{total}")

        if not valid_genes:
            raise RuntimeError("No valid gene genotype vectors were constructed")

        vector_matrix = torch.stack(vectors, dim=1)
        del vectors

        self.genes = valid_genes
        self.gene_to_index = {
            gene: index for index, gene in enumerate(self.genes)
        }
        number_of_genes = len(self.genes)
        self.gene_ld_matrix = torch.empty(
            (number_of_genes, number_of_genes),
            dtype=self.ld_storage_dtype,
            device=self.ld_storage_device,
        )

        for start in range(0, number_of_genes, self.ld_block_size):
            end = min(start + self.ld_block_size, number_of_genes)
            correlations = (
                vector_matrix[:, start:end].T @ vector_matrix
            ) / (self.N - 1)
            ld_block = correlations.square().clamp(0, 1)
            self.gene_ld_matrix[start:end] = ld_block.to(
                device=self.ld_storage_device,
                dtype=self.ld_storage_dtype,
            )
            if self.verbose:
                print(f"Calculated LD rows: {end}/{number_of_genes}")

        # Averaging removes small blockwise/low-precision asymmetries.
        self.gene_ld_matrix = (
            self.gene_ld_matrix + self.gene_ld_matrix.T
        ) / 2
        self.gene_ld_matrix.fill_diagonal_(1)

        del vector_matrix
        if self.compute_device.type == "cuda":
            torch.cuda.empty_cache()

        if self.verbose:
            memory_mb = (
                self.gene_ld_matrix.numel()
                * self.gene_ld_matrix.element_size()
                / 1024**2
            )
            print(
                f"Stored {number_of_genes} x {number_of_genes} gene LD "
                f"matrix ({memory_mb:.1f} MB)"
            )

    def save_ld(self, path: str | os.PathLike) -> None:
        """Save the precomputed LD matrix and its gene metadata."""
        path_string = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path_string))
        os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "n_samples": self.N,
                "n_loci": self.L,
                "genes": self.genes,
                "n_snps": {
                    gene: self.n_snps[gene] for gene in self.genes
                },
                "gene_ld_matrix": self.gene_ld_matrix.cpu(),
            },
            path_string,
        )

    def _load_ld(self, path: str | os.PathLike) -> None:
        """Load a previously calculated gene-level LD matrix."""
        payload = torch.load(
            os.fspath(path), map_location=self.ld_storage_device
        )
        if payload["n_samples"] != self.N:
            raise ValueError("Cached LD matrix has a different sample count")
        if payload["n_loci"] != self.L:
            raise ValueError("Cached LD matrix has a different locus count")

        self.genes = payload["genes"]
        for gene in self.genes:
            if gene not in self.gene_snps:
                raise ValueError(
                    f"Cached gene {gene!r} is absent from supplied gene_snps"
                )
            if payload["n_snps"][gene] != self.n_snps[gene]:
                raise ValueError(f"SNP count changed for {gene!r}")

        self.gene_to_index = {
            gene: index for index, gene in enumerate(self.genes)
        }
        self.gene_ld_matrix = payload["gene_ld_matrix"].to(
            device=self.ld_storage_device,
            dtype=self.ld_storage_dtype,
        )
        if self.verbose:
            print(f"Loaded LD for {len(self.genes)} genes from {path}")

    def gene_ld(self, gene_a: str, gene_b: str) -> float:
        """Return precomputed r-squared between two genes."""
        row = self.gene_to_index[gene_a]
        column = self.gene_to_index[gene_b]
        return float(self.gene_ld_matrix[row, column])

    def ld_matrix(self, genes: Collection[str]) -> torch.Tensor:
        """Retrieve the precomputed LD submatrix for a gene collection."""
        gene_list = list(genes)
        indices = torch.as_tensor(
            [self.gene_to_index[gene] for gene in gene_list],
            dtype=torch.long,
            device=self.ld_storage_device,
        )
        return self.gene_ld_matrix[
            indices[:, None], indices[None, :]
        ].float()

    @staticmethod
    def _upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
        number_of_genes = matrix.shape[0]
        if number_of_genes < 2:
            return torch.empty(0, dtype=matrix.dtype, device=matrix.device)
        rows, columns = torch.triu_indices(
            number_of_genes,
            number_of_genes,
            offset=1,
            device=matrix.device,
        )
        return matrix[rows, columns]

    def _candidate_pool(
        self,
        target_gene: str,
        excluded: set[str],
        maximum_candidates: int,
        pool_multiplier: int = 4,
    ) -> list[str]:
        """Return a randomized pool of genes close in log SNP count."""
        target_size = math.log1p(self.n_snps[target_gene])
        candidates = [gene for gene in self.genes if gene not in excluded]
        if not candidates:
            raise RuntimeError("No unused candidate genes remain")

        candidates.sort(
            key=lambda gene: abs(
                math.log1p(self.n_snps[gene]) - target_size
            )
        )
        broad_pool = candidates[
            : min(len(candidates), maximum_candidates * pool_multiplier)
        ]
        if len(broad_pool) <= maximum_candidates:
            return broad_pool
        return self.rng.sample(broad_pool, maximum_candidates)

    def _construct_candidate_set(
        self,
        target_genes: list[str],
        target_ld: torch.Tensor,
        maximum_candidates: int,
        choice_top_k: int,
        size_weight: float,
        ld_weight: float,
        excluded: set[str],
    ) -> list[str]:
        """Sequentially construct one candidate matched gene set."""
        number_of_genes = len(target_genes)
        if number_of_genes > 1:
            target_burden = (
                target_ld.sum(dim=1) - 1
            ) / (number_of_genes - 1)
            order = torch.argsort(
                target_burden, descending=True
            ).tolist()
        else:
            order = [0]

        ordered_targets = [target_genes[index] for index in order]
        ordered_target_ld = target_ld[order][:, order]
        selected: list[str] = []
        selected_indices: list[int] = []

        for position, target_gene in enumerate(ordered_targets):
            candidates = self._candidate_pool(
                target_gene,
                excluded | set(selected),
                maximum_candidates,
            )
            candidate_indices = torch.as_tensor(
                [self.gene_to_index[gene] for gene in candidates],
                dtype=torch.long,
                device=self.ld_storage_device,
            )
            target_size = math.log1p(self.n_snps[target_gene])
            candidate_sizes = torch.tensor(
                [math.log1p(self.n_snps[gene]) for gene in candidates],
                dtype=torch.float32,
                device=self.ld_storage_device,
            )
            size_loss = torch.abs(candidate_sizes - target_size)

            if position == 0:
                ld_loss = torch.zeros_like(size_loss)
            else:
                previous_indices = torch.as_tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=self.ld_storage_device,
                )
                candidate_ld = self.gene_ld_matrix[
                    candidate_indices[:, None], previous_indices[None, :]
                ].float()
                desired_ld = ordered_target_ld[
                    position, :position
                ].to(self.ld_storage_device)
                ld_loss = (
                    candidate_ld - desired_ld.unsqueeze(0)
                ).square().mean(dim=1)

            loss = size_weight * size_loss + ld_weight * ld_loss
            number_of_choices = min(choice_top_k, len(candidates))
            best_positions = torch.topk(
                loss, k=number_of_choices, largest=False
            ).indices.tolist()
            chosen_position = self.rng.choice(best_positions)
            chosen_gene = candidates[chosen_position]
            selected.append(chosen_gene)
            selected_indices.append(self.gene_to_index[chosen_gene])

        return selected

    def _matching_diagnostics(
        self,
        target_genes: list[str],
        target_ld: torch.Tensor,
        null_genes: list[str],
        null_ld: torch.Tensor,
        size_weight: float,
        ld_distribution_weight: float,
        ld_topology_weight: float,
    ) -> dict[str, Any]:
        """Calculate matching losses and summary diagnostics."""
        target_sizes = torch.tensor(
            sorted(math.log1p(self.n_snps[gene]) for gene in target_genes)
        )
        null_sizes = torch.tensor(
            sorted(math.log1p(self.n_snps[gene]) for gene in null_genes)
        )
        size_loss = torch.abs(target_sizes - null_sizes).mean()

        if len(target_genes) > 1:
            target_pairwise = self._upper_triangle(target_ld).sort().values
            null_pairwise = self._upper_triangle(null_ld).sort().values
            ld_distribution_loss = (
                target_pairwise - null_pairwise
            ).square().mean()
            target_burden = (
                (target_ld.sum(dim=1) - 1) / (len(target_genes) - 1)
            ).sort().values
            null_burden = (
                (null_ld.sum(dim=1) - 1) / (len(null_genes) - 1)
            ).sort().values
            ld_topology_loss = (
                target_burden - null_burden
            ).square().mean()
            target_mean_ld = target_pairwise.mean().item()
            null_mean_ld = null_pairwise.mean().item()
        else:
            ld_distribution_loss = torch.tensor(0.0)
            ld_topology_loss = torch.tensor(0.0)
            target_mean_ld = None
            null_mean_ld = None

        total = (
            size_weight * size_loss
            + ld_distribution_weight * ld_distribution_loss
            + ld_topology_weight * ld_topology_loss
        )
        return {
            "total": total.item(),
            "snp_count": size_loss.item(),
            "ld_distribution": ld_distribution_loss.item(),
            "ld_topology": ld_topology_loss.item(),
            "n_genes": len(null_genes),
            "target_snp_counts": sorted(
                self.n_snps[gene] for gene in target_genes
            ),
            "null_snp_counts": sorted(
                self.n_snps[gene] for gene in null_genes
            ),
            "target_mean_ld": target_mean_ld,
            "null_mean_ld": null_mean_ld,
        }

    def sample(
        self,
        gene_set: Collection[str],
        n_restarts: int = 50,
        maximum_candidates: int = 256,
        choice_top_k: int = 5,
        size_weight: float = 2.0,
        ld_weight: float = 15.0,
        ld_distribution_weight: float = 15.0,
        ld_topology_weight: float = 10.0,
        exclude_input: bool = True,
        return_diagnostics: bool = False,
    ) -> set[str] | tuple[set[str], dict[str, Any]]:
        """Return one LD- and SNP-count-matched null gene set."""
        if n_restarts <= 0:
            raise ValueError("n_restarts must be positive")
        if maximum_candidates <= 0:
            raise ValueError("maximum_candidates must be positive")
        if choice_top_k <= 0:
            raise ValueError("choice_top_k must be positive")

        target_genes = list(dict.fromkeys(gene_set))
        if not target_genes:
            raise ValueError("gene_set cannot be empty")

        missing = [
            gene for gene in target_genes if gene not in self.gene_to_index
        ]
        if missing:
            raise KeyError(
                "Genes unavailable in the LD matrix: " + ", ".join(missing)
            )

        excluded = set(target_genes) if exclude_input else set()

        # LD is undefined for a one-gene set, so only SNP count is matched.
        if len(target_genes) == 1:
            target_gene = target_genes[0]
            candidates = self._candidate_pool(
                target_gene, excluded, maximum_candidates
            )
            target_size = math.log1p(self.n_snps[target_gene])
            differences = [
                abs(math.log1p(self.n_snps[gene]) - target_size)
                for gene in candidates
            ]
            minimum = min(differences)
            best_candidates = [
                gene
                for gene, difference in zip(candidates, differences)
                if difference == minimum
            ]
            null_gene = self.rng.choice(best_candidates)
            result = {null_gene}
            if not return_diagnostics:
                return result
            return result, {
                "total": minimum,
                "snp_count": minimum,
                "ld_distribution": 0.0,
                "ld_topology": 0.0,
                "n_genes": 1,
                "target_snp_counts": [self.n_snps[target_gene]],
                "null_snp_counts": [self.n_snps[null_gene]],
                "target_mean_ld": None,
                "null_mean_ld": None,
            }

        target_ld = self.ld_matrix(target_genes)
        best_genes: list[str] | None = None
        best_diagnostics: dict[str, Any] | None = None

        for _ in range(n_restarts):
            null_genes = self._construct_candidate_set(
                target_genes=target_genes,
                target_ld=target_ld,
                maximum_candidates=maximum_candidates,
                choice_top_k=choice_top_k,
                size_weight=size_weight,
                ld_weight=ld_weight,
                excluded=excluded,
            )
            null_ld = self.ld_matrix(null_genes)
            diagnostics = self._matching_diagnostics(
                target_genes=target_genes,
                target_ld=target_ld,
                null_genes=null_genes,
                null_ld=null_ld,
                size_weight=size_weight,
                ld_distribution_weight=ld_distribution_weight,
                ld_topology_weight=ld_topology_weight,
            )
            if (
                best_diagnostics is None
                or diagnostics["total"] < best_diagnostics["total"]
            ):
                best_genes = null_genes
                best_diagnostics = diagnostics

        if best_genes is None or best_diagnostics is None:
            raise RuntimeError("Unable to construct a matched null gene set")

        result = set(best_genes)
        if return_diagnostics:
            return result, best_diagnostics
        return result

    def sample_many(
        self,
        gene_set: Collection[str],
        n_sets: int,
        n_restarts: int = 50,
        maximum_candidates: int = 256,
        choice_top_k: int = 5,
        size_weight: float = 2.0,
        ld_weight: float = 15.0,
        ld_distribution_weight: float = 15.0,
        ld_topology_weight: float = 10.0,
        exclude_input: bool = True,
        return_diagnostics: bool = False,
        show_progress: bool = True,
        n_jobs: int = 1,
        multiprocessing_start_method: str = "fork",
    ) -> list[set[str]] | tuple[list[set[str]], list[dict[str, Any]]]:
        """Return multiple LD- and SNP-count-matched null gene sets.

        This convenience method preserves the sampler's random-number stream
        across draws. Each set is sampled independently; sets are not required
        to be unique or mutually disjoint.
        """
        if n_sets <= 0:
            raise ValueError("n_sets must be positive")
        if n_jobs <= 0:
            raise ValueError("n_jobs must be positive")

        target_genes = tuple(dict.fromkeys(gene_set))
        sample_kwargs = {
            "gene_set": target_genes,
            "n_restarts": n_restarts,
            "maximum_candidates": maximum_candidates,
            "choice_top_k": choice_top_k,
            "size_weight": size_weight,
            "ld_weight": ld_weight,
            "ld_distribution_weight": ld_distribution_weight,
            "ld_topology_weight": ld_topology_weight,
            "exclude_input": exclude_input,
            "return_diagnostics": return_diagnostics,
        }

        if n_jobs == 1:
            results = (
                self.sample(**sample_kwargs)
                for _ in range(n_sets)
            )
        else:
            # Worker processes must not inherit CUDA tensors. A shallow copy
            # preserves the sampler metadata while replacing its tensors with
            # CPU versions; the caller's sampler remains unchanged.
            worker_sampler = copy.copy(self)
            worker_sampler.genotypes = self.genotypes.detach().cpu()
            worker_sampler.gene_ld_matrix = (
                self.gene_ld_matrix.detach().cpu()
            )
            worker_sampler.compute_device = torch.device("cpu")
            worker_sampler.ld_storage_device = torch.device("cpu")

            # Seeds are generated in the parent so results are reproducible and
            # independent of worker scheduling.
            seeds = [self.rng.getrandbits(64) for _ in range(n_sets)]
            context = mp.get_context(multiprocessing_start_method)
            number_of_workers = min(n_jobs, n_sets)
            pool = context.Pool(
                processes=number_of_workers,
                initializer=_initialize_sampling_worker,
                initargs=(worker_sampler, sample_kwargs),
            )
            chunksize = max(1, n_sets // (number_of_workers * 8))
            results = pool.imap(
                _sample_one_in_worker,
                seeds,
                chunksize=chunksize,
            )

        null_sets: list[set[str]] = []
        diagnostics: list[dict[str, Any]] = []

        completed = False
        try:
            iterator = tqdm(
                results,
                total=n_sets,
                desc="Sampling matched null sets",
                disable=not show_progress,
            )
            for result in iterator:
                if return_diagnostics:
                    null_set, null_diagnostics = result
                    null_sets.append(null_set)
                    diagnostics.append(null_diagnostics)
                else:
                    null_sets.append(result)
            completed = True
        finally:
            if n_jobs > 1:
                if completed:
                    pool.close()
                else:
                    pool.terminate()
                pool.join()

        if return_diagnostics:
            return null_sets, diagnostics
        return null_sets
