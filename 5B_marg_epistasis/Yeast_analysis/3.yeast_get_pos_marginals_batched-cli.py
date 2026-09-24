"""Calculate positional marginal effects for saved GP posterior beta samples."""

import argparse
import os
from pathlib import Path


# KeOps needs the cluster CUDA installation when the GPU modules are loaded.
os.environ["CUDA_PATH"] = "/apps/compilers/cuda/13.2.1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--indices", type=int, nargs="+", required=True,
        help="Zero-based posterior beta-sample indices",
    )
    parser.add_argument(
        "-k", "--order", type=int, required=True,
        help="Interaction order used for the positional marginal calculation",
    )
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Path to a GP checkpoint containing beta_samples",
    )
    parser.add_argument(
        "--out-path", type=Path, required=True,
        help="Directory for the sample-level CSV outputs",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="Dataset label used as the output filename prefix",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device (default: cuda:0)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    import torch
    from epikVC.models import load_GP_model
    from vcme.marginals import MargEpistasis

    if args.order < 1:
        raise ValueError("--order must be a positive integer")
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("Posterior sample indices must be unique")
    if not args.model_path.is_file() or args.model_path.stat().st_size == 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {args.model_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {args.device}")

    print(f"Loading {args.model_path} on {args.device}")
    gp = load_GP_model(str(args.model_path), device=args.device)
    if not hasattr(gp, "beta_samples"):
        raise ValueError(f"Checkpoint does not contain beta_samples: {args.model_path}")

    beta_samples = gp.beta_samples
    if beta_samples.ndim != 2 or len(beta_samples) == 0:
        raise ValueError(f"Expected a nonempty 2-D beta_samples tensor; got {beta_samples.shape}")

    invalid_indices = [i for i in args.indices if i < 0 or i >= len(beta_samples)]
    if invalid_indices:
        raise IndexError(
            f"Posterior sample indices {invalid_indices} are outside 0-{len(beta_samples) - 1}"
        )

    train_x, train_y = gp.genos, gp.y
    allele_count = int(gp.A)
    locus_count = int(gp.L)
    if train_x.ndim != 2 or train_x.shape[1] != allele_count * locus_count:
        raise ValueError(
            "Checkpoint genotype dimensions are inconsistent with its allele and locus counts"
        )
    if args.order > int(gp.k_max):
        raise ValueError(f"--order {args.order} exceeds checkpoint k_max={gp.k_max}")

    log_lda = gp.get_lda()
    loci = list(range(locus_count))
    args.out_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Processing posterior samples {args.indices}; order={args.order}; "
        f"loci={locus_count}; available samples={len(beta_samples)}"
    )
    for sample_index in args.indices:
        beta = beta_samples[sample_index]
        marg = MargEpistasis(
            allele_count,
            locus_count,
            train_x,
            train_y,
            beta,
            log_lda,
            chunk_size=10**10,
            beta_samples=None,
        )
        site_marginals, site_marginal_fractions = marg.get_VC_positions(args.order, loci)
        site_marginals.index.name = "SNP"
        site_marginals.name = "effect"
        site_marginal_fractions.index.name = "SNP"
        site_marginal_fractions.name = "fraction_of_order_variance"

        output_file = args.out_path / (
            f"{args.run_name}_k={args.order}_sample_{sample_index}.csv"
        )
        fraction_file = args.out_path / (
            f"{args.run_name}_k={args.order}_sample_{sample_index}_percent.csv"
        )
        output_tmp = output_file.with_suffix(".csv.tmp")
        fraction_tmp = fraction_file.with_suffix(".csv.tmp")
        site_marginals.to_csv(output_tmp)
        site_marginal_fractions.to_csv(fraction_tmp)
        output_tmp.replace(output_file)
        fraction_tmp.replace(fraction_file)
        print(f"Saved posterior sample {sample_index}: {output_file} and {fraction_file}")

    print("All requested posterior samples processed successfully.")


if __name__ == "__main__":
    main()
