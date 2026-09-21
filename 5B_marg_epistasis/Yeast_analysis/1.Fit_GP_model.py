"""Fit GP models for 37C or a selected FLU/PUL dose."""

import argparse
import gc
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("37C", "FLU", "PUL"), default="37C")
    parser.add_argument("--dose", help="Drug dose: FLU CON/QMIC/HMIC/FMIC or PUL CON/HMIC/FMIC/DMIC")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from yeast_inputs import DRUG_CONDITIONS

    if args.env == "37C" and args.dose is not None:
        parser.error("--dose is only valid for FLU or PUL")
    if args.env in DRUG_CONDITIONS and args.dose not in DRUG_CONDITIONS[args.env]:
        parser.error(f"--dose is required for {args.env}; choose from {DRUG_CONDITIONS[args.env]}")
    return args


def main():
    args = parse_args()
    os.environ["CUDA_PATH"] = "/apps/compilers/cuda/13.2.1"

    import torch
    from epikVC.models import make_GP_model
    from vcme.utils import yeast_data
    from yeast_inputs import make_drug_yeast_data

    r2_threshold = None
    maf_threshold = 0
    top_percent = 40 if args.env == "37C" else 60
    dataset_name = args.env if args.dose is None else f"{args.env}_{args.dose}"
    checkpoint_path = Path(
        f"../model_checkpoints/{dataset_name}_r2_threshold={r2_threshold}"
        f"_MAF_threshold={maf_threshold}_top{top_percent}_percent"
    )
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    print(f"Training {dataset_name} on {args.device}; GPUs available: {torch.cuda.device_count()}")
    if args.env == "37C":
        data = yeast_data(
            "/orange/juannanzhou/MarginalEpistasis/data/",
            "/orange/juannanzhou/dryad_data/",
            args.env,
            args.device,
            prune_snps=False,
            r2_threshold=r2_threshold,
            top_percent=top_percent,
            maf_threshold=maf_threshold,
        )
    else:
        data = make_drug_yeast_data(
            args.env, args.dose, args.device,
            prune_snps=False, r2_threshold=r2_threshold, maf_threshold=maf_threshold,
        )

    train_x, train_y, test_x, test_y, train_y_var = data.train_test_split(seed=666)
    A, L = data.A, data.L
    print(f"Samples: {len(data.y)}; loci: {L}; train: {len(train_y)}; test: {len(test_y)}")

    k_max = 8
    GP = make_GP_model(train_x, train_y, A, L, L, k_max, device=args.device)
    GP.fit_model(n_steps=200, learning_rate=0.1, mll=True)
    GP.fit_model(n_steps=40, learning_rate=0.1, mll=False)
    GP.get_alpha()
    GP.draw_pos_y(num_samples=100)
    GP.get_pos_beta()
    GP.save_checkpoint(str(checkpoint_path / f"GP_model_k={k_max}_top{top_percent}percent.model"))
    del GP
    gc.collect()
    torch.cuda.empty_cache()

    k_max = 1
    GP = make_GP_model(train_x, train_y, A, L, L, k_max, device=args.device)
    GP.fit_model(n_steps=100, learning_rate=0.1, mll=True)
    GP.save_checkpoint(str(checkpoint_path / f"GP_model_k={k_max}_top{top_percent}percent.model"))


if __name__ == "__main__":
    main()
