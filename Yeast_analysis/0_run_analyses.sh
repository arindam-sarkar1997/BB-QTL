python 1_Fit_GP_model.py 

python 2_yeast_marginals_cli.py -k 1 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=0.995_MAF_threshold=0/ --device cuda:0

python 2_yeast_marginals_cli.py -k 2 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=0.995_MAF_threshold=0/ --device cuda:0

python 2_yeast_marginals_cli.py -k 3 -model_name GP_model_k=8.model --env 37C --out_path ../results/yeast_analysis/5_19/ --checkpoint_path ../model_checkpoints/37C_r2_threshold=0.995_MAF_threshold=0/ --device cuda:0