import os
os.chdir('/blue/juannanzhou/juannanzhou/vcme_project/Yeast_analysis')

from epikVC.models import GPModel, make_GP_model, load_GP_model

checkpoint_path = '../model_checkpoints/37C_r2_threshold=None_MAF_threshold=0_top40_percent/'
geno_path = "/orange/juannanzhou/MarginalEpistasis/data/"
pheno_path = "/orange/juannanzhou/dryad_data/"
out_path = '../results/yeast_analysis/' 
pheno_name = '37C'
model_name = 'GP_model_k=8_top40percent.model'

GP = load_GP_model(checkpoint_path + model_name)
train_x, train_y = GP.genos, GP.y
log_lda = GP.get_lda()
A, L = 2, int(train_x.shape[1]/2)
print(f'L = {L}')
loci_candidate = list(range(L))

# Prepend alpha to beta_samples
beta_samples = torch.cat((GP.alpha.unsqueeze(0), GP.beta_samples), dim=0)

print(beta_samples.shape)