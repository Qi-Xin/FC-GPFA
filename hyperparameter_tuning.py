import numpy as np
import socket
import utility_functions as utils
import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from model_trainer import Trainer

### Load data
session_id = 757216464
stimuli_name = ''
# stimuli_name = 'gabors'

hostname = socket.gethostname()
if hostname[:8] == "ghidorah":
    path_prefix = '/home'
elif hostname == 'n01':
    path_prefix = '/home/export'
else:
    print(f"Unknown host: {hostname}")
    raise ValueError("Unknown host, can't set path prefix")
ckp_path = path_prefix+'/qix/user_data/FC-GPFA_checkpoint'
with open(path_prefix+'/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
    spikes = pickle.load(f)

# Only use 0-350ms; padding 100ms for accuate coupling effects
npadding = 100
spikes = [sp[:,:,-(500+npadding):-150] for sp in spikes]

# score below is actually the best test loss achieve when training the model, so the lower the better
best_score = [float('inf')]
def try_hp(params):
    trainer = Trainer(spikes, ckp_path, params, npadding=npadding)
    try:
        score = trainer.train(verbose=False, record_results=True)
        if score < best_score[0]:
            best_score[0] = score
            print(f"New best score: {best_score[0]}")
            print(f"with params: {params}")
            trainer.save_model_and_hp()
        return {'loss': score, 'status': STATUS_OK}
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return {'loss': float('inf'), 'status': STATUS_FAIL}

# Define the search space
param_dist = {
    'num_merge': hp.choice('num_merge', [2, 5, 10, 20]),
    'batch_size': hp.choice('batch_size', [4, 8, 16, 32, 64, 128, 600]),
    'num_B_spline_basis': hp.choice('num_B_spline_basis', [10, 20, 30, 50]),
    'nl_dim': hp.choice('nl_dim', [4, 8, 16, 32]),
    'num_layers': hp.choice('num_layers', [2, 4, 8]),
    'dim_feedforward': hp.choice('dim_feedforward', [32, 64, 128, 256]),
    'nfactor': hp.choice('nfactor', [8, 16, 32, 64, 128]),
    'nhead': hp.choice('nhead', [1, 2]),
    'learning_rate': hp.choice('learning_rate', [1e-4, 1e-3, 1e-2]),
    'learning_rate_decoder': hp.choice('learning_rate_decoder', [1e-3, 1e-2, 1e-1]),
    'learning_rate_cp': hp.choice('learning_rate_cp', [1e-3, 1e-2, 1e-1]),
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.5]),
    'beta': hp.choice('beta', [0.0, 0.1, 0.4, 1.0]),
    'epoch_warm_up': hp.choice('warm_up_epoch', [5]),
    'epoch_fix_latent': hp.choice('fix_latent_epoch', [10, 20, 50, 100]),
    'epoch_max': hp.choice('max_epoch', [300]),
    'epoch_patience': hp.choice('patience_epoch', [5]),
    'sample_latent': hp.choice('sample_latent', [True, False]),
    'decoder_architecture': hp.choice('decoder_architecture', [0, 1, 2]),
    'nsubspace':hp.choice('decoder_architecture', [1, 2, 4]),
    'K_tau': hp.choice('K_tau', [50, 100, 200, 300, 400]),
    'K_sigma2': hp.choice('K_sigma2', [0.25, 1.0, 4.0, 16.0]),
    'nlatent': hp.choice('nlatent', [1, 2, 4, 8]),
    'coupling_basis_num': hp.choice('coupling_basis_num', [3, 5, 8]),
    'coupling_basis_peaks_max': hp.choice('coupling_basis_peaks_max', [5, 10.2, 15]),
}

# Create a trials object to store details of each iteration
trials = Trials()

# Perform the optimization using Bayesian optimization
best = fmin(
    fn=try_hp,
    space=param_dist,
    algo=tpe.suggest,  # Use Bayesian optimization
    max_evals=500,  # Set the maximum number of evaluations
    trials=trials
)

# Extract the best parameters found during the optimization
print(f"Best Parameters: {best.items()}")