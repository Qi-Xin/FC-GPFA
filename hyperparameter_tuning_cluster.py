# parallel_hyperopt.py
import os
import numpy as np
import joblib
import socket
import pickle
import sys
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL
from model_trainer import Trainer
import utility_functions as utils

if sys.platform == 'linux':
    # Set paths
    hostname = socket.gethostname()
    if hostname[:8] == "ghidorah":
        path_prefix = '/home'
    elif hostname[:6] == "wright":
        path_prefix = '/home/export'
    elif hostname[:3] in ["n01", "n02", "n03"]:
        path_prefix = '/home/export'
    else:
        raise ValueError(f"Unknown host: {hostname}")
    data_path = path_prefix + '/qix/user_data/allen_spike_trains/single_sessions.joblib'
    ckp_path = path_prefix + '/qix/user_data/VAETransformer_checkpoint_hp_tuning_cluster'
else:
    data_path = 'D:/ecephys_cache_dir/single_sessions.joblib'
    ckp_path = 'D:/ecephys_cache_dir/VAETransformer_checkpoint_hp_tuning'

data_to_use = joblib.load(data_path)

# Global best score holder
best_score = [float('inf')]

# Trial function, with GPU assignment
def try_hp_on_gpu(params, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    trainer = Trainer(data_to_use, ckp_path, params)
    try:
        trainer.train(include_stimulus=True, include_coupling=False, include_self_history=False, fix_stimulus=True, fix_latents=True, verbose=False)
        trainer.make_optimizer(frozen_params=['sti_inhomo'])
        trainer.train(include_stimulus=True, include_coupling=False, include_self_history=False, fix_stimulus=False, fix_latents=True, verbose=False)
        trainer.make_optimizer(frozen_params=['transformer_encoder', 'to_latent', 'token_converter'])
        score = trainer.train(include_stimulus=True, include_coupling=True, include_self_history=False, fix_stimulus=False, fix_latents=True, verbose=False, record_results=True)
        trainer.save_model_and_hp(filename=None, test_loss=score)

        if score < best_score[0]:
            best_score[0] = score
            print(f"New best score: {best_score[0]}")
            print(f"with params: {params}")
            trainer.save_model_and_hp()

        return {'loss': score, 'status': STATUS_OK}

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        return {'loss': float('inf'), 'status': STATUS_FAIL}

# Define your hyperparameter search space
param_dist = {
    'num_B_spline_basis': hp.choice('num_B_spline_basis', [10, 30, 50]),
    'downsample_factor': hp.choice('downsample_factor', [5, 10]),
    'transformer_num_layers': hp.choice('transformer_num_layers', [1, 2, 4]),
    'transformer_d_model': hp.choice('transformer_d_model', [64, 128, 256]),
    'transformer_dim_feedforward': hp.choice('transformer_dim_feedforward', [256, 512, 1024]),
    'transformer_vae_output_dim': hp.choice('transformer_vae_output_dim', [12, 24, 48, 96]),
    'transformer_dropout': hp.choice('transformer_dropout', [0.0, 0.2]),
    'transformer_nhead': hp.choice('transformer_nhead', [1, 2, 4]),
    'stimulus_nfactor': hp.choice('stimulus_nfactor', [1, 2, 4]),
    'stimulus_decoder_inter_dim_factor': hp.choice('stimulus_decoder_inter_dim_factor', [1, 2, 4]),
    'beta': hp.choice('beta', [0.0, 0.3, 1.0]),
    'use_area_specific_decoder': hp.choice('use_area_specific_decoder', [True]),
    'use_area_specific_encoder': hp.choice('use_area_specific_encoder', [True]),
    'use_cls': hp.choice('use_cls', [False]),
    'coupling_basis_peaks_max': hp.choice('coupling_basis_peaks_max', [3, 5, 10]),
    'coupling_basis_num': hp.choice('coupling_basis_num', [3, 5]),
    'coupling_nsubspace': hp.choice('coupling_nsubspace', [1, 2]),
    'use_self_coupling': hp.choice('use_self_coupling', [True]),
    'K_sigma2': hp.choice('K_sigma2', [1.0]),
    'K_tau': hp.choice('K_tau', [100]),
    'coupling_strength_nlatent': hp.choice('coupling_strength_nlatent', [1]),
    'self_history_basis_peaks_max': hp.choice('self_history_basis_peaks_max', [1.5]),
    'self_history_basis_num': hp.choice('self_history_basis_num', [3]),
    'self_history_basis_nonlinear': hp.choice('self_history_basis_nonlinear', [1]),
    'penalty_smoothing_spline': hp.choice('penalty_smoothing_spline', [1e2, 1e3, 1e4]),
    'penalty_coupling_subgroup': hp.choice('penalty_coupling_subgroup', [1e-6, 1e-5, 1e-4, 1e-3]),
    'penalty_diff_loading': hp.choice('penalty_diff_loading', [None]),
    'penalty_loading_similarity': hp.choice('penalty_loading_similarity', [1e-5, 1e-4, 1e-3, 1e-2]),
    'batch_size': hp.choice('batch_size', [32, 64, 128]),
    'sample_latent': hp.choice('sample_latent', [True, False]),
    'lr': hp.choice('lr', [1e-3]),
    'lr_transformer': hp.choice('lr_transformer', [1e-4, 3e-4, 1e-3]),
    'lr_sti': hp.choice('lr_sti', [1e-3, 3e-3, 1e-2]),
    'lr_cp': hp.choice('lr_cp', [1e-3, 3e-3, 1e-2]),
    'lr_self_history': hp.choice('lr_self_history', [1e-2]),
    'epoch_warm_up': hp.choice('epoch_warm_up', [0]),
    'epoch_patience': hp.choice('epoch_patience', [3]),
    'epoch_max': hp.choice('epoch_max', [50]),
    'tol': hp.choice('tol', [1e-5]),
    'weight_decay': hp.choice('weight_decay', [0.0, 1e-5, 1e-4, 1e-3]),
}

# Parallel trial launcher
def run_parallel_trials(n_trials, gpu_ids):
    from hyperopt import rand, space_eval

    def run_single_trial(i):
        params = space_eval(param_dist, {k: rand.suggest(new_id=k, domain=v) for k, v in param_dist.items()})
        gpu_id = gpu_ids[i % len(gpu_ids)]
        return try_hp_on_gpu(params, gpu_id)

    results = joblib.Parallel(n_jobs=len(gpu_ids))(
        joblib.delayed(run_single_trial)(i) for i in range(n_trials)
    )
    return results

if __name__ == '__main__':
    gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]  # Customize as needed
    n_trials = 100  # Total number of evaluations
    run_parallel_trials(n_trials, gpu_ids)
