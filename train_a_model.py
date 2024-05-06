# Train a model
import numpy as np
import socket
import utility_functions as utils
import pickle
from model_trainer import Trainer
import torch
import matplotlib.pyplot as plt

### Load data
session_id = 757216464
stimuli_name = ''
# stimuli_name = 'gabors'
npadding = 100

hostname = socket.gethostname()
if hostname[:8] == "ghidorah":
    path_prefix = '/home'
elif hostname[:6] == "wright":
    path_prefix = '/home/export'
elif hostname[:3] == "n01":
    path_prefix = '/home/export'
else:
    raise ValueError(f"Unknown host: {hostname}")
ckp_path = path_prefix+'/qix/user_data/FC-GPFA_checkpoint'
with open(path_prefix+'/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
    spikes = pickle.load(f)

spikes = [sp[:,:,-(500+npadding):-150] for sp in spikes]

params = {
    'batch_size': 64,
    'beta': 0.2,
    'decoder_architecture': 0,
    'dim_feedforward': 128,
    'dropout': 0.0,
    'weight_decay': 0.0,
    'learning_rate': 0.001,
    'learning_rate_decoder': 0.01,
    'learning_rate_cp': 0.002,
    'epoch_warm_up': 10,
    'epoch_fix_latent': 5,
    'epoch_patience': 5,
    'epoch_max': 100,
    'nfactor': 4,
    'nhead': 1,
    'nl_dim': 8,
    'num_B_spline_basis': 20,
    'num_layers': 8,
    'num_merge': 10,
    'sample_latent': False,
    'K_tau': 100,
    'K_sigma2': 1.0,
    'nsubspace': 4,
    'nlatent': 1,
    'coupling_basis_num': 3,
    'coupling_basis_peaks_max': 10.2,
    'use_self_coupling': True
    }

trainer = Trainer(spikes, ckp_path, params, npadding=npadding)
trainer.train(verbose=True)
trainer.predict(return_torch=True, dataset='test')
trainer.save_model_and_hp()

homo_fr = torch.log(trainer.spikes_full_no_padding.mean())
print(f"Homogeneous baseline loss: {torch.mean(torch.exp(homo_fr) - trainer.spikes_full_no_padding * homo_fr):.5f}")
inhomo_fr = torch.log(trainer.spikes_full_no_padding.mean(axis=(0,1)))
print(f"Inhomogeneous baseline loss: {torch.mean(torch.exp(inhomo_fr) - trainer.spikes_full_no_padding * inhomo_fr):.5f}")
firing_rate, mu, std = trainer.predict(return_torch=True, dataset='all')
print(f"My model loss: {torch.mean(torch.exp(firing_rate) - trainer.spikes_full_no_padding * firing_rate):.5f}")
firing_rate, mu, std = trainer.predict(return_torch=True, dataset='train')
print(f"My model training set loss: {torch.mean(torch.exp(firing_rate) - trainer.spikes_full_no_padding[trainer.train_idx,:,:] * firing_rate):.5f}")
firing_rate, mu, std = trainer.predict(return_torch=True, dataset='test')
print(f"My model test set loss: {torch.mean(torch.exp(firing_rate) - trainer.spikes_full_no_padding[trainer.test_idx,:,:] * firing_rate):.5f}")

print(mu.std(axis=0))# If mu for each trial are the same, then we only get trivial solution. 