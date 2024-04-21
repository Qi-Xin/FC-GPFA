import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

import pickle
import numpy as np
import GLM

class VAETransformer(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward, nl_dim, spline_basis, nfactors, nneuron_list):
        super(VAETransformer, self).__init__()
        self.d_model = d_model
        self.nt, self.nbasis = spline_basis.shape
        self.narea = len(nneuron_list)
        self.nneuron_list = nneuron_list  # this should now be a list containing neuron counts for each area
        self.nneuron_tot = sum(nneuron_list)
        self.nfactors = nfactors
        self.nl_dim = nl_dim
        self.training = True
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
        self.decoder_fc = nn.Linear(nl_dim, self.nbasis * self.narea * self.nfactors)
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30

        self.readout_matrices = nn.ModuleList([nn.Linear(nfactors, neurons) for neurons in nneuron_list])

    def encode(self, src):
        # src: ntokens x batch_size x d_model (tmn)
        # Append CLS token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(-1, src.shape[1], -1)  # Expand CLS to batch size
        src = torch.cat((cls_tokens, src), dim=0)  # Concatenate CLS token
        encoded = self.transformer_encoder(src)
        cls_encoded = encoded[0]  # Only take the output from the CLS token
        latent_params = self.to_latent(cls_encoded)
        return latent_params[:, :latent_params.size(-1)//2], latent_params[:, latent_params.size(-1)//2:]  # mu, log_var

    def sample_a_latent(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        proj = self.decoder_fc(z)  # batch_size x (nbasis * narea * nfactors)
        proj = proj.view(-1, self.narea, self.nfactors, self.nbasis)  # batch_size x narea x nfactors x nbasis **mafb**
        
        # Apply spline_basis per area
        factors = torch.einsum('mafb,tb->matf', proj, self.spline_basis)  # batch_size x narea x nt x nfactors**matf**
        
        # Prepare to collect firing rates from each area
        firing_rates_list = []
        for i_area, readout_matrix in enumerate(self.readout_matrices):
            # Extract factors for the current area
            area_factors = factors[:, i_area, :, :]  # batch_size x nt x nfactors (mtf)
            firing_rates = readout_matrix(area_factors)  # batch_size x nt x nneuron_area[i_area] (mtn)
            firing_rates_list.append(firing_rates)

        # Concatenate along a new dimension and then flatten to combine areas and neurons
        firing_rates_combined = torch.cat(firing_rates_list, dim=2)  # batch_size x nt x nneuron_tot (mtn)        
        return firing_rates_combined

    def forward(self, src):
        mu, logvar = self.encode(src)
        z = self.sample_a_latent(mu, logvar)
        return self.decode(z), z, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Poisson loss
        poisson_loss = torch.mean(torch.exp(recon_x) - x * recon_x)
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return poisson_loss + beta * kl_div

def change_temporal_resolution(spikes, num_merge):
    return [change_temporal_resolution_single(spikes[i], num_merge) for i in range(len(spikes))]

def change_temporal_resolution_single(spike_train, num_merge):
    nneuron, nt, ntrial = spike_train.shape
    new_spike_train = np.zeros((nneuron, nt//num_merge, ntrial))
    for i in range(nt//num_merge):
        new_spike_train[:,i,:] = np.sum(spike_train[:,i*num_merge:(i+1)*num_merge,:], axis=1)
    return new_spike_train
    

if __name__ == '__main__':
    ############### Hyperparameters ###############
    nl_dim, num_layers, dim_feedforward = 3, 4, 64
    num_merge = 5 # Merge 5 time bins into 1, so new temporal resolution is 5ms
    session_id = 757216464

    ############### Model and Data Setup ###############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('/home/export/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
        spikes = pickle.load(f) # spikes[i]: (nneuron, nt, ntrial)
    spikes_full = torch.tensor(np.concatenate(spikes, axis=0)).float().to(device)
    spikes_low_res = change_temporal_resolution(spikes, num_merge) # (ntoken, ntrial, nneuron)
    spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=0)).float().to(device).permute(1, 2, 0)

    narea = len(spikes)
    nneuron_list = [spikes[i].shape[0] for i in range(narea)]
    nt, ntrial = spikes[0].shape[1], spikes[0].shape[2]
    d_model = sum(nneuron_list)
    B_spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=nt, dt=1, num=10, add_constant_basis=False)
    B_spline_basis = torch.tensor(B_spline_basis).float().to(device)

    kwargs = {'d_model': d_model, 'num_layers': num_layers, 'dim_feedforward': dim_feedforward, 'nl_dim': nl_dim, 
            'spline_basis': B_spline_basis, 'nfactors': 10, 'nneuron_list': nneuron_list}
    model = VAETransformer(**kwargs)
    model.to(device)

    x = spikes_full_low_res.to(device)
    # Forward pass and loss
    firing_rate, z, mu, logvar = model(x)
    loss = model.loss_function(firing_rate, spikes_full, mu, logvar)
    print(f'Loss: {loss.item()}')
