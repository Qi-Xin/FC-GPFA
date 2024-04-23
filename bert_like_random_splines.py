import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

import pickle
import numpy as np
import GLM
from matplotlib import pyplot as plt

class VAETransformer(nn.Module):
    def __init__(self, num_layers, dim_feedforward, nl_dim, spline_basis, nfactor, nneuron_list, dropout, nhead):
        super(VAETransformer, self).__init__()
        self.nneuron_list = nneuron_list  # this should now be a list containing neuron counts for each area
        self.d_model = sum(self.nneuron_list)
        self.nt, self.nbasis = spline_basis.shape
        self.narea = len(self.nneuron_list)
        self.nneuron_tot = self.d_model
        self.nfactor = nfactor
        self.nl_dim = nl_dim
        self.nhead = nhead
        self.training = True
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(self.d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
        self.decoder_fc = nn.Linear(nl_dim, self.nbasis * self.narea * self.nfactor)
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30

        self.readout_matrices = nn.ModuleList([nn.Linear(self.nfactor, neurons) for neurons in self.nneuron_list])

    def encode(self, src):
        # src: mnt
        src = src.permute(2, 0, 1)
        
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
        proj = self.decoder_fc(z)  # batch_size x (nbasis * narea * nfactor)
        proj = proj.view(-1, self.narea, self.nfactor, self.nbasis)  # batch_size x narea x nfactor x nbasis **mafb**
        
        # Apply spline_basis per area
        factors = torch.einsum('mafb,tb->matf', proj, self.spline_basis)  # batch_size x narea x nt x nfactor**matf**
        
        # Prepare to collect firing rates from each area
        firing_rates_list = []
        for i_area, readout_matrix in enumerate(self.readout_matrices):
            # Extract factors for the current area
            area_factors = factors[:, i_area, :, :]  # batch_size x nt x nfactor (mtf)
            firing_rates = readout_matrix(area_factors)  # batch_size x nt x nneuron_area[i_area] (mtn)
            firing_rates_list.append(firing_rates)

        # Concatenate along a new dimension and then flatten to combine areas and neurons
        firing_rates_combined = torch.cat(firing_rates_list, dim=2)  # batch_size x nt x nneuron_tot (mtn)        
        return firing_rates_combined.permute(0,2,1) # mnt

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
