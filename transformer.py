import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

import GLM

class VAETransformer(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward, nl_dim, spline_basis, n_factors, n_neuron_per_area):
        super(VAETransformer, self).__init__()
        self.d_model = d_model
        self.nt, self.n_basis = spline_basis.shape
        self.n_area = len(n_neuron_per_area)
        self.n_neuron_per_area = n_neuron_per_area  # this should now be a list containing neuron counts for each area
        self.n_neuron = sum(n_neuron_per_area)
        self.n_factor = n_factors
        self.nl_dim = nl_dim
        self.training = True
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
        self.decoder_fc = nn.Linear(nl_dim, 30 * n_area * n_factors)
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30

        self.readout_matrices = nn.ModuleList([nn.Linear(n_factors, neurons) for neurons in n_neuron_per_area])

    def encode(self, src):
        # Append CLS token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(-1, src.size(1), -1)  # Expand CLS to batch size
        src = torch.cat((cls_tokens, src), dim=0)  # Concatenate CLS token
        encoded = self.transformer_encoder(src)
        cls_encoded = encoded[0]  # Only take the output from the CLS token
        latent_params = self.to_latent(cls_encoded)
        return latent_params[:, :latent_params.size(-1)//2], latent_params[:, latent_params.size(-1)//2:]  # mu, log_var

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        proj = self.decoder_fc(z)  # batch_size x (n_basis * n_area * n_factors)
        proj = proj.view(-1, self.n_area, self.n_factors, self.n_basis)  # batch_size x n_area x n_factors x n_basis
        
        # Apply spline_basis per area
        factors = torch.einsum('mafb,tb->maft', proj, self.spline_basis)  # batch_size x n_area x n_factors x nt

        # Prepare to collect firing rates from each area
        all_firing_rates = []
        for area_idx, readout_matrix in enumerate(self.readout_matrices):
            # Extract factors for the current area
            area_factors = factors[:, area_idx, :, :]  # batch_size x n_factors x nt (mft)
            firing_rate = readout_matrix(area_factors)  # batch_size x n_neuron_area[area_idx] x nt
            all_firing_rates.append(firing_rate.permute(0, 2, 1).unsqueeze(1))  # batch_size x 1 x nt x n_neuron_area

        # Concatenate along a new dimension and then flatten to combine areas and neurons
        firing_rate_combined = torch.cat(all_firing_rates, dim=1)  # batch_size x n_area x nt x n_neuron_area
        firing_rate_combined = firing_rate_combined.reshape(-1, self.n_area * sum(self.n_neuron_area), firing_rate_combined.size(2))
        # batch_size x (n_area * total_neurons) x nt
        
        return firing_rate_combined

    def forward(self, src):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Poisson loss
        poisson_loss = torch.mean(torch.exp(recon_x) - x * recon_x)
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return poisson_loss + beta * kl_div

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nt, n_neuron, nl_dim, n_area = 100, 5, 3, 5
input_dim, num_layers, hidden_dim = 10, 4, 64


spline_basis = GLM.inhomo_baseline(ntrial=self.ntrial, 
                                    start=0,
                                    end=self.nt,
                                    dt=1, 
                                    **kwargs)

model = VAETransformer(input_dim, cls_dim, num_layers, hidden_dim, nl_dim, spline_basis, nt, n_neuron, n_area)
model.to(device)

# Example input
x = torch.randn(nt, 1, input_dim)

# Forward pass and loss
recon_x, mu, logvar = model(x)
loss = model.loss_function(recon_x, x, mu, logvar)
print(f'Loss: {loss.item()}')



# Constants
nt, n_neuron, nl_dim = 100, 10, 3
input_dim, cls_dim, num_layers, hidden_dim = 10, 10, 4, 64
spline_basis = torch.randn(T, 30)  # Dummy B-spline basis

# Model
model = VAETransformer(input_dim, cls_dim, num_layers, hidden_dim, nl_dim, spline_basis, T, N)

# Example input
spikes = torch.randn(T, 1, input_dim)

# Forward pass
firing_rate, z, mu, logvar = model(spikes)
loss = model.loss_function(firing_rate, spikes, mu, logvar)
print(f'Loss: {loss.item()}')
