import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

class VAETransformer(nn.Module):
    def __init__(self, d_model, num_layers, hidden_dim, nl_dim, spline_basis, T, N):
        super(VAETransformer, self).__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=1, dim_feedforward=hidden_dim)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, nl_dim * 2)  # Output mu and log-variance for each dimension

        self.decoder_fc = nn.Linear(nl_dim, 30)
        self.spline_basis = spline_basis  # Assume spline_basis is T x 30

        self.weight_vector = nn.Parameter(torch.randn(30, N))

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
        proj = self.decoder_fc(z)
        basis_expansion = torch.matmul(self.spline_basis, proj.T)  # T x N
        firing_rates = torch.matmul(basis_expansion, self.weight_vector)  # T x N
        return firing_rates

    def forward(self, src):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Poisson loss
        poisson_loss = torch.mean(torch.exp(recon_x) - x * recon_x)
        # KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return poisson_loss + beta * kl_div

# Constants
T, N, nl_dim = 100, 10, 3
input_dim, cls_dim, num_layers, hidden_dim = 10, 10, 4, 64
spline_basis = torch.randn(T, 30)  # Dummy B-spline basis

# Model
model = VAETransformer(input_dim, cls_dim, num_layers, hidden_dim, nl_dim, spline_basis, T, N)

# Example input
x = torch.randn(T, 1, input_dim)

# Forward pass
recon_x, mu, logvar = model(x)
loss = model.loss_function(recon_x, x, mu, logvar)
print(f'Loss: {loss.item()}')
