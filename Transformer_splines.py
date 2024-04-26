import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist
import math

class VAETransformer(nn.Module):
    def __init__(self, num_layers, dim_feedforward, nl_dim, spline_basis, nfactor, nneuron_list, dropout, 
                 nhead, decoder_architecture):
        print(VAETransformer)
        super(VAETransformer, self).__init__()
        self.nneuron_list = nneuron_list  # this should now be a list containing neuron counts for each area
        self.d_model = sum(self.nneuron_list)
        self.nt, self.nbasis = spline_basis.shape
        self.narea = len(self.nneuron_list)
        self.nneuron_tot = self.d_model
        self.nfactor = nfactor
        self.nl_dim = nl_dim
        self.nhead = nhead
        self.sample_latent = False
        self.decoder_architecture = decoder_architecture
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, 
                                                            nhead=self.nhead,
                                                            dim_feedforward=dim_feedforward, 
                                                            activation='gelu',
                                                            dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(self.d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        ##################################
        # Enhanced decoder_fc with additional layers and non-linearities
        if self.decoder_architecture == 0:
            self.decoder_fc = nn.Linear(nl_dim, self.nbasis * self.narea * self.nfactor)
            torch.nn.init.kaiming_uniform_(self.decoder_fc.weight, mode='fan_in', nonlinearity='relu')
        else:
            self.decoder_fc = nn.Sequential(
                nn.Linear(nl_dim, nl_dim * self.decoder_architecture),  # Expand dimension
                nn.ReLU(),                      # Non-linear activation
                nn.Linear(nl_dim* self.decoder_architecture, self.nbasis * self.narea * self.nfactor)  # Final output to match required dimensions
            )
            torch.nn.init.kaiming_uniform_(self.decoder_fc[0].weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.decoder_fc[2].weight, mode='fan_in', nonlinearity='relu')
        ##################################
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30
        self.readout_matrices = nn.ModuleList([nn.Linear(self.nfactor, neurons) for neurons in self.nneuron_list])
        self.positional_encoding = PositionalEncoding(self.d_model)

    def encode(self, src):
        # src: mnt
        src = src.permute(2, 0, 1)
        # src: ntokens x batch_size x d_model (tmn)
        
        # Append CLS token to the beginning of each sequence
        cls_tokens = self.cls_token.expand(-1, src.shape[1], -1)  # Expand CLS to batch size
        src = torch.cat((cls_tokens, src), dim=0)  # Concatenate CLS token
        
        src = self.positional_encoding(src)  # Apply positional encoding
        encoded = self.transformer_encoder(src) # Put it through the transformer encoder
        ##################################
        # cls_encoded = encoded[0,:,:]  # Only take the output from the CLS token
        cls_encoded = encoded.mean(dim=0)  # average pooling over all tokens
        ##################################
        latent_params = self.to_latent(cls_encoded)
        return latent_params[:, :latent_params.size(-1)//2], latent_params[:, latent_params.size(-1)//2:]  # mu, log_var

    def sample_a_latent(self, mu, logvar):
        if self.sample_latent:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(1*std)
            return mu + eps * std
            # return mu
        else:
            return mu
        
    def decode(self, z):
        # proj = torch.einsum('ltn,ml->mnt', self.decoder_matrix, z)
        # return proj-3
        
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
        return firing_rates_combined.permute(0,2,1) -5 # mnt

    def forward(self, src):
        mu, logvar = self.encode(src)
        z = self.sample_a_latent(mu, logvar)
        return self.decode(z), z, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=0.2):
        # Poisson loss
        poisson_loss = (torch.exp(recon_x) - x * recon_x).mean()
        
        # KL divergence
        kl_div = torch.mean(-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()))
        kl_div *= self.nl_dim/(self.nneuron_tot*self.nt)
        return poisson_loss + beta * kl_div
        # return poisson_loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(1000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)