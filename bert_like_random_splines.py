import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

import pickle
import numpy as np
import GLM
from matplotlib import pyplot as plt

class VAETransformer(nn.Module):
    def __init__(self, num_layers, dim_feedforward, nl_dim, spline_basis, nfactors, nneuron_list, dropout, nhead):
        super(VAETransformer, self).__init__()
        self.nneuron_list = nneuron_list  # this should now be a list containing neuron counts for each area
        self.d_model = sum(self.nneuron_list)
        self.nt, self.nbasis = spline_basis.shape
        self.narea = len(self.nneuron_list)
        self.nneuron_tot = self.d_model
        self.nfactors = nfactors
        self.nl_dim = nl_dim
        self.nhead = nhead
        self.training = True
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(self.d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
        self.decoder_fc = nn.Linear(nl_dim, self.nbasis * self.narea * self.nfactors)
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30

        self.readout_matrices = nn.ModuleList([nn.Linear(self.nfactors, neurons) for neurons in self.nneuron_list])

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


    
if __name__ == '__main__':
    import utility_functions as utils
    
    ### Hyperparameters
    nl_dim, num_layers, dim_feedforward = 5, 4, 64
    num_merge = 5
    session_id = 757216464
    epochs = 100
    learning_rate = 1e-2
    warm_up_epochs = 3
    num_B_spline_basis = 15
    nfactors = 5
    batch_size = 128
    dropout = 0.5
    patience = 5
    
    ### Set seed and directory
    utils.set_seed(0)
    import socket
    hostname = socket.gethostname()
    if hostname[:8] == "ghidorah":
        path_prefix = '/home'
    elif hostname[:6] == "wright":
        path_prefix = '/home/export'

    ### Model and Data Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    with open(path_prefix+'/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
        spikes = pickle.load(f)
    spikes_full = torch.tensor(np.concatenate(spikes, axis=1)).float()
    spikes_low_res = utils.change_temporal_resolution(spikes, num_merge)
    spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float()

    ### Splitting data into train and test sets
    num_samples = spikes_full_low_res.shape[0]
    indices = list(range(num_samples))
    split = int(np.floor(0.8 * num_samples))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = torch.utils.data.TensorDataset(spikes_full_low_res[train_idx], spikes_full[train_idx])
    test_dataset = torch.utils.data.TensorDataset(spikes_full_low_res[test_idx], spikes_full[test_idx])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    narea = len(spikes)
    nneuron_list = [spike.shape[1] for spike in spikes]
    nt, ntrial = spikes[0].shape[2], spikes[0].shape[0]
    d_model = sum(nneuron_list)
    B_spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=nt, dt=1, num=num_B_spline_basis, add_constant_basis=True)
    B_spline_basis = torch.tensor(B_spline_basis).float().to(device)

    model = VAETransformer(d_model, num_layers, dim_feedforward, nl_dim, B_spline_basis, nfactors, nneuron_list, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_loss = float('inf')
    no_improve_epoch = 0
    best_model_path = path_prefix+'/qix/user_data/FC-GPFA_checkpoint/best_model.pth'

    ### Function to adjust learning rate
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    ### Training and Testing Loops
    for epoch in range(epochs):
        model.train()
        model.training = True
        train_loss = 0.0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            firing_rate, z, mu, logvar = model(data)
            loss = model.loss_function(firing_rate, targets, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        model.training = False
        test_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                firing_rate, z, mu, logvar = model(data)
                loss = model.loss_function(firing_rate, targets, mu, logvar, beta=0.0)
                test_loss += loss.item() * data.size(0)

        test_loss /= len(test_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Checkpointing and Early Stopping Logic
        if test_loss < best_loss:
            best_loss = test_loss
            no_improve_epoch = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_epoch += 1
            print(f'No improvement in Test Loss for {no_improve_epoch} epoch(s).')
            if no_improve_epoch >= patience:
                print('Early stopping triggered.')
                break
