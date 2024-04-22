import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist

import pickle
import numpy as np
import GLM
import random
from matplotlib import pyplot as plt

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU

class VAETransformer(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward, nl_dim, spline_basis, nfactors, nneuron_list, dropout):
        super(VAETransformer, self).__init__()
        self.d_model = d_model
        self.nt, self.nbasis = spline_basis.shape
        self.nneuron_list = nneuron_list  # this should now be a list containing neuron counts for each area
        self.narea = len(self.nneuron_list)
        self.nneuron_tot = sum(self.nneuron_list)
        self.nfactors = nfactors
        self.nl_dim = nl_dim
        self.training = True
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=1,
                                                dim_feedforward=dim_feedforward, dropout=dropout,
                                                batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
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

def change_temporal_resolution(spikes, num_merge):
    return [change_temporal_resolution_single(spikes[i], num_merge) for i in range(len(spikes))]

def change_temporal_resolution_single(spike_train, num_merge):
    ntrial, nneuron, nt = spike_train.shape
    new_spike_train = np.zeros((ntrial, nneuron, nt//num_merge))
    for t in range(nt//num_merge):
        new_spike_train[:,:,t] = np.sum(spike_train[:,:,t*num_merge:(t+1)*num_merge], axis=2)
    return new_spike_train
    
if __name__ == '__main__':
    set_seed(0)

    # Hyperparameters
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

    # Model and Data Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    with open('/home/export/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
        spikes = pickle.load(f)
    spikes_full = torch.tensor(np.concatenate(spikes, axis=1)).float()
    spikes_low_res = change_temporal_resolution(spikes, num_merge)
    spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float()

    # Splitting data into train and test sets
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
    best_model_path = '/home/export/qix/user_data/FC-GPFA_checkpoint/best_model.pth'

    # Function to adjust learning rate
    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training and Testing Loops
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


'''
if __name__ == '__main__':
    set_seed(0)
    ############### Hyperparameters ###############
    nl_dim, num_layers, dim_feedforward = 3, 4, 64
    num_merge = 5 # Merge 5 time bins into 1, so new temporal resolution is 5ms
    session_id = 757216464
    epochs = 10  # Define the number of training epochs
    learning_rate = 1e-4  # Set the learning rate for the optimizer
    warm_up_epochs = 3  # Set the number of warm-up epochs
    num_B_spline_basis = 10
    nfactors = 10

    ############### Model and Data Setup ###############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    with open('/home/export/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
        ### spike and firing rate standard format: mnt 
        spikes = pickle.load(f) # spikes[i]: (ntrial, n_neuron, nt) mnt
    spikes_full = torch.tensor(np.concatenate(spikes, axis=1)).float().to(device)
    spikes_low_res = change_temporal_resolution(spikes, num_merge) # (ntoken, ntrial, n_neuron)
    spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float().to(device)

    narea = len(spikes)
    nneuron_list = [spikes[i].shape[1] for i in range(narea)]
    nt, ntrial = spikes[0].shape[2], spikes[0].shape[0]
    d_model = sum(nneuron_list)
    B_spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=nt, dt=1, num=num_B_spline_basis, add_constant_basis=False)
    B_spline_basis = torch.tensor(B_spline_basis).float().to(device)

    kwargs = {'d_model': d_model, 'num_layers': num_layers, 'dim_feedforward': dim_feedforward, 'nl_dim': nl_dim, 
            'spline_basis': B_spline_basis, 'nfactors': nfactors, 'nneuron_list': nneuron_list}
    model = VAETransformer(**kwargs)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Initialize the Adam optimizer

    # Function to adjust learning rate
    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = learning_rate * (epoch / warm_up_epochs if epoch < warm_up_epochs else 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        adjust_learning_rate(optimizer, epoch)  # Adjust learning rate during warm-up
        optimizer.zero_grad()  # Clear existing gradients

        x = spikes_full_low_res.to(device)
        # Forward pass and loss calculation
        firing_rate, z, mu, logvar = model(x)
        loss = model.loss_function(firing_rate, spikes_full, mu, logvar)

        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, LR: {optimizer.param_groups[0]["lr"]}')  # Print loss and LR for the epoch
'''

'''
if __name__ == '__main__':
    ############### Hyperparameters ###############
    nl_dim, num_layers, dim_feedforward = 3, 4, 64
    num_merge = 5 # Merge 5 time bins into 1, so new temporal resolution is 5ms
    session_id = 757216464

    ############### Model and Data Setup ###############
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('/home/export/qix/user_data/allen_spike_trains/'+str(session_id)+'.pkl', 'rb') as f:
        ### spike and firing rate standard format: mnt 
        spikes = pickle.load(f) # spikes[i]: (ntrial, n_neuron, nt) mnt
    spikes_full = torch.tensor(np.concatenate(spikes, axis=1)).float().to(device)
    spikes_low_res = change_temporal_resolution(spikes, num_merge) # (ntoken, ntrial, n_neuron)
    spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float().to(device)

    narea = len(spikes)
    nneuron_list = [spikes[i].shape[1] for i in range(narea)]
    nt, ntrial = spikes[0].shape[2], spikes[0].shape[0]
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
'''