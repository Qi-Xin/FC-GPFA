import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist
import math
import numpy as np
import matplotlib.pyplot as plt

class VAETransformer_FCGPFA(nn.Module):
    def __init__(self, num_layers, dim_feedforward, nl_dim, spline_basis, nfactor, nneuron_list, dropout, 
                 nhead, decoder_architecture, 
                 npadding, nsubspace, K, nlatent, coupling_basis):
        super(VAETransformer_FCGPFA, self).__init__()
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
        
        ### FCGPFA's additional settings
        self.npadding = npadding
        self.accnneuron = [0]+np.cumsum(self.nneuron_list).tolist()
        self.nsubspace = nsubspace
        self.K = K
        self.nlatent = nlatent

        ### VAETransformer's parameters
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, 
                                                            nhead=self.nhead,
                                                            dim_feedforward=dim_feedforward, 
                                                            activation='gelu',
                                                            dropout=dropout,
                                                            batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        self.to_latent = nn.Linear(self.d_model, nl_dim * 2)  # Output mu and log-variance for each dimension
        
        if self.decoder_architecture == 0:
            self.decoder_fc = nn.Linear(nl_dim, self.nbasis * self.narea * self.nfactor)
            torch.nn.init.kaiming_uniform_(self.decoder_fc.weight, mode='fan_in', nonlinearity='relu')
        else:
            # Enhanced decoder_fc with additional layers and non-linearities
            self.decoder_fc = nn.Sequential(
                nn.Linear(nl_dim, nl_dim * self.decoder_architecture),  # Expand dimension
                nn.ReLU(),                      # Non-linear activation
                # Final output to match required dimensions
                nn.Linear(nl_dim* self.decoder_architecture, self.nbasis * self.narea * self.nfactor)  
            )
            torch.nn.init.kaiming_uniform_(self.decoder_fc[0].weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.decoder_fc[2].weight, mode='fan_in', nonlinearity='relu')
            
        self.spline_basis = spline_basis  # Assume spline_basis is nt x 30
        self.readout_matrices = nn.ModuleList([nn.Linear(self.nfactor, neurons) for neurons in self.nneuron_list])
        self.positional_encoding = PositionalEncoding(self.d_model)
        
        ### FCGPFA's parameters
        # NOT needed to do gradient descent on these parameters
        self.latents = None
        self.mu = None
        self.hessian = None
        self.coupling_outputs_subspace = [[None]*self.narea for _ in range(self.narea)]
        self.coupling_outputs = [[None]*self.narea for _ in range(self.narea)]
        self.coupling_basis = coupling_basis
        
        # DO gradient descent on these parameters
        self.cp_latents_readout = nn.Parameter(0.01 * (torch.randn(self.narea, self.narea, self.nlatent) * 2 - 1))
        self.cp_time_varying_coef_offset = nn.Parameter(1 * (torch.ones(self.narea, self.narea, 1, 1)))
        
        self.cp_beta_coupling = nn.ModuleList([
            nn.ParameterList([
                    nn.Parameter(0.2*torch.ones(coupling_basis.shape[1], self.nsubspace))
                for jarea in range(self.narea)])
            for iarea in range(self.narea)])
        
        self.cp_weight_sending = nn.ModuleList([
            nn.ParameterList([
                    nn.Parameter(1/np.sqrt(self.nneuron_list[iarea]*self.nsubspace)*\
                        torch.ones(self.nneuron_list[iarea], self.nsubspace))
                for jarea in range(self.narea)])
            for iarea in range(self.narea)])
        
        self.cp_weight_receiving = nn.ModuleList([
            nn.ParameterList([
                    nn.Parameter(1/np.sqrt(self.nneuron_list[jarea]*self.nsubspace)*\
                        torch.ones(self.nneuron_list[jarea], self.nsubspace))
                for jarea in range(self.narea)])
            for iarea in range(self.narea)])

    def get_latents(self, lr=5e-1, max_iter=1000, tol=1e-2, verbose=False, fix_latents=False):
        device = self.cp_latents_readout.device
        if fix_latents:
            self.latents = torch.zeros(self.ntrial, self.nlatent, self.nt, device=self.cp_latents_readout.device)
            # self.latents[:,:,:150] = -1
            return None
        # Get the best latents under the current model
        with torch.no_grad():
            # weight: mnlt
            # bias: mnt
            weight = torch.zeros(self.ntrial, self.nneuron_tot, self.nlatent, self.nt, device=device)
            bias = torch.zeros(self.ntrial, self.nneuron_tot, self.nt, device=device)
            bias += self.firing_rates_stimulus
            for iarea in range(self.narea):
                for jarea in range(self.narea):
                    if iarea == jarea:
                        continue
                    weight[:, self.accnneuron[jarea]:self.accnneuron[jarea+1], :, :] += (
                        self.coupling_outputs[iarea][jarea][:, :, None, :] *
                        self.cp_latents_readout[None, None, iarea, jarea, :, None]
                    )
                    bias[:, self.accnneuron[jarea]:self.accnneuron[jarea+1], :] += (
                        self.coupling_outputs[iarea][jarea] *
                        self.cp_time_varying_coef_offset[iarea, jarea, 0, 0]
                    )

        self.mu, self.hessian, self.lambd, self.elbo = gpfa_poisson_fix_weights(
            self.spikes_full[:,:,self.npadding:], weight, self.K, 
            initial_mu=None, initial_hessian=None, bias=bias, 
            lr=lr, max_iter=max_iter, tol=tol, verbose=True)
        self.latents = self.mu

        return self.elbo
    
    def get_coupling_outputs(self):
        # coupling_basis, cp_beta_coupling -> coupling_filters
        self.coupling_filters = [[torch.einsum('tb,bs->ts', self.coupling_basis, self.cp_beta_coupling[iarea][jarea])
            for jarea in range(self.narea)] for iarea in range(self.narea)]

        for jarea in range(self.narea):
            for iarea in range(self.narea):
                if iarea == jarea:
                    continue
                # spikes(mit), coupling_filters(ts), cp_weight_sending(is) -> coupling_outputs in subspace(mst)
                self.coupling_outputs_subspace[iarea][jarea] = torch.einsum(
                    'mist,is->mst', 
                    conv_subspace(self.spikes_full[:,self.accnneuron[iarea]:self.accnneuron[iarea+1],:], 
                                  self.coupling_filters[iarea][jarea], npadding=self.npadding),
                    self.cp_weight_sending[iarea][jarea]
                )
                self.coupling_outputs[iarea][jarea] = torch.einsum('mst,js->mjt', 
                                                self.coupling_outputs_subspace[iarea][jarea],
                                                self.cp_weight_receiving[iarea][jarea],
                )
        return None
    
    def get_firing_rates_coupling(self):
        # Generate time-varying coupling strength coefficients
        self.time_varying_coef = torch.einsum('ijl,mlt -> ijmt', self.cp_latents_readout, self.latents) \
                                    + self.cp_time_varying_coef_offset
        # coupling_outputs in subspace, weight_receving, time_varying_coef (total coupling effects) 
        # -> log_firing_rate
        self.firing_rates_coupling = torch.zeros(self.ntrial, self.nneuron_tot, self.nt, 
                                                 device=self.cp_latents_readout.device)
        for jarea in range(self.narea):
            for iarea in range(self.narea):
                if iarea == jarea:
                    continue
                self.firing_rates_coupling[:,self.accnneuron[jarea]:self.accnneuron[jarea+1],:] += \
                    self.coupling_outputs[iarea][jarea] * self.time_varying_coef[iarea, jarea, :, None, :]

    def forward(self, src, spikes_full, fix_latents=False):
        self.spikes_full = spikes_full
        self.ntrial = spikes_full.shape[0]
        
        ### VAETransformer's forward pass
        mu, logvar = self.encode(src)
        z = self.sample_a_latent(mu, logvar)
        self.firing_rates_stimulus = self.decode(z)
        
        ### FCGPFA's forward pass
        # Get latents
        self.get_coupling_outputs()
        self.get_latents(fix_latents=fix_latents)
        self.get_firing_rates_coupling()
        
        self.firing_rates_combined = self.firing_rates_stimulus + self.firing_rates_coupling
        return self.firing_rates_combined, z, mu, logvar
    
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
        firing_rates_stimulus = torch.cat(firing_rates_list, dim=2)  # batch_size x nt x nneuron_tot (mtn)        
        return firing_rates_stimulus.permute(0,2,1) -5 # mnt

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
    
#%% Define decoder algorithm
def gpfa_poisson_fix_weights(Y, weights, K, initial_mu=None, initial_hessian=None, 
                            bias=None, lr=5e-1, max_iter=100, tol=1e-2, 
                            print_iter=1, verbose=False):
    """
    Performs fixed weights GPFA with Poisson observations.

    Args:
        Y (torch.Tensor): Tensor of shape (ntrial, nneuron, nt) representing the spike counts.
        weights (torch.Tensor): Tensor of shape (ntrial, nneuron, nlatent, nt) representing the weights.
        K (torch.Tensor): Tensor of shape (nt, nt) representing the covariance matrix of the latents.
        bias (float, optional): Bias term. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-2.
        verbose (bool, optional): Whether to print iteration information. Defaults to False.

    Returns:
        mu (torch.Tensor): Tensor of shape (ntrial, nlatent, nt) representing the estimated latents.
        hessian (torch.Tensor): Tensor of shape (ntrial, nlatent, nt, nt) representing the Hessian matrix.
        lambd (torch.Tensor): Tensor of shape (ntrial, nneuron, nt) representing the estimated Poisson rates.
    """
    
    device = Y.device
    ntrial, nneuron, nlatent, nt = weights.shape

    # Expand Y dimensions if needed
    if Y.ndimension() == 2:
        Y = Y.unsqueeze(0)

    # Initialize latents
    if initial_mu is not None:
        mu = initial_mu
    else:
        mu = torch.zeros(ntrial, nlatent, nt, device=device)
    mu_record = []
    
    # Inverse of K with regularization
    inv_K = torch.linalg.inv(K + 1e-3 * torch.eye(nt, device=device)).float()
    if initial_hessian is not None:
        hessian = initial_hessian
    else:
        hessian = inv_K.unsqueeze(0).unsqueeze(0).repeat(ntrial, nlatent, 1, 1)
    if bias is None:
        bias = torch.tensor(4.0, device=device)
    else:
        bias = bias
    loss_old = float('-inf')
    flag = False

    for i in (range(max_iter)):
        # Updated log_lambd calculation to handle the new shape of weights
        log_lambd = torch.einsum('mnlt,mlt->mnt', weights, mu) + bias
        # lambd = torch.exp(log_lambd)
        hessian_inv = torch.linalg.inv(hessian)
        # lambd = torch.exp(log_lambd)
        lambd = torch.exp(log_lambd + 1/2*(torch.diagonal(hessian_inv, dim1=-2, dim2=-1).unsqueeze(1)*weights**2).sum(axis=2))
        inv_K_times_hessian_inv = inv_K@hessian_inv
        # print((torch.einsum('mlt,mlt->mlt', mu@inv_K, mu)).shape)
        # print(1/2*(mu@inv_K@mu.T).shape)
        # loss = torch.sum(Y * log_lambd) - torch.sum(lambd) - 1/2*(mu@inv_K*mu).sum()
        loss = torch.sum(Y * log_lambd) - torch.sum(lambd) - 1/2*(mu@inv_K*mu).sum()\
            - 1/2*(torch.diagonal(inv_K_times_hessian_inv, dim1=-2, dim2=-1)).sum()\
                + 1/2*torch.logdet(inv_K_times_hessian_inv).sum() - nt
        # print(loss)
        # print(weights)
        # print(f"log_lambd.max():{log_lambd.max()}")
        # print(f"torch.sum(lambd):{torch.sum(lambd)}")
        if np.abs(loss.item())>1e7:
            pass
            # raise ValueError('Loss is too big')
        if np.isnan(loss.item()):
            print(i)
            plt.plot(log_lambd[:, :, :].cpu().numpy().max(axis=(0,1)).T)
            plt.plot(log_lambd[:, :, :].cpu().numpy().min(axis=(0,1)).T)
            plt.figure()
            plt.plot(mu[:, 0, :].cpu().numpy().T)
            raise ValueError('Loss is NaN')
            
        
        if verbose and i % print_iter == 0:
            print(f'Iteration {i}: Loss change {loss.item() - loss_old}')
            # plt.plot(mu[0, 0, :].cpu().numpy(), label=f'Iteration {i+1}')
            # plt.legend()
        if loss.item() - loss_old < tol and i >= 1 :
            flag = True
            if verbose:
                print(f'Converged at iteration {i} with loss {loss.item()}')
            break

        # Update gradient calculation
        grad = torch.einsum('mnlt,mnt->mlt', weights, Y - lambd) - torch.matmul(mu, inv_K)
        
        # Update Hessian calculation to reflect the new dimensions and calculations
        hessian = -inv_K.unsqueeze(0).unsqueeze(0) - make_4d_diagonal(torch.einsum('mnlt,mnt->mlt', weights**2, lambd))
        mu_update = torch.linalg.solve(hessian, grad.unsqueeze(-1)).squeeze(-1)
        # mu_update = torch.linalg.lstsq(hessian, grad.unsqueeze(-1)).solution.squeeze(-1)
        mu_new = mu - lr * mu_update

        loss_old = loss.item()

        # if torch.norm(lr * mu_update) < tol:
        #     flag = True
        #     if verbose:
        #         print(f'Converged at iteration {i} with loss {loss.item()}')
        #     break
        mu = mu_new
        
        # # Record mu in each iteration
        # mu_record.append(mu.clone().detach())
        # stride = 3
        # if i < 10*stride and i % stride == 0:
        #     plt.plot(mu[0, 0, :].cpu().numpy(), label=f'Iteration {i+1}')
        # if i == 10*stride-1:
        #     plt.legend()
        #     plt.show()
        
    if flag is False:
        print(f'Not Converged with norm {torch.norm(lr * mu_update)} at the last iteration')
    return mu, hessian, lambd, loss.item()

def get_K(sigma2=1.0, L=100.0, nt=500, use_torch=False, device='cpu'):
    """
    Get the covariance matrix K for GPFA.

    Parameters:
    - sigma2 (float): The variance of the Gaussian kernel.
    - L (float): The length scale of the Gaussian kernel.
    - nt (int): The number of time bins.
    - device (str or torch.device): The device to create the tensor on.

    Returns:
    - K (Tensor): The covariance matrix. Shape: (nt, nt).
    """
    x = np.linspace(0, nt-1, nt)
    diff = np.subtract.outer(x, x)
    K = sigma2 * np.exp(-diff**2 / L**2)
    # Convert to a PyTorch tensor and then move to the specified device
    if use_torch:
        return torch.from_numpy(K).to(device)
    else:
        return K
    
def make_4d_diagonal(mat):
    """
    Take a matrix of shape (n, m, l) and return a 3D array of shape (n, m, l, l) where
    the original matrix is repeated along the last axis.
    """
    # Initialize an empty 3D tensor with the required shape
    mat_diag = torch.zeros((mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[2]), device=mat.device)

    # Use advanced indexing to fill in the diagonals
    i = torch.arange(mat.shape[2], device=mat.device)
    mat_diag[:, :, i, i] = mat

    return mat_diag

# mat = torch.rand(2, 3, 4)  # Example input matrix of shape (2, 3, 4)
# mat_4d_diag = make_4d_diagonal(mat)

# print(mat[0,0,:])
# mat_4d_diag[0,0,:,:]

def conv(raw_input, kernel, npadding=None, enforce_causality=True):
    """
    Applies convolution operation on the input tensor using the given kernel.

    Args:
        raw_input (torch.Tensor): Input tensor of shape (ntrial, nneuroni, nt).
        kernel (torch.Tensor): Convolution kernel of shape (nneuroni, ntau, nneuronj).
        npadding (int, optional): Number of padding time to remove from the output. Defaults to None.
        enforce_causality (bool, optional): Whether to enforce causality by zero-padding the kernel. Defaults to True.

    Returns:
        torch.Tensor: Convolved tensor of shape (ntrial, nneuroni, nneuronj, nt).

    Raises:
        AssertionError: If the number of neurons in the kernel is not the same as the input.

    """
    
    device = raw_input.device
    ntrial, nneuroni, nt = raw_input.shape
    if enforce_causality:
        zero_pad = torch.zeros((kernel.shape[0], 1, kernel.shape[2]), dtype=torch.float32, device=device)
        kernel = torch.cat((zero_pad, kernel), dim=1)
    ntau = kernel.shape[1]
    assert kernel.shape[0] == nneuroni, 'The number of neurons in the kernel should be the same as the input'
    nneuronj = kernel.shape[2]
    
    nn = nt + ntau - 1
    G = torch.fft.ifft(torch.fft.fft(raw_input, n=nn, dim=2).unsqueeze(3) * torch.fft.fft(kernel, n=nn, dim=1).unsqueeze(0), dim=2)
    G = G.real
    G[torch.abs(G) < 1e-5] = 0
    G = G[:,:,:nt,:]
    if npadding is not None:
        G = G[:,:,npadding:,:]
    return G.transpose(-1,-2)

# # Test conv
# raw_input = torch.tensor([[1, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 1, 0, 0, 0, 0]], dtype=torch.float32, device='cpu')
# raw_input = raw_input.unsqueujhieze(0)

# # Sample kernel
# kernel = torch.zeros((3, 5, 2), dtype=torch.float32, device='cpu')
# kernel[0, :, 0] = torch.tensor([0.5, 0.3, 0.1, 0, 0])
# kernel[2, :, 1] = torch.tensor([0.6, 0.4, 0.2, 0, 0])

# # Call the conv function
# X = conv(raw_input, kernel)

# # Print the result
# print(X.shape)
# print(X[0, :, :, :])
# print(X[0, 0, 0,:])
# print(X[0, 2, 1,:])

def conv_subspace(raw_input, kernel, npadding=None, enforce_causality=True):
    """
    Applies convolution operation on the input tensor using the given kernel.

    Args:
        raw_input (torch.Tensor): Input tensor of shape (ntrial, nneuroni, nt).
        kernel (torch.Tensor): Convolution kernel of shape (ntau, nsubspace).
        npadding (int, optional): Number of padding time to remove from the output. Defaults to None.
        enforce_causality (bool, optional): Whether to enforce causality by zero-padding the kernel. Defaults to True.

    Returns:
        torch.Tensor: Convolved tensor of shape (ntrial, nneuroni, nsubspace, nt).

    Raises:
        AssertionError: If the number of neurons in the kernel is not the same as the input.

    """
    
    device = raw_input.device
    kernel = kernel[None, :, :]
    ntrial, nneuroni, nt = raw_input.shape
    if enforce_causality:
        zero_pad = torch.zeros((kernel.shape[0], 1, kernel.shape[2]), dtype=torch.float32, device=device)
        kernel = torch.cat((zero_pad, kernel), dim=1)
    ntau = kernel.shape[1]
    
    nn = nt + ntau - 1
    G = torch.fft.ifft(torch.fft.fft(raw_input, n=nn, dim=2).unsqueeze(3) * torch.fft.fft(kernel, n=nn, dim=1).unsqueeze(0), dim=2)
    G = G.real
    G[torch.abs(G) < 1e-5] = 0
    G = G[:,:,:nt,:]
    if npadding is not None:
        G = G[:,:,npadding:,:]
    return G.transpose(-1,-2)