#%% Importing libraries
import os
from collections import defaultdict
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import copy
import numpy as np
import numpy.random
from numpy.fft import fft as fft
from numpy.fft import ifft as ifft
import pickle
from tqdm import tqdm

from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
import scipy.stats
from scipy.stats import wilcoxon, chi2
import scipy.interpolate 
import scipy.signal
from scipy import linalg
from scipy.special import rel_entr
import statsmodels.api as sm
import statsmodels.genmod.generalized_linear_model as smm

import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim

import utility_functions as utils
import GLM
from DataLoader import Allen_dataset
import FCGPFA

#%% Define the model
#%% New functions
def gpfa_poisson_fix_weights(Y, weights, K, initial_mu=None, initial_hessian=None, 
                                     bias=None, lr=2e-1, max_iter=10, tol=1e-2, 
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
    inv_K = torch.linalg.inv(K + 1e-3 * torch.eye(nt, device=device))
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
        torch.Tensor: Convolved tensor of shape (ntrial, nneuroni, nt + ntau - 1).

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