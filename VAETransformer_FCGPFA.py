import torch
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.distributions as dist
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import utility_functions as utils

class VAETransformer_FCGPFA(nn.Module):
    
    def __init__(
            self, 
            transformer_num_layers: int, 
            transformer_d_model: int, 
            transformer_dim_feedforward: int, 
            transformer_vae_output_dim: int, 
            stimulus_basis: torch.Tensor, 
            stimulus_nfactor: int, 
            transformer_dropout: float, 
            transformer_nhead: int, 
            stimulus_decoder_inter_dim_factor: int, 
            narea: int,
            npadding: int, # Coupling's feature:
            coupling_nsubspace: int, # Coupling's feature:
            coupling_basis: torch.Tensor, # Coupling's feature:
            use_self_coupling: bool, # Coupling's feature:
            coupling_strength_nlatent: int, # Coupling's feature:
            coupling_strength_cov_kernel: float, # Coupling's parameters
            self_history_basis: torch.Tensor, # Self-history's feature:
            session_id2nneuron_list: dict, # all session id and corresponding nneuron_list
            use_area_specific_decoder: bool,
            use_cls: bool,
            use_area_specific_encoder: bool,
        ):
        # Things that are specific to each session:
        # nneuron_list_dict, num_neurons_dict, accnneuron_dict
        # sti_readout_matrix_dict, token_converter_dict
        
        super().__init__()
        self.transformer_num_layers = transformer_num_layers
        self.transformer_d_model = transformer_d_model
        self.transformer_dim_feedforward = transformer_dim_feedforward
        self.transformer_vae_output_dim = transformer_vae_output_dim
        self.transformer_nhead = transformer_nhead
        self.nt, self.stimulus_nbasis = stimulus_basis.shape
        self.stimulus_nfactor = stimulus_nfactor
        self.stimulus_decoder_inter_dim_factor = stimulus_decoder_inter_dim_factor
        self.transformer_dropout = transformer_dropout
        self.narea = narea
        self.sample_latent = False  # This will be changed in "Trainer"
        self.num_neurons_dict = {}
        self.use_area_specific_decoder = use_area_specific_decoder
        self.use_cls = use_cls
        self.use_area_specific_encoder = use_area_specific_encoder
        
        ### Coupling's additional settings
        self.nneuron_list_dict = {}
        self.accnneuron_dict = {}
        self.npadding = npadding
        self.coupling_nsubspace = coupling_nsubspace
        self.coupling_strength_cov_kernel = coupling_strength_cov_kernel
        self.coupling_strength_nlatent = coupling_strength_nlatent
        self.use_self_coupling = use_self_coupling

        ### VAETransformer's parameters
        # self.cls_token = nn.Parameter(torch.randn(1, 1, self.transformer_d_model))
        self.token_converter_dict = nn.ModuleDict()
        self.RNN = False
        if self.RNN == False:
            transformer_encoder_layer = TransformerEncoderLayer(d_model=self.transformer_d_model, 
                                                                nhead=self.transformer_nhead,
                                                                dim_feedforward=self.transformer_dim_feedforward, 
                                                                activation='gelu',
                                                                dropout=self.transformer_dropout,
                                                                batch_first=True)
            self.transformer_encoder = TransformerEncoder(transformer_encoder_layer, num_layers=transformer_num_layers)
        else:
            self.transformer_encoder = nn.GRU(
                input_size=self.transformer_d_model,  # same as d_model
                hidden_size=self.transformer_d_model // 2,  # to match output dim if bidirectional
                num_layers=self.transformer_num_layers,
                batch_first=True,
                dropout=self.transformer_dropout if self.transformer_num_layers > 1 else 0,
                bidirectional=True
            )

        # Output mu and log-variance for each dimension
        if self.use_area_specific_encoder:
            assert self.transformer_vae_output_dim % self.narea == 0, 'transformer_vae_output_dim must be divisible by narea'
            self.per_area_vae_output_dim = self.transformer_vae_output_dim // self.narea
            self.to_latent = nn.Linear(self.transformer_d_model, 
                                        self.per_area_vae_output_dim * 2)
        else:
            self.to_latent = nn.Linear(self.transformer_d_model, 
                                        self.transformer_vae_output_dim * 2)

        # Initialize the stimulus decoder
        if use_area_specific_decoder:
            self.sti_decoder = area_specific_sti_decoder(
                self.transformer_vae_output_dim, 
                self.stimulus_decoder_inter_dim_factor, 
                self.stimulus_nbasis, 
                self.narea, 
                self.stimulus_nfactor,
            )
        else:
            self.sti_decoder = get_naive_sti_decoder(
                self.transformer_vae_output_dim, 
                self.stimulus_decoder_inter_dim_factor, 
                self.stimulus_nbasis, 
                self.narea, 
                self.stimulus_nfactor,
            )
            
        self.stimulus_basis = stimulus_basis  # Assume stimulus_basis is of size (nt, number of basis)
        self.sti_readout_matrix_dict = nn.ModuleDict()
        
        for session_id in session_id2nneuron_list.keys():
            nneuron_list = session_id2nneuron_list[session_id]
            self.nneuron_list_dict[session_id] = nneuron_list
            self.accnneuron_dict[session_id] = [0]+np.cumsum(nneuron_list).tolist()
            self.num_neurons_dict[session_id] = sum(nneuron_list)
            if self.use_area_specific_encoder:
                for iarea in range(self.narea):
                    self.token_converter_dict[session_id + "_"+ str(iarea)] = nn.Linear(
                        nneuron_list[iarea], 
                        self.transformer_d_model
                    )
            else:
                self.token_converter_dict[session_id] = nn.Linear(
                    self.num_neurons_dict[session_id], 
                    self.transformer_d_model
                )
            self.sti_readout_matrix_dict[session_id] = nn.ModuleList([
                nn.Linear(self.stimulus_nfactor, nneurons, bias=True) for nneurons in self.nneuron_list_dict[session_id]
            ])
            # # Initialize weights and biases to very small values close to zero
            # for layer in self.sti_readout_matrix_dict[session_id]:
            #     nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
            #     # nn.init.normal_(layer.weight, mean=0.1, std=1e-1)
            #     # nn.init.normal_(layer.bias, mean=0.1, std=1e-1)
            #     nn.init.constant_(layer.bias, 0.1) # log(1.5)

        # Positional encoding doesn't have dropout for now
        self.positional_encoding = PositionalEncoding(self.transformer_d_model)
        self.sti_inhomo = nn.Parameter(torch.zeros(self.narea, self.stimulus_nfactor, self.stimulus_nbasis))
        
        ### FCGPFA's parameters
        # NOT needed to do gradient descent on these parameters
        self.latents = None
        self.mu = None
        self.hessian = None
        self.coupling_basis = coupling_basis
        self.self_history_basis = self_history_basis
        self.init_cp_params()
        self.init_self_history_params()

    def init_cp_params(self):
        # self.cp_latents_readout = nn.Parameter(
        #     0.2 * (torch.randn(self.narea, self.narea, self.coupling_strength_nlatent) * 2 - 1)
        # )
        # self.cp_time_varying_coef_offset = nn.Parameter(
        #     1.0 * (torch.ones(self.narea, self.narea, 1, 1))
        # )
        
        self.cp_beta_coupling_dict = nn.ModuleDict({
            str(session_id): nn.ModuleList([
                nn.ParameterList([
                    nn.Parameter(1.0*(torch.zeros((self.coupling_basis.shape[1], self.coupling_nsubspace))+\
                        0.1*torch.randn((self.coupling_basis.shape[1], self.coupling_nsubspace))))
                    for jarea in range(self.narea)])
                for iarea in range(self.narea)])
            for session_id in self.nneuron_list_dict.keys()
        })
        
        self.cp_weight_sending_dict = nn.ModuleDict({
            str(session_id): nn.ModuleList([
                nn.ParameterList([
                    nn.Parameter(1/np.sqrt(self.nneuron_list_dict[session_id][iarea]*self.coupling_nsubspace)*\
                        (torch.zeros(self.nneuron_list_dict[session_id][iarea], self.coupling_nsubspace)+\
                            0.1*torch.randn(self.nneuron_list_dict[session_id][iarea], self.coupling_nsubspace)))
                    for jarea in range(self.narea)])
                for iarea in range(self.narea)])
            for session_id in self.nneuron_list_dict.keys()
        })
        
        self.cp_weight_receiving_dict = nn.ModuleDict({
            str(session_id): nn.ModuleList([
                nn.ParameterList([
                    nn.Parameter(1/np.sqrt(self.nneuron_list_dict[session_id][jarea]*self.coupling_nsubspace)*\
                        (torch.zeros(self.nneuron_list_dict[session_id][jarea], self.coupling_nsubspace)+\
                            0.1*torch.randn(self.nneuron_list_dict[session_id][jarea], self.coupling_nsubspace)))
                    for jarea in range(self.narea)])
                for iarea in range(self.narea)])
            for session_id in self.nneuron_list_dict.keys()
        })

        self.coupling_filters_dict = {}
        self.coupling_outputs_subspace = [[None]*self.narea for _ in range(self.narea)]
        self.coupling_outputs = [[None]*self.narea for _ in range(self.narea)]

    def init_self_history_params(self):
        self.beta_self_history_dict = nn.ParameterDict({
            str(session_id): nn.Parameter(
                -1.0*(torch.ones((
                    self.self_history_basis.shape[1], 
                    self.num_neurons_dict[session_id]
                )))
            )
            for session_id in self.num_neurons_dict.keys()
        })

        self.self_history_filters_dict = {}

    def get_self_history(self, src):
        ### Get self-history filters
        current_session_id = src['session_id']
        # basis, beta_self_history -> self_history_filters
        self.self_history_filters_dict[current_session_id] = torch.einsum(
            'tb,bn->tn', 
            self.self_history_basis, 
            self.beta_self_history_dict[current_session_id]
        )
        
        ### Get self-history outputs
        accnneuron = self.accnneuron_dict[current_session_id]
        spike_trains = src["spike_trains"].permute(2,1,0)

        # spikes(mnt), self_history_filters(t) -> self_history_outputs(mnt)
        self.self_history_outputs = conv_individual(
            spike_trains,
            self.self_history_filters_dict[current_session_id], npadding=self.npadding
        )

    def get_latents(self, lr=5e-1, max_iter=1000, tol=1e-2, verbose=False, fix_latents=False):
        # device = self.cp_latents_readout.device
        if fix_latents:
            # self.latents = torch.zeros(self.ntrial, self.coupling_strength_nlatent, self.nt, device=self.cp_latents_readout.device)
            # self.latents = torch.ones(self.ntrial, self.coupling_strength_nlatent, self.nt, device=self.cp_latents_readout.device)
            # self.latents[:,:,:150] = -1
            return None
        '''
        # Get the best latents under the current model
        with torch.no_grad():
            # weight: mnlt
            # bias: mnt
            weight = torch.zeros(self.ntrial, self.num_neurons, self.coupling_strength_nlatent, self.nt, device=device)
            bias = torch.zeros(self.ntrial, self.num_neurons, self.nt, device=device)
            bias += self.firing_rates_stimulus
            for iarea in range(self.narea):
                for jarea in range(self.narea):
                    # if iarea != jarea, then we need to update weight and bias
                    # if iarea == jarea, then we need to update bias, if and only if use_self_coupling is True
                    # if iarea == jarea, we never update weight no matter use_self_coupling is True or False
                    if iarea != jarea:
                        weight[:, self.accnneuron[jarea]:self.accnneuron[jarea+1], :, :] += (
                            self.coupling_outputs[iarea][jarea][:, :, None, :] *
                            self.cp_latents_readout[None, None, iarea, jarea, :, None]
                        )
                    if iarea != jarea or self.use_self_coupling:
                        bias[:, self.accnneuron[jarea]:self.accnneuron[jarea+1], :] += (
                            self.coupling_outputs[iarea][jarea] *
                            self.cp_time_varying_coef_offset[iarea, jarea, 0, 0]
                        )
        
        self.mu, self.hessian, self.lambd, self.elbo = gpfa_poisson_fix_weights(
            self.spikes_full[:,:,self.npadding:], weight, self.coupling_strength_cov_kernel, 
            initial_mu=None, initial_hessian=None, bias=bias, 
            lr=lr, max_iter=max_iter, tol=tol, verbose=False)
        self.latents = self.mu

        return self.elbo
        '''
    
    def get_coupling_outputs(self, src):
        current_session_id = src['session_id']
        # coupling_basis, cp_beta_coupling -> coupling_filters
        self.coupling_filters_dict[current_session_id] = [
            [torch.einsum(
                'tb,bs->ts', 
                self.coupling_basis, 
                self.cp_beta_coupling_dict[current_session_id][iarea][jarea]
            ) for jarea in range(self.narea)]
            for iarea in range(self.narea)]

        accnneuron = self.accnneuron_dict[current_session_id]
        coupling_filters = self.coupling_filters_dict[current_session_id]
        cp_weight_sending = self.cp_weight_sending_dict[current_session_id]
        cp_weight_receiving = self.cp_weight_receiving_dict[current_session_id]
        spike_trains = src["spike_trains"].permute(2,1,0)

        for jarea in range(self.narea):
            for iarea in range(self.narea):
                if not self.use_self_coupling and iarea == jarea:
                    continue
                # spikes(mit), coupling_filters(ts), cp_weight_sending(is) -> coupling_outputs in subspace(mst)
                self.coupling_outputs_subspace[iarea][jarea] = torch.einsum(
                    'mist,is->mst', 
                    conv_subspace(
                        spike_trains[:,accnneuron[iarea]:accnneuron[iarea+1],:], 
                        coupling_filters[iarea][jarea], npadding=self.npadding
                    ),
                    cp_weight_sending[iarea][jarea]
                )
                self.coupling_outputs[iarea][jarea] = torch.einsum('mst,js->mjt', 
                                                self.coupling_outputs_subspace[iarea][jarea],
                                                cp_weight_receiving[iarea][jarea],
                )
        return None

    def get_firing_rates_coupling(self, src, fix_latents=True):
        # Generate time-varying coupling strength coefficients
        if fix_latents:
            pass
        else:
            self.time_varying_coef = torch.einsum(
                'ijl,mlt -> ijmt', self.cp_latents_readout, self.latents
            ) + self.cp_time_varying_coef_offset
        # coupling_outputs in subspace, weight_receving, time_varying_coef (total coupling effects) 
        # -> log_firing_rate
        accnneuron = self.accnneuron_dict[src['session_id']]
        num_neurons = self.num_neurons_dict[src['session_id']]
        ntrial = src['spike_trains'].shape[2]
        nt = src['spike_trains'].shape[0] - self.npadding
        self.firing_rates_coupling = torch.zeros(ntrial, num_neurons, nt, 
                                                 device=src['spike_trains'].device)
        for jarea in range(self.narea):
            for iarea in range(self.narea):
                if iarea == jarea and not self.use_self_coupling:
                    continue
                    
                coupling_output = self.coupling_outputs[iarea][jarea]
                if not fix_latents and iarea != jarea:
                    coupling_output = coupling_output * self.time_varying_coef[iarea, jarea, :, None, :]
                    
                self.firing_rates_coupling[
                    :,accnneuron[jarea]:accnneuron[jarea+1],:
                ] += coupling_output
        return None

    def get_ci(self, alpha=0.05):
        self.std = torch.sqrt(torch.diagonal(-torch.linalg.inv(self.hessian), dim1=-2, dim2=-1))
        zscore = scipy.stats.norm.ppf(1-alpha/2)
        self.ci = [self.mu - zscore * self.std, self.mu + zscore * self.std]
        self.ci_time_varying_coef = [
            torch.einsum('ijl,mlt -> ijmt', self.cp_latents_readout, self.ci[0]) \
                + self.cp_time_varying_coef_offset, 
            torch.einsum('ijl,mlt -> ijmt', self.cp_latents_readout, self.ci[1]) \
                + self.cp_time_varying_coef_offset
            ]

    def normalize_coupling_coefficients(self):
        ### Solve identifiability issue
        ### 1. Make the coupling strength coefficients' absolute values sum to 1
        ### 2. Make the sending matrix be positive
        ### 3. Make the receiving matrix be positive
        with torch.no_grad():
            for session_id in self.nneuron_list_dict.keys():
                for iarea in range(self.narea):
                    for jarea in range(self.narea):
                        for k in range(self.coupling_nsubspace):
                            sending_weight = self.cp_weight_sending_dict[session_id][iarea][jarea][:,k]
                            sending_rescale = torch.sign(sending_weight.mean())*sending_weight.norm()
                            self.cp_weight_sending_dict[session_id][iarea][jarea][:,k] /=sending_rescale
                            receiving_weight = self.cp_weight_receiving_dict[session_id][iarea][jarea][:,k]
                            receiving_rescale = torch.sign(receiving_weight.mean())*receiving_weight.norm()
                            self.cp_weight_receiving_dict[session_id][iarea][jarea][:,k] /= receiving_rescale
                            self.cp_beta_coupling_dict[session_id][iarea][jarea][:,k] *= (
                                sending_rescale*receiving_rescale
                            )
                self.coupling_filters_dict[session_id] = [
                    [torch.einsum(
                        'tb,bs->ts', 
                        self.coupling_basis, 
                        self.cp_beta_coupling_dict[session_id][iarea][jarea]
                    ) for jarea in range(self.narea)]
                    for iarea in range(self.narea)]


    def forward(self, 
                src,
                fix_stimulus=False,
                fix_latents=False, 
                include_stimulus=True,
                include_coupling=True,
                include_self_history=False,
        ):
        assert include_coupling or include_stimulus, 'Need to have either coupling or stimulus'

        self.current_session_id = src['session_id']
        low_res_spike_trains = src['low_res_spike_trains']
        self.ntrial = low_res_spike_trains.shape[2]
        self.sti_z, self.sti_mu, self.sti_logvar = (
            None, 
            torch.zeros(self.ntrial, self.transformer_vae_output_dim), 
            torch.zeros(self.ntrial, self.transformer_vae_output_dim)
        ) # Set here so they at least have values
        
        ### VAETransformer's forward pass
        if include_stimulus:
            if fix_stimulus:
                self.get_inhomo_firing_rates_stimulus()
            else:
                self.get_inhomo_firing_rates_stimulus()
                trial_invariant_stimulus = self.firing_rates_stimulus
                self.sti_mu, self.sti_logvar = self.encode(low_res_spike_trains)
                self.sti_z = self.sample_a_latent(self.sti_mu, self.sti_logvar)
                self.firing_rates_stimulus = self.decode(self.sti_z) + trial_invariant_stimulus
        
        ### FCGPFA's forward pass 
        # Get latents
        if include_coupling:
            self.get_coupling_outputs(src)
            self.get_latents(fix_latents=fix_latents)
            self.get_firing_rates_coupling(src, fix_latents=fix_latents)
        if include_self_history:
            self.get_self_history(src)
        
        ### Return the combined firing rates
        if include_coupling and (not include_stimulus):
            self.firing_rates_combined = self.firing_rates_coupling
        elif include_stimulus and (not include_coupling):
            self.firing_rates_combined = self.firing_rates_stimulus
        else:
            self.firing_rates_combined = (
                self.firing_rates_stimulus + self.firing_rates_coupling
            )
        if include_self_history:
            self.firing_rates_combined += self.self_history_outputs
        return self.firing_rates_combined.permute(2,1,0)
    
    def encode(self, src):
        ### Depends on parameters: self.use_cls and self.treat_separately
        # src: tnm
        src = src.permute(0, 2, 1)

        if self.use_area_specific_encoder:
            latent_params_list = []
            for iarea in range(self.narea):
                # src: ntokens x batch_size x d_model (tmn)
                start_idx = self.accnneuron_dict[self.current_session_id][iarea]
                end_idx = self.accnneuron_dict[self.current_session_id][iarea+1]
                area_src = src[:, :, start_idx:end_idx]
                token_seq = self.token_converter_dict[self.current_session_id + "_"+ str(iarea)](area_src)
                if self.use_cls:
                    ### Append CLS token to the beginning of each sequence
                    cls_tokens = self.cls_token.expand(-1, token_seq.shape[1], -1)  # Expand CLS to batch size
                    token_seq = torch.cat((cls_tokens, token_seq), dim=0)  # Concatenate CLS token
                token_seq = self.positional_encoding(token_seq)  # Apply positional encoding
                if self.RNN == False:
                    encoded = self.transformer_encoder(token_seq) # Put it through the transformer encoder
                else:
                    encoded, _= self.transformer_encoder(token_seq) # Put it through the RNN encoder
                if self.use_cls:
                    cls_encoded = encoded[0,:,:]  # Only take the output from the CLS token
                else:
                    cls_encoded = encoded.mean(dim=0)  # average pooling over all tokens
                latent_params = self.to_latent(cls_encoded)
                latent_params_list.append(latent_params)
            mu_list = [l[:, :self.per_area_vae_output_dim] for l in latent_params_list]
            log_var_list = [l[:, self.per_area_vae_output_dim:] for l in latent_params_list]
            mu = torch.cat(mu_list, dim=1)
            log_var = torch.cat(log_var_list, dim=1)
            return mu, log_var
        else:
            # token_seq: ntokens x batch_size x d_model (tmn)
            token_seq = self.token_converter_dict[self.current_session_id](src)
            if self.use_cls:
                ### Append CLS token to the beginning of each sequence
                cls_tokens = self.cls_token.expand(-1, token_seq.shape[1], -1)  # Expand CLS to batch size
                token_seq = torch.cat((cls_tokens, token_seq), dim=0)  # Concatenate CLS token
            token_seq = self.positional_encoding(token_seq)  # Apply positional encoding
            encoded = self.transformer_encoder(token_seq) # Put it through the transformer encoder
            if self.use_cls:
                cls_encoded = encoded[0,:,:]  # Only take the output from the CLS token
            else:
                cls_encoded = encoded.mean(dim=0)  # average pooling over all tokens
            latent_params = self.to_latent(cls_encoded)
            mu = latent_params[:, :self.transformer_vae_output_dim]
            log_var = latent_params[:, self.transformer_vae_output_dim:]
        return mu, log_var  # mu, log_var

    def sample_a_latent(self, mu, sti_logvar):
        if self.sample_latent:
            std = torch.exp(0.5 * sti_logvar)
            eps = torch.randn_like(1*std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        # batch_size x (nbasis * narea * nfactor)
        self.proj = self.sti_decoder(z)
        # batch_size x narea x nfactor x nbasis **mafb**
        self.proj = self.proj.view(-1, self.narea, self.stimulus_nfactor, self.stimulus_nbasis)
        
        # Apply stimulus_basis per area
        # batch_size x narea x nt x nfactor**matf**
        self.factors = torch.einsum('mafb,tb->matf', self.proj, self.stimulus_basis)
        
        # Prepare to collect firing rates from each area
        firing_rates_list = []
        for i_area, readout_matrix in enumerate(self.sti_readout_matrix_dict[self.current_session_id]):
            # Extract factors for the current area
            area_factors = self.factors[:, i_area, :, :]  # batch_size x nt x nfactor (mtf)
            firing_rates = readout_matrix(area_factors)  # batch_size x nt x nneuron_area[i_area] (mtn)
            firing_rates_list.append(firing_rates)

        # Concatenate along a new dimension and then flatten to combine areas and neurons
        firing_rates_stimulus = torch.cat(firing_rates_list, dim=2)  # batch_size x nt x num_neurons (mtn)        
        return firing_rates_stimulus.permute(0,2,1) # mnt
    
    def get_inhomo_firing_rates_stimulus(self):
        # self.sti_inhomo: narea x nfactor x nbasis **afb**
        # narea x nt x nfactor **atf**
        self.factors = torch.einsum('afb,tb->atf', self.sti_inhomo, self.stimulus_basis)
        firing_rates_list = []
        for i_area, readout_matrix in enumerate(self.sti_readout_matrix_dict[self.current_session_id]):
            # nt x nfactor (tf)
            area_factors = self.factors[i_area, :, :]
            # nt x nneuron_area[i_area] (tn)
            firing_rates = readout_matrix(area_factors)  
            firing_rates_list.append(firing_rates)
        self.firing_rates_stimulus = torch.cat(firing_rates_list, dim=1)  # nt x num_neurons (tn)
        # mtn
        self.firing_rates_stimulus = self.firing_rates_stimulus[None,:,:].repeat(self.ntrial, 1, 1)
        self.firing_rates_stimulus = self.firing_rates_stimulus.permute(0,2,1) # mnt
        # self.firing_rates_stimulus = self.sti_inhomo[0, 0, 0]*torch.ones_like(self.firing_rates_stimulus)

    def loss_function(self, recon_x, x, mu, sti_logvar, beta=0.2):
        # Poisson loss
        poisson_loss = (torch.exp(recon_x) - x * recon_x).mean()

        # KL divergence
        kl_div = torch.mean(-0.5 * (1 + sti_logvar - mu.pow(2) - sti_logvar.exp()))
        kl_div *= self.transformer_vae_output_dim/(recon_x.shape[0]*recon_x.shape[1])

        return poisson_loss + beta * kl_div
        # return poisson_loss


def get_naive_sti_decoder(
            transformer_vae_output_dim,
            stimulus_decoder_inter_dim_factor,
            stimulus_nbasis,
            narea,
            stimulus_nfactor,
    ):
    if stimulus_decoder_inter_dim_factor == 0:
        sti_decoder = nn.Linear(transformer_vae_output_dim, 
                                stimulus_nbasis * narea * stimulus_nfactor)
        torch.nn.init.kaiming_uniform_(sti_decoder.weight, mode='fan_in', nonlinearity='relu')
    else:
        # Enhanced sti_decoder with additional layers and non-linearities
        sti_decoder = nn.Sequential(
            nn.Linear(transformer_vae_output_dim, 
                    transformer_vae_output_dim * stimulus_decoder_inter_dim_factor),
            nn.ReLU(), # Non-linear activation
            # Final output to match required dimensions
            nn.Linear(transformer_vae_output_dim * stimulus_decoder_inter_dim_factor, 
                        stimulus_nbasis * narea * stimulus_nfactor)  
        )
        torch.nn.init.kaiming_uniform_(sti_decoder[0].weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(sti_decoder[2].weight, mode='fan_in', nonlinearity='relu')
        sti_decoder[2].weight.data *= 1e-3
        if sti_decoder[2].bias is not None:
            nn.init.constant_(sti_decoder[2].bias, 0.0)
    return sti_decoder


class area_specific_sti_decoder(nn.Module):
    def __init__(
            self,
            transformer_vae_output_dim,
            stimulus_decoder_inter_dim_factor,
            stimulus_nbasis,
            narea,
            stimulus_nfactor,
        ):
        super(area_specific_sti_decoder, self).__init__()
        self.narea = narea
        assert transformer_vae_output_dim % narea == 0, \
            'transformer_vae_output_dim must be divisible by narea if using area-specific decoder'
        self.per_area_vae_output_dim = transformer_vae_output_dim // narea
        self.naive_sti_decoder = nn.ModuleList([
            get_naive_sti_decoder(
                self.per_area_vae_output_dim, 
                stimulus_decoder_inter_dim_factor, 
                stimulus_nbasis,
                1, # we need to predict one area for area-specific decoder
                stimulus_nfactor,
            )
            for _ in range(narea)
        ])

    def forward(self, z):
        return torch.cat([
            self.naive_sti_decoder[i](
                z[:, i*self.per_area_vae_output_dim:(i+1)*self.per_area_vae_output_dim]
            )
            for i in range(self.narea)
        ], dim=1)


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
        x = x + self.pe[:x.size(0)].to(x.device)
        return self.dropout(x)
    
#%% Define decoder algorithm
def gpfa_poisson_fix_weights(Y, weights, K, initial_mu=None, initial_hessian=None, 
                            bias=None, lr=5e-1, max_iter=500, tol=1e-2, 
                            print_iter=1, verbose=False, use_loss_to_stop=True):
    """
    Performs fixed weights GPFA with Poisson observations.

    Args:
        Y (torch.Tensor): Tensor of shape (ntrial, nneuron, nt) representing the spike counts.
        weights (torch.Tensor): Tensor of shape (ntrial, nneuron, coupling_strength_nlatent, nt) representing the weights.
        K (torch.Tensor): Tensor of shape (nt, nt) representing the covariance matrix of the latents.
        bias (float, optional): Bias term. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-2.
        verbose (bool, optional): Whether to print iteration information. Defaults to False.

    Returns:
        mu (torch.Tensor): Tensor of shape (ntrial, coupling_strength_nlatent, nt) representing the estimated latents.
        hessian (torch.Tensor): Tensor of shape (ntrial, coupling_strength_nlatent, nt, nt) representing the Hessian matrix.
        lambd (torch.Tensor): Tensor of shape (ntrial, nneuron, nt) representing the estimated Poisson rates.
    """
    
    device = Y.device
    ntrial, nneuron, coupling_strength_nlatent, nt = weights.shape

    # Expand Y dimensions if needed
    if Y.ndimension() == 2:
        Y = Y.unsqueeze(0)

    # Initialize latents
    if initial_mu is not None:
        mu = initial_mu
    else:
        mu = torch.zeros(ntrial, coupling_strength_nlatent, nt, device=device)
    mu_record = []
    
    # Inverse of K with regularization
    inv_K = torch.linalg.inv(K + 1e-3 * torch.eye(nt, device=device)).float()
    if initial_hessian is not None:
        hessian = initial_hessian
    else:
        hessian = inv_K.unsqueeze(0).unsqueeze(0).repeat(ntrial, coupling_strength_nlatent, 1, 1)
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
        # hessian += torch.eye(nt, device=device).unsqueeze(0).unsqueeze(0) * 1e-3
        hessian_inv = torch.linalg.inv(hessian)
        # lambd = torch.exp(log_lambd)
        lambd = torch.exp(
            torch.clamp(
                log_lambd + 1/2*(torch.diagonal(hessian_inv, dim1=-2, dim2=-1).unsqueeze(1)*weights**2).sum(axis=2), 
                max=30)
            )
        inv_K_times_hessian_inv = inv_K@hessian_inv

        
        ##############################
        if use_loss_to_stop:
            loss = torch.sum(Y * log_lambd) - torch.sum(lambd) - 1/2*(mu@inv_K*mu).sum()\
                - 1/2*(torch.diagonal(inv_K_times_hessian_inv, dim1=-2, dim2=-1)).sum()\
                    + 1/2*torch.log((torch.det(inv_K_times_hessian_inv)+1e-10)).sum() - nt
                    # + 1/2*torch.logdet(inv_K_times_hessian_inv).sum() - nt

            ############### For debug use ################
            if np.abs(loss.item())>1e7:
                pass
                # raise ValueError('Loss is too big')
            if loss.item()==float('inf') or torch.isnan(loss):
                print(i)
                print(loss.item())
                print(torch.sum(Y * log_lambd))
                print(torch.sum(lambd))
                print(1/2*(mu@inv_K*mu).sum())
                print(1/2*(torch.diagonal(inv_K_times_hessian_inv, dim1=-2, dim2=-1)).sum())
                print(1/2*torch.log(torch.relu(torch.det(inv_K_times_hessian_inv))).sum())
                raise ValueError('Loss is NaN')
            ##############################################
            
            if verbose and i % print_iter == 0:
                print(f'Iteration {i}: Loss change {loss.item() - loss_old}')

            if loss.item() - loss_old < tol and i >= 1 :
                flag = True
                if verbose:
                    print(f'Converged at iteration {i} with loss {loss.item()}')
                break
            
            loss_old = loss.item()
        ###############################

        # Update gradient calculation
        grad = torch.einsum('mnlt,mnt->mlt', weights, Y - lambd) - torch.matmul(mu, inv_K)
        
        # Update Hessian calculation to reflect the new dimensions and calculations
        hessian = -inv_K.unsqueeze(0).unsqueeze(0) - make_4d_diagonal(torch.einsum('mnlt,mnt->mlt', weights**2, lambd))
        # if torch.linalg.matrix_rank(hessian[0,0,:,:]) < nt:
        #     raise ValueError('Rank of hessian is not full')
        mu_update = torch.linalg.solve(hessian, grad.unsqueeze(-1)).squeeze(-1)
        # mu_update = torch.linalg.lstsq(hessian, grad.unsqueeze(-1)).solution.squeeze(-1)
        mu_new = mu - lr * mu_update

        if torch.isnan(mu_update).any():
            plt.plot(mu[:,0,:].cpu().numpy().T)
            print(torch.isnan(torch.linalg.lstsq(hessian, grad.unsqueeze(-1)).solution.squeeze(-1)).any())
            print(hessian[0,0,:,:])
            ranks = torch.tensor([torch.linalg.matrix_rank(hessian[i,0,:,:]) for i in range(hessian.shape[0])])
            print(f"rank of hessian: {ranks}")
            ranks = torch.tensor([torch.linalg.matrix_rank(1e5*torch.eye(hessian.shape[-1],device=hessian.device)+hessian[i,0,:,:]) 
                     for i in range(hessian.shape[0])])
            print(f"rank of hessian: {ranks}")
            raise ValueError('NaN in mu_new')
        
        #############################
        if not use_loss_to_stop:

            if torch.norm(lr * mu_update) < tol:
                flag = True
                if verbose:
                    loss = torch.sum(Y * log_lambd) - torch.sum(lambd) - 1/2*(mu@inv_K*mu).sum()\
                        - 1/2*(torch.diagonal(inv_K_times_hessian_inv, dim1=-2, dim2=-1)).sum()\
                            + 1/2*torch.log(torch.relu(torch.det(inv_K_times_hessian_inv))).sum() - nt
                    print(f'Converged at iteration {i} with loss {loss.item()}')
                break
        
        #############################
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
        raise ValueError('Not Converged')
    
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
        kernel (torch.Tensor): Convolution kernel of shape (ntau, coupling_nsubspace).
        npadding (int, optional): Number of padding time to remove from the output. Defaults to None.
        enforce_causality (bool, optional): Whether to enforce causality by zero-padding the kernel. Defaults to True.

    Returns:
        torch.Tensor: Convolved tensor of shape (ntrial, nneuroni, coupling_nsubspace, nt).

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


def conv_individual(raw_input, kernel, npadding=None, enforce_causality=True):
    """
    Applies convolution operation on each neuron's own spike history using the given kernel.

    Args:
        raw_input (torch.Tensor): Input tensor of shape (ntrial, nneuroni, nt).
        kernel (torch.Tensor): Convolution kernel of shape (ntau, nneuroni).
        npadding (int, optional): Number of padding time points to remove from the output. Defaults to None.
        enforce_causality (bool, optional): Whether to enforce causality by zero-padding the kernel. Defaults to True.

    Returns:
        torch.Tensor: Convolved tensor of shape (ntrial, nneuroni, nt).

    Raises:
        AssertionError: If the number of neurons in the kernel is not the same as the input.
    """
    assert kernel.shape[1] == raw_input.shape[1], "Kernel must match number of neurons in input"
    device = raw_input.device

    ntrial, nneuroni, nt = raw_input.shape
    ntau = kernel.shape[0]

    if enforce_causality:
        # prepend one time point of zeros to the kernel for causality
        zero_pad = torch.zeros((1, nneuroni), dtype=kernel.dtype, device=device)
        kernel = torch.cat((zero_pad, kernel), dim=0)

    ntau = kernel.shape[0]
    nn = nt + ntau - 1  # FFT size

    # FFT of input: (ntrial, nneuroni, nn)
    input_fft = torch.fft.fft(raw_input, n=nn, dim=2)

    # FFT of kernels: (nneuroni, nn)
    kernel_fft = torch.fft.fft(kernel.T, n=nn)  # transpose to (nneuroni, ntau)

    # Element-wise multiply in frequency domain
    # Result shape: (ntrial, nneuroni, nn)
    conv_fft = input_fft * kernel_fft[None, :, :]

    # Inverse FFT to get time domain signal
    conv_time = torch.fft.ifft(conv_fft, dim=2).real
    conv_time[torch.abs(conv_time) < 1e-5] = 0

    # Truncate to match input time dimension
    conv_time = conv_time[:, :, :nt]

    if npadding is not None:
        conv_time = conv_time[:, :, npadding:]

    return conv_time
