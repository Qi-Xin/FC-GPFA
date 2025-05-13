import torch
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from VAETransformer_FCGPFA import VAETransformer_FCGPFA, get_K
import utility_functions as utils
import GLM
import matplotlib.pyplot as plt
from DataLoader import Allen_dataloader_multi_session, Simple_dataloader_from_spikes
import socket
'''
First, load data "spikes", set path, set hyperparameters, and use these three to create a Trainer object.
Then, call the trainer.train() method to train the model, which use early stop. 
If the results is good, you can save the model along with hyperparameters (aka the trainer) 
    by calling trainer.save_model_and_hp() method.
'''

class Trainer:
    def __init__(self, dataloader, path, params):
        self.dataloader = dataloader
        self.path = path
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.results_file = "hyperparameter_tuning_results.json"
        hostname = socket.gethostname().split('.')[0]
        self.model_id = (
            hostname + "_" + str(self.dataloader.session_ids) \
                +"_"+(datetime.now() - timedelta(hours=4)).strftime('%Y%m%d_%H%M%S')
        )
        self.temp_best_model_path = self.path+'/temp_best_model'+hostname+'.pth'
        ### Change batch size
        if hasattr(self.dataloader, 'change_batch_size'):
            self.dataloader.change_batch_size(self.params['batch_size'])
        
        ### Get some dependent parameters
        first_batch = next(iter(self.dataloader.train_loader))
        self.narea = len(first_batch["nneuron_list"])
        try:
            self.npadding = self.dataloader.sessions[
                next(iter(self.dataloader.sessions.keys()))
            ].npadding
        except:
            self.npadding = self.dataloader.npadding
        self.nt = first_batch["spike_trains"].shape[0]
        self.nt -= self.npadding
        self.session_id2nneuron_list = {}
        if hasattr(self.dataloader, 'sessions'):
            for session_id in self.dataloader.session_ids:
                self.session_id2nneuron_list[str(session_id)] = self.dataloader.sessions[session_id].nneuron_list
        else:
            self.session_id2nneuron_list['0'] = self.dataloader.nneuron_list

        # Initialize model
        self.initialize_model(verbose=True)

    # def make_optimizer(self, frozen_params=[]):
    #     # self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
    #     ###############################
    #     transformer_group = ['transformer_encoder', 'to_latent', 'token_converter', 'cls_token']
    #     sti_group = ['sti_readout', 'sti_decoder', 'sti_inhomo']
    #     cp_group = ['cp_latents_readout', 'cp_time_varying_coef_offset', 'cp_beta_coupling', 
    #                 'cp_weight_sending', 'cp_weight_receiving']
        
    #     # Define different learning rates
    #     transformer_lr = self.params['lr_transformer']
    #     sti_lr = self.params['lr_sti']  # Higher learning rate for decoder_matrix
    #     cp_lr = self.params['lr_cp']  # Learning rate for coupling parameters
    #     weight_decay = self.params['weight_decay']

    #     # Configure optimizer with parameter groups
    #     params_not_assigned = [n for n, p in self.model.named_parameters()
    #         if all([key_word not in n for key_word in transformer_group+sti_group+cp_group])]
    #     if len(params_not_assigned)!=0:
    #         print(params_not_assigned)
    #         raise ValueError("Some parameters are not assigned to any group.")
    #     self.optimizer = optim.Adam([
    #         {'params': [p for n, p in self.model.named_parameters() 
    #                     if (
    #                         any([key_word in n for key_word in transformer_group]) 
    #                         and (not any(key_word in n for key_word in frozen_params))
    #                     )], 
    #          'lr': transformer_lr},
    #         {'params': [p for n, p in self.model.named_parameters() 
    #                     if (
    #                         any([key_word in n for key_word in sti_group])
    #                         and (not any(key_word in n for key_word in frozen_params))
    #                     )], 
    #          'lr': sti_lr},
    #         {'params': [p for n, p in self.model.named_parameters() 
    #                     if (
    #                         any([key_word in n for key_word in cp_group])
    #                         and (not any(key_word in n for key_word in frozen_params))
    #                     )], 
    #          'lr': cp_lr},
    #     ], weight_decay=weight_decay)
    
    def make_optimizer(self, frozen_params=[]):
        """
        Creates an Adam optimizer with different learning rates for different model component groups.
        
        Args:
            frozen_params (list): List of substrings; parameters matching any of these will be frozen.
        """
        # Define parameter groups and their learning rates
        param_group_specs = [
            (['transformer_encoder', 'to_latent', 'token_converter', 'cls_token'], self.params['lr_transformer']),
            (['sti_readout', 'sti_decoder', 'sti_inhomo'], self.params['lr_sti']),
            (['cp_latents_readout', 'cp_time_varying_coef_offset', 'cp_beta_coupling', 
            'cp_weight_sending', 'cp_weight_receiving'], self.params['lr_cp']),
            (['self_history'], self.params['lr_self_history']),
        ]

        # Build optimizer param groups
        param_groups = []
        all_group_keywords = []

        for keywords, lr in param_group_specs:
            all_group_keywords.extend(keywords)
            group_params = [
                p for n, p in self.model.named_parameters()
                if any(kw in n for kw in keywords) and not any(frozen in n for frozen in frozen_params)
            ]
            param_groups.append({'params': group_params, 'lr': lr})

        # Check for any unassigned parameters
        unassigned = [
            n for n, _ in self.model.named_parameters()
            if all(kw not in n for kw in all_group_keywords)
        ]
        if unassigned:
            print("Unassigned parameters:", unassigned)
            raise ValueError("Some parameters are not assigned to any group.")

        # Create optimizer
        self.optimizer = optim.Adam(param_groups, weight_decay=self.params['weight_decay'])
         
    def initialize_model(self, verbose=False):
        stimulus_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=self.nt, dt=1, 
                                           num=self.params['num_B_spline_basis'], 
                                           add_constant_basis=True)
        stimulus_basis = torch.tensor(stimulus_basis).float().to(self.device)
        
        coupling_basis = GLM.make_pillow_basis(**{'peaks_max':self.params['coupling_basis_peaks_max'], 
                                                  'num':self.params['coupling_basis_num'], 
                                                  'nonlinear':0.5})
        coupling_basis = torch.tensor(coupling_basis).float().to(self.device)
        self_history_basis = GLM.make_pillow_basis(**{'peaks_max':self.params['self_history_basis_peaks_max'], 
                                                  'num':self.params['self_history_basis_num'], 
                                                  'nonlinear':self.params['self_history_basis_nonlinear']})
        self_history_basis = torch.tensor(self_history_basis).float().to(self.device)
        K = torch.tensor(get_K(nt=self.nt, L=self.params['K_tau'], sigma2=self.params['K_sigma2'])).to(self.device)
        self.D = torch.tensor(utils.second_order_diff_matrix(self.nt)).float().to(self.device) # shape: (nt-2, nt)

        self.model = VAETransformer_FCGPFA(
            transformer_num_layers=self.params['transformer_num_layers'],
            transformer_d_model=self.params['transformer_d_model'],
            transformer_dim_feedforward=self.params['transformer_dim_feedforward'],
            transformer_vae_output_dim=self.params['transformer_vae_output_dim'],
            stimulus_basis=stimulus_basis,
            stimulus_nfactor=self.params['stimulus_nfactor'],
            transformer_dropout=self.params['transformer_dropout'],
            transformer_nhead=self.params['transformer_nhead'],
            stimulus_decoder_inter_dim_factor=self.params['stimulus_decoder_inter_dim_factor'],
            narea=self.narea,
            npadding=self.npadding, 
            coupling_nsubspace=self.params['coupling_nsubspace'],
            coupling_basis=coupling_basis,
            use_self_coupling=self.params['use_self_coupling'],
            coupling_strength_nlatent=self.params['coupling_strength_nlatent'],
            coupling_strength_cov_kernel=K,
            self_history_basis=self_history_basis,
            session_id2nneuron_list=self.session_id2nneuron_list,
            use_area_specific_decoder=self.params['use_area_specific_decoder'],
            use_area_specific_encoder=self.params['use_area_specific_encoder'],
            use_cls=self.params['use_cls'],
        ).to(self.device)
        self.make_optimizer()

        if verbose:
            print(f"Model initialized. Training on {self.device}")

    def process_batch(self, batch):
        batch["spike_trains"] = batch["spike_trains"].to(self.device)
        batch["low_res_spike_trains"] = utils.change_temporal_resolution_single(
            batch["spike_trains"][self.npadding:,:,:], 
            self.params['downsample_factor']
        )
    
    def train(
            self,
            verbose=True,
            record_results=False,
            include_stimulus=True,
            include_coupling=True, 
            include_self_history=True,
            fix_stimulus=False,
            fix_latents=False, 
        ):

        if verbose:
            print(f"Start training model with parameters: {self.params}")
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        best_train_loss_wo_penalty = float('inf')
        no_improve_epoch = 0

        
        # Function to adjust learning rate
        def adjust_lr(optimizer, epoch):
            if len(optimizer.param_groups) == 1:
                optimizer.param_groups[0]['lr'] = \
                    self.params['lr']*(epoch+1)/self.params['epoch_warm_up']
            else:
                optimizer.param_groups[0]['lr'] = \
                    self.params['lr_transformer']*(epoch+1)/self.params['epoch_warm_up']
                optimizer.param_groups[1]['lr'] = \
                    self.params['lr_sti']*(epoch+1)/self.params['epoch_warm_up']
                optimizer.param_groups[1]['lr'] = \
                    self.params['lr_cp']*(epoch+1)/self.params['epoch_warm_up']
        
        ### Training and Testing Loops
        for epoch in tqdm(range(self.params['epoch_max']), disable=verbose):
            # Warm up
            if epoch < self.params['epoch_warm_up']:
                adjust_lr(self.optimizer, epoch)
            self.model.train()
            self.model.sample_latent = self.params['sample_latent']
            train_loss = 0.0
            train_loss_wo_penalty = 0.0
            total_trial = 0
            for batch in tqdm(self.dataloader.train_loader, disable=not verbose):
                self.process_batch(batch)
                self.optimizer.zero_grad()
                firing_rate = self.model(
                    batch,
                    include_stimulus=include_stimulus,
                    include_coupling=include_coupling,
                    include_self_history=include_self_history,
                    fix_stimulus=fix_stimulus,
                    fix_latents=fix_latents,
                )
                loss = self.model.loss_function(
                    firing_rate, 
                    batch["spike_trains"][self.npadding:,:,:], 
                    self.model.sti_mu, 
                    self.model.sti_logvar, 
                    beta=self.params['beta']
                )
                train_loss_wo_penalty += loss.item() * batch["spike_trains"].size(2)
                loss += self.get_penalty()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * batch["spike_trains"].size(2)
                total_trial += batch["spike_trains"].size(2)
            train_loss /= total_trial
            train_loss_wo_penalty /= total_trial


            self.model.eval()
            self.model.sample_latent = False
            test_loss = 0.0
            total_trial = 0
            with torch.no_grad():
                for batch in tqdm(self.dataloader.val_loader, disable=not verbose):
                    self.process_batch(batch)
                    firing_rate = self.model(
                        batch,
                        include_stimulus=include_stimulus,
                        include_coupling=include_coupling,
                        include_self_history=include_self_history,
                        fix_stimulus=fix_stimulus,
                        fix_latents=fix_latents,
                    )
                    loss = self.model.loss_function(
                        firing_rate, 
                        batch["spike_trains"][self.npadding:,:,:], 
                        self.model.sti_mu, 
                        self.model.sti_logvar, 
                        beta=self.params['beta']
                    )
                    loss += self.get_penalty()
                    test_loss += loss.item() * batch["spike_trains"].size(2)
                    total_trial += batch["spike_trains"].size(2)
            test_loss /= total_trial
            
            # if epoch % 5 == 2:
            #     plt.figure()
            #     plt.plot(self.model.latents[:, 0, :].cpu().numpy().T)

            if verbose:
                print(f"Epoch {epoch+1}/{self.params['epoch_max']}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Checkpointing and Early Stopping Logic
            if test_loss < best_test_loss - self.params['tol']:
                no_improve_epoch = 0
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_train_loss_wo_penalty = train_loss_wo_penalty
                torch.save(self.model.state_dict(), self.temp_best_model_path)
            else:
                no_improve_epoch += 1
                if verbose:
                    print(f'No improvement in Test Loss for {no_improve_epoch} epoch(s).')
                    print(f'Current Best Test Loss: {best_test_loss:.4f}')
                if no_improve_epoch >= self.params['epoch_patience']:
                    if verbose:
                        print('Early stopping triggered.')
                    break

        self.model.load_state_dict(torch.load(self.temp_best_model_path))
        if record_results:
            self.log_results(best_train_loss_wo_penalty, best_train_loss, best_test_loss)
        return best_test_loss

    def predict(
            self, 
            dataset='test',
            batch_indices=[0,1,2,3,4],
            manual_batches=None,
            include_stimulus=True,
            include_coupling=False,
            include_self_history=False,
            fix_stimulus=False,
            fix_latents=False,
            return_torch=True, 
            return_trial_indices=True,
            return_spike_trains=False,
        ):
        self.model.eval()
        self.model.sample_latent = False
        sti_mu_list = []
        sti_logvar_list = []
        firing_rate_list = []
        trial_indices_list = []
        if dataset == 'train':
            loader = self.dataloader.train_loader
        elif dataset == 'test':
            loader = self.dataloader.test_loader
        elif dataset == 'val':
            loader = self.dataloader.val_loader
        else:
            raise ValueError("Invalid dataset. Choose from 'val', 'train', or 'test'.")
        
        if manual_batches is not None:
            batch_indices = np.arange(len(manual_batches))
        with torch.no_grad():
            for batch_idx in batch_indices:
                if manual_batches is None:
                    batch = loader.dataloader.get_batch(batch_idx, dataset)
                else:
                    batch = manual_batches[batch_idx]
                self.process_batch(batch)
                firing_rate = self.model(
                    batch,
                    include_stimulus=include_stimulus,
                    include_coupling=include_coupling,
                    include_self_history=include_self_history,
                    fix_stimulus=fix_stimulus,
                    fix_latents=fix_latents,
                )
                sti_mu_list.append(self.model.sti_mu)
                sti_logvar_list.append(torch.exp(0.5 * self.model.sti_logvar))
                firing_rate_list.append(firing_rate)
                if return_trial_indices and 'batch_indices' in batch:
                    trial_indices_list.append(batch['batch_indices'])
        outputs = [torch.concat(firing_rate_list, dim=2).cpu(),
                  torch.concat(sti_mu_list, dim=0).cpu(), 
                  torch.concat(sti_logvar_list, dim=0).cpu()]
        if not return_torch:
            outputs = [x.numpy() for x in outputs]
        if return_trial_indices:
            outputs.append(np.concatenate(trial_indices_list, axis=0))
        if return_spike_trains:
            outputs.append(batch["spike_trains"][self.npadding:,:,:].cpu())
        return outputs

    def get_penalty(self):
        penalty = 0.0
        if self.params['penalty_coupling_subgroup'] is not None:
            for iarea in range(self.narea):
                for jarea in range(self.narea):
                    penalty += (
                        self.params['penalty_coupling_subgroup'] \
                            * self.model.cp_weight_receiving_dict[
                                self.model.current_session_id
                            ][iarea][jarea].norm(dim=1).mean()
                    )
                    penalty += (
                        self.params['penalty_coupling_subgroup'] \
                            * self.model.cp_weight_sending_dict[
                                self.model.current_session_id
                            ][iarea][jarea].norm(dim=1).mean()
                    )
        if self.params['penalty_loading_similarity'] is not None:
            for iarea in range(self.narea):
                for jarea in range(self.narea):
                    penalty += (
                        self.params['penalty_loading_similarity'] \
                            * self.model.cp_weight_receiving_dict[
                                self.model.current_session_id
                            ][iarea][jarea].var()
                    )
                    penalty += (
                        self.params['penalty_loading_similarity'] \
                            * self.model.cp_weight_sending_dict[
                                self.model.current_session_id
                            ][iarea][jarea].var()
                    )
        if self.params['penalty_smoothing_spline'] is not None:
            if self.model.factors.dim() == 3:
                second_diff = torch.einsum('atf,dt->adf', self.model.factors, self.D)
            else:
                second_diff = torch.einsum('matf,dt->madf', self.model.factors, self.D)
            penalty += self.params['penalty_smoothing_spline'] * torch.mean(second_diff ** 2)
        if self.params['penalty_diff_loading'] is not None:
            # add penalty for coupling effect's loading matrix and stimulus effect's loading matrix
            for iarea in range(self.narea):
                for jarea in range(self.narea):
                    if iarea == jarea:
                        continue
                    # nneurons x nfactor
                    sti_loading = self.model.sti_readout_matrix_dict[
                        self.model.current_session_id][iarea]
                    # nneurons x nsubspace
                    cp_loading = self.model.cp_weight_receiving_dict[
                        self.model.current_session_id][iarea][jarea]
                    penalty += (
                        self.params['penalty_diff_loading'] * torch.norm(sti_loading.T @ cp_loading)**2
                    )
        return penalty
    
    def save_model_and_hp(self, filename=None, test_loss=None):
        if filename is None:
            filename = self.path + '/' + self.model_id + '.pth'
        else:
            filename = self.path + '/' + filename + '.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
            'test_loss': test_loss,
        }, filename)
        print(f"Trainer instance (model and hyperparameters) saved to {filename}")
        
    def load_model_and_hp(self, filename=None):
        if filename is None:
            print(f"Loading default model from {self.path}")
            filename = self.path
        # checkpoint = torch.load(filename)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        map_location = lambda storage, loc: storage.cuda() \
            if torch.cuda.is_available() else storage.cpu()
        checkpoint = torch.load(filename, map_location=map_location)

        self.params = checkpoint['params']        
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Trainer instance (model and hyperparameters) loaded from {filename}")

    def log_results(self, best_train_loss_wo_penalty, best_train_loss, best_test_loss):
        results = {
            "model_id": self.model_id,
            "params": self.params,
            "best_train_loss_wo_penalty": best_train_loss_wo_penalty,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
        }
        with open(self.results_file, 'a') as file:
            json.dump(results, file, indent=4, sort_keys=False)  # Indent each level by 4 spaces
            file.write('\n')  # Write a newline after each set of results
        # If running on cluster, save the results again to a different file
        hostname = socket.gethostname()
        if hostname[:3] == "n01":
            path_prefix = '/home/export'
            with open(path_prefix+self.results_file, 'a') as file:
                json.dump(results, file, indent=4, sort_keys=False)  # Indent each level by 4 spaces
                file.write('\n')  # Write a newline after each set of results
