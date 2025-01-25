import torch
import torch.optim as optim
import json
import os
import numpy as np
from tqdm import tqdm
from VAETransformer_FCGPFA import VAETransformer_FCGPFA, get_K
import utility_functions as utils
import GLM
import matplotlib.pyplot as plt
from DataLoader import Allen_dataloader_multi_session, Simple_dataloader_from_spikes
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
        self.results_file = "training_results.json"
        self.penalty_overlapping = self.params['penalty_overlapping']

        ### Change batch size
        self.dataloader.change_batch_size(self.params['batch_size'])
        
        ### Get some dependent parameters
        first_batch = next(iter(self.dataloader.train_loader))
        self.narea = len(first_batch["nneuron_list"])
        self.npadding = self.dataloader.sessions[
            next(iter(self.dataloader.sessions.keys()))
        ].npadding
        self.nt = first_batch["spike_trains"].shape[0]
        self.nt -= self.npadding
         
    def initialize_model(self, verbose=False):
        stimulus_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=self.nt, dt=1, 
                                           num=self.params['num_B_spline_basis'], 
                                           add_constant_basis=True)
        stimulus_basis = torch.tensor(stimulus_basis).float().to(self.device)
        
        coupling_basis = GLM.make_pillow_basis(**{'peaks_max':self.params['coupling_basis_peaks_max'], 
                                                  'num':self.params['coupling_basis_num'], 
                                                  'nonlinear':0.5})
        coupling_basis = torch.tensor(coupling_basis).float().to(self.device)
        K = torch.tensor(get_K(nt=self.nt, L=self.params['K_tau'], sigma2=self.params['K_sigma2'])).to(self.device)

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
        ).to(self.device)
        ################################
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        # transformer_group = ['transformer_encoder', 'to_latent', 'token_converter']
        # sti_group = ['sti_readout_matrices', 'sti_decoder', 'sti_inhomo']
        # cp_group = ['cp_latents_readout', 'cp_time_varying_coef_offset', 'cp_beta_coupling', 
        #             'cp_weight_sending', 'cp_weight_receiving']
        
        # # Define different learning rates
        # transformer_lr = self.params['lr_transformer']
        # sti_lr = self.params['lr_sti']  # Higher learning rate for decoder_matrix
        # cp_lr = self.params['lr_cp']  # Learning rate for coupling parameters
        # weight_decay = self.params['weight_decay']

        # # Configure optimizer with two parameter groups
        # params_not_assigned = [n for n, p in self.model.named_parameters()
        #     if all([key_word not in n for key_word in transformer_group+sti_group+cp_group])]
        # if len(params_not_assigned)!=0:
        #     print(params_not_assigned)
        #     raise ValueError("Some parameters are not assigned to any group.")
        # self.optimizer = optim.Adam([
        #     {'params': [p for n, p in self.model.named_parameters() 
        #                 if any([key_word in n for key_word in transformer_group])], 
        #      'lr': transformer_lr},
        #     {'params': [p for n, p in self.model.named_parameters() 
        #                 if any([key_word in n for key_word in sti_group])], 
        #      'lr': sti_lr},
        #     {'params': [p for n, p in self.model.named_parameters() 
        #                 if any([key_word in n for key_word in cp_group])], 
        #      'lr': cp_lr},
        # ], weight_decay=weight_decay)
        ################################
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
            fix_stimulus=False,
            fix_latents=False, 
            include_stimulus=True,
            include_coupling=True, 
        ):

        if verbose:
            print(f"Start training model with parameters: {self.params}")
        utils.set_seed(0)
        self.initialize_model(verbose=verbose)
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        no_improve_epoch = 0
        temp_best_model_path = self.path+'/temp_best_model.pth'
        
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
        for epoch in range(self.params['epoch_max']):
            # Warm up
            if epoch < self.params['epoch_warm_up']:
                adjust_lr(self.optimizer, epoch)
            self.model.train()
            self.model.sample_latent = self.params['sample_latent']
            train_loss = 0.0
            for batch in tqdm(self.dataloader.train_loader):
                self.process_batch(batch)
                self.optimizer.zero_grad()
                firing_rate = self.model(
                    batch,
                    fix_stimulus=fix_stimulus,
                    fix_latents=fix_latents,
                    include_stimulus=include_stimulus,
                    include_coupling=include_coupling
                )
                loss = self.model.loss_function(firing_rate, 
                                                batch["spike_trains"][self.npadding:,:,:], 
                                                self.model.sti_mu, 
                                                self.model.sti_logvar, 
                                                beta=self.params['beta'])
                if self.penalty_overlapping is not None:
                    loss += self.penalty_overlapping * self.model.overlapping_scale
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch["spike_trains"].size(2)
            train_loss /= len(self.dataloader.train_loader)

            self.model.eval()
            self.model.sample_latent = False
            test_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.dataloader.val_loader):
                    self.process_batch(batch)
                    firing_rate = self.model(
                        batch,
                        fix_stimulus=fix_stimulus,
                        fix_latents=fix_latents,
                        include_stimulus=include_stimulus,
                        include_coupling=include_coupling
                    )
                    loss = self.model.loss_function(firing_rate, 
                                                    batch["spike_trains"][self.npadding:,:,:], 
                                                    self.model.sti_mu, 
                                                    self.model.sti_logvar, 
                                                    beta=self.params['beta'])
                    if self.penalty_overlapping is not None:
                        loss += self.penalty_overlapping * self.model.overlapping_scale
                    test_loss += loss.item() * batch["spike_trains"].size(2)
            test_loss /= len(self.dataloader.test_loader)
            
            # if epoch % 5 == 2:
            #     plt.figure()
            #     plt.plot(self.model.latents[:, 0, :].cpu().numpy().T)

            if verbose:
                print(f"Epoch {epoch+1}/{self.params['epoch_max']}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Checkpointing and Early Stopping Logic
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_train_loss = train_loss
                no_improve_epoch = 0
                torch.save(self.model.state_dict(), temp_best_model_path)
            else:
                no_improve_epoch += 1
                if verbose:
                    print(f'No improvement in Test Loss for {no_improve_epoch} epoch(s).')
                if no_improve_epoch >= self.params['epoch_patience']:
                    if verbose:
                        print('Early stopping triggered.')
                    break
        
        self.model.load_state_dict(torch.load(temp_best_model_path))
        if record_results:
            self.log_results(best_train_loss, best_test_loss)
        return best_test_loss

    def predict(
            self, 
            dataset='test',
            batch_indices=[0,1,2,3,4],
            return_torch=True, 
            fix_stimulus=False,
            fix_latents=False, 
            include_stimulus=True,
            include_coupling=False, 
        ):
        self.model.eval()
        self.model.sample_latent = False
        sti_mu_list = []
        sti_logvar_list = []
        firing_rate_list = []
        if dataset == 'train':
            loader = self.dataloader.train_loader
        elif dataset == 'test':
            loader = self.dataloader.test_loader
        elif dataset == 'val':
            loader = self.dataloader.val_loader
        else:
            raise ValueError("Invalid dataset. Choose from 'val', 'train', or 'test'.")
        
        with torch.no_grad():
            for batch_idx in batch_indices:
                batch = loader.get_batch(batch_idx, dataset)
                self.process_batch(batch)
                firing_rate = self.model(
                    batch,
                    fix_stimulus=fix_stimulus,
                    fix_latents=fix_latents,
                    include_stimulus=include_stimulus,
                    include_coupling=include_coupling
                )
                sti_mu_list.append(self.model.sti_mu)
                sti_logvar_list.append(torch.exp(0.5 * self.model.sti_logvar))
                firing_rate_list.append(firing_rate)
        if return_torch:
            return (torch.concat(firing_rate_list, dim=2).cpu(), 
                    torch.concat(sti_mu_list, dim=0).cpu(),
                    torch.concat(sti_logvar_list, dim=0).cpu())
        else:
            return (torch.concat(firing_rate_list, dim=2).cpu().numpy(), 
                    torch.concat(sti_mu_list, dim=0).cpu().numpy(),
                    torch.concat(sti_logvar_list, dim=0).cpu().numpy())
        
    def save_model_and_hp(self):
        filename = self.path + '/best_model_and_hp.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
        }, filename)
        print(f"Trainer instance (model and hyperparameters) saved to {filename}")
        
    def load_model_and_hp(self, filename=None):
        if filename is None:
            print(f"Loading default model from {self.path}")
            filename = self.path + '/best_model_and_hp.pth'
        # checkpoint = torch.load(filename)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        map_location = lambda storage, loc: storage.cuda() \
            if torch.cuda.is_available() else storage.cpu()
        checkpoint = torch.load(filename, map_location=map_location)

        self.params = checkpoint['params']        
        self.initialize_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Trainer instance (model and hyperparameters) loaded from {filename}")

    def log_results(self, train_loss, test_loss):
        results = {
            "params": self.params,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        with open(self.results_file, 'a') as file:
            json.dump(results, file, indent=4, sort_keys=False)  # Indent each level by 4 spaces
            file.write('\n')  # Write a newline after each set of results
