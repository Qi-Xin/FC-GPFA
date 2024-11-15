import torch
import torch.optim as optim
import json
import os
import numpy as np

from VAETransformer_FCGPFA import VAETransformer_FCGPFA, get_K
import utility_functions as utils
import GLM
import matplotlib.pyplot as plt

'''
First, load data "spikes", set path, set hyperparameters, and use these three to create a Trainer object.
Then, call the trainer.train() method to train the model, which use early stop. 
If the results is good, you can save the model along with hyperparameters (aka the trainer) 
    by calling trainer.save_model_and_hp() method.
'''

class Trainer:
    def __init__(self, spikes, path, params, npadding):
        self.spikes = spikes
        self.path = path
        self.npadding = npadding
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.results_file = "training_results.json"
        self.penalty_overlapping = params['penalty_overlapping']
        
        ### Get some dependent parameters
        self.narea = len(self.spikes)
        self.nneuron_list = [sp.shape[1] for sp in self.spikes]
        self.nt, self.ntrial = self.spikes[0].shape[2], self.spikes[0].shape[0]
        self.nt -= self.npadding

    def process_data(self, verbose=False):
        ### Get tokenized spike trains by downsampling
        self.spikes_full = np.concatenate(self.spikes, axis=1)
        self.spikes_full_low_res = utils.change_temporal_resolution_single(
            self.spikes_full[:,:,self.npadding:], self.params['num_merge']
            )
        self.spikes_full = torch.tensor(self.spikes_full).float()
        self.spikes_full_no_padding = self.spikes_full[:,:,self.npadding:]
        self.spikes_full_low_res = torch.tensor(self.spikes_full_low_res).float()
        
        ### Splitting data into train and test sets
        indices = list(range(self.ntrial))
        split = int(np.floor(0.8 * self.ntrial))
        utils.set_seed(1)
        np.random.shuffle(indices)
        self.train_idx, self.test_idx = indices[:split], indices[split:]
        train_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[self.train_idx], 
                                                       self.spikes_full[self.train_idx])
        test_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[self.test_idx], 
                                                      self.spikes_full[self.test_idx])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=self.params['batch_size'], 
                                                        shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                       batch_size=len(self.test_idx), 
                                                       shuffle=False)
        if verbose:
            print(f"Data processed. Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    def initialize_model(self, verbose=False):
        spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=self.nt, dt=1, 
                                           num=self.params['num_B_spline_basis'], 
                                           add_constant_basis=True)
        spline_basis = torch.tensor(spline_basis).float().to(self.device)
        
        coupling_basis = GLM.make_pillow_basis(**{'peaks_max':self.params['coupling_basis_peaks_max'], 
                                                  'num':self.params['coupling_basis_num'], 
                                                  'nonlinear':0.5})
        coupling_basis = torch.tensor(coupling_basis).float().to(self.device)
        K = torch.tensor(get_K(nt=self.nt, L=self.params['K_tau'], sigma2=self.params['K_sigma2'])).to(self.device)

        self.model = VAETransformer_FCGPFA(
            num_layers=self.params['num_layers'], 
            d_model=self.params['d_model'],
            dim_feedforward=self.params['dim_feedforward'], 
            nl_dim=self.params['nl_dim'], 
            spline_basis=spline_basis, 
            nfactor=self.params['nfactor'], 
            nneuron_list=self.nneuron_list,
            dropout=self.params['dropout'], 
            nhead=self.params['nhead'],
            decoder_architecture=self.params['decoder_architecture'],
            npadding=self.npadding, 
            nsubspace=self.params['nsubspace'], 
            K=K, 
            nlatent=self.params['nlatent'], 
            coupling_basis=coupling_basis,
            use_self_coupling=self.params['use_self_coupling'],
            ).to(self.device)
        ################################
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        
        transformer_group = ['transformer_encoder', 'to_latent', 'token_converter']
        sti_group = ['sti_readout_matrices', 'sti_decoder', 'sti_inhomo']
        cp_group = ['cp_latents_readout', 'cp_time_varying_coef_offset', 'cp_beta_coupling', 
                    'cp_weight_sending', 'cp_weight_receiving']
        
        # Define different learning rates
        transformer_lr = self.params['lr_transformer']
        sti_lr = self.params['lr_sti']  # Higher learning rate for decoder_matrix
        cp_lr = self.params['lr_cp']  # Learning rate for coupling parameters
        weight_decay = self.params['weight_decay']

        # Configure optimizer with two parameter groups
        params_not_assigned = [n for n, p in self.model.named_parameters()
            if all([key_word not in n for key_word in transformer_group+sti_group+cp_group])]
        if len(params_not_assigned)!=0:
            print(params_not_assigned)
            raise ValueError("Some parameters are not assigned to any group.")
        self.optimizer = optim.Adam([
            {'params': [p for n, p in self.model.named_parameters() 
                        if any([key_word in n for key_word in transformer_group])], 
             'lr': transformer_lr},
            {'params': [p for n, p in self.model.named_parameters() 
                        if any([key_word in n for key_word in sti_group])], 
             'lr': sti_lr},
            {'params': [p for n, p in self.model.named_parameters() 
                        if any([key_word in n for key_word in cp_group])], 
             'lr': cp_lr},
        ], weight_decay=weight_decay)
        ################################
        if verbose:
            print(f"Model initialized. Training on {self.device}")

    def train(self, verbose=True, record_results=False, 
              fix_latents=False, fix_stmulus=False,
              only_coupling=False, only_stimulus=False):
        assert not (only_coupling and only_stimulus), 'Cannot have both only_coupling and only_stimulus'
        if verbose:
            print(f"Start training model with parameters: {self.params}")
        utils.set_seed(0)
        self.process_data(verbose=verbose)
        self.initialize_model(verbose=verbose)
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        no_improve_epoch = 0
        temp_best_model_path = self.path+'/temp_best_model.pth'
        
        # Function to adjust learning rate
        def adjust_lr(optimizer, epoch):
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
            for spikes_full_low_res_batch, spikes_full_batch in self.train_loader:
                spikes_full_low_res_batch = spikes_full_low_res_batch.to(self.device)
                spikes_full_batch = spikes_full_batch.to(self.device)
                self.optimizer.zero_grad()
                firing_rate = self.model(spikes_full_low_res_batch, 
                                                        spikes_full_batch, 
                                                        fix_latents=fix_latents, 
                                                        fix_stimulus=fix_stmulus, 
                                                        only_coupling=only_coupling,
                                                        only_stimulus=only_stimulus)
                loss = self.model.loss_function(firing_rate, spikes_full_batch[:,:,self.npadding:], 
                                                self.model.sti_mu, self.model.sti_logvar, 
                                                beta=self.params['beta'])
                if self.penalty_overlapping is not None:
                    loss += self.penalty_overlapping * self.model.overlapping_scale
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * spikes_full_batch.size(0)
            train_loss /= len(self.train_loader.dataset)

            self.model.eval()
            self.model.sample_latent = False
            test_loss = 0.0
            with torch.no_grad():
                for spikes_full_low_res_batch, spikes_full_batch in self.test_loader:
                    spikes_full_low_res_batch = spikes_full_low_res_batch.to(self.device)
                    spikes_full_batch = spikes_full_batch.to(self.device)
                    firing_rate = self.model(spikes_full_low_res_batch, 
                                                            spikes_full_batch,
                                                            fix_latents=fix_latents,
                                                            fix_stimulus=fix_stmulus,
                                                            only_coupling=only_coupling,
                                                            only_stimulus=only_stimulus)
                    loss = self.model.loss_function(firing_rate, spikes_full_batch[:,:,self.npadding:], 
                                                    self.model.sti_mu, self.model.sti_logvar, 
                                                    beta=self.params['beta'])
                    if self.penalty_overlapping is not None:
                        loss += self.penalty_overlapping * self.model.overlapping_scale
                    test_loss += loss.item() * spikes_full_batch.size(0)
            test_loss /= len(self.test_loader.dataset)
            
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

    def predict(self, return_torch=True, dataset='all',
                fix_latents=False, fix_stmulus=False,
                only_coupling=False, only_stimulus=False):
        self.model.eval()
        self.model.sample_latent = False
        sti_mu_list = []
        sti_logvar_list = []
        firing_rate_list = []
        if dataset == 'all':
            all_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res, self.spikes_full)
            all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.spikes_full.shape[0], shuffle=False)
        elif dataset == 'train':
            all_loader = self.train_loader
        elif dataset == 'test':
            all_loader = self.test_loader
        else:
            raise ValueError("Invalid dataset. Choose from 'all', 'train', or 'test'.")
        
        with torch.no_grad():
            for spikes_full_low_res_batch, spikes_full_batch in all_loader:
                spikes_full_low_res_batch = spikes_full_low_res_batch.to(self.device)
                spikes_full_batch = spikes_full_batch.to(self.device)
                firing_rate = self.model(spikes_full_low_res_batch, 
                                                            spikes_full_batch, 
                                                            fix_latents=fix_latents,
                                                            fix_stimulus=fix_stmulus,
                                                            only_coupling=only_coupling,
                                                            only_stimulus=only_stimulus)
                sti_mu_list.append(self.model.sti_mu)
                sti_logvar_list.append(torch.exp(0.5 * self.model.sti_logvar))
                firing_rate_list.append(firing_rate)
        if return_torch:
            return (torch.concat(firing_rate_list, dim=0).cpu(), 
                    torch.concat(sti_mu_list, dim=0).cpu(),
                    torch.concat(sti_logvar_list, dim=0).cpu())
        else:
            return (torch.concat(firing_rate_list, dim=0).cpu().numpy(), 
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
        self.process_data()
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
