import torch
import torch.optim as optim
import json
import os
import numpy as np

from Transformer_splines import VAETransformer
import utility_functions as utils
import GLM

'''
First, load data "spikes", set path, set hyperparameters, and use these three to create a Trainer object.
Then, call the trainer.train() method to train the model, which use early stop. 
If the results is good, you can save the model along with hyperparameters (aka the trainer) 
    by calling trainer.save_model_and_hp() method.
'''

class Trainer:
    def __init__(self, spikes, path, params=None):
        self.spikes = spikes
        self.path = path
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.results_file = "training_results.json"
        
        ### Get some dependent parameters
        self.narea = len(self.spikes)
        self.nneuron_list = [sp.shape[1] for sp in self.spikes]
        self.nt, self.ntrial = self.spikes[0].shape[2], self.spikes[0].shape[0]
        self.d_model = sum(self.nneuron_list)

    def process_data(self, verbose=False):
        ### Get tokenized spike trains by downsampling
        self.spikes_full = torch.tensor(np.concatenate(self.spikes, axis=1)).float()
        spikes_low_res = utils.change_temporal_resolution(self.spikes, self.params['num_merge'])
        self.spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float()
        
        ### Splitting data into train and test sets
        indices = list(range(self.ntrial))
        split = int(np.floor(0.8 * self.ntrial))
        utils.set_seed(0)
        np.random.shuffle(indices)
        self.train_idx, self.test_idx = indices[:split], indices[split:]
        train_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[self.train_idx], 
                                                       self.spikes_full[self.train_idx])
        test_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[self.test_idx], 
                                                      self.spikes_full[self.test_idx])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=False)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.params['batch_size'], shuffle=False)
        if verbose:
            print(f"Data processed. Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    def initialize_model(self, verbose=False):
        spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=self.nt, dt=1, num=self.params['num_B_spline_basis'], 
                                             add_constant_basis=True)
        spline_basis = torch.tensor(spline_basis).float().to(self.device)

        self.model = VAETransformer(num_layers=self.params['num_layers'], 
                                    dim_feedforward=self.params['dim_feedforward'], 
                                    nl_dim=self.params['nl_dim'], 
                                    spline_basis=spline_basis, 
                                    nfactor=self.params['nfactor'], 
                                    nneuron_list=self.nneuron_list,
                                    dropout=self.params['dropout'], 
                                    nhead=self.params['nhead']).to(self.device)
        ################################
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])
        
        # Define different learning rates
        standard_lr = self.params['learning_rate']
        decoder_lr = self.params['learning_rate_decoder']  # Higher learning rate for decoder_matrix

        # Configure optimizer with two parameter groups
        self.optimizer = optim.Adam([
            {'params': [p for n, p in self.model.named_parameters() if 'decoder_fc' not in n], 'lr': standard_lr},
            {'params': self.model.decoder_fc.parameters(), 'lr': decoder_lr},
        ])
        ################################
        if verbose:
            print(f"Model initialized. Training on {self.device}")

    def train(self, verbose=True, record_results=False):
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
        def adjust_learning_rate(optimizer, epoch):
            standard_lr = self.params['learning_rate'] * (epoch + 1) / self.params['warm_up_epoch']
            decoder_lr = self.params['learning_rate_decoder'] * (epoch + 1) / self.params['warm_up_epoch']
            optimizer.param_groups[0]['lr'] = standard_lr
            optimizer.param_groups[1]['lr'] = decoder_lr
        
        ### Training and Testing Loops
        for epoch in range(self.params['max_epoch']):
            # Warm up
            if epoch < self.params['warm_up_epoch']:
                adjust_learning_rate(self.optimizer, epoch)
            self.model.train()
            self.model.training = True
            train_loss = 0.0
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                firing_rate, z, mu, logvar = self.model(data)
                loss = self.model.loss_function(firing_rate, targets, mu, logvar, beta=self.params['beta'])
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
            train_loss /= len(self.train_loader.dataset)

            self.model.eval()
            self.model.training = False
            test_loss = 0.0
            with torch.no_grad():
                for data, targets in self.test_loader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    firing_rate, z, mu, logvar = self.model(data)
                    loss = self.model.loss_function(firing_rate, targets, mu, logvar, beta=0.0)
                    test_loss += loss.item() * data.size(0)
            test_loss /= len(self.test_loader.dataset)

            if verbose:
                print(f"Epoch {epoch+1}/{self.params['max_epoch']}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
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
                if no_improve_epoch >= self.params['patience_epoch']:
                    if verbose:
                        print('Early stopping triggered.')
                    break
        
        # self.model.load_state_dict(torch.load(temp_best_model_path))
        if record_results:
            self.log_results(best_train_loss, best_test_loss)
        return best_test_loss

    def predict(self, return_torch=True, dataset='all'):
        self.model.eval()
        self.model.training = False
        mu_list = []
        std_list = []
        firing_rate_list = []
        if dataset == 'all':
            all_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res, self.spikes_full)
            all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.params['batch_size'], shuffle=False)
        elif dataset == 'train':
            all_loader = self.train_loader
        elif dataset == 'test':
            all_loader = self.test_loader
        else:
            raise ValueError("Invalid dataset. Choose from 'all', 'train', or 'test'.")
        
        with torch.no_grad():
            for data, targets in all_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                firing_rate, z, mu, logvar = self.model(data)
                mu_list.append(mu)
                std_list.append(torch.exp(0.5 * logvar))
                firing_rate_list.append(firing_rate)
        if return_torch:
            return (torch.concat(firing_rate_list, dim=0).cpu(), 
                    torch.concat(mu_list, dim=0).cpu(),
                    torch.concat(std_list, dim=0).cpu())
        else:
            return (torch.concat(firing_rate_list, dim=0).cpu().numpy(), 
                    torch.concat(mu_list, dim=0).cpu().numpy(),
                    torch.concat(std_list, dim=0).cpu().numpy())
        
    def save_model_and_hp(self):
        filename = self.path + '/best_model_and_hp.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
        }, filename)
        
    def load_model_and_hp(self, filename=None):
        if filename is None:
            print(f"Loading default model from {self.path}")
            filename = self.path + '/best_model_and_hp.pth'
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.params = checkpoint['params']
        self.process_data()
        self.initialize_model()

    def log_results(self, train_loss, test_loss):
        results = {
            "params": self.params,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        with open(self.results_file, 'a') as file:
            json.dump(results, file, indent=4, sort_keys=True)  # Indent each level by 4 spaces
            file.write('\n')  # Write a newline after each set of results
