import torch
import torch.optim as optim
import json
import os
import numpy as np

from bert_like_random_splines import VAETransformer
import utility_functions as utils
import GLM

'''
session_id = 757216464

### Dataset parameters
num_merge = 5
batch_size = 128

### GLM parameters
num_B_spline_basis = 15 

### Transformer encoder parameters
nl_dim = 5
num_layer = 4
dim_feedforward = 64 
nfactor = 5
nhead = 1

### Training parameters
learning_rate = 1e-2
dropout = 0.5
warm_up_epoch = 3
max_epoch = 100
patience_epoch = 5
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

    def process_data(self):
        ### Get tokenized spike trains by downsampling
        self.spikes_full = torch.tensor(np.concatenate(self.spikes, axis=1)).float()
        spikes_low_res = utils.change_temporal_resolution(self.spikes, self.params['num_merge'])
        self.spikes_full_low_res = torch.tensor(np.concatenate(spikes_low_res, axis=1)).float()
        
        ### Splitting data into train and test sets
        indices = list(range(self.ntrial))
        split = int(np.floor(0.8 * self.ntrial))
        utils.set_seed(0)
        np.random.shuffle(indices)
        train_idx, test_idx = indices[:split], indices[split:]
        train_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[train_idx], self.spikes_full[train_idx])
        test_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res[test_idx], self.spikes_full[test_idx])

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.params['batch_size'], shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.params['batch_size'], shuffle=False)

    def initialize_model(self):
        B_spline_basis = GLM.inhomo_baseline(ntrial=1, start=0, end=self.nt, dt=1, num=self.params['num_B_spline_basis'], 
                                             add_constant_basis=True)
        B_spline_basis = torch.tensor(B_spline_basis).float().to(self.device)

        self.model = VAETransformer(self.params['num_layer'], self.params['dim_feedforward'],
                                    self.params['nl_dim'], B_spline_basis, self.params['nfactor'], self.nneuron_list,
                                    self.params['dropout']).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['learning_rate'])

    def train(self, verbose=True):
        utils.set_seed(0)
        self.process_data()
        self.initialize_model()
        best_test_loss = float('inf')
        best_train_loss = float('inf')
        no_improve_epoch = 0
        temp_best_model_path = self.path+'/temp_best_model.pth'
        
        # Function to adjust learning rate
        def adjust_learning_rate(optimizer, epoch):
            lr = self.params['learning_rate'] * (epoch + 1) / self.params['warm_up_epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
        ### Training and Testing Loops
        for epoch in range(self.params['max_epoch']):
            # Warm up
            if epoch < self.params['warm_up_epoch']:
                adjust_learning_rate(self.optimizer, epoch)
            self.train()
            self.training = True
            train_loss = 0.0
            for data, targets in self.train_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                firing_rate, z, mu, logvar = self.model(data)
                loss = self.model.loss_function(firing_rate, targets, mu, logvar)
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
                print(f'Epoch {epoch+1}/{self.params['max_epoch']}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
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
        
        self.model.load_state_dict(torch.load(temp_best_model_path))
        self.log_results(best_train_loss, best_test_loss)
        return best_test_loss

    def predict_all(self):
        self.model.eval()
        self.model.training = False
        mu_list = []
        firing_rate_list = []
        all_dataset = torch.utils.data.TensorDataset(self.spikes_full_low_res, self.spikes_full)
        all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=self.params['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for data, targets in all_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                firing_rate, z, mu, logvar = self.model(data)
                mu_list.append(mu)
                firing_rate_list.append(firing_rate)
        return torch.concat(firing_rate_list, dim=1).cpu().numpy(), torch.concat(mu_list, dim=1).cpu().numpy()

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
            json.dump(results, file)
            file.write('\n')