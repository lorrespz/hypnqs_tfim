
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from lorentz_tfim_definitions import *

class wfModel(nn.Module):
    def __init__(self, rnn_layer, dense_layer):
        super(wfModel, self).__init__()
        self.rnn = rnn_layer
        self.dense = dense_layer
        self.k_tensor = rnn_layer.k_tensor
 
    def forward(self, inputs, rnn_state):        
        rnn_output, rnn_state = self.rnn(inputs, rnn_state) 
        rnn_output = project_lorentz_manual(rnn_output)   
        rnn_output_eucl = hc_math.logmap0(rnn_output,  k=self.k_tensor, is_tan_normalize=False)
        output = self.dense(rnn_output_eucl)
        return output, rnn_state 
     

class Lorentzwavefunction(object):
    def __init__(self, systemsize_x, systemsize_y, cell_type, units, spatial_clamp, non_lin, seed):
        self.cell_type = cell_type
        self.units = units
        self.inputdim = 2
        self.Nx = systemsize_x
        self.Ny = systemsize_y
        self.spatial_clamp = spatial_clamp
        self.non_lin = non_lin
        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cell_type == 'LorentzRNN':
            self.cell = LorentzRNN(self.inputdim,self.units, self.spatial_clamp, self.non_lin,  1.0)
            self.name = f'1DRNN_{cell_type}_{self.units}'
        if cell_type == 'LorentzGRU':
            self.cell = LorentzGRU(self.inputdim,self.units, self.spatial_clamp, self.non_lin, 1.0)
            self.name = f'1DRNN_{cell_type}_{self.units}'

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.rnn = self.cell
        self.k_tensor = self.rnn.k_tensor
        self.dense = nn.Sequential(nn.Linear(self.units, 2), nn.Softmax(dim=-1))
        self.model = wfModel(self.rnn, self.dense)
        self.name = f'{cell_type}_{units}'

    def get_manifold_parameters(self):
        # 1. Get the split from the Lorentz cell
        cell_eucl, cell_hyp = self.cell.get_manifold_parameters()        
        # 2. Dense layer is always Euclidean
        dense_params = list(self.dense.parameters())        
        # 3. Combine Euclidean parts
        total_eucl = cell_eucl + dense_params     
        return total_eucl, cell_hyp


    def sample(self, numsamples):
        b = np.zeros((numsamples, self.inputdim))
        inputs = torch.tensor(b, dtype=self.dtype)
        self.outputdim = self.inputdim
        self.numsamples = inputs.shape[0]

        samples = []
        #For Poincare based hypRNN/hypGRU:
        #rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype)
        # Instead of zeros, use the manifold's origin point (often called the 'pole')
        # For Lorentz, this is usually (sqrt(k), 0, 0, ...)
        rnn_state = hc_math.expmap0(torch.zeros((self.numsamples, self.units)), k=self.k_tensor)

        for ny in range(self.Ny):
            for nx in range(self.Nx):
                output, rnn_state = self.model(inputs, rnn_state)
                sample_temp = torch.multinomial(output, 1).view(-1)
                samples.append(sample_temp)
                inputs = F.one_hot(sample_temp, num_classes=self.outputdim).to(self.dtype)
        return torch.stack(samples, dim=1)

    def log_probability(self, samples):
        self.outputdim = self.inputdim
        self.numsamples = samples.shape[0]
        inputs = torch.stack([torch.zeros(self.numsamples), torch.zeros(self.numsamples)], dim=1).to(self.dtype)
        probs = []
        rnn_state = hc_math.expmap0(torch.zeros((self.numsamples, self.units)), k=self.k_tensor)

        for ny in range(self.Ny):
            for nx in range(self.Nx):
                output, rnn_state = self.model(inputs, rnn_state)
                probs.append(output)
                idx = ny * self.Nx + nx
                inputs = F.one_hot(samples[:, idx].long(), num_classes=self.outputdim).to(self.dtype)

        probs = torch.stack(probs, dim=1).to(torch.float64)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).to(torch.float64)
        return torch.sum(torch.log(torch.sum(probs * one_hot_samples, dim=2)), dim=1)
