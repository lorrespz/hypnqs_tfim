import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from poincare_tfim_definitions import *
from poincare_util import *

class wfModel(nn.Module):
    def __init__(self, rnn_layer, dense_layer, is_hyp):
        super(wfModel, self).__init__()
        self.rnn = rnn_layer
        self.dense = dense_layer
        self.hyp = is_hyp

    def forward(self, inputs, rnn_state):
        if not self.hyp:
            rnn_output, rnn_state = self.rnn(inputs, rnn_state)
        if self.hyp:
            rnn_outp, rnn_state = self.rnn(inputs, rnn_state)
            rnn_output = th_log_map_zero(rnn_outp, 1.0)
        output = self.dense(rnn_output)
        return output, rnn_state

class RNNwavefunction(object):
    def __init__(self, systemsize, cell_type, units, r_max=None, seed=111):
        self.N = systemsize
        self.dtype = torch.float32
        self.units = units
        self.inputdim=2
        self.h_non_lin='tanh'
        self.r_max=r_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if cell_type == 'EuclRNN':
            self.cell = EuclRNN(self.inputdim, self.units)
            self.name = f'1DRNN_N={self.N}_{cell_type}_{self.units}'
        if cell_type == 'EuclGRU':
            self.cell = EuclGRU(self.inputdim,self.units)
            self.name = f'1DRNN_N={self.N}_{cell_type}_{self.units}'

        self.rnn = self.cell
        self.dense = nn.Sequential(nn.Linear(self.units, 2), nn.Softmax(dim=-1))
        self.model = wfModel(self.rnn, self.dense, is_hyp=False)
        self.name = f'{cell_type}_{units}'


    def get_manifold_parameters(self):
        return self.cell.get_manifold_parameters()[0] + list(self.dense.parameters()), []

    def sample(self,numsamples):
        b=np.zeros((numsamples,self.inputdim)).astype(np.float64)
        inputs = torch.tensor(b, dtype=self.dtype)

        self.outputdim = self.inputdim
        self.numsamples = inputs.shape[0]
        samples = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype)

        for n in range(self.N):
            output, rnn_state =self.model(inputs, rnn_state)
            sample_temp=torch.multinomial(output, 1).view(-1) #Sample from the probability
            samples.append(sample_temp)
            inputs = F.one_hot(sample_temp, num_classes=self.outputdim).to(self.dtype)

        self.samples = torch.stack(samples, dim=1)
        return self.samples

    def log_probability(self,samples):
        self.outputdim=self.inputdim
        self.numsamples=samples.shape[0]
        a = torch.zeros(self.numsamples, dtype=self.dtype)
        b = torch.zeros(self.numsamples, dtype=self.dtype)
        inputs = torch.stack([a, b], dim=1)

            
        probs=[]
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype)

        for n in range(self.N):  
            output, rnn_state =self.model(inputs, rnn_state)
            probs.append(output)
            inputs= F.one_hot(samples[:, n].long(), num_classes=self.outputdim).to(self.dtype)

        probs = torch.stack(probs, dim=1).to(torch.float64)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).to(torch.float64)
        self.log_probs = torch.sum(torch.log(torch.sum(probs * one_hot_samples, dim=2)), dim=1)
        return self.log_probs

class RNNwavefunction_hyp(object):
    def __init__(self, systemsize, cell_type, bias_geom, hyp_non_lin, units, r_max,seed=111):
        self.cell_type = cell_type
        self.units = units
        self.inputdim = 2
        self.N=systemsize
        self.dtype = torch.float32
        self.h_non_lin = hyp_non_lin
        self.h_bias_geo = bias_geom
        self.r_max= r_max
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if cell_type == 'HypRNN':
            self.cell = HypRNN(self.inputdim,self.units, self.r_max, 'eucl', self.h_bias_geo, 1.0, self.h_non_lin)
            self.name = f'1DRNN_N={self.N}_{cell_type}_{self.units}_{self.h_bias_geo}_{self.h_non_lin}'
        if cell_type == 'HypGRU':
            self.cell = HypGRU(self.inputdim,self.units, self.r_max, 'eucl', self.h_bias_geo, 1.0, self.h_non_lin)
            self.name = f'1DRNN_N={self.N}_{cell_type}_{self.units}_{self.h_bias_geo}_{self.h_non_lin}'

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.rnn = self.cell
        self.dense = nn.Sequential(nn.Linear(self.units, 2), nn.Softmax(dim=-1))
        self.model = wfModel(self.rnn, self.dense, is_hyp=True)
        self.name = f'{cell_type}_{units}'

    def get_manifold_parameters(self):
        cell_eucl, cell_hyp = self.cell.get_manifold_parameters()
        dense_params = list(self.dense.parameters())
        return cell_eucl + dense_params, cell_hyp

    def sample(self,numsamples):
        b=np.zeros((numsamples,self.inputdim)).astype(np.float64)
        inputs = torch.tensor(b, dtype=self.dtype)

        self.outputdim = self.inputdim
        self.numsamples = inputs.shape[0]
        samples = []
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype)

        for n in range(self.N):
            output, rnn_state =self.model(inputs, rnn_state)
            sample_temp=torch.multinomial(output, 1).view(-1) #Sample from the probability
            samples.append(sample_temp)
            inputs = F.one_hot(sample_temp, num_classes=self.outputdim).to(self.dtype)

        self.samples = torch.stack(samples, dim=1)
        return self.samples

    def log_probability(self,samples):
        self.outputdim=self.inputdim
        self.numsamples=samples.shape[0]
        a = torch.zeros(self.numsamples, dtype=self.dtype)
        b = torch.zeros(self.numsamples, dtype=self.dtype)
        inputs = torch.stack([a, b], dim=1)


        probs=[]
        rnn_state = torch.zeros((self.numsamples, self.units), dtype=self.dtype)

        for n in range(self.N):
            output, rnn_state =self.model(inputs, rnn_state)
            probs.append(output)
            inputs= F.one_hot(samples[:, n].long(), num_classes=self.outputdim).to(self.dtype)

        probs = torch.stack(probs, dim=1).to(torch.float64)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).to(torch.float64)
        self.log_probs = torch.sum(torch.log(torch.sum(probs * one_hot_samples, dim=2)), dim=1)
        return self.log_probs
