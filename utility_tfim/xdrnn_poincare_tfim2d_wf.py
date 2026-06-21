import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from math import ceil
from xdrnn_poincare import *

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
            rnn_output = util.th_log_map_zero(rnn_outp, 1.0)
        output = self.dense(rnn_output)
        return output, rnn_state

class PoincareRNNwavefunction(object):
    def __init__(self,systemsize_x, systemsize_y, _units, r_max, non_lin_act, _num_in,  seed):
        self.Nx=systemsize_x 
        self.Ny=systemsize_y
        self.units = _units    
        self.inputdim=2
        self.num_in = 2
        self.non_lin = non_lin_act
        self.outputdim=self.inputdim
        self.r_max=r_max
        self.dtype = torch.float64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.name = f'Poincare2dRNN_u={self.units}'

        self.rnn=Poincare2dRNNcell(self.units, self.num_in, r_max, inputs_geom='eucl', bias_geom='hyp', c_val=1.0, non_lin=self.non_lin)
        self.dense = nn.Sequential(nn.Linear(self.units, 2), nn.Softmax(dim=-1))
        self.model = wfModel(self.rnn, self.dense, is_hyp=True)

    def get_manifold_parameters(self):
        # Correctly splits bias into hyperbolic group
        cell_eucl, cell_hyp = self.rnn.get_manifold_parameters()
        dense_params = list(self.dense.parameters())
        return cell_eucl + dense_params, cell_hyp
    

    def sample(self, numsamples):
        self.numsamples = numsamples
        device = next(self.model.parameters()).device
        
        # Initialize a tensor to hold all samples: [Nx, Ny, numsamples]
        full_samples = torch.zeros((self.Nx, self.Ny, self.numsamples), device=device, dtype=torch.long)
        
        rnn_states = {}
        inputs = {}

        # --- Boundary Conditions  ---
        for ny in range(self.Ny):
            nx_bound = -1 if ny % 2 == 0 else self.Nx
            rnn_states[f"{nx_bound}{ny}"] = torch.zeros((numsamples, self.units), dtype=self.dtype, device=device)
            inputs[f"{nx_bound}{ny}"] = torch.zeros((numsamples, self.inputdim), dtype=self.dtype, device=device)

        for nx in range(self.Nx):
            rnn_states[f"{nx}-1"] = torch.zeros((numsamples, self.units), dtype=self.dtype, device=device)
            inputs[f"{nx}-1"] = torch.zeros((numsamples, self.inputdim), dtype=self.dtype, device=device)

        # --- Sampling Loop ---
        for ny in range(self.Ny):
            # Zigzag scan
            x_range = range(self.Nx) if ny % 2 == 0 else range(self.Nx - 1, -1, -1)
            prev_x_offset = -1 if ny % 2 == 0 else 1
            
            for nx in x_range:
                # Inputs from left/right (nx_prev) and above (ny-1)
                in_x = inputs[f"{nx + prev_x_offset}{ny}"]
                in_y = inputs[f"{nx}{ny - 1}"]
                state_x = rnn_states[f"{nx + prev_x_offset}{ny}"]
                state_y = rnn_states[f"{nx}{ny - 1}"]

                # Model Call
                output, rnn_states[f"{nx}{ny}"] = self.model((in_x, in_y), (state_x, state_y))
                
                # Sampling (output must be probabilities for multinomial)
                sample_temp = torch.multinomial(output, 1).view(-1)
                
                # Store directly in tensor
                full_samples[nx, ny, :] = sample_temp
                
                # Update inputs for next step
                inputs[f"{nx}{ny}"] = F.one_hot(sample_temp, num_classes=self.outputdim).to(self.dtype)

        # Permute to [numsamples, Nx, Ny]
        self.samples = full_samples.permute(2, 0, 1)
        return self.samples


    def log_probability(self, samples):
        # samples shape: [batch, Nx, Ny]
        self.numsamples = samples.shape[0]
        device = samples.device
        
        # TF: tf.transpose(samples, perm=[1, 2, 0]) -> [Nx, Ny, batch]
        samples_T = samples.permute(1, 2, 0)
        
        # Initialize all_probs to store the output distribution for each site
        # Shape: [Nx, Ny, batch, outputdim]
        all_probs = torch.zeros((self.Nx, self.Ny, self.numsamples, self.outputdim), 
                                device=device, dtype=self.dtype)
        
        rnn_states = {}
        inputs = {}

        # --- 1. Boundary Initializations ---
        for ny in range(self.Ny):
            nx_bound = -1 if ny % 2 == 0 else self.Nx
            rnn_states[f"{nx_bound}{ny}"] = torch.zeros((self.numsamples, self.units), dtype=self.dtype, device=device)
            inputs[f"{nx_bound}{ny}"] = torch.zeros((self.numsamples, self.inputdim), dtype=self.dtype, device=device)

        for nx in range(self.Nx):
            rnn_states[f"{nx}-1"] = torch.zeros((self.numsamples, self.units), dtype=self.dtype, device=device)
            inputs[f"{nx}-1"] = torch.zeros((self.numsamples, self.inputdim), dtype=self.dtype, device=device)

        # --- 2. Autoregressive Pass (Snake Scan) ---
        for ny in range(self.Ny):
            x_range = range(self.Nx) if ny % 2 == 0 else range(self.Nx - 1, -1, -1)
            prev_x_offset = -1 if ny % 2 == 0 else 1
            
            for nx in x_range:
                # Inputs/States from the "past" (neighbors already processed)
                in_x = inputs[f"{nx + prev_x_offset}{ny}"]
                in_y = inputs[f"{nx}{ny - 1}"]
                state_x = rnn_states[f"{nx + prev_x_offset}{ny}"]
                state_y = rnn_states[f"{nx}{ny - 1}"]

                # Forward pass: self.model is the wrapper for rnn + dense + softmax
                output, rnn_states[f"{nx}{ny}"] = self.model((in_x, in_y), (state_x, state_y))
                
                # Store probabilities: output shape is [batch, outputdim]
                all_probs[nx, ny] = output
                
                # Feed ACTUAL sample site into next step (one-hot)
                # TF: tf.one_hot(samples_[nx,ny], ...)
                inputs[f"{nx}{ny}"] = F.one_hot(samples_T[nx, ny].long(), num_classes=self.outputdim).to(self.dtype)

        # --- 3. Probability Extraction and Summation ---
        # TF: tf.transpose(tf.stack(probs), perm=[2, 0, 1, 3]) 
        # Our all_probs is [Nx, Ny, batch, dim] -> we want [batch, Nx, Ny, dim]
        probs = all_probs.permute(2, 0, 1, 3)
        
        # TF: tf.one_hot(samples, depth=self.inputdim)
        one_hot_samples = F.one_hot(samples.long(), num_classes=self.inputdim).to(self.dtype)

        # TF logic: reduce_sum of log(reduce_sum(probs * one_hot, axis=3))
        # This picks the probability of the actual state and sums the logs.
        picked_probs = torch.sum(probs * one_hot_samples, dim=3)
        
        # Added tiny epsilon (1e-12) to log for numerical stability
        self.log_probs = torch.sum(torch.log(picked_probs + 1e-12), dim=(1, 2))

        return self.log_probs