import torch
import torch.nn as nn
import torch.nn.functional as F
import math

###################################################################################################333

class Euclidean2dRNNcell(nn.Module):

    def __init__(self, num_units, num_in):
        super(Euclidean2dRNNcell, self).__init__()

        self._num_in = num_in
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units

        self.Wh = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.Wv = nn.Parameter(torch.empty(self._num_units, self._num_units))

        self.Uh = nn.Parameter(torch.empty(self._num_in,self._num_units))
        self.Uv = nn.Parameter(torch.empty(self._num_in,self._num_units))

        self.b = nn.Parameter(torch.zeros(1, self._num_units))
        self.register_buffer('current_max_h', torch.zeros(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        #initializer=tf.keras.initializers.glorot_normal() in TF version
        nn.init.xavier_uniform_(self.Wh, gain=0.01)
        nn.init.xavier_uniform_(self.Wv, gain=0.01)
        nn.init.xavier_uniform_(self.Uh, gain=0.01)
        nn.init.xavier_uniform_(self.Uv, gain=0.01)
        nn.init.zeros_(self.b)

    def forward(self, inputs, states):

        # prepare input linear combination
        input_mul_left = torch.matmul(inputs[0], self.Uh) #Horizontal
        input_mul_up = torch.matmul(inputs[1], self.Uv) #Vertical

        state_mul_left = torch.matmul(states[0], self.Wh) #Horizontal
        state_mul_up = torch.matmul(states[1], self.Wv)  #Vertical

        preact = input_mul_left + state_mul_left + input_mul_up + state_mul_up  + self.b 
        new_state = F.elu(preact) 

        with torch.no_grad():
            current_norms = torch.norm(new_state, dim=-1)
            # Update the buffer with the highest radius found in this batch/step
            self.current_max_h.copy_(torch.max(current_norms).detach())

        return new_state, new_state

    def get_manifold_parameters(self):
        return list(self.parameters()), []
