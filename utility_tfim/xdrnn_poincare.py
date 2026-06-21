import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import poincare_util as util 

###################################################################################################333

class Poincare2dRNNcell(nn.Module):
    def __init__(self, num_units, num_in, r_max, inputs_geom='eucl', bias_geom='hyp', c_val=1.0, non_lin='id'):
        super(Poincare2dRNNcell, self).__init__()

        self._num_in = num_in # same as input_dim in 1D version
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units
        self.c_val = torch.tensor(c_val)
        self.inputs_geom = inputs_geom
        self.bias_geom = bias_geom
        self.non_lin = non_lin
        self.r_max = r_max

        self.Wh = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.Wv = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.Uh = nn.Parameter(torch.empty(self._num_in,self._num_units))
        self.Uv = nn.Parameter(torch.empty(self._num_in,self._num_units))
        self.b = nn.Parameter(torch.zeros(1, self._num_units))

        self.register_buffer('current_max_r', torch.zeros(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        #initializer=tf.keras.initializers.glorot_normal() in TF version
        nn.init.xavier_uniform_(self.Wh, gain=0.01)
        nn.init.xavier_uniform_(self.Wv, gain=0.01)
        nn.init.xavier_uniform_(self.Uh, gain=0.01)
        nn.init.xavier_uniform_(self.Uv, gain=0.01)
        nn.init.zeros_(self.b)

    def norm_clamp(self,h):
        # restrict the vector to Rmax value
        norm = torch.norm(h, dim=-1, keepdim=True)
        rescale = torch.where(norm > self.r_max, self.r_max / (norm + 1e-7), torch.ones_like(norm))
        h = h * rescale
        return h

    def project(self, x, eps=1e-5):
        # Boundary for c=1 is 1.0. Keeps hidden state inside Poincare Ball
        max_norm = (1.0 - eps) / (self.c_val**0.5)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm > max_norm
        projected = x * (max_norm / (norm + 1e-10))
        return torch.where(cond, projected, x)


    def forward(self, inputs, states):
        hyp_b = self.b if self.bias_geom == 'hyp' else util.th_exp_map_zero(self.b, self.c_val)
        # apply R_max
        state_0 = self.norm_clamp(states[0])
        state_1 = self.norm_clamp(states[1])

        input_mul_left = self.norm_clamp(util.th_mob_mat_mul(self.Uh, inputs[0], self.c_val)) #Horizontal
        input_mul_up = self.norm_clamp(util.th_mob_mat_mul(self.Uv, inputs[1], self.c_val)) #Vertical

        state_mul_left = self.norm_clamp(util.th_mob_mat_mul(self.Wh, state_0, self.c_val)) #Horizontal
        state_mul_up = self.norm_clamp(util.th_mob_mat_mul(self.Wv, state_1, self.c_val))#Vertical

        # Euclidean version: res = input_mul_left + state_mul_left + input_mul_up + state_mul_up  + self.b
        # applying second Rmax clamp
        pre_res= self.norm_clamp(util.th_mob_add(util.th_mob_add(util.th_mob_add(util.th_mob_add(input_mul_left, state_mul_left, self.c_val), 
                                        input_mul_up,self.c_val), state_mul_up, self.c_val),hyp_b, self.c_val))
        res= util.th_hyp_non_lin(pre_res, non_lin=self.non_lin, hyp_output=True, c=self.c_val)

        new_state = self.project(res)

        with torch.no_grad():
            current_norms = torch.norm(new_state, dim=-1)
            # Update the buffer with the highest radius found in this batch/step
            self.current_max_r.copy_(torch.max(current_norms).detach())

        return new_state, new_state

    def get_manifold_parameters(self):
        eucl_params, hyp_params = [], []
        for name, param in self.named_parameters():
            if name.startswith('b'): hyp_params.append(param)
            else: eucl_params.append(param)
        return eucl_params, hyp_params
