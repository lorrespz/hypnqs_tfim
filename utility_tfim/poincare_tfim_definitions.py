import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import poincare_util as util 

class EuclRNN(nn.Module):
    def __init__(self, input_dim, num_units):
        super().__init__()
        self.num_units = num_units
        self.W = nn.Parameter(torch.empty(num_units, num_units))
        self.U = nn.Parameter(torch.empty(input_dim, num_units))
        self.b = nn.Parameter(torch.zeros(1, num_units))
        self.reset_parameters()

    def reset_parameters(self):
        # FIXED: Use consistent gain with Hyperbolic for fair comparison
        nn.init.xavier_uniform_(self.W, gain=0.01)
        nn.init.xavier_uniform_(self.U, gain=0.01)

    def forward(self, inputs, state):
        new_h = torch.tanh(torch.matmul(state, self.W) + torch.matmul(inputs, self.U) + self.b)
        return new_h, new_h

    def get_manifold_parameters(self):
        return list(self.parameters()), []

class EuclGRU(nn.Module):
    def __init__(self, input_depth, num_units):
        super().__init__()
        self._num_units = num_units
        
        for gate in ['z', 'r', 'h']:
            setattr(self, f'W{gate}', nn.Parameter(torch.empty(num_units, num_units)))
            setattr(self, f'U{gate}', nn.Parameter(torch.empty(input_depth, num_units)))
            setattr(self, f'b{gate}', nn.Parameter(torch.zeros(1, num_units)))
        
        self.reset_parameters()

    def reset_parameters(self):
        # FIXED: Gain of 0.01 to match Hyperbolic initialization
        for gate in ['z', 'r', 'h']:
            nn.init.xavier_uniform_(getattr(self, f'W{gate}'), gain=1.0)
            nn.init.xavier_uniform_(getattr(self, f'U{gate}'), gain=1.0)

    def forward(self, inputs, state):
        z = torch.sigmoid(torch.matmul(state, self.Wz) + torch.matmul(inputs, self.Uz) + self.bz)
        r = torch.sigmoid(torch.matmul(state, self.Wr) + torch.matmul(inputs, self.Ur) + self.br)
        h_tilde = torch.tanh(torch.matmul(r * state, self.Wh) + torch.matmul(inputs, self.Uh) + self.bh)
        new_h = (1 - z) * state + z * h_tilde
        return new_h, new_h

    def get_manifold_parameters(self):
        return list(self.parameters()), []

class HypRNN(nn.Module):
    def __init__(self, input_dim, num_units, r_max, inputs_geom='eucl', bias_geom='eucl', c_val=1.0, non_lin='id'):
        super().__init__()
        self.num_units = num_units
        self.c_val = torch.tensor(c_val)
        self.inputs_geom = inputs_geom
        self.bias_geom = bias_geom
        self.non_lin = non_lin
        self.r_max = r_max

        self.W = nn.Parameter(torch.empty(num_units, num_units))
        self.U = nn.Parameter(torch.empty(input_dim, num_units))
        self.b = nn.Parameter(torch.zeros(1, num_units))
        self.register_buffer('current_max_r', torch.zeros(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W, gain=0.01)
        nn.init.xavier_uniform_(self.U, gain=0.01)
        nn.init.zeros_(self.b)

    def project(self, x, eps=1e-5):
        # Boundary for c=1 is 1.0. Keeps hidden state inside Poincare Ball
        max_norm = (1.0 - eps) / (self.c_val**0.5)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm > max_norm
        projected = x * (max_norm / (norm + 1e-10))
        return torch.where(cond, projected, x)

    def norm_clamp(self,h):
        norm = torch.norm(h, dim=-1, keepdim=True)
        rescale = torch.where(norm > self.r_max, self.r_max / (norm + 1e-7), torch.ones_like(norm))
        h = h * rescale
        return h

    def one_rnn_transform(self, W, h, U, x, b):
        hyp_b = b if self.bias_geom == 'hyp' else util.th_exp_map_zero(b, self.c_val)
        W_otimes_h = util.th_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.th_mob_mat_mul(U, x, self.c_val)
        #W_otimes_h = self.norm_clamp(W_otimes_h)   
        #U_otimes_x = self.norm_clamp(U_otimes_x)
        
        res = util.th_mob_add(util.th_mob_add(W_otimes_h, U_otimes_x, self.c_val), hyp_b, self.c_val)
        return self.project(res)

    def forward(self, inputs, state):
        state = self.norm_clamp(state)
        hyp_x = inputs if self.inputs_geom == 'hyp' else util.th_exp_map_zero(inputs, self.c_val)
        new_h = self.one_rnn_transform(self.W, state, self.U, hyp_x, self.b)
        new_h = util.th_hyp_non_lin(new_h, non_lin=self.non_lin, hyp_output=True, c=self.c_val)

        # --- LOGGING RADIUS BEFORE CLAMP ---
        with torch.no_grad():
            # We calculate the norm here so we can use it for the buffer AND the clamp
            current_norms = torch.norm(new_h, dim=-1)
            # Update the buffer with the highest radius found in this batch/step
            self.current_max_r.copy_(torch.max(current_norms).detach())

        # Final clamp to keep it within the safety zone
        new_h = self.norm_clamp(new_h)
    
        return new_h, new_h

    def get_manifold_parameters(self):
        eucl_params, hyp_params = [], []
        for name, param in self.named_parameters():
            if name.startswith('b'): hyp_params.append(param) # Biases are Riemannian
            else: eucl_params.append(param) # Weights are Euclidean
        return eucl_params, hyp_params
        
class HypGRU(nn.Module):
    def __init__(self, input_dim, num_units, r_max, inputs_geom='eucl', bias_geom='hyp', c_val=1.0, non_lin='id'):
        super().__init__()
        self.num_units = num_units
        self.c_val = torch.tensor(c_val)
        self.inputs_geom = inputs_geom
        self.bias_geom = bias_geom
        self.non_lin = non_lin
        self.r_max = r_max
        self.register_buffer('current_max_r', torch.zeros(1))

        for gate in ['z', 'r', 'h']:
            setattr(self, f'W{gate}', nn.Parameter(torch.empty(num_units, num_units)))
            setattr(self, f'U{gate}', nn.Parameter(torch.empty(input_dim, num_units)))
            setattr(self, f'b{gate}', nn.Parameter(torch.zeros(1, num_units)))

        self.reset_parameters()

    def reset_parameters(self):
        for gate in ['z', 'r', 'h']:
            nn.init.xavier_uniform_(getattr(self, f'W{gate}'), gain=1.0)
            nn.init.xavier_uniform_(getattr(self, f'U{gate}'), gain=1.0)
            nn.init.zeros_(getattr(self, f'b{gate}'))

    def project(self, x, eps=1e-5):
        max_norm = (1.0 - eps) / (self.c_val**0.5)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        cond = norm > max_norm
        projected = x * (max_norm / (norm + 1e-10))
        return torch.where(cond, projected, x)

    def one_rnn_transform(self, W, h, U, x, b):
        hyp_b = b if self.bias_geom == 'hyp' else util.th_exp_map_zero(b, self.c_val)
        W_otimes_h = util.th_mob_mat_mul(W, h, self.c_val)
        U_otimes_x = util.th_mob_mat_mul(U, x, self.c_val)
        res = util.th_mob_add(util.th_mob_add(W_otimes_h, U_otimes_x, self.c_val), hyp_b, self.c_val)
        return self.project(res)

    def forward(self, inputs, state):
        #additional projection
        state = self.project(state, eps=1e-4)

        hyp_x = inputs if self.inputs_geom == 'hyp' else util.th_exp_map_zero(inputs, self.c_val)
        # z= Sigmoid(W_z h_{i-1}+ U_z x_i + b_z)
        z = util.th_hyp_non_lin(self.one_rnn_transform(self.Wz, state, self.Uz, hyp_x, self.bz),
                                non_lin='sigmoid', hyp_output=False, c=self.c_val)
        # r= Sigmoid(W_r h_{i-1}+ U_r x_i + b_r)
        r = util.th_hyp_non_lin(self.one_rnn_transform(self.Wr, state, self.Ur, hyp_x, self.br),
                                non_lin='sigmoid', hyp_output=False, c=self.c_val)

        #r_point_h = util.th_mob_pointwise_prod(state, r, self.c_val)
        # h_tilde = (W_h (r_i*h_{i-1}) + U_h x_i + b_h)
        r_point_h = self.project(util.th_mob_pointwise_prod(state, r, self.c_val), eps=1e-4)

        h_tilde = util.th_hyp_non_lin(self.one_rnn_transform(self.Wh, r_point_h, self.Uh, hyp_x, self.bh),
                                      non_lin=self.non_lin, hyp_output=True, c=self.c_val)
        h_tilde =self.project(h_tilde, eps=1e-4)
        
        # Update equation: h_{i-1} + z_i*(-h_{i-1} + h_tilde_i), h_{i-1} = state
        minus_h_oplus_htilde = util.th_mob_add(-state, h_tilde, self.c_val)
        update_step = util.th_mob_pointwise_prod(minus_h_oplus_htilde, z, self.c_val)
        
        pre_new_h = util.th_mob_add(state, update_step, self.c_val)
        with torch.no_grad():
            # We calculate the norm here so we can use it for the buffer AND the clamp
            current_norms = torch.norm(pre_new_h, dim=-1)
            # Update the buffer with the highest radius found in this batch/step
            self.current_max_r.copy_(torch.max(current_norms).detach())

        new_h = self.project(pre_new_h)

        return new_h, new_h

    def get_manifold_parameters(self):
        eucl_params, hyp_params = [], []
        for name, param in self.named_parameters():
            if name.startswith('b'): hyp_params.append(param)
            else: eucl_params.append(param)
        return eucl_params, hyp_params
