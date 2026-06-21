import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from lorentz_util_loading import *

###################################################################################################

class Lorentz2dRNNcell(nn.Module):
    def __init__(self, num_units, num_in, spatial_clamp, non_lin, k=1.0):
        super(Lorentz2dRNNcell, self).__init__()

        self._num_in = num_in # same as input_dim in 1D version
        self._num_units = num_units
        self.k = k
        self.k_tensor= torch.as_tensor(self.k, dtype=torch.float32)
        self.name = f'Lorent2dRNN_{num_units}'
        self.spatial_clamp = spatial_clamp       
        self.non_lin = non_lin
        self.norm_spatial = 20

        self.Wh = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.Wv = nn.Parameter(torch.empty(self._num_units, self._num_units))
        self.Uh = nn.Parameter(torch.empty(self._num_units, self._num_in))
        self.Uv = nn.Parameter(torch.empty(self._num_units, self._num_in))
        self.b = nn.Parameter(torch.zeros(1, self._num_units))

        self.register_buffer('max_h0', torch.tensor(1.0))            # Track h0 (the "Lorentz Factor")
        self.register_buffer('max_violation', torch.tensor(0.0))     # Track drift off the hyperboloid
        self.register_buffer('max_spatial_norm', torch.tensor(0.0))  # Track Euclidean spread
        
        self.reset_parameters()

    def reset_parameters(self):
        #initializer=tf.keras.initializers.glorot_normal() in TF version
        nn.init.xavier_uniform_(self.Wh, gain=0.01)
        nn.init.xavier_uniform_(self.Wv, gain=0.01)
        nn.init.xavier_uniform_(self.Uh, gain=0.01)
        nn.init.xavier_uniform_(self.Uv, gain=0.01)
        nn.init.zeros_(self.b)


    def project_lorentz_manual(self, x, k=1.0):
        spatial = x[..., 1:]
        
        # We only clamp to prevent INF, not to restrict the manifold
        spatial = torch.clamp(spatial, min=-self.norm_spatial, max=self.norm_spatial)     
        spatial_norm_sq = torch.sum(spatial**2, dim=-1, keepdim=True)
        
        # Based on lmath.py _inner0: -x0*sqrt(k) = -k => x0 = sqrt(k) at origin
        # Constraint: x0^2 - spatial^2 = k  => x0 = sqrt(spatial^2 + k)
        new_x0 = torch.sqrt(spatial_norm_sq + k)
        
        return torch.cat([new_x0, spatial], dim=-1)


    def _apply_spatial_clamp(self, x, eps=1e-5):
        # More aggressive scaling than 'projection_lorentz_manual'
        # 1. Force the hidden state to stay within a strict radius
        max_norm = (self.spatial_clamp - eps) / (self.k**0.5)
        
        # Calculate the spatial norm (sum of squares of all but the first component)
        spatial_norm = torch.norm(x[..., 1:], p=2, dim=-1, keepdim=True)
        
        # If the spatial norm is too high, scale it down 
        mask = (spatial_norm > max_norm).float()
        scale = (max_norm / (spatial_norm + 1e-10)) * mask + (1.0 - mask)
        
        # Apply scaling to spatial components
        x_spatial = x[..., 1:] * scale
        x_0 = torch.sqrt(self.k + torch.sum(x_spatial**2, dim=-1, keepdim=True))
        
        return torch.cat([x_0, x_spatial], dim=-1)

    def _check_manifold_violation(self, x, name=""):
        # The Hyperboloid model satisfies B(x, x) = -c
        # We calculate the deviation from -c
        x0 = x[..., 0:1]
        spatial_sq = torch.sum(x[..., 1:]**2, dim=-1, keepdim=True)
        # B(x, x) = -x0^2 + ||x_spatial||^2
        # Violation = |B(x, x) - (-c)| = |-x0^2 + ||x_spatial||^2 + c|
        violation = torch.abs(-x0**2 + spatial_sq + self.k)
        
        if torch.isnan(violation).any():
            print(f"CRITICAL: NaN detected at {name}")
            return True
            
        if violation.max() > 1e-2: # Threshold for "significant" drift
            print(f"Warning: High manifold violation at {name}: {violation.max().item():.6f}")
        return False

    # Applies a non linearity sigma to a hyperbolic h using: exp_0(sigma(log_0(h)))
    def eucl_non_lin_activation(self, eucl_h):
        if self.non_lin == 'id':
            return eucl_h
        elif self.non_lin == 'relu':
            return torch.nn.functional.relu(eucl_h)
        elif self.non_lin == 'elu':
            return torch.nn.functional.elu(eucl_h)
        elif self.non_lin == 'tanh':
            return torch.tanh(eucl_h)
        elif self.non_lin == 'sigmoid':
            return torch.sigmoid(eucl_h)

    def hyp_non_lin_activation(self, hyp_h):
        if self.non_lin == 'id':
            return hyp_h 
        else:
            eucl_h = self.eucl_non_lin_activation(hc_math.logmap0(hyp_h, k=self.k_tensor, is_tan_normalize=False))
            return self.project_lorentz_manual(hc_math.expmap0(eucl_h, k=self.k_tensor))

    def forward(self, inputs, states):
        # 1. Map Euclidean input to manifold
        hyp_x0 = self.project_lorentz_manual(hc_math.expmap0(inputs[0], k=self.k_tensor))
        hyp_x1 = self.project_lorentz_manual(hc_math.expmap0(inputs[1], k=self.k_tensor))
        # First spatial_clamp
        state_0 = self._apply_spatial_clamp(states[0])
        state_1 = self._apply_spatial_clamp(states[1])

        input_mul_left = hc_math.mobius_matvec(self.Uh, hyp_x0) #Horizontal
        input_mul_up = hc_math.mobius_matvec(self.Uv, hyp_x1)  #Vertical
        #self._check_manifold_violation(input_mul_left, "input_mul_left")
        #self._check_manifold_violation(input_mul_up, "input_mul_up")

        state_mul_left = hc_math.mobius_matvec(self.Wh, state_0)  #Horizontal
        state_mul_up = hc_math.mobius_matvec(self.Wv, state_1) #Vertical
        #self._check_manifold_violation(state_mul_left, "state_mul_left")
        #self._check_manifold_violation(state_mul_up, "state_mul_up")

        # Map bias to manifold  (bias is Euclidean params optimized with Adam)  
        hyp_b = self.project_lorentz_manual(hc_math.expmap0(self.b,  k=self.k_tensor))

        # Euclidean version: res = input_mul_left + state_mul_left + input_mul_up + state_mul_up  + self.b
        input_state_mul_left= hc_math.mobius_add(input_mul_left, state_mul_left)
        i_ml_s_mu = hc_math.mobius_add(input_state_mul_left, input_mul_up)
        mat_term = hc_math.mobius_add(i_ml_s_mu, state_mul_up)
        res = hc_math.mobius_add(mat_term, hyp_b)
        res_p =self.project_lorentz_manual(res)
        new_state = self.hyp_non_lin_activation(res_p)

        #new_state =self._apply_spatial_clamp(res)
        self._check_manifold_violation(new_state, "Final state")

        #Check all parameters relating the Lorentz geometry:
        with torch.no_grad():
            # a. Temporal Component (h0) - tells us how "curved" the state is
            # if h0=1.0 or close to 1.0: practically Euclidean space (near the origin)
            h0 = new_state[..., 0]
            self.max_h0.copy_(torch.max(h0).detach())

            # b. Spatial Spread 
            spatial_norm = torch.norm(new_state[..., 1:], p=2, dim=-1)
            self.max_spatial_norm.copy_(torch.max(spatial_norm).detach())

            # c. Manifold Violation - tells us if the J3 frustration is breaking the math
            # Calculation: |-h0^2 + ||spatial||^2 + k|
            violation = torch.abs(-(h0**2) + (spatial_norm**2) + self.k)
            self.max_violation.copy_(torch.max(violation).detach())

        return new_state, new_state

    def get_manifold_parameters(self):
        # We choose these to be Euclidean because they act as transformation matrices
        eucl_vars = [self.Wh, self.Wv, self.Uh, self.Uv]
        
        # We choose this to be Hyperbolic because it represents a point 
        # being projected onto the manifold in the forward pass.
        hyp_vars = [self.b]
        
        return eucl_vars, hyp_vars
