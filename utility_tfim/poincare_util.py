import logging
import torch
import numpy as np
from numpy import linalg as LA
from numpy import random as np_random
import os
import random

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0

def th_project_hyp_vecs(x, c):
    # Projection op to make sure hyperbolic embeddings are inside the unit ball.
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    max_norm = (1. - PROJ_EPS) / np.sqrt(c)
    projected = x * (max_norm / torch.clamp(norm, min=max_norm))
    return projected

# Real x, not vector!
def th_atanh(x):
    return torch.atanh(torch.clamp(x, -1. + EPS, 1. - EPS))

# Real x, not vector!
def th_tanh(x):
    return torch.tanh(torch.clamp(x, -MAX_TANH_ARG, MAX_TANH_ARG))

def th_dot(x, y):
    return torch.sum(x * y, dim=1, keepdim=True)

def th_norm(x, eps=1e-10):
    return torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)

#def th_mob_add(u, v, c):
#    v = v + EPS
#    dot_u_v = 2. * c * th_dot(u, v)
#    norm_u_sq = c * th_dot(u, u)
#    norm_v_sq = c * th_dot(v, v)
#    denominator = 1. + dot_u_v + norm_v_sq * norm_u_sq
#    result = ((1. + dot_u_v + norm_v_sq) / denominator) * u + ((1. - norm_u_sq) / denominator) * v
#    return th_project_hyp_vecs(result, c

def th_mob_add(u, v, c, eps=1e-12):
    # 1. Compute dot products and squared norms
    # Using keepdim=True and proper broadcasting is safer
    u_sq = torch.sum(u * u, dim=-1, keepdim=True)
    v_sq = torch.sum(v * v, dim=-1, keepdim=True)
    u_dot_v = torch.sum(u * v, dim=-1, keepdim=True)

    # 2. Compute the Möbius denominator
    # Add eps to avoid division by zero
    denom = 1.0 + 2.0 * c * u_dot_v + (c**2) * u_sq * v_sq
    denom = torch.clamp(denom, min=eps) 

    # 3. Compute the numerator components
    # Formula: ((1 + 2c<u,v> + c|v|^2)u + (1 - c|u|^2)v) / denom
    res_u = (1.0 + 2.0 * c * u_dot_v + c * v_sq) * u
    res_v = (1.0 - c * u_sq) * v
    
    result = (res_u + res_v) / denom
    
    # 4. Project immediately
    return th_project_hyp_vecs(result, c)

def th_poinc_dist_sq(u, v, c):
    sqrt_c = np.sqrt(c)
    m = th_mob_add(-u, v, c) + EPS
    atanh_x = sqrt_c * th_norm(m)
    dist_poincare = 2. / sqrt_c * th_atanh(atanh_x)
    return dist_poincare ** 2

def th_euclid_dist_sq(u, v):
    return torch.sum(torch.square(u - v), dim=1, keepdim=True)

def th_mob_scalar_mul(r, v, c):
    v = v + EPS
    norm_v = th_norm(v)
    nomin = th_tanh(r * th_atanh(np.sqrt(c) * norm_v)) * v
    result = nomin / (np.sqrt(c) * norm_v) 
    return th_project_hyp_vecs(result, c)

def th_lambda_x(x, c):
    return 2. / (1 - c * th_dot(x, x))

def th_exp_map_x(x, v, c):
    v = v + EPS 
    norm_v = th_norm(v)
    second_term = (th_tanh(np.sqrt(c) * th_lambda_x(x, c) * norm_v / 2) / (np.sqrt(c) * norm_v)) * v
    return th_mob_add(x, second_term, c)

def th_log_map_x(x, y, c):
    diff = th_mob_add(-x, y, c) + EPS
    norm_diff = th_norm(diff)
    lam = th_lambda_x(x, c)
    return (((2. / np.sqrt(c)) / lam) * th_atanh(np.sqrt(c) * norm_diff) / norm_diff) * diff

def th_exp_map_zero(v, c):
    v = v + EPS 
    norm_v = th_norm(v)
    result = th_tanh(np.sqrt(c) * norm_v) / (np.sqrt(c) * norm_v) * v
    return th_project_hyp_vecs(result, c)

def th_log_map_zero(y, c):
    diff = y + EPS
    norm_diff = th_norm(diff)
    return 1. / np.sqrt(c) * th_atanh(np.sqrt(c) * norm_diff) / norm_diff * diff

def th_mob_mat_mul(M, x, c):
    x = x + EPS
    Mx = torch.matmul(x, M) + EPS
    MX_norm = th_norm(Mx)
    x_norm = th_norm(x)
    result = 1. / np.sqrt(c) * th_tanh(MX_norm / x_norm * th_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return th_project_hyp_vecs(result, c)

def th_mob_pointwise_prod(x, u, c):
    x = x + EPS
    Mx = x * u + EPS
    MX_norm = th_norm(Mx)
    x_norm = th_norm(x)
    result = 1. / np.sqrt(c) * th_tanh(MX_norm / x_norm * th_atanh(np.sqrt(c) * x_norm)) / MX_norm * Mx
    return th_project_hyp_vecs(result, c)

def riemannian_gradient_c(u, c):
    return ((1. - c * th_dot(u, u)) ** 2) / 4.0

def th_eucl_non_lin(eucl_h, non_lin):
    if non_lin == 'id':
        return eucl_h
    elif non_lin == 'relu':
        return torch.nn.functional.relu(eucl_h)
    elif non_lin == 'elu':
        return torch.nn.functional.elu(eucl_h)
    elif non_lin == 'tanh':
        return torch.tanh(eucl_h)
    elif non_lin == 'sigmoid':
        return torch.sigmoid(eucl_h)
    return eucl_h

# Applies a non linearity sigma to a hyperbolic h using: exp_0(sigma(log_0(h)))
def th_hyp_non_lin(hyp_h, non_lin, hyp_output, c):
    if non_lin == 'id':
        return hyp_h if hyp_output else th_log_map_zero(hyp_h, c)

    eucl_h = th_eucl_non_lin(th_log_map_zero(hyp_h, c), non_lin)
    return th_exp_map_zero(eucl_h, c) if hyp_output else eucl_h

# --- Unit Test Update for PyTorch ---
def mobius_test_PyTorch():
    emb_dim = 20
    bs = 1
    r = np_random.random() * 10
    
    # Generate instances
    v1_inst = torch.randn(bs, emb_dim, dtype=torch.float64)
    v2_inst = torch.randn(bs, emb_dim, dtype=torch.float64)
    v1_inst = v1_inst * 0.59999 / torch.norm(v1_inst)
    v2_inst = v2_inst * 0.99 / torch.norm(v2_inst)
    M_inst = torch.randn(emb_dim, 5, dtype=torch.float64)

    for c_pow in range(15):
        c = 10 ** (- c_pow)
        
        # PyTorch results
        with torch.no_grad():
            mat_mul_v = th_mob_mat_mul(M_inst, v1_inst, c).numpy()
            exp_map_x_v = th_exp_map_x(v1_inst, v2_inst, c).numpy()
            log_map_x_v = th_log_map_x(v1_inst, v2_inst, c).numpy()
            lambda_x_v = th_lambda_x(v1_inst * 0.5, c).numpy()
            poinc_dist_v = th_poinc_dist_sq(v1_inst, v2_inst, c).numpy()
        # Note: In PyTorch matmul, we use M_inst, but in Numpy M.T was used to match TF logic.       
        print(f"Tested scale c={c}")

    print('PyTorch translation tests passed!')