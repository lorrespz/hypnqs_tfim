import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from math import ceil

from xdrnn_euclidean_tfim2d_wf import *
from xdrnn_poincare_tfim2d_wf import *
from xdrnn_lorentz_tfim2d_wf import *
from poincare_rsgd import *

import torch
import numpy as np
from math import ceil

def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, wf, lp_s=None):
    numsamples = samples.shape[0]
    N = Nx * Ny
    device = samples.device
    
    local_energies = torch.zeros(numsamples, device=device, dtype=torch.float64)

    # --- 1. Diagonal Energy  ---
    for i in range(Nx - 1):
        values = samples[:, i, :] + samples[:, i+1, :]
        valuesT = torch.ones_like(values, dtype=torch.float64)
        valuesT[values == 1] = -1.0
        local_energies += torch.sum(valuesT * (-Jz[i, :]), dim=1)

    for j in range(Ny - 1):
        values = samples[:, :, j] + samples[:, :, j+1]
        valuesT = torch.ones_like(values, dtype=torch.float64)
        valuesT[values == 1] = -1.0
        local_energies += torch.sum(valuesT * (-Jz[:, j]), dim=1)

    # --- 2. Off-Diagonal (Spin Flips) ---
    if Bx != 0:
        all_flips = []
        # We stack the flipped configurations now
        for i in range(Nx):
            for j in range(Ny):
                flipped = samples.clone()
                flipped[:, i, j] = 1 - flipped[:, i, j]
                all_flips.append(flipped)
        
        # Shape: [N * numsamples, Nx, Ny]
        queue_flips = torch.cat(all_flips, dim=0)
        
        len_flips = N * numsamples
        steps = ceil(len_flips / numsamples)
        lp_flips_list = []
        
        for i in range(steps):
            start = (i * len_flips) // steps
            end = ((i + 1) * len_flips) // steps if i < steps - 1 else len_flips
            
            with torch.no_grad():
                lp_i = wf.log_probability(queue_flips[start:end])
                lp_flips_list.append(lp_i)
        
        # Reshape flipped log-probs to [N, numsamples]
        lp_flips = torch.cat(lp_flips_list).view(N, numsamples)
        
        # --- 3. Handle original log_probs ---
        if lp_s is None:
            with torch.no_grad():
                lp_orig = wf.log_probability(samples)
        else:
            # Detach to ensure no unwanted gradients are tracked here
            lp_orig = lp_s.detach() if torch.is_tensor(lp_s) else torch.tensor(lp_s, device=device)

        # Off-diagonal contribution: -Bx * sum(exp(0.5 * (lp_flip - lp_orig)))
        # axis=0 in original sum is the N dimension (flips)
        off_diag = -Bx * torch.sum(torch.exp(0.5 * lp_flips - 0.5 * lp_orig), dim=0)
        local_energies += off_diag

    return local_energies.cpu().numpy()
#--------------------------
def cost_fn(Eloc, log_probs_):
    # Eloc: Local energies, usually a tensor of shape [numsamples]
    # log_probs_: Log-probabilities of the samples, shape [numsamples]
    
    # Formula: <log_psi * Eloc> - <Eloc> * <log_psi>
    cost = torch.mean(log_probs_ * Eloc) - torch.mean(Eloc) * torch.mean(log_probs_)
    
    return cost

def train_step(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp=None):
    opt_eucl.zero_grad()
    if opt_hyp:  opt_hyp.zero_grad()

    tsamp = wf.sample(numsamples)
    log_prob_tensor = wf.log_probability(tsamp)
    with torch.no_grad():
        Eloc = torch.tensor(Ising2D_local_energies(Jz, Bx, Nx, Ny, tsamp, wf, lp_s=log_prob_tensor))
    loss_value = cost_fn(Eloc, log_prob_tensor)
    loss_value.backward()

    # --- GRADIENT CLIPPING  ---
    # This prevents gradient explosion by capping the gradient norm
    #only clip if wf is not HypRNN which performs better w/o grad clipping
    torch.nn.utils.clip_grad_norm_(wf.model.parameters(), max_norm=1.0)    
    opt_eucl.step()
    if opt_hyp: opt_hyp.step()
    return loss_value, Eloc.numpy(), wf.model

# ---------------- Running VMC with 2DRNNs -------------------------------------
def run_2DTFIM(wf, numsteps = 200, Nx = 10, Ny = 10, Bx = +2,
                        numsamples = 80, lr1=1e-2,lr2=1e-3, var_tol = 1.0,
                        seed = 111, fname = 'results/'):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    Jz = np.ones((Nx,Ny))
    if not os.path.exists(fname): os.makedirs(fname)
    device = torch.device("cpu")

    meanEnergy, varEnergy, best_E_list = [], [], []
    max_patience = 200
    patience = 0
    eucl_vars, hyp_vars = wf.get_manifold_parameters()
    opt_eucl = torch.optim.Adam(eucl_vars, lr=lr1)
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=50, verbose=True)
    opt_hyp = None
    if hyp_vars:
        if 'Lorentz' in wf.name:
            # Lorentz hyperbolic network
            opt_hyp = torch.optim.Adam(hyp_vars, lr=lr2)
        else:
            # Poincare hyperbolic network
            opt_hyp = RSGD(params=hyp_vars, lr=lr2, c_val=wf.rnn.c_val, hyp_opt='rsgd')
        sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=50, verbose=True)

    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp)
        meanE = np.mean(E)
        varE = np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        sched_eucl.step(meanE)
        if hyp_vars:
            sched_hyp.step(meanE)
            if 'Lorentz' in wf.name:
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_seed={seed}_meanE.npy', meanEnergy)
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_seed={seed}_varE.npy', varEnergy)
            else:
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_{wf.non_lin}_seed={seed}_meanE.npy', meanEnergy)
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_{wf.non_lin}_seed={seed}_varE.npy', varEnergy)
        else:
            np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_seed={seed}_meanE.npy', meanEnergy)
            np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_seed={seed}_varE.npy', varEnergy)
        
        if step==0:
            best_E = meanE
            best_E_list.append(best_E)
            patience = 0
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.abs(varE) < var_tol:
            best_E_list.append(meanE)
            if 'Lorentz' in wf.name:
                torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_{wf.non_lin}_Lmax={wf.spatial_clamp}_seed={seed}_checkpoint.pt')
            elif 'Poincare' in wf.name:
                torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_{wf.non_lin}_rmax={wf.r_max}_seed={seed}_checkpoint.pt')
            else:
                torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_seed={seed}_checkpoint.pt') 
            print(f'Best model saved at step {step} with E={meanE:.5f} and var = {varE:.5f}')
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"Early stopping at step {step} - no improvement for {max_patience} steps.")
            break

        lorentz_h_max = []
        poincare_h_max = []
        euclidean_h_max=[]
        if step % 10 == 0:        
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            if hyp_vars:
                curr_lr_h = opt_hyp.param_groups[0]['lr']
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}|varE: {varE:.5f}| LR: {curr_lr_e:.2e}| Hyp LR: {curr_lr_h:.2e}')
                if 'Lorentz' in wf.name:
                    max_h0 = wf.model.rnn.max_h0.item()
                    max_h_spatial_norm = wf.model.rnn.max_spatial_norm.item()
                    max_h_violation = wf.model.rnn.max_violation.item()
                    lorentz_h_max.append([max_h0, max_h_spatial_norm])
                    np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_seed={seed}_max_h.npy', lorentz_h_max)
                    print(f'max_h0 = {max_h0} | max_h_spatial_norm = {max_h_spatial_norm}| max_h_violation = {max_h_violation}')   
                else:
                    r_val = wf.model.rnn.current_max_r.item()                  
                    print(f' Max Radius R_max: {r_val:.4f}')
                    poincare_h_max.append(r_val)
                    np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_{wf.non_lin}_seed={seed}_max_h.npy', poincare_h_max)
            else:
                h_val = wf.model.rnn.current_max_h.item()              
                euclidean_h_max.append(h_val)
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_rmax={wf.r_max}_seed={seed}_max_h.npy', euclidean_h_max)    
                print(f' Max value of hidden state: {h_val:.4f}')
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}, |LR: {curr_lr_e:.2e}')
        
    return meanEnergy, varEnergy

