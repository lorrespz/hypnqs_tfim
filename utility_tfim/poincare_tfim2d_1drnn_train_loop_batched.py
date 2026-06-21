import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from math import ceil
from poincare_tfim2d_1drnn import *
from poincare_rsgd import *

#####################################################################################################
#   Batched local energy calculation
#####################################################################################################
def Ising2D_local_energies_batched(Jz, Bx, Nx, Ny, samples, wf, lp_s=None, batch_size=16):
    device = samples.device
    numsamples = samples.shape[0]
    local_energies = torch.zeros(numsamples, device=device, dtype=torch.float64)
    s = samples.view(numsamples, Nx, Ny)
    
    # --- 1. Diagonal Energy (Uniform Jz=1.0) ---
    for i in range(Nx - 1):
        # s is [numsamples, Nx, Ny]
        # s[:, i, :] + s[:, i+1, :]
        values = s[:, i, :] + s[:, i+1, :]
        # mapping: values 2->1, 0->1, 1->-1
        # This is equivalent to (2*prod - 1)
        valuesT = torch.where(values == 1, -1.0, 1.0)
        local_energies += torch.sum(valuesT * (-Jz[i, :]), dim=1)

    # Y-direction interactions
    for j in range(Ny - 1):
        values = s[:, :, j] + s[:, :, j+1]
        valuesT = torch.where(values == 1, -1.0, 1.0)
        local_energies += torch.sum(valuesT * (-Jz[:, j]), dim=1)
        
    # --- 2. Off-Diagonal (Spin Flips) with Batching ---
    if Bx != 0:
        if lp_s is None:
            with torch.no_grad():
                lp_orig = wf.log_probability(samples)
        else:
            lp_orig = lp_s.detach() if torch.is_tensor(lp_s) else torch.tensor(lp_s, device=device)

        flip_ratios = torch.zeros(numsamples, device=device, dtype=torch.float64)
        N = Nx * Ny
        
        # Batching: Reduce RNN passes by 'batch_size'
        for i_start in range(0, N, batch_size):
            i_end = min(i_start + batch_size, N)
            current_batch = i_end - i_start
            
            # Create a larger input tensor [batch_size * numsamples, N]
            batch_samples = samples.repeat(current_batch, 1)
            
            # Flip sites in the batch
            # This creates 'batch_size' versions of each sample, each with a different site flipped
            for b in range(current_batch):
                site_idx = i_start + b
                batch_samples[b*numsamples : (b+1)*numsamples, site_idx] = \
                    1.0 - batch_samples[b*numsamples : (b+1)*numsamples, site_idx]
            
            with torch.no_grad():
                # One call for 'batch_size' flips
                lp_flipped_batch = wf.log_probability(batch_samples).view(current_batch, numsamples)
            
            # Accumulate contributions
            for b in range(current_batch):
                flip_ratios += torch.exp(0.5 * (lp_flipped_batch[b] - lp_orig))
            
        local_energies += -Bx * flip_ratios
        
    return local_energies.cpu().numpy()
#####################################################################################################
#    Train step (batched)
#####################################################################################################
def train_step_batched(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp=None, batch_size=16):
    opt_eucl.zero_grad()
    if opt_hyp:  opt_hyp.zero_grad()

    tsamp = wf.sample(numsamples)
    log_prob_tensor = wf.log_probability(tsamp)
    with torch.no_grad():
        Eloc = torch.tensor(Ising2D_local_energies_batched(Jz, Bx, Nx, Ny, tsamp, wf, 
                                    lp_s=log_prob_tensor, batch_size=batch_size))
    loss_value = cost_fn(Eloc, log_prob_tensor)
    loss_value.backward()

    # --- GRADIENT CLIPPING  ---
    # This prevents gradient explosion by capping the gradient norm
    torch.nn.utils.clip_grad_norm_(wf.model.parameters(), max_norm=1.0)
    
    opt_eucl.step()
    if opt_hyp: opt_hyp.step()
    return loss_value, Eloc.numpy(), wf.model
#####################################################################################################
#    Train loop (batched)
#####################################################################################################
def run_2DTFIM_batched(wf, numsteps=1001, Nx=10, Ny=10, Bx=+3, numsamples=80, batch_size=16,
                    lr1=1e-2,lr2=1e-3, var_tol=0.5, seed=111, fname='results/'):
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
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=40, verbose=True)
    opt_hyp = None
    if hyp_vars:
        opt_hyp = RSGD(params=hyp_vars, lr=lr2, c_val=wf.cell.c_val, hyp_opt='rsgd')
        sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=40, verbose=True)

    for step in range(numsteps):
        cost, E, wfmodel = train_step_batched(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp, batch_size=16)
        meanE, varE = np.mean(E), np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        sched_eucl.step(meanE)
        if hyp_vars:
            sched_hyp.step(meanE)
        np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_meanE.npy', meanEnergy)
        np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_varE.npy', varEnergy)

        if step == 0: 
            best_E_list.append(meanE)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.abs(varE) < var_tol:
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_checkpoint.pt')
            print(f'Best model saved at step {step} with E={meanE:.5f} and var = {varE:.5f}')
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f"Early stopping at step {step} - no improvement for {max_patience} steps.")
            break
        r_val_list = []
        if step % 10 == 0:        
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            if hyp_vars:
                # Access the buffer from the model's RNN layer
                r_val = wf.model.rnn.current_max_r.item() 
                r_val_list.append(r_val)
                curr_lr_h = opt_hyp.param_groups[0]['lr']
                # save the list of rmax
                np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_rmax={wf.r_max}_ns={numsamples}_rmax.npy',r_val_list)
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}| Max Radius: {r_val:.4f} | Hyp LR: {curr_lr_h:.2e}| LR: {curr_lr_e:.2e}')
            else:
                print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}, varE: {varE:.5f}, |LR: {curr_lr_e:.2e}')
        
    return meanEnergy, varEnergy