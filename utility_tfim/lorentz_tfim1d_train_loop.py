import torch
import torch.optim as optim
import numpy as np
import os
import time
import sys
import random
from math import ceil

from lorentz_tfim1d_wf import *

# Loading Functions --------------------------
def Ising1D_local_energies_old(Jz, Bx, N, samples, wf, lp_s=None):
    numsamples = samples.shape[0]
    queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) 
    #Array to store all the diagonal and non diagonal matrix elements 
    local_energies = np.zeros((numsamples), dtype = np.float64)

    for i in range(N-1): #diagonal elements
        values = samples[:,i]+samples[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2] = +1 #If both spins are up
        valuesT[values==0] = +1 #If both spins are down
        valuesT[values==1] = -1 #If they are opposite
        local_energies += valuesT*(-Jz[i])

    queue_samples[0] = samples #storing the diagonal samples

    if Bx != 0:
        flip_queue = np.zeros((N, numsamples, N), dtype=np.int32)
        for i in range(N):  #Non-diagonal elements
            valuesT = np.copy(samples)
            #valuesT[:,i][samples[:,i]==1] = 0 #Flip
            #valuesT[:,i][samples[:,i]==0] = 1 #Flip
            valuesT[:,i] = 1 - valuesT[:,i] # Flip spin i
            flip_queue[i] = valuesT

        flip_queue_reshaped = np.reshape(flip_queue, [N * numsamples, N])

    #------------- Calculate log probs of samples + off diagonal samples -------
    # Log_probs for the flips
    lp_flips_list = []
    len_flips = N * numsamples
    steps = ceil(len_flips / numsamples) # Batching by numsamples

    for i in range(steps):
        cut = slice((i*len_flips)//steps, ((i+1)*len_flips)//steps if i < steps-1 else len_flips)
        with torch.no_grad():
            # Call RNN on the flipped configurations
            lp_i = wf.log_probability(torch.tensor(flip_queue_reshaped[cut]).to(wf.device))
            lp_flips_list.append(lp_i.cpu().numpy())

    lp_flips = np.concatenate(lp_flips_list).reshape(N, numsamples)

    # --- 3. Combine with lp_s (Original Samples) ---
    if lp_s is None:
         # If not provided, we unfortunately have to calculate the original samples now
        with torch.no_grad():
            lp_orig = wf.log_probability(samples).cpu().numpy()
    else:
        # Use the pre-calculated one from the train_step
        lp_orig = lp_s.detach().cpu().numpy() if torch.is_tensor(lp_s) else lp_s

    local_energies += -Bx * np.sum(np.exp(0.5 * lp_flips - 0.5 * lp_orig), axis=0)
    return local_energies
#----------------------------------------------------------------------------------------------------
def Ising1D_local_energies(Jz, Bx, N, samples, wf, lp_s=None):
    numsamples = samples.shape[0]

    # --- 1. Efficient Vectorized Diagonal Energies (No Loops!) ---
    # Convert binary {0, 1} spins to physical eigenvalues {1, -1}
    # If the input samples are already {1, -1}, skip this transformation step.
    spins_physical = 1 - 2 * samples  # 0 -> 1 (up), 1 -> -1 (down)

    # Calculate all Jz terms across all sites and samples instantly: s_i * s_{i+1}
    # Jz shape: (N-1,), spins_physical[:, :-1] shape: (numsamples, N-1)
    diagonal_terms = spins_physical[:, :-1] * spins_physical[:, 1:]
    local_energies = -np.sum(diagonal_terms * Jz, axis=1)

    # --- 2. Non-Diagonal Elements (Transverse Field Bx) ---
    if Bx != 0:
        # Vectorized generation of all single spin flips
        # Shape: (N, numsamples, N)
        flip_queue = np.repeat(samples[np.newaxis, :, :], N, axis=0)

        # Advanced indexing to flip the diagonal entries of the queue
        idx = np.arange(N)
        flip_queue[idx, :, idx] = 1 - flip_queue[idx, :, idx]

        # Flatten into batches for the neural network pass
        flip_queue_reshaped = flip_queue.reshape(N * numsamples, N)

        # --- 3. Compute Log Probs of Flips in Batches ---
        lp_flips_list = []
        # Step through the array cleanly using exact numsamples strides
        for i in range(0, N * numsamples, numsamples):
            batch_samples = flip_queue_reshaped[i : i + numsamples]

            with torch.no_grad():
                # Direct conversion to tensor on the proper device
                batch_tensor = torch.from_numpy(batch_samples).to(wf.device)
                lp_i = wf.log_probability(batch_tensor)
                lp_flips_list.append(lp_i.cpu().numpy())

        lp_flips = np.concatenate(lp_flips_list).reshape(N, numsamples)

        # --- 4. Combine with Baseline Log Probs ---
        if lp_s is None:
            with torch.no_grad():
                # Convert original samples safely
                orig_tensor = torch.from_numpy(samples).to(wf.device)
                lp_orig = wf.log_probability(orig_tensor).cpu().numpy()
        else:
            lp_orig = lp_s.detach().cpu().numpy() if torch.is_tensor(lp_s) else lp_s

        # Accumulate off-diagonal exponential expectations
        local_energies += -Bx * np.sum(np.exp(0.5 * lp_flips - 0.5 * lp_orig), axis=0)

    return local_energies
#----------------------------------------------------------------------------------------------------
def cost_fn(Eloc, log_probs_):
    return torch.mean(log_probs_ * Eloc) - torch.mean(Eloc) * torch.mean(log_probs_)

#----------------------------------------------------------------------------------------------------
def train_step(wf, numsamples, Jz, N, Bx, opt_eucl, opt_hyp=None):
    opt_eucl.zero_grad()
    if opt_hyp:  opt_hyp.zero_grad()

    tsamp = wf.sample(numsamples)
    log_prob_tensor = wf.log_probability(tsamp)
    with torch.no_grad():
        Eloc = torch.tensor(Ising1D_local_energies(Jz, Bx, N, tsamp, wf, lp_s=log_prob_tensor))
    loss_value = cost_fn(Eloc, log_prob_tensor)
    loss_value.backward()

    # --- GRADIENT CLIPPING  ---
    # This prevents gradient explosion by capping the gradient norm
    torch.nn.utils.clip_grad_norm_(wf.model.parameters(), max_norm=1.0)
    opt_eucl.step()
    if opt_hyp: opt_hyp.step()
    return loss_value, Eloc.numpy(), wf.model
#----------------------------------------------------------------------------------------------------
def run_1DTFIM(wf, numsteps=1001, N=100, Bx=+1, numsamples=100,
                    lr1=1e-2,lr2=1e-3, var_tol=0.5, seed=111, fname='results/'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    Jz = np.ones((N))
    if not os.path.exists(fname): os.makedirs(fname)
    device = torch.device("cpu")
    meanEnergy, varEnergy, best_E_list = [], [], []
    max_patience = 200
    patience = 0
    eucl_vars, hyp_vars = wf.get_manifold_parameters()

    opt_eucl = torch.optim.Adam(eucl_vars, lr=lr1)
    # We use standard Adam because these are D-dim tangent vectors (not (D+1)-dim Lorentz manifold vectors)
    opt_hyp = torch.optim.Adam(hyp_vars, lr=lr2)
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=40, verbose=True)
    sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=40, verbose=True)

    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, Jz, N, Bx, opt_eucl, opt_hyp)
        meanE, varE = np.mean(E), np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        sched_eucl.step(meanE)
        if hyp_vars:
            sched_hyp.step(meanE)
        np.save(f'{fname}/{wf.name}_N={N}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_meanE.npy', meanEnergy)
        np.save(f'{fname}/{wf.name}_{N}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_varE.npy', varEnergy)

        if step == 0:
            best_E_list.append(meanE)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.abs(varE) < var_tol:
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_N={N}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_checkpoint.pt')
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
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            curr_lr_h = opt_hyp.param_groups[0]['lr']
            #access the buffer from RNN cell:
            max_h0 = wf.model.rnn.max_h0.item()
            max_h_spatial_norm = wf.model.rnn.max_spatial_norm.item()
            max_h_violation = wf.model.rnn.max_violation.item()
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}|varE: {varE:.5f}| Hyp LR: {curr_lr_h:.2e}| LR: {curr_lr_e:.2e}')
            print(f'max_h0 = {max_h0} | max_h_spatial_norm = {max_h_spatial_norm}| max_h_violation = {max_h_violation}')
    return meanEnergy, varEnergy

