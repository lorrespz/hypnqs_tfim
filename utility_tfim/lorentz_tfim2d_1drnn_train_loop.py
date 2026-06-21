import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from math import ceil
from lorentz_tfim2d_1drnn_wf import *
#####################################################################################################
#    Local energy
#####################################################################################################
def Ising2D_local_energies_old(Jz, Bx, Nx, Ny, samples, wf, lp_s=None):
    numsamples = samples.shape[0]
    queue_samples = np.zeros(((Nx*Ny+1), numsamples, Nx*Ny), dtype=np.int32)
    samples_np = samples.cpu().numpy()
    samples_reshaped = np.reshape(samples_np, [numsamples, Nx, Ny])
    N = Nx*Ny
    local_energies = np.zeros((numsamples), dtype=np.float64)
    # --- 1. Diagonal Energy ---
    for i in range(Nx-1):
        values = samples_reshaped[:,i] + samples_reshaped[:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2], valuesT[values==0], valuesT[values==1] = 1, 1, -1
        local_energies += np.sum(valuesT * (-Jz[i,:]), axis=1)

    for i in range(Ny-1):
        values = samples_reshaped[:,:,i] + samples_reshaped[:,:,i+1]
        valuesT = np.copy(values)
        valuesT[values==2], valuesT[values==0], valuesT[values==1] = 1, 1, -1
        local_energies += np.sum(valuesT * (-Jz[:,i]), axis=1)

    queue_samples[0] = samples_np
    # --- 2. Off-Diagonal (Spin Flips) Logic ---
    if Bx != 0:
        # Create queue for ONLY the flipped configurations (N configurations)
        # We don't need queue_samples[0] if lp_s is provided
        flip_queue = np.zeros((N, numsamples, N), dtype=np.int32)
        for i in range(N):
            valuesT = np.copy(samples_np)
            valuesT[:,i] = 1 - valuesT[:,i] # Flip spin i
            flip_queue[i] = valuesT
        
        # Flatten for the RNN call: (N * numsamples, N)
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
         # If not provided, calculate the original samples now
        with torch.no_grad():
            lp_orig = wf.log_probability(samples).cpu().numpy()
    else:
        # Use the pre-calculated one from the train_step
        lp_orig = lp_s.detach().cpu().numpy() if torch.is_tensor(lp_s) else lp_s

    local_energies += -Bx * np.sum(np.exp(0.5 * lp_flips - 0.5 * lp_orig), axis=0)
    return local_energies
#####################################################################################################
# Fast local energy calculations
#####################################################################################################
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, wf, lp_s=None):
    # Ensure Jz is a tensor for broadcasting/vectorized ops
    if not torch.is_tensor(Jz):
        Jz = torch.tensor(Jz, device=samples.device, dtype=torch.float64)
        
    numsamples = samples.shape[0]
    device = samples.device
    local_energies = torch.zeros(numsamples, device=device, dtype=torch.float64)
    
    # Reshape for easier spatial index management
    # Shape: [numsamples, Nx, Ny]
    s = samples.view(numsamples, Nx, Ny)
    
    # --- 1. Diagonal Energy (Replicated from old code) ---
    # X-direction interactions
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
        
    # --- 2. Off-Diagonal (Spin Flips) ---
    if Bx != 0:
        if lp_s is None:
            with torch.no_grad():
                lp_orig = wf.log_probability(samples)
        else:
            lp_orig = lp_s.detach() if torch.is_tensor(lp_s) else torch.tensor(lp_s, device=device)

        # Iterate over all sites to calculate flip probability ratios
        # We perform this site-by-site to keep memory footprint at O(numsamples * N)
        flip_ratios = torch.zeros(numsamples, device=device, dtype=torch.float64)
        
        for i in range(Nx * Ny):
            # Create a flipped version of samples on the fly
            # 1 - s flips 0 to 1 and 1 to 0
            flipped_samples = samples.clone()
            flipped_samples[:, i] = 1.0 - flipped_samples[:, i]
            
            with torch.no_grad():
                lp_flipped = wf.log_probability(flipped_samples)
            
            # Sum the contribution of each flip site
            # exp(0.5 * (lp_flipped - lp_orig)) = sqrt(Psi_flipped^2 / Psi_orig^2)
            flip_ratios += torch.exp(0.5 * (lp_flipped - lp_orig))
            
        local_energies += -Bx * flip_ratios
        
    return local_energies.cpu().numpy()
#####################################################################################################
def cost_fn(Eloc, log_probs_):
    return torch.mean(log_probs_ * Eloc) - torch.mean(Eloc) * torch.mean(log_probs_)
#####################################################################################################
#    Train step
#####################################################################################################
def train_step(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp):
    opt_eucl.zero_grad()
    opt_hyp.zero_grad()

    tsamp = wf.sample(numsamples)
    log_prob_tensor = wf.log_probability(tsamp)

    # Calculate local energies (ensure no_grad inside this function)
    with torch.no_grad():
        E = Ising2D_local_energies(Jz, Bx, Nx, Ny, tsamp, wf, log_prob_tensor)
    
    # NQS Cost function: 2 * < (E - <E>) * log_psi >
    E_mean = np.mean(E)
    E_diff = torch.tensor(E - E_mean, dtype=wf.dtype).to(wf.device)
    loss = torch.mean(E_diff * log_prob_tensor)
    loss.backward()

    # --- Clipping Strategy ---
    # Euclidean weights can handle larger gradients
    torch.nn.utils.clip_grad_norm_(opt_eucl.param_groups[0]['params'], max_norm=1.0)
    # Hyperbolic biases are sensitive because of expmap (sinh/cosh)
    torch.nn.utils.clip_grad_norm_(opt_hyp.param_groups[0]['params'], max_norm=0.5)
    opt_eucl.step()
    opt_hyp.step()
    return loss.item(), E, wf.model
#####################################################################################################
#    Train loop
#####################################################################################################
def run_2DTFIM(wf, numsteps=1001, Nx=10, Ny=10, Bx=+3, numsamples=50, 
                    lr1=1e-2,lr2=1e-2, var_tol=0.5, seed=111, fname='results/'):
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    Jz = np.ones((Nx,Ny))
    if not os.path.exists(fname): os.makedirs(fname)
    device = torch.device("cpu")
    meanEnergy, varEnergy, best_E_list = [], [], []
    max_patience = 200
    patience = 0

    #Get parameters
    eucl_vars, hyp_vars = wf.get_manifold_parameters()

    # lr1 for weights (e.g., 1e-3)
    opt_eucl = torch.optim.Adam(eucl_vars, lr=lr1)    
    # lr2 for hyperbolic biases (e.g., 1e-4 or lr1/10)
    # We use standard Adam because these are D-dim tangent vectors (not (D+1)-dim Lorentz manifold vectors)
    opt_hyp = torch.optim.Adam(hyp_vars, lr=lr2)
    sched_eucl = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_eucl, mode='min', factor=0.5, patience=40, verbose=True)
    sched_hyp = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_hyp, mode='min', factor=0.5, patience=40, verbose=True)

    for step in range(numsteps):
        cost, E, wfmodel = train_step(wf, numsamples, Jz, Nx, Ny, Bx, opt_eucl, opt_hyp)
        meanE, varE = np.mean(E), np.var(E)
        meanEnergy.append(meanE)
        varEnergy.append(varE)
        sched_eucl.step(meanE)
        sched_hyp.step(meanE)
        np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_meanE.npy', meanEnergy)
        np.save(f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_varE.npy', varEnergy)
        # Update the schedulers based on the mean energy
        
        if step == 0: 
            best_E_list.append(meanE)
            patience = 0
        elif np.real(meanE) < min(np.real(best_E_list)) and np.abs(varE) < var_tol:
            best_E_list.append(meanE)
            torch.save(wfmodel.state_dict(), f'{fname}/{wf.name}_{Nx}x{Nx}_ns={numsamples}_spatial_cl={wf.spatial_clamp}_{wf.non_lin}_checkpoint.pt')
            print(f'Best model saved at step {step} with E={meanE:.4f} and var = {varE:.4f}')
            patience = 0
        else:
            patience += 1
        if patience >= max_patience:
            print(f"Early stopping at step {step} - no improvement for {max_patience} steps.")
            break
        if step % 10 == 0: 
            curr_lr_e = opt_eucl.param_groups[0]['lr']
            curr_lr_h = opt_hyp.param_groups[0]['lr']
            #access the buffer from RNN cell:
            max_h0 = wf.model.rnn.max_h0.item()
            max_h_spatial_norm = wf.model.rnn.max_spatial_norm.item()
            max_h_violation = wf.model.rnn.max_violation.item()
            print(f'step: {step}, loss: {cost:.5f}, mean energy: {meanE:.5f}|varE: {varE:.5f}| Hyp LR: {curr_lr_h:.2e}| LR: {curr_lr_e:.2e}')
            print(f'max_h0 = {max_h0} | max_h_spatial_norm = {max_h_spatial_norm}| max_h_violation = {max_h_violation}')            
    return meanEnergy, varEnergy
