import torch
import torch.optim as optim
import numpy as np
import os
import time
import random
from math import ceil

#2D NQS
from xdrnn_euclidean_tfim2d_wf import *
from xdrnn_poincare_tfim2d_wf import *
from xdrnn_lorentz_tfim2d_wf import *

#1D NQS
from poincare_tfim2d_1drnn_wf import *
from lorentz_tfim2d_1drnn_wf import *
from poincare_rsgd import *
from collections import defaultdict

@torch.no_grad()
def compute_correlations(trained_nqs, num_samples=50000):
    """
    Computes spin-spin correlations using the custom sample generation layout.
    Assumes the sampler.sample() returns a tensor of shape [num_samples, Nx, Ny].
    """
    # 1. Generate samples 
    # Shape: [num_samples, Nx, Ny]
    trained_nqs.model.eval()
    samples = trained_nqs.sample(num_samples)
    
    # 2. Map binary or 0/1 states to physical spin values {-1, 1}
    # If the model outputs 0 and 1, we map: 0 -> -1, 1 -> 1
    # If the output is already -1 and 1,  remove this line.
    spins = 2 * samples.to(torch.float64) - 1.0
    
    num_samples, Nx, Ny = spins.shape
    distance_groups = defaultdict(list)
    
    # 3. Calculate correlations between all unique pairs of spatial sites
    # We loop over all combinations of site 1 (x1, y1) and site 2 (x2, y2)
    for y1 in range(Ny):
        for x1 in range(Nx):
            for y2 in range(Ny):
                for x2 in range(Nx):
                    # Avoid self-correlation and double-counting (compute only upper triangle)
                    if (y2 > y1) or (y2 == y1 and x2 > x1):
                        
                        # Calculate physical Euclidean distance
                        dx = abs(x1 - x2)
                        dy = abs(y1 - y2)
                        
                        # UNCOMMENT lines below if the 2D Hamiltonian uses Periodic Boundary Conditions (PBC):
                        # dx = min(dx, Nx - dx)
                        # dy = min(dy, Ny - dy)
                        
                        r = np.sqrt(dx**2 + dy**2)
                        
                        # Grab the full vector of spin configurations for these two positions
                        spin_site_1 = spins[:, x1, y1] # Shape: [num_samples]
                        spin_site_2 = spins[:, x2, y2] # Shape: [num_samples]
                        
                        # Monte Carlo average: mean( s_1 * s_2 ) across all samples
                        c_ij = torch.mean(spin_site_1 * spin_site_2).item()
                        
                        distance_groups[r].append(c_ij)
                        
    # 4. Collapse distance groups into their mean correlation averages
    unique_r = []
    mean_C_r = []
    
    for r in sorted(distance_groups.keys()):
        unique_r.append(r)
        mean_C_r.append(np.mean(distance_groups[r]))
        
    return np.array(unique_r), np.array(mean_C_r)


@torch.no_grad()
def compute_correlations_1d(trained_nqs, num_samples=50000):
    """
    Computes 1D spin-spin correlations.
    Assumes sample() returns [num_samples, L] where L = Nx * Ny.
    """
    trained_nqs.model.eval()
    samples = trained_nqs.sample(num_samples) # Shape: [num_samples, L]

    # Map to spins {-1, 1}
    spins = 2 * samples.to(torch.float64) - 1.0
    num_samples, L = spins.shape

    distance_groups = defaultdict(list)

    # Calculate correlations between all unique pairs (i, j)
    for i in range(L):
        for j in range(i + 1, L): # Only compute upper triangle

            # Simple 1D distance
            r = abs(i - j)

            # If the 1D chain has Periodic Boundary Conditions:
            # r = min(r, L - r)

            spin_i = spins[:, i]
            spin_j = spins[:, j]

            # Monte Carlo average
            c_ij = torch.mean(spin_i * spin_j).item()
            distance_groups[r].append(c_ij)

    # Collapse groups
    unique_r = sorted(distance_groups.keys())
    mean_C_r = [np.mean(distance_groups[r]) for r in unique_r]

    return np.array(unique_r), np.array(mean_C_r)


   

