from itertools import combinations
import torch

def dist(a,b):
    """
    Computes euclidean distance between torch vectors.
    
    Parameters
    -----------
    a: 1D torch.tensor 
    b: 1D torch.tensor
    
    Returns
    -----------
    euclidean distance
    """
    
    return (a - b).pow(2).sum().sqrt()

def compute_pair_distances(X):
    
    CA_CA_pair_distances = torch.tensor([dist(X[i], X[j]) for (i, j) in combinations(list(range(len(X))), 2)
    if abs(i - j) > 3])
    
    return CA_CA_pair_distances

def q(coords, native_coords, beta_const = 5.0, lambda_const = 1.2):
    
    
    native_pair_distances = compute_pair_distances(native_coords)
    native_contacts_idx = native_pair_distances < 12
    r0 = native_pair_distances[native_contacts_idx]
    
    pair_distances = compute_pair_distances(coords)
    r = pair_distances[native_contacts_idx]
    
    return torch.mean(1 / (1 + torch.exp(beta_const * (r - (lambda_const * r0)))))
