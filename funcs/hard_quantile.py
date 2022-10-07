import torch
import numpy as np
import torch.nn as nn


def quantile_plus(vec_x, tau): #tau would be 1-alpha
    assert len(vec_x.shape) == 3 # (num_te, N+1, 1)
    N = vec_x.shape[1] # this is actually N+1 in our notations
    sorted_vec_x, _ = torch.sort(vec_x, dim=1)
    ind_quantile = np.ceil((tau)*(N)) - 1 ### python index starts from 0
    return sorted_vec_x[:, int(ind_quantile)]

def soft_quantile_via_softmin(vec_x, tau, d):
    assert len(vec_x.shape) == 3 # (num_te, N+1, 1) # Goal: (num_te, 1, 1)
    # first: compute prob. weight via pinball loss
    weight_via_pin = pinball_loss(torch.transpose(vec_x, 1, 2), vec_x, tau) # torch.transpose(vec_x, 1, 2): num_te, 1, N+1
    # weight_via_pin: (num_te, N+1, 1)
    exp_neg_weight = torch.exp(-weight_via_pin/d) # (num_te, N+1, 1)
    # normalize
    exp_neg_weight_normalized = exp_neg_weight/torch.sum(exp_neg_weight, dim=1, keepdim=True) # (num_te, N+1, 1)
    return torch.sum(exp_neg_weight_normalized * vec_x, dim=1) # (num_te, 1) weighted output -> with d=0 -> should be same as hard quantile

def pinball_loss(a, x_hat, tau):
    # a: (num_te, 1, N+1)
    # x_hat: (num_te, M, 1)
    # x_hat-a, a-x_hat: (num_te, M, N+1)
    relu = nn.ReLU()
    loss = (1-tau)*torch.sum(relu(x_hat - a), dim=2, keepdim=True) + tau * torch.sum(relu(a - x_hat), dim=2, keepdim=True) # (num_te, M,1)
    return loss