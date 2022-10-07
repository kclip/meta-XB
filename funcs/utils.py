import torch
import numpy as np
import random

def reset_random_seed(random_seed):
    if_fix_random_seed = True
    if if_fix_random_seed:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)
    else:
        pass


def randomly_divide_tr_te(tr_dataset, N):
    # N: for SC -- val based
    # N: for jk+ -- training
    X_tot = tr_dataset[0] # (num_samples, *)
    Y_tot = tr_dataset[1] # (num_samples, 1)

    rand_perm = torch.randperm(X_tot.shape[0])

    rand_perm_tr = rand_perm[:N]
    rand_perm_te = rand_perm[N:]

    X_tr = X_tot[rand_perm_tr]
    Y_tr = Y_tot[rand_perm_tr]

    X_te = X_tot[rand_perm_te]
    Y_te = Y_tot[rand_perm_te]

    if len(X_tr.shape) == 1:
        X_tr = X_tr.unsqueeze(dim=0)
    if len(Y_tr.shape) == 1:
        Y_tr = Y_tr.unsqueeze(dim=0)
    if len(X_te.shape) == 1:
        X_te = X_te.unsqueeze(dim=0)
    if len(Y_te.shape) == 1:
        Y_te = Y_te.unsqueeze(dim=0)

    return (X_tr, Y_tr), (X_te, Y_te)

def divide_tr_te(tr_dataset, N):
    # N: for SC -- val based
    # N: for jk+ -- training
    X_tot = tr_dataset[0] # (num_samples, *)
    Y_tot = tr_dataset[1] # (num_samples, 1)

    X_tr = X_tot[:N]
    Y_tr = Y_tot[:N]

    X_te = X_tot[N:]
    Y_te = Y_tot[N:]

    if len(X_tr.shape) == 1:
        X_tr = X_tr.unsqueeze(dim=0)
    if len(Y_tr.shape) == 1:
        Y_tr = Y_tr.unsqueeze(dim=0)
    if len(X_te.shape) == 1:
        X_te = X_te.unsqueeze(dim=0)
    if len(Y_te.shape) == 1:
        Y_te = Y_te.unsqueeze(dim=0)

    return (X_tr, Y_tr), (X_te, Y_te)