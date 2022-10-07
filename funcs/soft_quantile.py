import torch
import numpy as np
from funcs.hard_quantile import quantile_plus, soft_quantile_via_softmin

## softquan_softmin_coeff = c_Q in the paper

## depending on sq_mode, if sq_mode=='OT': soft quantile via OT (https://arxiv.org/abs/1905.11885)
## if sq_mode == 'pinball', proposed soft quantile via pinball (placed in /funcs/hard_quantile.py (of course this it not 'hard' but just placed in that dir.))

class Soft_Quantile:
    def __init__(self, n, tau, sq_mode, softquan_softmin_coeff, if_compute_rank=False, c_sigmoid_for_ada_NC = 1.0, t=None, epsilon_entropy=0.01, p_for_h=2, sinkhorn_th=0.01, max_iter=2000):
        # n: number of samples
        # tau would be 1-alpha
        # vec_a: (n, 1)
        # vec_x: (num_te, n, 1)
        # vec_y, vec_b: (m, 1)
        if t is None:
            t = 1/n
        self.vec_a = torch.ones(n,1)/n # (n,1)
        self.n = n

        self.sq_mode = sq_mode
        self.tau = tau

        if if_compute_rank:
            self.m = n
            self.vec_b = torch.ones(self.m,1)/self.m # (n,1)
            self.vec_y = torch.zeros(self.vec_b.shape)
            for ind_m in range(self.vec_y.shape[0]): # strictly increasing vector
                self.vec_y[ind_m] = (ind_m)/(self.vec_y.shape[0]-1)
        else: # soft quantile
            self.vec_y = torch.unsqueeze(torch.tensor([0, 0.5, 1]), 1) # (3,1)
            self.vec_b = torch.unsqueeze(torch.tensor([tau-t/2, t, 1-tau-t/2]), 1) # (3,1)
            self.m = 3
        self.epsilon_entropy = epsilon_entropy
        self.p_for_h = p_for_h
        self.ones_n = torch.ones(n,1)
        self.ones_m = torch.ones(self.m,1)
        self.sinkhorn_th = sinkhorn_th
        self.max_iter = max_iter  # sinkhorn max iter
        self.c_sigmoid_for_ada_NC = c_sigmoid_for_ada_NC # for better sigmoid!
        self.softquan_softmin_coeff = softquan_softmin_coeff # smoothness para (d)
        assert self.c_sigmoid_for_ada_NC == 1.0 # do it at soft indicator function!

    def forward(self, vec_x):
        if self.sq_mode == 'OT':
            device_info = vec_x.get_device()
            if device_info == -1:
                self.device = "cpu"
            else:
                self.device = "cuda:" + str(device_info)
            self.vec_b = self.vec_b.to(self.device)
            self.vec_a = self.vec_a.to(self.device)
            ## standardize x
            num_te = vec_x.shape[0]
            x_tilde = self.g_tilde_original(vec_x) #(num_te, n, 1)
            C = self.C_compute(x_tilde, self.vec_y.to(self.device)) # (num_te, n, m)
            alpha = torch.zeros(num_te, self.n, 1).to(self.device)
            beta = torch.zeros(num_te, self.m, 1).to(self.device)
            alpha, beta = self.sinkhorn(self.sinkhorn_th, alpha, beta, C, self.max_iter)
            soft_quantile_vec = self.soft_quantile_compute(alpha, beta, C, vec_x) # (num_te, m, 1)
            return soft_quantile_vec[:, 1]
        elif self.sq_mode == 'pinball':
            return soft_quantile_via_softmin(vec_x, self.tau, self.softquan_softmin_coeff)
        else:
            raise NotImplementedError
    
    def forward_rank(self, vec_x): # soft rank
        device_info = vec_x.get_device()
        if device_info == -1:
            self.device = "cpu"
        else:
            self.device = "cuda:" + str(device_info)
        self.vec_b = self.vec_b.to(self.device)
        self.vec_a = self.vec_a.to(self.device)
        num_te = vec_x.shape[0]
        x_tilde = self.g_tilde_original(vec_x) #(num_te, n, 1)
        C = self.C_compute(x_tilde, self.vec_y.to(self.device)) # (num_te, n, m)
        alpha = torch.zeros(num_te, self.n, 1).to(self.device)
        beta = torch.zeros(num_te, self.m, 1).to(self.device)
        alpha, beta = self.sinkhorn(self.sinkhorn_th, alpha, beta, C, self.max_iter)
        soft_ranking_vec = self.soft_ranking_compute(alpha, beta, C) # (num_te, m, 1)
        return soft_ranking_vec

    def soft_quantile_compute(self, alpha, beta, C, vec_x):
        # (m,1) * (num_te, m, n) * (num_te, n, 1)
        tmp_1 = (1/self.vec_b)  # (m, 1)
        tmp_2 = (torch.exp( -self.compute_for_R(alpha, beta, C)/self.epsilon_entropy ))
        tmp_3 = vec_x
        tmp_2_3 = torch.bmm(tmp_2, tmp_3) # (num_te, m, 1)
        tmp_1_broad = torch.ones(tmp_2_3.shape[0], 1).to(self.device) *torch.squeeze(tmp_1)    #(num_te, 1) * (m) -> (num_te, m)
        return tmp_1_broad.unsqueeze(dim=2) * (tmp_2_3)


    def soft_ranking_compute(self, alpha, beta, C):
        b_bar = torch.zeros(self.vec_b.shape).to(self.device)
        for ind_m in range(b_bar.shape[0]):
            if ind_m == 0:
                b_bar[ind_m] = self.vec_b[ind_m]
            else:
                b_bar[ind_m] = b_bar[ind_m-1] + self.vec_b[ind_m]
        tmp_1 = (1/self.vec_a)  # (n, 1)
        tmp_for_2 = -self.compute_for_R(alpha, beta, C) #(num_te, m, n)
        tmp_2 = (torch.exp( torch.transpose(tmp_for_2, 1, 2)/self.epsilon_entropy )) #(num_te, n, m)
        tmp_3 = torch.ones(tmp_for_2.shape[0], 1).to(self.device) * torch.squeeze(b_bar) # (num_te, 1) * m -> (num_te, m)
        tmp_2_3 = torch.bmm(tmp_2, tmp_3.unsqueeze(dim=2)) # (num_te, n, 1)
        tmp_1_broad = torch.ones(tmp_2_3.shape[0], 1).to(self.device) * torch.squeeze(tmp_1) # (num_te, n)
        return tmp_1_broad.unsqueeze(dim=2) * (tmp_2_3) * self.n * self.c_sigmoid_for_ada_NC  # (num_te, n, 1)

    
    def sinkhorn(self, sinkhorn_th, alpha, beta, C, max_iter):
        num_te = C.shape[0]
        cnt = 0
        delta = 9999999999
        while delta > sinkhorn_th:
            tmp_1 = self.compute_for_R(alpha, beta, C) # (num_te, m, n)
            broadcasting_1 = torch.ones(num_te, 1).to(self.device) * torch.squeeze(self.epsilon_entropy * torch.log(self.vec_b)) # (num_te,1) * m -> (num_te, m)
            beta += broadcasting_1.unsqueeze(dim=2) + self.min_epsilon( tmp_1 )  # (num_te, m, 1)
            tmp_2 =  torch.transpose(self.compute_for_R(alpha, beta, C), 1, 2) # (num_te, n, m) 
            broadcasting_2 = torch.ones(num_te, 1).to(self.device) * torch.squeeze(self.epsilon_entropy * torch.log(self.vec_a)) # (num_te, 1) * n -> (num_te, n)
            alpha += broadcasting_2.unsqueeze(dim=2).to(self.device) + self.min_epsilon( tmp_2  )   # (num_te, n, 1)
            tmp_3 = -self.compute_for_R(alpha, beta, C)/self.epsilon_entropy
            delta = self.discrepancy_for_batch( tmp_3 , self.vec_b)
            cnt += 1
            if cnt == max_iter:
                break
            else:
                pass
        return alpha, beta

        
    def compute_for_R(self, alpha, beta, C):
        tmp_1 = torch.transpose(C, 1, 2).to(self.device) # (num_te, m, n)
        tmp_2 = alpha * torch.ones(1, self.m).to(self.device) # (num_te, n, 1) * (1, m) -> (num_te, n, m)
        tmp_2 = torch.transpose(tmp_2, 1, 2) # (num_te, m, n)
        tmp_3 = beta * torch.ones(1, self.n).to(self.device) # (num_te, m, 1) * (1, n) -> (num_te, m, n)
        return tmp_1 - tmp_2 - tmp_3
            
    def discrepancy_for_batch(self, tmp_3, b_2):
        # tmp_3: (num_te, m, n)
        tmp = torch.exp(tmp_3)
        b_1 = torch.sum(tmp, dim=2) # (num_te, m)
        # b_1: (num_te, m) , b_2: (m, 1)
        tmp_b_2 = torch.ones(b_1.shape[0], 1).to(self.device) * torch.squeeze(b_2) # (num_te, 1) * (m) -> (num_te, m)
        disc_tot = pow(torch.norm(b_1 - tmp_b_2, dim=1),2) # (num_te)
        return torch.max(disc_tot) # max discrepancy among num_te

    def discrepancy_between_measures(self, b_1, b_2):
        return pow(torch.norm(b_1 - b_2),2)

    def min_epsilon(self, M):
        # M: (num_te, m', n')
        out = -self.epsilon_entropy * torch.logsumexp(-M/self.epsilon_entropy, dim=2) # (num_te, m')
        out = torch.unsqueeze(out, 2) # (num_te, m', 1)
        return out


    def h(self, u):
        # u = y_j - x_i
        tmp = torch.pow(torch.norm(u, dim=1), self.p_for_h)
        return torch.pow(torch.norm(u, dim=1), self.p_for_h)
    
    def C_compute(self, x_tilde, y):
        # x_tilde: normalized, squased x
        C = torch.zeros(x_tilde.shape[0], self.n, self.m)
        for ind_x in range(self.n):
            for ind_y in range(self.m):
                C[:, ind_x, ind_y] = self.h( y[ind_y] - x_tilde[:, ind_x]  )
        return C

    def g_tilde_original(self, x):
        # x: (num_te, n, 1)
        sum_x = torch.sum(x, dim=1) # (num_te, 1)
        tmp = sum_x * torch.ones(1,self.n).to(self.device) # (num_te, n)
        subtracted_x = x - tmp.unsqueeze(dim=2) # (num_te, n, 1)
        norm_value = torch.norm(subtracted_x, dim=1) # (num_te, 1)
        norm_value *= (1/np.sqrt(self.n)) # (num_te, 1)
        denom = norm_value * torch.ones(1,self.n).to(self.device) # (num_te, n)
        stand_x = torch.div(x, denom.unsqueeze(dim=2))
        x_tilde_tmp = (torch.atan(stand_x) + np.pi/2)/np.pi
        return x_tilde_tmp
    
def Trans(vector):
    return torch.transpose(vector,0,1)

