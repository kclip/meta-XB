import torch
import numpy as np
from funcs.soft_quantile import Soft_Quantile
from funcs.hard_quantile import quantile_plus
from funcs.utils_for_set_prediction import compute_pred_set_as_indicator_matrix, compute_prob_vec, SVC, NC_compute

class Split_Conformal:
    def __init__(self, alpha, num_classes, c_sigmoid=None, c_sigmoid_for_ada_NC=None):
        self.alpha = alpha
        self.num_classes = num_classes
        self.c_sigmoid_for_ada_NC = c_sigmoid_for_ada_NC
        self.c_sigmoid = c_sigmoid
    def forward(self, phi, NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable,  u_for_tr_dataset, u_for_X_te, Y_te):
        # here tr_dataset is used only for calibration
        # phi is already trained with proper training set 
        num_te = X_te.shape[0]
        if quantile_mode == 'hard':
            if_soft_inf_val_differentiable = False # since only for soft
        
        N = tr_dataset[0].shape[0] 
        self.SQ = Soft_Quantile(n=N+1, tau=1-self.alpha, sq_mode=None, softquan_softmin_coeff=None)
        NC_y_dict = self.compute_NC_scores(phi, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te)
        pred_set_as_indicator_matrix = compute_pred_set_as_indicator_matrix(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, self.alpha, self.num_classes, self.SQ,  self.c_sigmoid, if_JK_mm=False, tau_for_soft_min=None)
        
        return pred_set_as_indicator_matrix
    
    def compute_NC_scores(self, curr_phi, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te):
        #  tr_dataset is used for validation set
        # fitting is already done -- curr_phi
        N = tr_dataset[0].shape[0] #len(tr_dataset)
        NC_y_dict = {} # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        # for each training point, fit model (phi) and get corresponding LOO NC score
        if 'maml' in NC_mode: # 
            curr_phi_para_list = list(map(lambda p: p[0], zip(curr_phi.parameters())))
        else:
            curr_phi_para_list = curr_phi

        for i in range(N):
            NC_y_dict[i] = {}
            curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
            curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
            prob_vec_i = compute_prob_vec(curr_phi_para_list, curr_x_tr, curr_phi, NC_mode, self.num_classes)
            if 'adaptive' in NC_mode:
                u_i = u_for_tr_dataset[i].unsqueeze(dim=0) # (1,1)
            else:
                u_i = None
            NC_i = NC_compute(prob_vec_i, torch.squeeze(curr_y_tr), NC_mode, u_i, self.c_sigmoid_for_ada_NC)
            NC_y_dict[i]['NC_i'] = NC_i # (1, 1)
            # for each training point, get corresponding NC score for new point (X_te: num_te, *)
            for y_prime in range(self.num_classes):
                prob_vec_y_prime = compute_prob_vec(curr_phi_para_list, X_te, curr_phi, NC_mode, self.num_classes) # (num_te, num_classes)
                NC_y_prime = NC_compute(prob_vec_y_prime, y_prime, NC_mode, u_for_X_te, self.c_sigmoid_for_ada_NC)
                NC_y_dict[i][y_prime] = NC_y_prime # (num_te, 1) 
                # this is same for every i but for the consistency with code with CV, we keep it as this.
        return NC_y_dict



def Trans(vector):
    return torch.transpose(vector,0,1)


