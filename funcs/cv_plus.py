import torch
import numpy as np
from funcs.soft_quantile import Soft_Quantile
from funcs.utils_for_set_prediction import compute_pred_set_as_indicator_matrix, compute_prob_vec, SVC, NC_compute
from funcs.jk_plus import reset_running_mean_bn

class CrossVal_plus:
    def __init__(self, K, alpha, xi, num_classes, lr_inner=0.1, inner_iter=1, c_sigmoid=None, c_sigmoid_for_ada_NC=1, if_JK_mm=False,  keep_grad=True, tau_for_soft_min=None, sq_mode=None, tau_for_soft_quan=None):
        self.K = K # K-fold
        self.xi = xi
        self.alpha = alpha
        self.c_sigmoid = c_sigmoid
        self.num_classes = num_classes
        self.c_sigmoid_for_ada_NC = c_sigmoid_for_ada_NC
        self.lr_inner = lr_inner
        self.inner_iter = inner_iter
        self.keep_grad = keep_grad
        self.tau_for_soft_min = tau_for_soft_min
        self.if_JK_mm = if_JK_mm
        self.sq_mode = sq_mode
        self.tau_for_soft_quan = tau_for_soft_quan
    def forward(self, NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te, Y_te):
        # make tr_dataset into X and y
        # X_te: (num_te, *)
        # u_for_tr_dataset: (N, 1)
        # u_for_X_te: (num_te, 1)
        num_te = X_te.shape[0]
        if quantile_mode == 'hard':
            if_soft_inf_val_differentiable = False # since only for soft
        N = tr_dataset[0].shape[0] #len(tr_dataset)
        assert N % self.K == 0 # same-sized folds
        # split into K-folds
        m = N//self.K
        rand_perm_full_indices = torch.randperm(N)
        dict_folds_indices = {}
        dict_folds_curr_phis = {} 
        for ind_fold in range(self.K):
            dict_folds_indices[ind_fold] = rand_perm_full_indices[ind_fold*m:ind_fold*m+m]
            dict_folds_curr_phis[ind_fold] = None
        self.SQ = Soft_Quantile(n=N+1, tau=1-self.alpha, sq_mode=self.sq_mode, softquan_softmin_coeff=self.tau_for_soft_quan)
        NC_y_dict = self.compute_NC_scores(tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, dict_folds_indices, dict_folds_curr_phis)
        pred_set_as_indicator_matrix = compute_pred_set_as_indicator_matrix(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, self.alpha, self.num_classes, self.SQ,  self.c_sigmoid, if_JK_mm=self.if_JK_mm, tau_for_soft_min=self.tau_for_soft_min)
        # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
        # using this, we can compute both inefficiency and validity for both evalulation and training
        ce_loss = 0 # not implemented here
        return pred_set_as_indicator_matrix, ce_loss
        
    def compute_NC_scores(self, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, dict_folds_indices, dict_folds_curr_phis):
        xi = self.xi
        N = tr_dataset[0].shape[0] 
        NC_y_dict = {} # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        # for each training point, fit model (phi) and get corresponding LOO NC score
        for i in range(N):
            NC_y_dict[i] = {}
            curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
            curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
            fold_ind = assigning_fold(dict_folds_indices, i)

            if dict_folds_curr_phis[fold_ind] is not None: # fitting for this fold already done
                curr_phi_LOO = dict_folds_curr_phis[fold_ind]
            else:
                LOO_fold_dataset = get_LOO_foldwise(tr_dataset, dict_folds_indices[fold_ind], N)
                if 'maml' in NC_mode: # 'maml_adaptive', 'maml_ori' ... 'maml' + '_' + 'adaptive' or 'ori'
                    curr_phi_LOO = GD(xi, LOO_fold_dataset, self.lr_inner, self.inner_iter, self.keep_grad)
                elif 'SVC' in NC_mode:
                    curr_phi_LOO = SVC(LOO_fold_dataset, self.num_classes) # curr_phi_LOO is SVC fitted model
                else:
                    raise NotImplementedError
                # save so that no need for refitting again for this fold
                dict_folds_curr_phis[fold_ind] = curr_phi_LOO
            prob_vec_i_LOO = compute_prob_vec(curr_phi_LOO, curr_x_tr, xi, NC_mode, self.num_classes)
            if 'adaptive' in NC_mode:
                u_i = u_for_tr_dataset[i].unsqueeze(dim=0) # (1,1)
            else:
                u_i = None
            NC_i = NC_compute(prob_vec_i_LOO, torch.squeeze(curr_y_tr), NC_mode, u_i, self.c_sigmoid_for_ada_NC)
            NC_y_dict[i]['NC_i'] = NC_i # (1, 1)
            # for each training point, get corresponding NC score for new point (X_te: num_te, *)
            for y_prime in range(self.num_classes):
                prob_vec_y_prime_LOO = compute_prob_vec(curr_phi_LOO, X_te, xi, NC_mode, self.num_classes) # (num_te, num_classes)
                NC_y_prime = NC_compute(prob_vec_y_prime_LOO, y_prime, NC_mode, u_for_X_te, self.c_sigmoid_for_ada_NC)
                NC_y_dict[i][y_prime] = NC_y_prime # (num_te, 1) 
        return NC_y_dict
        
def Trans(vector):
    return torch.transpose(vector,0,1)

def GD(xi, LOO_dataset, lr_inner, inner_iter, keep_grad):
    # LOO_dataset: (X_tr, Y_tr) # X_tr: N-1, *, Y_tr: N-1, 1
    X_tr = LOO_dataset[0]
    Y_tr = LOO_dataset[1]
    # batch norm reset to ensure exchangeability
    reset_running_mean_bn(xi)
    para_list_from_net = list(map(lambda p: p[0], zip(xi.parameters())))
    for iter in range(inner_iter):
        if iter == 0:
            out = xi(X_tr, para_list_from_net, xi.bn_running_stats, xi.alpha_TN)
            loss = torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr))
            grad = torch.autograd.grad(loss, para_list_from_net, create_graph=keep_grad)
            intermediate_updated_para_list = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, para_list_from_net)))
        else:
            reset_running_mean_bn(xi)
            out = xi(X_tr, intermediate_updated_para_list, xi.bn_running_stats, xi.alpha_TN)
            loss = torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr))
            grad = torch.autograd.grad(loss, intermediate_updated_para_list, create_graph=keep_grad)
            intermediate_updated_para_list = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, intermediate_updated_para_list)))
    reset_running_mean_bn(xi)
    _ = xi(X_tr, intermediate_updated_para_list, xi.bn_running_stats, xi.alpha_TN)
    return intermediate_updated_para_list


def assigning_fold(dict_folds_indices, sample_ind):
    for ind_fold in dict_folds_indices.keys():
        if sample_ind in dict_folds_indices[ind_fold]:
            return ind_fold
        else:
            pass

def get_LOO_foldwise(tr_dataset, excluding_indices_list, N):
    including_indices_list = [int(item) for item in torch.randperm(N) if item not in excluding_indices_list]
    LOO_X_tr = tr_dataset[0][including_indices_list] 
    LOO_Y_tr = tr_dataset[1][including_indices_list] 
    LOO_fold_dataset = (LOO_X_tr, LOO_Y_tr)
    return LOO_fold_dataset