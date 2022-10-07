import torch
import numpy as np
from funcs.utils_for_set_prediction import compute_pred_set_as_indicator_matrix_for_auxiliary_few_shot_meta, compute_prob_vec, SVC, NC_compute

class Auxiliary_Few_Shot_Quan_Est:
    def __init__(self, alpha, corr_lambda, xi, predictor_net, num_classes, lr_inner=0.1, inner_iter=1):
        self.xi = xi
        self.alpha = alpha
        print(self.alpha)
        if self.alpha == 0:
            pass
        else:
            assert (self.alpha - np.floor(self.alpha)) > 0 # float!! due to conditional task correction!
        self.corr_lambda = corr_lambda
        self.num_classes = num_classes
        self.lr_inner = lr_inner
        self.inner_iter = inner_iter
        self.predictor_net = predictor_net
        self.c_sigmoid_for_ada_NC = None
        self.SQ = None
        self.c_sigmoid = None
        self.if_JK_mm = None
        self.tau_for_soft_min = None
    def forward(self, NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te, Y_te):
        # make tr_dataset into X and y
        # X_te: (num_te, *)
        # u_for_tr_dataset: (N, 1)
        # u_for_X_te: (num_te, 1)
        num_te = X_te.shape[0]
        if quantile_mode == 'hard':
            if_soft_inf_val_differentiable = False # since only for soft
        N = tr_dataset[0].shape[0] #len(tr_dataset)    
        NC_y_dict, corrected_quantile = self.compute_NC_scores(quantile_mode, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te)
        pred_set_as_indicator_matrix = compute_pred_set_as_indicator_matrix_for_auxiliary_few_shot_meta(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, corrected_quantile, self.num_classes, self.SQ,  self.c_sigmoid, if_JK_mm=self.if_JK_mm, tau_for_soft_min=self.tau_for_soft_min)
        # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
        # using this, we can compute both inefficiency and validity for both evalulation and training
        ce_loss = 0 # no need for any loss since meta-VB is done at file meta_tr_benchmark.py, this is only used during evaluation phase
        return pred_set_as_indicator_matrix, ce_loss

    def compute_NC_scores(self, quantile_mode,  tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te):
        xi = self.xi
        N = tr_dataset[0].shape[0] #len(tr_dataset)
        softmax = torch.nn.Softmax(dim=1)
        NC_y_dict = {} # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        NC_y_dict[0] = {}
        ## first get the quantile by predictor net
        # first compute LOOs
        LOOs_for_quantile_predictor = []
        for i in range(N):
            curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
            curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
            LOO_X_tr = torch.cat([tr_dataset[0][:i],tr_dataset[0][i+1:]], dim=0)
            LOO_Y_tr = torch.cat([tr_dataset[1][:i],tr_dataset[1][i+1:]], dim=0)
            LOO_dataset = (LOO_X_tr, LOO_Y_tr) #tr_dataset[:i]+tr_dataset[i+1:]
            if 'maml' in NC_mode: 
                curr_phi_LOO = GD(xi, LOO_dataset, self.lr_inner, self.inner_iter, False)
            else:
                raise NotImplementedError
            prob_vec_i_LOO = compute_prob_vec(curr_phi_LOO, curr_x_tr, xi, NC_mode, self.num_classes)
            if 'adaptive' in NC_mode:
                u_i = u_for_tr_dataset[i].unsqueeze(dim=0) # (1,1)
            else:
                u_i = None
            NC_i = NC_compute(prob_vec_i_LOO, torch.squeeze(curr_y_tr), NC_mode, u_i, None)
            LOOs_for_quantile_predictor.append(NC_i)
        est_quantile = self.predictor_net(LOOs_for_quantile_predictor)
        corrected_quantile = est_quantile + self.corr_lambda
        # get full phi
        # corrected_quantile use this for threshold!
        curr_phi_FULL = GD(xi, tr_dataset, self.lr_inner, self.inner_iter, False)
        prob_vec_y_prime_full = compute_prob_vec(curr_phi_FULL, X_te, xi, NC_mode, self.num_classes) # (num_te, num_classes)
        for y_prime in range(self.num_classes):
            NC_y_prime = NC_compute(prob_vec_y_prime_full, y_prime, NC_mode, u_for_X_te, self.c_sigmoid_for_ada_NC)
            NC_y_dict[0][y_prime] = NC_y_prime # (num_te, 1) 
        return NC_y_dict, corrected_quantile
        
    
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
    # just to update BN
    reset_running_mean_bn(xi)
    _ = xi(X_tr, intermediate_updated_para_list, xi.bn_running_stats, xi.alpha_TN)
    return intermediate_updated_para_list


def reset_running_mean_bn(xi):
    using_bn = False
    bn_running_stats = []
    for name, m in xi.named_modules():
        if 'bn' in name:
            bn_running_stats.append([None, None])
            using_bn = True
    xi.bn_running_stats = bn_running_stats
    if using_bn:
        pass
    else:
        xi.bn_running_stats = None
