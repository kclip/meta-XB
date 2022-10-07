import torch
import numpy as np
from funcs.soft_quantile import Soft_Quantile
from funcs.utils_for_set_prediction import compute_pred_set_as_indicator_matrix, compute_prob_vec, SVC, NC_compute

# this is only when using CE regularization during `meta-training` (only used for miniimagenet case here)
CE_IMPACT_TE = 1
CE_IMPACT_LOO = 1

class Jacknife_plus:
    def __init__(self, alpha, xi, num_classes, lr_inner=0.1, inner_iter=1, c_sigmoid=None, c_sigmoid_for_ada_NC=1, if_JK_mm=False,  keep_grad=True, tau_for_soft_min=None, sq_mode=None, tau_for_soft_quan=None):
        self.xi = xi
        self.alpha = alpha
        self.c_sigmoid = c_sigmoid
        self.num_classes = num_classes
        self.c_sigmoid_for_ada_NC = c_sigmoid_for_ada_NC
        self.lr_inner = lr_inner
        self.inner_iter = inner_iter
        self.keep_grad = keep_grad
        self.if_JK_mm = if_JK_mm
        self.tau_for_soft_min = tau_for_soft_min
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
        self.SQ = Soft_Quantile(n=N+1, tau=1-self.alpha, sq_mode=self.sq_mode, softquan_softmin_coeff=self.tau_for_soft_quan)
        NC_y_dict, ce_loss_over_N = self.compute_NC_scores(quantile_mode, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te)
        pred_set_as_indicator_matrix = compute_pred_set_as_indicator_matrix(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, self.alpha, self.num_classes, self.SQ,  self.c_sigmoid, if_JK_mm=self.if_JK_mm, tau_for_soft_min=self.tau_for_soft_min)
        # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
        # using this, we can compute both inefficiency and validity for both evalulation and training
        return pred_set_as_indicator_matrix, ce_loss_over_N #pred_set, inefficiency, clustering_loss

    def compute_NC_scores(self, quantile_mode,  tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te):
        xi = self.xi
        N = tr_dataset[0].shape[0] #len(tr_dataset)
        softmax = torch.nn.Softmax(dim=1)
        #   print('num_te:', X_te.shape[0])
        NC_y_dict = {} # from i= 0, ..., N-1: {i: {y_grid: NC_ygrid}}
        # for each training point, fit model (phi) and get corresponding LOO NC score
        ce_loss_over_N = 0
        ce_loss_over_N_for_tr_dataset = 0
        for i in range(N):
            NC_y_dict[i] = {}
            curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
            curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
            LOO_X_tr = torch.cat([tr_dataset[0][:i],tr_dataset[0][i+1:]], dim=0)
            LOO_Y_tr = torch.cat([tr_dataset[1][:i],tr_dataset[1][i+1:]], dim=0)

            if len(LOO_X_tr.shape) == 1: # single sample
                LOO_X_tr = LOO_X_tr.unsqueeze(dim=0)
            if len(LOO_Y_tr.shape) == 1:
                LOO_Y_tr = LOO_Y_tr.unsqueeze(dim=0)

            LOO_dataset = (LOO_X_tr, LOO_Y_tr) #tr_dataset[:i]+tr_dataset[i+1:]
            
            
            if 'maml' in NC_mode: # 'maml_adaptive', 'maml_ori' ... 'maml' + '_' + 'adaptive' or 'ori'
                # do local adaptation once
                # xi here initialization
                # phi here being local adapted NN
                curr_phi_LOO = GD(xi, LOO_dataset, self.lr_inner, self.inner_iter, self.keep_grad)
            elif 'SVC' in NC_mode:
                curr_phi_LOO = SVC(LOO_dataset, self.num_classes) # curr_phi_LOO is SVC fitted model
            else:
                raise NotImplementedError
            prob_vec_i_LOO = compute_prob_vec(curr_phi_LOO, curr_x_tr, xi, NC_mode, self.num_classes)
            if 'adaptive' in NC_mode:
                u_i = u_for_tr_dataset[i].unsqueeze(dim=0) # (1,1)
            else:
                u_i = None
            NC_i = NC_compute(prob_vec_i_LOO, torch.squeeze(curr_y_tr), NC_mode, u_i, self.c_sigmoid_for_ada_NC)
            NC_y_dict[i]['NC_i'] = NC_i # (1, 1)

            if quantile_mode == 'hard':
                ce_loss_over_N_for_tr_dataset = 0
            else:
                ce_loss_over_N_for_tr_dataset += torch.nn.functional.cross_entropy(softmax(prob_vec_i_LOO), torch.squeeze(curr_y_tr, dim=1))
            # for each training point, get corresponding NC score for new point (X_te: num_te, *)
            prob_vec_y_prime_LOO = compute_prob_vec(curr_phi_LOO, X_te, xi, NC_mode, self.num_classes) # (num_te, num_classes)
            for y_prime in range(self.num_classes):
                NC_y_prime = NC_compute(prob_vec_y_prime_LOO, y_prime, NC_mode, u_for_X_te, self.c_sigmoid_for_ada_NC)
                NC_y_dict[i][y_prime] = NC_y_prime # (num_te, 1) 
            ### CE loss
            if quantile_mode == 'hard':
                ce_loss_over_N = 0
            else:
                ce_loss_over_N += torch.nn.functional.cross_entropy(softmax(prob_vec_y_prime_LOO), torch.squeeze(Y_te))
        ce_loss_over_N /= N
        ce_loss_over_N_for_tr_dataset /= N        
        return NC_y_dict, CE_IMPACT_TE*ce_loss_over_N + CE_IMPACT_LOO*ce_loss_over_N_for_tr_dataset

def Trans(vector):
    return torch.transpose(vector,0,1)


def GD_for_SC(xi, LOO_dataset, lr_inner, inner_iter):
    # LOO_dataset: (X_tr, Y_tr) # X_tr: N-1, *, Y_tr: N-1, 1
    X_tr = LOO_dataset[0]
    Y_tr = LOO_dataset[1]
    for iter in range(inner_iter):
        # batch norm reset to ensure exchangeability
        reset_running_mean_bn(xi)
        xi.zero_grad()
        out = xi(X_tr, None, xi.bn_running_stats, xi.alpha_TN)
        loss = torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr))
        loss.backward()
        for f in xi.parameters():
            f.data.sub_(f.grad.data * lr_inner)
    # just to update BN
    reset_running_mean_bn(xi)
    _ = xi(X_tr, None, xi.bn_running_stats, xi.alpha_TN)
    return xi
    
def GD(xi, LOO_dataset, lr_inner, inner_iter, keep_grad):
    # LOO_dataset: (X_tr, Y_tr) # X_tr: N-1, *, Y_tr: N-1, 1
    X_tr = LOO_dataset[0]
    Y_tr = LOO_dataset[1]
    # batch norm reset to ensure exchangeability
    reset_running_mean_bn(xi)
    para_list_from_net = list(map(lambda p: p[0], zip(xi.parameters())))
    for iter in range(inner_iter):
        #print('iter', iter)
        if iter == 0:
            out = xi(X_tr, para_list_from_net, xi.bn_running_stats, xi.alpha_TN)
            loss = torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr, dim=1))
            grad = torch.autograd.grad(loss, para_list_from_net, create_graph=keep_grad)
            intermediate_updated_para_list = list(map(lambda p: p[1] - lr_inner * p[0], zip(grad, para_list_from_net)))
        else:
            reset_running_mean_bn(xi)
            out = xi(X_tr, intermediate_updated_para_list, xi.bn_running_stats, xi.alpha_TN)
            loss = torch.nn.functional.cross_entropy(out, torch.squeeze(Y_tr, dim=1))
            #print('iter ', iter, 'loss: ', float(loss))
            grad = torch.autograd.grad(loss, intermediate_updated_para_list, create_graph=keep_grad)
            #intermediate_updated_para_list = list(map(lambda p: p[1] - lr_inner * p[0]/2, zip(grad, intermediate_updated_para_list)))
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
