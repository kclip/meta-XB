import torch
import numpy as np
from funcs.hard_quantile import quantile_plus
from funcs.soft_quantile import Soft_Quantile
from sklearn import svm
from data.data_gen_gam import get_gt_prob_from_X

SOFT_IFTY_EPS = 0.001 # this is delta in the paper
new_scheme_for_ada_without_sr = True # if this is False, then the direct sigmoid for soft adaptive NC score, see e.g., Appendix. B
assert new_scheme_for_ada_without_sr == True

def minimum_of_available_NC(NC_y_dict, y_prime, N, quantile_mode, tau_for_soft_min):
    # tau_for_soft_min = c_S
    tmp_whole_NC_y_dict_y_prime = []
    for i in range(N):
        tmp_whole_NC_y_dict_y_prime.append(NC_y_dict[i][y_prime]) #[ (num_te,1), ... ]
    tmp_whole_NC_y_dict_y_prime = torch.cat(tmp_whole_NC_y_dict_y_prime, dim=1) # (num_te, N)
    if quantile_mode == 'hard':
        min_NC_y_dict_y_prime, _ = torch.min(tmp_whole_NC_y_dict_y_prime, dim=1, keepdim=True) # (num_te,1)
    elif quantile_mode == 'soft':
        min_NC_y_dict_y_prime = soft_min(tmp_whole_NC_y_dict_y_prime, tau_for_soft_min)
    else:
        raise NotImplementedError
    return min_NC_y_dict_y_prime

def compute_pred_set_as_indicator_matrix_for_auxiliary_few_shot_meta(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, corrected_quantile, num_classes, SQ,  c_sigmoid, if_JK_mm, tau_for_soft_min=None):
    pred_set_as_indicator_matrix = []
    for y_prime in range(num_classes):
        q_plus_v_prime = corrected_quantile - NC_y_dict[0][y_prime]   # (num_te, 1) -- only taking the y' that 
        if quantile_mode == 'hard':
            curr_indicator_column = hard_indicator(q_plus_v_prime) # (num_te, 1)
        elif quantile_mode == 'soft':
            raise NotImplementedError
        else:
            raise NotImplementedError
        pred_set_as_indicator_matrix.append(curr_indicator_column) # [(num_te,1), (num_te,1), ... ]
    pred_set_as_indicator_matrix = torch.cat(pred_set_as_indicator_matrix, dim=1) # (num_te, num_classes)
    # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
    # using this, we can compute both inefficiency and validity for both evalulation and training
    return pred_set_as_indicator_matrix

def compute_pred_set_as_indicator_matrix(NC_y_dict, N, quantile_mode, if_soft_inf_val_differentiable, alpha, num_classes, SQ,  c_sigmoid, if_JK_mm, tau_for_soft_min=None):
    pred_set_as_indicator_matrix = []
    device_info = NC_y_dict[0]['NC_i'].get_device()
    if device_info == -1:
        device = "cpu"
    else:
        device = "cuda:" + str(device_info)
    for y_prime in range(num_classes):
        v_prime = []
        if_debug = True
        if if_JK_mm:
        # make (num_te, N) matrix, then find min for each num_te --> (num_te, 1)
            min_NC_y_dict_y_prime = minimum_of_available_NC(NC_y_dict, y_prime, N, quantile_mode, tau_for_soft_min)
        else:
            pass

        for i in range(N):
            if if_JK_mm:    
                curr_v_prime =  NC_y_dict[i]['NC_i'] - min_NC_y_dict_y_prime # (1,1) - (num_te, 1) -> (num_te, 1)
            else:  
                curr_v_prime =  NC_y_dict[i]['NC_i'] - NC_y_dict[i][y_prime] # (1,1) - (num_te, 1) -> (num_te, 1)
            v_prime.append(curr_v_prime)
        if quantile_mode == 'hard':
            tmp_for_dtype = torch.rand(1)
            max_value = torch.tensor([torch.finfo(tmp_for_dtype.dtype).max]).to(device) # (1)
            max_value = max_value * torch.ones(curr_v_prime.shape).to(device)
            v_prime.append(max_value)
        elif quantile_mode == 'soft':
            v_prime_tensor = torch.cat(v_prime, dim=1) # v_prime: [(num_te,1), (num_te,1), ....] -> (num_te,N)
            v_prime_max = torch.max(v_prime_tensor, dim=1, keepdim=True) # (num_te,1)
            v_prime.append(v_prime_max[0] + SOFT_IFTY_EPS)
        else:
            raise NotImplementedError
        v_prime = torch.cat(v_prime, dim=1) # [(num_te,1), (num_te,1), ..., (num_te,1)] -> (num_te, N+1)
        v_prime = v_prime.unsqueeze(dim=2) # (num_te, N+1, 1)
        if quantile_mode == 'hard':
            q_plus_v_prime = quantile_plus(v_prime, 1-alpha)
            curr_indicator_column = hard_indicator(q_plus_v_prime) # (num_te, 1)
        elif quantile_mode == 'soft':
            q_plus_v_prime = SQ.forward(v_prime)
            curr_indicator_column = soft_indicator(q_plus_v_prime, c_sigmoid) # (num_te, 1)
        else:
            raise NotImplementedError
        pred_set_as_indicator_matrix.append(curr_indicator_column) # [(num_te,1), (num_te,1), ... ]
    pred_set_as_indicator_matrix = torch.cat(pred_set_as_indicator_matrix, dim=1) # (num_te, num_classes)
    # pred_set_as_indicator_matrix 0, 1 matrix for hard, 0~1 matrix for soft
    # using this, we can compute both inefficiency and validity for both evalulation and training
    return pred_set_as_indicator_matrix

def compute_prob_vec(var, X, xi, NC_mode, num_classes):
    assert len(X.shape) > 1 # batch dim at 0
    if 'maml' in NC_mode:
        return xi(X, var, xi.bn_running_stats, xi.alpha_TN, if_softmax_out=True)
    elif 'SVC' in NC_mode:
        if type(var) == np.int64:
            prob_filled = np.zeros((int(X.shape[0]), num_classes)) # (num_te, num_classes)
            eps = 0.001 # numerical stability for log NC (original NC)
            for ind_class in range(num_classes):
                if ind_class == var:
                    prob_filled[:, ind_class] = 1 - eps * (num_classes - 1)
                else:
                    prob_filled[:, ind_class] = eps
        else:
            # var is model
            if len(X.shape) > 2:
                X_reshape = X.reshape(X.shape[0], -1)
            else:
                X_reshape = X
            curr_prob_only_seen_labels = var.predict_proba(X_reshape) 
            # need to put other labels as 0 since we are now interested in few shot!
            prob_filled = np.zeros((int(X.shape[0]), var.num_classes)) # (num_te, num_classes)
            ind_seen_labels = 0
            for ind_class in range(var.num_classes):
                if ind_class in var.classes_:
                    prob_filled[:, ind_class] = curr_prob_only_seen_labels[:, ind_seen_labels]
                    ind_seen_labels += 1
                else:
                    pass
        return torch.from_numpy(prob_filled)
    else:
        raise NotImplementedError



def SVC(LOO_dataset, num_classes):
    model = svm.SVC(kernel = 'linear', C = 1, probability = True, random_state = 2020)
    # fitting
    X_tr = LOO_dataset[0].numpy()
    Y_tr = torch.squeeze(LOO_dataset[1]).numpy()
    if len(np.unique(Y_tr)) == 1:
        # case with single label
        model = np.unique(Y_tr)[0]
    else:
        if len(X_tr.shape) > 2:
            X_tr_reshape = X_tr.reshape(X_tr.shape[0], -1)
        else:
            X_tr_reshape = X_tr
        model.fit(X_tr_reshape, Y_tr)
        model.num_classes = num_classes
    return model


def soft_indicator(quan_plus, tau, epsilon=0.5):
    tmp = 1+torch.exp(-quan_plus/tau)
    return 1/tmp

def hard_indicator(quan_plus):
    # (num_te, 1)
    return (quan_plus >= 0).type(torch.uint8) 

def NC_compute(prob_vec, y, NC_mode, u=None, c_sigmoid_for_ada_NC=1):
    # prob_vec: (num_te, num_classes) # num_te=1 for NC_i
    num_classes = prob_vec.shape[1]
    if 'adaptive' in NC_mode:
        # u: (num_te, 1) # num_te=1 for NC_i
        if isinstance(y, int) or len(y.shape) == 0:
            prob_y = prob_vec[:, y] # (num_te)
        else: # for auxiliary cdf or CV+ speed-up 
            tmp_y_one_hot = torch.nn.functional.one_hot(torch.squeeze(y), num_classes=num_classes)
            assert prob_vec.shape == tmp_y_one_hot.shape
            prob_y = torch.sum(prob_vec*tmp_y_one_hot, dim=1)
        device_info = prob_y.get_device()
        if device_info == -1:
            device = "cpu"
        else:
            device = "cuda:" + str(device_info)
        if 'soft_ada' in NC_mode:
            if new_scheme_for_ada_without_sr: # proposed soft adaptive NC (Appendix B)
                prob_y_broadcast = prob_y.unsqueeze(dim=1) * torch.ones(1,num_classes).to(device) # (num_te,1) * (1,C) -> (num_te,C)
                thresholding_prob_vec = torch.nn.functional.relu(prob_y_broadcast - prob_vec) # (num_te, C)
                soft_more_probable_labels_than_y = torch.sum(soft_indicator(prob_y_broadcast - prob_vec, c_sigmoid_for_ada_NC).to(device), dim=1)  #(num_te, C) -> sum -> (num_te, 1)
                sum_prob_upto_curr_y = 1+torch.sum(thresholding_prob_vec, dim=1) - (soft_more_probable_labels_than_y)*prob_y # (num_te)
            else:
                # direct sigmoid for soft adaptive NC
                prob_y_broadcast = prob_y.unsqueeze(dim=1) * torch.ones(1,num_classes).to(device)
                thresholding_soft_rank_prob = soft_indicator(prob_vec - prob_y_broadcast, c_sigmoid_for_ada_NC).to(device)
                tmp_sr = thresholding_soft_rank_prob * prob_vec # (num_te, C) * (num_te, C)
                sum_prob_upto_curr_y = torch.sum(tmp_sr, dim=1) # (num_te) # soft!            
            sum_prob_upto_curr_y_actual = sum_prob_upto_curr_y
        else: # hard ranking
            prob_y_broadcast = prob_y.unsqueeze(dim=1) * torch.ones(1,num_classes).to(device) # (num_te,1) * (1,C) -> (num_te,C)
            thresholding_prob_vec = torch.nn.functional.relu(prob_vec - prob_y_broadcast)
            more_probable_labels_than_y = torch.count_nonzero(thresholding_prob_vec, dim=1) # (num_te)
            sum_prob_upto_curr_y_actual = torch.sum(thresholding_prob_vec, dim=1) + (more_probable_labels_than_y)*prob_y + prob_y # (num_te)
        assert u.shape[0] == prob_y.shape[0]
        portion_from_u = u * prob_y.unsqueeze(dim=1) # (num_te, 1)
        return sum_prob_upto_curr_y_actual.unsqueeze(dim=1) - portion_from_u # (num_te, 1)
    else: 
        eps_for_log = 1e-10
        if isinstance(y, int) or len(y.shape) == 0:
            prob_y = prob_vec[:, y] # (num_te)
            return -torch.log(prob_y.unsqueeze(dim=1)+eps_for_log) # (num_te, 1)
        else:  # for auxiliary cdf or CV+ speed-up -- deprecated
            tmp_y_one_hot = torch.nn.functional.one_hot(torch.squeeze(y), num_classes=num_classes)
            assert prob_vec.shape == tmp_y_one_hot.shape
            prob_y = torch.sum(prob_vec*tmp_y_one_hot, dim=1)
            return -torch.log(prob_y.unsqueeze(dim=1)+eps_for_log)
        
def soft_min(full_matrix, tau_for_soft_min):
    neg_matrix = -full_matrix
    # negative max = min
    weight_for_soft_min = softmax_with_temperature(neg_matrix, tau_for_soft_min) # (num_te, N)
    return torch.sum(weight_for_soft_min * full_matrix, dim=1, keepdim=True) # (num_te, N) * (num_te, N) -> sum -> (num_te,1)


def softmax_with_temperature(out, tau):
    # out: (num_te, N)
    # tau: 0 < tau -- if tau near 0 -- argmax, if tau near infty -- passing the average
    tmp = torch.exp(out/tau) # (num_te, N)
    denom = torch.sum(tmp, dim=1, keepdim=True) # (num_te,1)
    numer = tmp # (num_te, N)
    return numer/denom # (num_te, N)


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
