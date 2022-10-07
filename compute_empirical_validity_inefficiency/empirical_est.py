import torch
import numpy as np
from funcs.jk_plus import Jacknife_plus
from funcs.cv_plus import CrossVal_plus
from data.data_gen_gam import Data_gen_GAM
from funcs.utils import reset_random_seed
import copy
from funcs.split_conformal import Split_Conformal
from funcs.jk_plus import GD_for_SC
from funcs.utils import randomly_divide_tr_te, divide_tr_te
from data.data_gen import Data_gen_toy
from funcs.auxiliary_few_shot import Auxiliary_Few_Shot_Quan_Est
from funcs.utils_for_set_prediction import SVC
import math

IF_ADA_DETERMINISTIC = True # for deterministic adaptive NC score -- assuming always true in the paper

def generate_rep_dataset(exp_mode, full_dataset_dict, dim_x, num_classes, N, num_rep_phi, num_rep_tr_set, num_rep_te_samples, rand_seed_for_datagen, if_special_setting = False, fixed_m=None, gt_phi=None):
    # if_special_setting now only for gam for visualization purpose
    if full_dataset_dict is None:
        print('full_dataset_dict_te generation!')
        assert gt_phi is None
        full_dataset_dict = {}
        for ind_rep_phi in range(num_rep_phi):
            if num_rep_tr_set > 100:
                print('task ind', ind_rep_phi)
            reset_random_seed(ind_rep_phi+rand_seed_for_datagen)
            if exp_mode == 'toy':
                Data_generator = Data_gen_toy(dim_x, num_classes)
            elif exp_mode == 'toy_vis_gam':
                Data_generator = Data_gen_GAM(num_classes)
                #print('GAM info:', Data_generator.dict)
            dataset_dict_per_phi = {}
            for ind_rep_tr_set in range(num_rep_tr_set):
                reset_random_seed(rand_seed_for_datagen + ind_rep_tr_set + ind_rep_phi*num_rep_tr_set)
                if exp_mode == 'toy_vis_gam':
                    if if_special_setting:
                        tr_dataset = Data_generator.gen(N, None, if_test_fix_ordering_for_vis=False, fixed_training_sample=((0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(0,0),(2,2),(4,1))) #(15,7)
                        te_dataset = Data_generator.gen(num_rep_te_samples, fixed_m, if_test_fix_ordering_for_vis=True, fixed_training_sample=None)
                    else:
                        tr_dataset = Data_generator.gen(N, None, False, None)
                        te_dataset = Data_generator.gen(num_rep_te_samples, None, False, None)
                else:
                    if if_special_setting:
                        raise NotImplementedError
                    else:
                        tr_dataset = Data_generator.gen(N)
                        te_dataset = Data_generator.gen(num_rep_te_samples)
                dataset_dict_per_phi['tr_' + str(ind_rep_tr_set)] = tr_dataset
                dataset_dict_per_phi['te_' + str(ind_rep_tr_set)] = te_dataset
            full_dataset_dict[ind_rep_phi] = dataset_dict_per_phi
            full_dataset_dict['gt_beta_' + str(ind_rep_phi)] = Data_generator.beta
    else:
        pass
    return full_dataset_dict

def compute_empirical_est(exp_mode, device, full_dataset_dict, alpha, xi, num_classes, set_prediction_mode, NC_mode, quantile_mode, num_rep_phi, num_rep_tr_set, num_rep_te_samples, if_soft_inf_val_differentiable, if_compute_conditional_coverage,  phi=None, iter_ConfTr=500, N=None, mb_size_task=None, mb_size_tr_set=None, mb_size_te_samples=None, c_sigmoid=None, c_sigmoid_for_ada_NC=None,  delta=0.1, M=1000, actual_test_ratio=0.75, scheme_details=None, c_S=None, sq_mode=None, c_Q=None):
    if N is None:
        if isinstance(full_dataset_dict, dict):
            N = full_dataset_dict[int(0)]['tr_'+str(int(0))][0].shape[0]
        else:
            # dataloader class for OTA and miniimagenet
            N = full_dataset_dict.N
    else:
        pass

    if exp_mode == 'toy':
        IF_ADA_DETERMINISTIC = True
        lr_inner = 0.1 
        inner_iter = 1 
    elif exp_mode == 'toy_vis_gam':
        IF_ADA_DETERMINISTIC = True
        lr_inner = 0.1
        inner_iter = 1 
    elif exp_mode == 'OTA':
        # modulation classification
        IF_ADA_DETERMINISTIC = True
        lr_inner = 0.1  
        inner_iter = 1 
    elif exp_mode == 'miniimagenet':
        IF_ADA_DETERMINISTIC = True
        # dataloader class for miniimagenet
        lr_inner = 0.01 
        inner_iter = 4 
    else:
        raise NotImplementedError

    if quantile_mode == 'hard':
        keep_grad = False
    else:
        keep_grad = True # for differentiable soft inefficiency!
    
    if mb_size_task is None:
        raise NotImplementedError
    # for GD
    if 'JK+' in set_prediction_mode:
        if 'mm' in set_prediction_mode:
            if_JK_mm = True
        else:
            if_JK_mm = False
        JK_plus = Jacknife_plus(alpha=alpha, xi=xi, num_classes = num_classes, lr_inner=lr_inner, inner_iter=inner_iter, c_sigmoid=c_sigmoid, c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC,  if_JK_mm=if_JK_mm, keep_grad=keep_grad, tau_for_soft_min=c_S, sq_mode=sq_mode, tau_for_soft_quan=c_Q)
    elif set_prediction_mode == 'SC': #split conformal
        SC = Split_Conformal(alpha=alpha, num_classes=num_classes, c_sigmoid=c_sigmoid, c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC)
        # xi is used as random initialization for phi
    elif 'CV+' in set_prediction_mode:
        # 'CV+5' 5 stands for K  -- i.e., 'CV+K'
        K = int(set_prediction_mode[3])
        if 'mm' in set_prediction_mode:
            if_JK_mm = True
        else:
            if_JK_mm = False
        CV_plus = CrossVal_plus(K=K, alpha=alpha, xi=xi, num_classes = num_classes, lr_inner=lr_inner, inner_iter=inner_iter, c_sigmoid=c_sigmoid, c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC, if_JK_mm=if_JK_mm, keep_grad=keep_grad, tau_for_soft_min=c_S, sq_mode=sq_mode, tau_for_soft_quan=c_Q)
    elif set_prediction_mode == 'auxiliary_few_shot_meta':
        Auxiliary_Few_Shot = Auxiliary_Few_Shot_Quan_Est(alpha=alpha, corr_lambda=xi[2] , xi=xi[0], predictor_net = xi[1], num_classes = num_classes, lr_inner=lr_inner, inner_iter=inner_iter)
    else:
        raise NotImplementedError

    if M < 0:
        if_compute_conditional_coverage = False
    else:
        pass
    
    ## num_rep_phi, num_rep_tr_set, num_rep_te_samples is full data set size
    rand_perm_phi = torch.randperm(num_rep_phi)[:mb_size_task]
    rand_perm_tr_sets = torch.randperm(num_rep_tr_set)[:mb_size_tr_set]
    ineff = 0
    validity = 0
    ce_loss = 0
    conditional_ineff = 0
    conditional_validity = 0
    dict_for_pd = {}
    dict_for_pd['marginal coverage'] = []
    dict_for_pd['inefficiency'] = []
    dict_for_pd['conditional coverage'] = []
    dict_for_pd['conditional inefficiency'] = []
    dict_for_pd['NC mode'] = []
    dict_for_pd['set prediction mode'] = []
    dict_for_pd['details'] = []
    ind_actual_rep_phi = 0
    num_nan_phi = 0
    for ind_rep_phi in rand_perm_phi:
        #print('meta-test task ind: ', ind_actual_rep_phi)
        conditional_X_info = None # this will be determined at the very first realizations (see, e.g., S1.2 https://arxiv.org/pdf/2006.02544.pdf)
        curr_task_marginal_validity = 0
        curr_task_inefficiency = 0
        curr_task_conditional_validity = 0
        curr_task_conditional_ineff = 0
        num_nan_val = 0
        num_nan_ineff = 0
        ind_actual_rep_tr_set = 0
        for ind_rep_tr_set in rand_perm_tr_sets:
            if IF_ADA_DETERMINISTIC:
                u_for_tr_dataset = torch.ones(N, 1).to(device) # this is the prob. for choosing smaller set -- for determnistic casae, alwayas choose the safer one!
                u_for_X_te = torch.ones(mb_size_te_samples, 1).to(device)
            else:
                u_for_tr_dataset = torch.rand(N, 1).to(device)
                u_for_X_te = torch.rand(mb_size_te_samples, 1).to(device)
            if isinstance(full_dataset_dict, dict):
                if len(full_dataset_dict[int(ind_rep_phi)].keys()) == 2:
                    # during meta-training
                    tr_dataset = full_dataset_dict[int(ind_rep_phi)]['tr_'+str(int(0))]
                    te_dataset = full_dataset_dict[int(ind_rep_phi)]['te_'+str(int(0))]
                    assert te_dataset is None # need to marginalize over training data set to get marginal measures!
                    (tr_dataset, te_dataset) = randomly_divide_tr_te(tr_dataset, N)
                else:
                    # during meta-testing
                    tr_dataset = full_dataset_dict[int(ind_rep_phi)]['tr_'+str(int(ind_rep_tr_set))]
                    te_dataset = full_dataset_dict[int(ind_rep_phi)]['te_'+str(int(ind_rep_tr_set))]
            else:
                # automatically randomly divide and also consideirng as much as training set rep. as needed by random sampling
                tr_dataset, te_dataset = full_dataset_dict.gen(N, device)
            if set_prediction_mode == 'SC':
                # divide tr_dataset into two parts -- one for ConfTR (as proper training set) the other for calibration
                if isinstance(full_dataset_dict, dict):
                    if exp_mode == 'toy_vis_gam':
                        (proper_tr_dataset, cali_tr_dataset) = divide_tr_te(tr_dataset, N-N//2) # proper_tr: N-N//2, cali_tr: N//2 -- in case N is odd!
                    else:
                        (proper_tr_dataset, cali_tr_dataset) = randomly_divide_tr_te(tr_dataset, N-N//2) # proper_tr: N-N//2, cali_tr: N//2 -- in case N is odd!
                else:
                    proper_tr_dataset, _ = full_dataset_dict.gen(N-N//2, device)
                    cali_tr_dataset, _ = full_dataset_dict.gen(N//2, device) 
                if ind_actual_rep_tr_set == 0:
                    # we need to update phi
                    phi = copy.deepcopy(xi) # initialization
                    # update phi using half of the training data set
                    if 'maml' in NC_mode:
                        phi = GD_for_SC(phi, proper_tr_dataset, lr_inner, inner_iter)
                    elif 'SVC' in NC_mode:
                        phi = SVC(proper_tr_dataset, num_classes)
                    else:
                        raise NotImplementedError
                else:
                    # use the obtained phi at the first training data set realization to save time
                    # half of the data set is considered to be used for training so only remaining half will be used for calibration
                    # we need to update phi
                    phi = copy.deepcopy(xi) # initialization
                    # update phi using half of the training data set
                    if 'maml' in NC_mode:
                        phi = GD_for_SC(phi, proper_tr_dataset, lr_inner, inner_iter)
                    elif 'SVC' in NC_mode:
                        phi = SVC(proper_tr_dataset, num_classes)
                    else:
                        raise NotImplementedError

            else:
                pass
            # testing
            # dataset: (X, Y) # X: (num_samples, *), Y: (num_samples, 1)
            X_te_tot = te_dataset[0]
            Y_te_tot = te_dataset[1]
            X_te = X_te_tot[:mb_size_te_samples] # randomly divided already in case of meta-learning
            Y_te = Y_te_tot[:mb_size_te_samples]
            # now actual test
            if set_prediction_mode == 'SC':
                pred_set_as_indicator_matrix = SC.forward(phi, NC_mode, quantile_mode, cali_tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te,  Y_te)
                ce_loss_over_N = 0
            elif 'JK+' in set_prediction_mode:
                pred_set_as_indicator_matrix, ce_loss_over_N = JK_plus.forward(NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te,  Y_te)
            elif 'CV+' in set_prediction_mode:
                pred_set_as_indicator_matrix, ce_loss_over_N = CV_plus.forward(NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te,  Y_te)
            elif set_prediction_mode == 'auxiliary_few_shot_meta':
                pred_set_as_indicator_matrix, ce_loss_over_N = Auxiliary_Few_Shot.forward(NC_mode, quantile_mode, tr_dataset, X_te, if_soft_inf_val_differentiable, u_for_tr_dataset, u_for_X_te, Y_te)
            else:
                raise NotImplementedError
            curr_validity, curr_inefficiency = compute_val_ineff(pred_set_as_indicator_matrix, Y_te)
            validity += curr_validity
            ineff += curr_inefficiency
            ce_loss += ce_loss_over_N

            if if_compute_conditional_coverage:
                conditional_coverage, conditional_X_info = compute_conditional_coverage(pred_set_as_indicator_matrix, X_te, Y_te, conditional_X_info, delta, M, actual_test_ratio)
                conditional_ineff, conditional_X_info = compute_conditional_inefficiency(pred_set_as_indicator_matrix, X_te, Y_te, conditional_X_info, delta, M, actual_test_ratio)
            else:
                conditional_coverage = None
                conditional_ineff = None
            curr_task_marginal_validity += curr_validity
            curr_task_inefficiency += curr_inefficiency
            if conditional_coverage is not None:
                if ind_actual_rep_tr_set == 0:
                    num_nan_val += 1 # Nan
                else:
                    if math.isnan(float(curr_task_conditional_validity)):
                        num_nan_val += 1 # nothing in the conditioning x 
                    else:
                        curr_task_conditional_validity += conditional_coverage
            if conditional_ineff is not None:
                if ind_actual_rep_tr_set == 0:
                    num_nan_ineff += 1 # Nan
                else:
                    if math.isnan(float(curr_task_conditional_ineff)):
                        num_nan_ineff += 1
                    else:
                        curr_task_conditional_ineff += conditional_ineff

            ind_actual_rep_tr_set += 1
        assert num_nan_val == num_nan_ineff
        if num_nan_val > 900:
            if_curr_phi_cond_invalid = True
        else:
            if_curr_phi_cond_invalid = False
        dict_for_pd['marginal coverage'].append(float(curr_task_marginal_validity/(mb_size_te_samples*len(rand_perm_tr_sets)))) # since single data set
        dict_for_pd['inefficiency'].append(float(curr_task_inefficiency/(mb_size_te_samples*len(rand_perm_tr_sets)))) # since single data set     
        if conditional_coverage is None:
            dict_for_pd['conditional coverage'].append(0) # since single data set
            dict_for_pd['conditional inefficiency'].append(0)
        else:
            dict_for_pd['conditional coverage'].append(float(curr_task_conditional_validity/(len(rand_perm_tr_sets)-num_nan_val))) # since single data set
            dict_for_pd['conditional inefficiency'].append(float(curr_task_conditional_ineff/(len(rand_perm_tr_sets)-num_nan_val))) # since single data set
            if if_curr_phi_cond_invalid:
                num_nan_phi += 1
            else:
                conditional_ineff += float(curr_task_conditional_ineff/(len(rand_perm_tr_sets)-num_nan_val))
                conditional_validity += float(curr_task_conditional_validity/(len(rand_perm_tr_sets)-num_nan_val))
        dict_for_pd['NC mode'].append(NC_mode)
        dict_for_pd['set prediction mode'].append(set_prediction_mode)
        dict_for_pd['details'].append(scheme_details)
        ind_actual_rep_phi += 1
    validity = validity.type('torch.FloatTensor')
    ineff = ineff.type('torch.FloatTensor')
    validity /= (mb_size_te_samples*mb_size_tr_set*mb_size_task)
    ineff /= (mb_size_te_samples*mb_size_tr_set*mb_size_task)
    if conditional_coverage is None:
        pass
    else:
        conditional_ineff  /= (mb_size_task-num_nan_phi) # already averaged over tr and te
        conditional_validity /= (mb_size_task-num_nan_phi) # already averaged over tr and te
    ce_loss /= (mb_size_tr_set*mb_size_task)
    return (validity, ineff), ce_loss, (conditional_validity, conditional_ineff), dict_for_pd


def Trans(vector):
    return torch.transpose(vector,0,1)

def compute_val_ineff(indicator_matrix, Y_te):
    # Y_te: (num_te, 1), indicator_matrix: (num_te, num_classes)
    one_hot_label = torch.nn.functional.one_hot(torch.squeeze(Y_te), num_classes=indicator_matrix.shape[1])
    assert one_hot_label.shape == indicator_matrix.shape
    validity_matrix = one_hot_label * indicator_matrix
    validity = torch.sum(validity_matrix)
    ineff = torch.sum(indicator_matrix)
    return validity, ineff


### code below for WSC, taken and adapted from https://github.com/msesia/arc/blob/2f6e5c40c0b673be6f245d352cac93e2169ca2f6/arc/coverage.py#L5

def compute_conditional_coverage(indicator_matrix, X_te, Y_te, conditional_X_info, delta=0.1, M=1000, actual_test_ratio=0.75):
    # Y_te: (num_te, 1), indicator_matrix: (num_te, num_classes)
    # X_te: (num_te, dim_x)
    # conditional_X_info = (v_star, a_star, b_star)
    indicator_matrix = indicator_matrix.type('torch.FloatTensor')
    indicator_matrix = indicator_matrix.cpu()
    one_hot_label = torch.nn.functional.one_hot(torch.squeeze(Y_te), num_classes=indicator_matrix.shape[1])
    one_hot_label = one_hot_label.cpu()
    X_te = torch.reshape(X_te, (X_te.shape[0],-1))
    X_te = X_te.cpu()
    Y_te = Y_te.cpu()
    assert one_hot_label.shape == indicator_matrix.shape
    validity_matrix = one_hot_label * indicator_matrix
    coverage = torch.sum(validity_matrix, dim=1) # (num_te)
    num_te = X_te.shape[0]

    if conditional_X_info is None:
        actual_test_num = int(actual_test_ratio * num_te) # currently 0 -- first tr. data set for slab info
        actual_tr_num = num_te - actual_test_num
        X_te_1 = X_te[:actual_tr_num]
        Y_te_1 = Y_te[:actual_tr_num]
        coverage_1 = coverage[:actual_tr_num]
        X_te_2 = X_te[actual_tr_num:]
        Y_te_2 = Y_te[actual_tr_num:]
        coverage_2 = coverage[actual_tr_num:]
        _, v_star, a_star, b_star = wsc(X_te_1, Y_te_1, coverage_1, delta, M)
        conditional_X_info = (v_star, a_star, b_star)
    else:
        (v_star, a_star, b_star) = conditional_X_info
        # use full for computing conditional measure
        X_te_2 = X_te
        Y_te_2 = Y_te
        coverage_2 = coverage
    # conditional coverage
    conditional_coverage = wsc_vab(X_te_2, Y_te_2, coverage_2, v_star, a_star, b_star)
    return conditional_coverage, conditional_X_info


def compute_conditional_inefficiency(indicator_matrix, X_te, Y_te, conditional_X_info, delta=0.1, M=1000, actual_test_ratio=0.75):
    # Y_te: no need
    # Y_te: (num_te, 1), indicator_matrix: (num_te, num_classes)
    # X_te: (num_te, dim_x)
    # conditional_X_info = (v_star, a_star, b_star)
    indicator_matrix = indicator_matrix.type('torch.FloatTensor')
    ineff = torch.sum(indicator_matrix, dim=1) # (num_te)
    num_te = X_te.shape[0]
    X_te = X_te.cpu()
    X_te = torch.reshape(X_te, (X_te.shape[0],-1))
    Y_te = Y_te.cpu()

    if conditional_X_info is None:
        actual_test_num = int(actual_test_ratio * num_te)
        actual_tr_num = num_te - actual_test_num
        X_te_1 = X_te[:actual_tr_num]
        Y_te_1 = Y_te[:actual_tr_num]
        ineff_1 = ineff[:actual_tr_num]
        X_te_2 = X_te[actual_tr_num:]
        Y_te_2 = Y_te[actual_tr_num:]
        ineff_2 = ineff[actual_tr_num:]
        _, v_star, a_star, b_star = wsc(X_te_1, Y_te_1, -ineff_1, delta, M) # finding worst slab <-> finding minimum negative ineff <-> finding largest ineff
        conditional_X_info = (v_star, a_star, b_star)
    else:
        (v_star, a_star, b_star) = conditional_X_info
        # use full for computing conditional measure
        X_te_2 = X_te
        Y_te_2 = Y_te
        ineff_2 = ineff
    # conditional ineff
    conditional_ineff = wsc_vab(X_te_2, Y_te_2, ineff_2, v_star, a_star, b_star)
    return conditional_ineff, conditional_X_info

def wsc_vab(X_te_2, Y_te_2, coverage_2, v, a, b):
    num_te = Y_te_2.shape[0]
    z = torch.squeeze(X_te_2 @ v)
    idx = np.where((z>=a)*(z<=b))
    conditional_coverage = torch.mean(coverage_2[idx])
    return conditional_coverage

def wsc(X_te_1, Y_te_1, coverage_1, delta, M):
    # find a*, b*, v*
    V = sample_sphere(M, dim_x=X_te_1.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    for m in range(M):
        wsc_list[m], a_list[m], b_list[m] = wsc_v(V[m], coverage_1, X_te_1, Y_te_1, delta, M)
    idx_star = np.argmin(np.array(wsc_list))
    a_star = a_list[idx_star]
    b_star = b_list[idx_star]
    v_star = V[idx_star]
    wsc_star = wsc_list[idx_star]
    return wsc_star, v_star, a_star, b_star



def wsc_v(v, coverage_tot, X_te, Y_te, delta, M):
    # X_te: (num_te, dim_x)
    # v: (dim_x, 1)
    num_te = coverage_tot.shape[0]
    z = torch.squeeze(X_te @ v) #(num_te)
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    cover_ordered = coverage_tot[z_order]
    ai_max = int(np.round((1.0-delta)*num_te))
    ai_best = 0
    bi_best = num_te-1
    cover_min = 1
    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai+int(np.round(delta*num_te)),num_te)
        coverage = np.cumsum(cover_ordered[ai:num_te]) / np.arange(1,num_te-ai+1)
        coverage[np.arange(0,bi_min-ai)]=1 # 1 means possible best which prevents selecting in the region -- ok with ineff. since + is better than possible since we are considering negative sign of ineff
        bi_star = ai+np.argmin(coverage)
        cover_star = coverage[bi_star-ai]
        if cover_star < cover_min:
            ai_best = ai
            bi_best = bi_star
            cover_min = cover_star
    return cover_min, z_sorted[ai_best], z_sorted[bi_best]

def sample_sphere(n, dim_x):
    v = np.random.randn(dim_x, n)
    v /= np.linalg.norm(v, axis=0)
    return v.T



