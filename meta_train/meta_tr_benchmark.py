import torch
import numpy as np
from compute_empirical_validity_inefficiency.empirical_est import generate_rep_dataset
from torch.utils.tensorboard import SummaryWriter
from funcs.utils import randomly_divide_tr_te
from funcs.jk_plus import Jacknife_plus, GD
from funcs.utils_for_set_prediction import soft_indicator, hard_indicator
from data.data_gen_miniimagenet import Data_gen_miniimagenet
from data.data_gen_ota import Data_gen_OTA
import torch.nn as nn
import os
import copy
from meta_train.meta_training import vanailla_maml_CE_loss
from funcs.utils_for_set_prediction import compute_prob_vec, NC_compute
from funcs.hard_quantile import quantile_plus

#### meta-VB implementation (https://arxiv.org/abs/2102.08898)
IF_PREDICTOR_SET_GEN_ONCE = False # for large data set (e.g., miniimagenet), we compute data set for quantile estimator at once to save time
NUM_PREDICT_DATASET = 1000 # size of training data set for quantile estimator if IF_PREDICTOR_SET_GEN_ONCE is True 


class Meta_Tr_Few_Shot_Auxiliary:
    def __init__(self, exp_mode,  N, num_classes, dim_x,  meta_tr_setting, meta_val_setting, num_tasks_ratio_meta_cal, mb_size_te_samples):
        (num_total_tasks_meta, num_tr_te_per_task_meta, rand_seed_for_datagen_mtr) = meta_tr_setting
        (num_rep_phi_eval, num_rep_tr_set_eval, num_rep_te_samples_eval, rand_seed_for_datagen_mval) = meta_val_setting
        self.num_rep_phi_eval = num_rep_phi_eval # working as meta-val
        self.num_rep_tr_set_eval = num_rep_tr_set_eval # working as meta-val
        self.num_rep_te_samples_eval = num_rep_te_samples_eval # working as meta-val
        if num_total_tasks_meta == 1:
            num_tasks_ratio_meta_cal = 1
        else:
            pass
        self.num_total_tasks_meta_tr_cal = int(num_total_tasks_meta*num_tasks_ratio_meta_cal)
        self.num_total_tasks_meta_tr_tr = num_total_tasks_meta-int(num_total_tasks_meta*num_tasks_ratio_meta_cal)
        if (exp_mode == 'toy') or (exp_mode == 'toy_vis_gam'):
            self.full_dataset_dict_meta_te = generate_rep_dataset(exp_mode, None, dim_x, num_classes, N, self.num_rep_phi_eval, self.num_rep_tr_set_eval, self.num_rep_te_samples_eval, rand_seed_for_datagen_mval)
            self.full_dataset_dict_meta_tr_cal = generate_rep_dataset(exp_mode, None, dim_x, num_classes, num_tr_te_per_task_meta, self.num_total_tasks_meta_tr_cal, 1, 0, rand_seed_for_datagen_mtr) # random split for meta-tr
            self.full_dataset_dict_meta_tr_tr =  generate_rep_dataset(exp_mode, None, dim_x, num_classes, num_tr_te_per_task_meta, self.num_total_tasks_meta_tr_tr, 1, 0, rand_seed_for_datagen_mtr) # random split for meta-tr
        elif exp_mode == 'OTA':
            # not using self.full_dataset_dict_meta_te for model selection!
            assert num_tasks_ratio_meta_cal == 0.5
            #self.full_dataset_dict_meta_te = Data_gen_OTA(N, mode='test', num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta)
            self.full_dataset_dict_meta_tr_cal = Data_gen_OTA(N, mode='trcal', num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta) # trcal needs particular fixed m_i
            self.full_dataset_dict_meta_tr_tr = Data_gen_OTA(N, mode='trtr', num_classes=num_classes, supp_and_query=N+mb_size_te_samples) # this is for meta-training scoring func.
        elif exp_mode == 'miniimagenet':
            self.full_dataset_dict_meta_tr_cal = Data_gen_miniimagenet(N, mode='val', num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta, num_total_task=self.num_total_tasks_meta_tr_cal)
            self.full_dataset_dict_meta_tr_tr  = Data_gen_miniimagenet(N, mode='train', num_classes=num_classes, supp_and_query=N+mb_size_te_samples, num_total_task=self.num_total_tasks_meta_tr_tr)
        else:
            raise NotImplementedError
        self.exp_mode = exp_mode
        self.num_total_tasks_meta = num_total_tasks_meta
        self.num_tr_te_per_task_meta = num_tr_te_per_task_meta
        self.num_classes = num_classes
        self.dim_x = dim_x
        self.N = N
        assert self.num_tr_te_per_task_meta > N 

        # should be same in compute_empirical_est function part during meta-training for fair comparison
        if self.exp_mode == 'toy':
            # toy
            self.IF_ADA_DETERMINISTIC = True # always True for this work
            self.lr_inner = 0.1
            self.inner_iter = 1
        elif self.exp_mode == 'toy_vis_gam':
            self.IF_ADA_DETERMINISTIC = True
            self.lr_inner = 0.1
            self.inner_iter = 1
        elif self.exp_mode == 'OTA':
            # modulation classification
            self.IF_ADA_DETERMINISTIC = True
            self.lr_inner =  0.1
            self.inner_iter = 1
        elif self.exp_mode == 'miniimagenet':
            self.IF_ADA_DETERMINISTIC = True
            # dataloader class for miniimagenet
            self.lr_inner = 0.01
            self.inner_iter = 1 
        else:
            raise NotImplementedError


    def forward(self, folding_num, saved_net_path, device, alpha, xi,  NC_mode, num_meta_iterations, num_meta_iterations_predictor, mb_size_task, mb_size_tr_set, mb_size_te_samples, lr_meta=0.001, lr_meta_predictor=0.0001, if_conditional_task_correction=True):
        # training scoring network with k-fold
        if self.num_total_tasks_meta_tr_tr < folding_num:
            if self.num_total_tasks_meta_tr_tr == 0:
                folding_num = 0
                num_total_tasks_for_trval = 0
            else:
                folding_num = self.num_total_tasks_meta_tr_tr
                num_total_tasks_for_trval = self.num_total_tasks_meta_tr_tr // folding_num # 
        else:
            num_total_tasks_for_trval = self.num_total_tasks_meta_tr_tr // folding_num # 
        
        if self.num_total_tasks_meta_tr_tr < mb_size_task: # may also need to adjust more considering folding (considered num_tasks are fine)
            mb_size_task = self.num_total_tasks_meta_tr_tr
        else:
            pass

        if (self.exp_mode == 'toy') or (self.exp_mode == 'toy_vis_gam'):
            dict_for_k_fold_dict = k_fold(self.full_dataset_dict_meta_tr_tr, folding_num)
        elif self.exp_mode == 'OTA':
            dict_for_k_fold_dict = k_fold_OTA(folding_num, self.num_classes, self.N, self.num_tr_te_per_task_meta)
        elif self.exp_mode == 'miniimagenet':
            dict_for_k_fold_dict = k_fold_MINI(folding_num, self.num_classes, self.N, self.num_tr_te_per_task_meta, self.num_total_tasks_meta_tr_tr)
        else:
            raise NotImplementedError
        
        meta_loss = 0 # dummy value for printing
        loss_predictor_tot = 0 # also dummy
        # check path for k_fold_meta_trained_net exists
        if xi.alpha_TN is None:
            tmp_saved_path_for_xi = None
        elif xi.alpha_TN.requires_grad is True:
            tmp_saved_path_for_xi = './tmp_path_for_xi'
            torch.save(xi, tmp_saved_path_for_xi)
        else:   
            tmp_saved_path_for_xi = None

        if if_conditional_task_correction:
            delta = 0.9
            num_samples_for_cdf = self.num_tr_te_per_task_meta - self.N
            num_tasks_cal= self.num_total_tasks_meta_tr_cal
            alpha = compute_condtional_correction(delta, alpha, num_samples_for_cdf, num_tasks_cal)
            print('conditionally corrected alpha', alpha)
        else:
            pass

        if 'adaptive' in NC_mode:
            NC_mode = 'maml_adaptive_hard_rank'
        else:
            NC_mode = 'maml_original_hard_rank'

        for ind_fold in range(folding_num):
            meta_trained_net_for_k_fold_path = saved_net_path + 'fold_ind_' + str(ind_fold) + '/meta_trained_scoring'
            if os.path.exists(meta_trained_net_for_k_fold_path):
                pass 
            else:
                print('meta training score function for fold ind', ind_fold)
                # meta_train with CE 
                if os.path.isdir(saved_net_path + 'fold_ind_' + str(ind_fold)):
                    pass
                else:
                    os.makedirs(saved_net_path + 'fold_ind_' + str(ind_fold))
                if tmp_saved_path_for_xi is None:
                    curr_fold_xi = copy.deepcopy(xi) # we need many k_folds
                else:
                    curr_fold_xi = torch.load(tmp_saved_path_for_xi)
                curr_fold_dict_for_scoring = dict_for_k_fold_dict['for_scoring_func'][ind_fold] # dict for meta-train
                if self.exp_mode == 'toy':
                    if len(curr_fold_dict_for_scoring.keys()) == 0: # no task inside due to few meta-tr tasks
                        torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path)
                    else:
                        if_coeff_opt = False # nothing to balance!
                        meta_optimizer = torch.optim.Adam(curr_fold_xi.parameters(), lr_meta)
                        torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path)
                        for iter in range(num_meta_iterations): # num_meta_iterations will be number of meta-tr tasks for non-toy cases -- but it doesn't really matter! only meta-cal task num matters!
                            if iter % 1000 == 0:
                                print('saving at iter: ', iter, 'meta loss scoring: ', float(meta_loss))
                                torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path)
                                torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path + str(iter)) # just as a backup
                            meta_optimizer.zero_grad()
                            meta_loss, inner_loss = vanailla_maml_CE_loss(curr_fold_xi, self.N, curr_fold_dict_for_scoring, mb_size_task, mb_size_tr_set, mb_size_te_samples, 'None', self.lr_inner, self.inner_iter, device)
                            meta_loss.backward(retain_graph=False)
                            meta_optimizer.step()
                        torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path) # final net
                else:
                    if_coeff_opt = False # nothing to balance!
                    meta_optimizer = torch.optim.Adam(curr_fold_xi.parameters(), lr_meta)
                    torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path)
                    for iter in range(num_meta_iterations): # num_meta_iterations will be number of meta-tr tasks for non-toy cases -- but it doesn't really matter! only meta-cal task num matters!
                        if iter % 1000 == 0:
                            print('saving at iter: ', iter, 'meta loss scoring: ', float(meta_loss))
                            torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path)
                            torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path + str(iter)) # just as a backup
                        meta_optimizer.zero_grad()
                        meta_loss, inner_loss = vanailla_maml_CE_loss(curr_fold_xi, self.N, curr_fold_dict_for_scoring, mb_size_task, mb_size_tr_set, mb_size_te_samples, 'None', self.lr_inner, self.inner_iter, device)
                        meta_loss.backward(retain_graph=False)
                        meta_optimizer.step()
                    torch.save(curr_fold_xi, meta_trained_net_for_k_fold_path) # final net
        # actual xi that will be used during runtime -- no fold
        meta_trained_net_full = saved_net_path + 'full_set/meta_trained_scoring'
        if os.path.exists(meta_trained_net_full):
            print('load saved scoring net')
            full_xi = torch.load(meta_trained_net_full)
        else:
            print('train scoring net')
            if os.path.isdir(saved_net_path + 'full_set'):
                pass
            else:
                os.makedirs(saved_net_path + 'full_set')
            if tmp_saved_path_for_xi is None:
                full_xi = copy.deepcopy(xi) # we need many k_folds
            else:
                full_xi = torch.load(tmp_saved_path_for_xi)
            
            if_coeff_opt = False # nothing to balance!
            meta_optimizer = torch.optim.Adam(full_xi.parameters(), lr_meta)
            if self.exp_mode == 'toy':
                if len(self.full_dataset_dict_meta_tr_tr.keys()) == 0: # no task inside due to few meta-tr tasks
                    torch.save(full_xi, meta_trained_net_full)
                else:
                    for iter in range(num_meta_iterations):
                        if iter % 1000 == 0:
                            print('saving at iter: ', iter, 'meta loss full scoring: ', float(meta_loss))
                            torch.save(full_xi, meta_trained_net_full)
                            torch.save(full_xi, meta_trained_net_full + str(iter)) # just as a backup
                        meta_optimizer.zero_grad()
                        meta_loss, inner_loss = vanailla_maml_CE_loss(full_xi, self.N, self.full_dataset_dict_meta_tr_tr, mb_size_task, mb_size_tr_set, mb_size_te_samples, 'None', self.lr_inner, self.inner_iter, device)
                        meta_loss.backward(retain_graph=False)
                        meta_optimizer.step()
                    torch.save(full_xi, meta_trained_net_full)
            else:
                for iter in range(num_meta_iterations):
                    if iter % 1000 == 0:
                        print('saving at iter: ', iter, 'meta loss full scoring: ', float(meta_loss))
                        torch.save(full_xi, meta_trained_net_full)
                        torch.save(full_xi, meta_trained_net_full + str(iter)) # just as a backup
                    meta_optimizer.zero_grad()
                    meta_loss, inner_loss = vanailla_maml_CE_loss(full_xi, self.N, self.full_dataset_dict_meta_tr_tr, mb_size_task, mb_size_tr_set, mb_size_te_samples, 'None', self.lr_inner, self.inner_iter, device)
                    meta_loss.backward(retain_graph=False)
                    meta_optimizer.step()
                torch.save(full_xi, meta_trained_net_full)
        # training predictor network
        quantile_predictor_net_path = saved_net_path + 'predictor_net/' + 'deep_sets' #+ 'alpha_' + str(alpha)
        if os.path.exists(quantile_predictor_net_path):
            print('load saved predictor net')
            predictor_net = Predictor_Deep_Sets()
            predictor_net = predictor_net.to(device)
            predictor_net.load_state_dict(torch.load(quantile_predictor_net_path))  
        else:
            print('train predictor net')
            if os.path.isdir(saved_net_path + 'predictor_net/'):
                pass
            else:
                os.makedirs(saved_net_path + 'predictor_net/')
            # now train
            if num_total_tasks_for_trval == 0:
                num_meta_iterations_predictor = 0
            else:
                mb_task_for_predictor = 16//folding_num #64//folding_num
            predictor_net = Predictor_Deep_Sets()
            predictor_net = predictor_net.to(device)
            meta_optimizer_predictor = torch.optim.Adam(predictor_net.parameters(), lr_meta_predictor)
            # for init for task 0 case
            torch.save(predictor_net.state_dict(), quantile_predictor_net_path)  

            if IF_PREDICTOR_SET_GEN_ONCE:
                print('generating predictor data set first')
                whole_dataset_for_predictor_net = []
                for ind_sample in range(NUM_PREDICT_DATASET):
                    if ind_sample % 100 == 0:
                        print('generating sample:', ind_sample)
                    dict_for_predictor_dataset_single_sample = {}
                    for ind_fold in range(folding_num):
                        dict_for_predictor_dataset_single_sample['ind_fold_' + str(ind_fold)] = {}
                        meta_trained_net_for_k_fold_path = saved_net_path + 'fold_ind_' + str(ind_fold) + '/meta_trained_scoring'
                        xi = torch.load(meta_trained_net_for_k_fold_path)
                        # data set for predictor train
                        curr_fold_dict_for_predictor = dict_for_k_fold_dict['for_cdf_predictor'][ind_fold] # dict for meta-train
                        rand_perm_tasks = torch.randperm(num_total_tasks_for_trval)
                        rand_tasks_mb = rand_perm_tasks[:mb_task_for_predictor]
                        ind_task_from_0 = 0
                        for ind_task in rand_tasks_mb: # mb_task_for_predictor 64//folding_num following the original paper
                            input_data, true_emp_quantile = compute_dataset_for_predictor_net(curr_fold_dict_for_predictor, self.N, self.IF_ADA_DETERMINISTIC, self.lr_inner, self.inner_iter, xi, NC_mode, self.num_classes, device, alpha)
                            dict_for_predictor_dataset_single_sample['ind_fold_' + str(ind_fold)]['ind_task_' + str(ind_task_from_0)] = (input_data, true_emp_quantile)
                            ind_task_from_0 += 1
                    whole_dataset_for_predictor_net.append(dict_for_predictor_dataset_single_sample)
            else:
                pass

            for iter in range(num_meta_iterations_predictor):
                # for each fold get the loss     
                if iter % 1000 == 0:
                    print('saving at iter: ', iter, 'predictor loss: ', float(loss_predictor_tot))
                    torch.save(predictor_net.state_dict(), quantile_predictor_net_path)
                    torch.save(predictor_net.state_dict(), quantile_predictor_net_path + str(iter)) # just as a backup
                    ######
                if IF_PREDICTOR_SET_GEN_ONCE:
                    rand_index = int(torch.randperm(NUM_PREDICT_DATASET)[0])
                    curr_dict_for_pred = whole_dataset_for_predictor_net[rand_index]
                else:
                    pass

                meta_optimizer_predictor.zero_grad()
                loss_predictor_tot = 0
                for ind_fold in range(folding_num):
                    if IF_PREDICTOR_SET_GEN_ONCE:
                        for ind_task_from_zero in range(mb_task_for_predictor):
                            input_data, true_emp_quantile = curr_dict_for_pred['ind_fold_' + str(ind_fold)]['ind_task_' + str(ind_task_from_zero)]
                            est_quantile = predictor_net(input_data)
                            loss_predictor_tot += pow((torch.squeeze(est_quantile) - torch.squeeze(true_emp_quantile)), 2)
                    else:

                        meta_trained_net_for_k_fold_path = saved_net_path + 'fold_ind_' + str(ind_fold) + '/meta_trained_scoring'
                        xi = torch.load(meta_trained_net_for_k_fold_path)
                        # data set for predictor train
                        curr_fold_dict_for_predictor = dict_for_k_fold_dict['for_cdf_predictor'][ind_fold] # dict for meta-train
                        rand_perm_tasks = torch.randperm(num_total_tasks_for_trval)
                        rand_tasks_mb = rand_perm_tasks[:mb_task_for_predictor]
                        for ind_task in rand_tasks_mb: # mb_task_for_predictor 64//folding_num following the original paper
                            input_data, true_emp_quantile = compute_dataset_for_predictor_net(ind_task, curr_fold_dict_for_predictor, self.N, self.IF_ADA_DETERMINISTIC, self.lr_inner, self.inner_iter, xi, NC_mode, self.num_classes, device, alpha)
                            est_quantile = predictor_net(input_data)
                            loss_predictor_tot += pow((torch.squeeze(est_quantile) - torch.squeeze(true_emp_quantile)), 2)
                loss_predictor_tot /= (folding_num * len(rand_tasks_mb))
                loss_predictor_tot.backward()
                meta_optimizer_predictor.step()
            torch.save(predictor_net.state_dict(), quantile_predictor_net_path) # final
        ### now meta-cal
        ### first compute Lambda
        total_emp_cdf = 0
        LOOs_for_quantile_predictor_dict = {}
        NCs_for_cdf_dict = {}
        for ind_task in range(self.num_total_tasks_meta_tr_cal):
            if isinstance(self.full_dataset_dict_meta_tr_cal, dict):
                tr_dataset = self.full_dataset_dict_meta_tr_cal[ind_task]['tr_'+str(int(0))]
                te_dataset = self.full_dataset_dict_meta_tr_cal[ind_task]['te_'+str(int(0))]
                assert te_dataset is None # need to marginalize over training data set to get marginal measures!
                (tr_dataset, te_dataset) = randomly_divide_tr_te(tr_dataset, self.N)
            else:
                tr_dataset, te_dataset = self.full_dataset_dict_meta_tr_cal.gen(self.N, device)
            X_te_tot = te_dataset[0]
            Y_te_tot = te_dataset[1]
            if self.IF_ADA_DETERMINISTIC:
                u_for_tr_dataset = torch.ones(self.N, 1).to(device)
                u_for_X_te = torch.ones(X_te_tot.shape[0], 1).to(device)
            else:
                u_for_tr_dataset = torch.rand(self.N, 1).to(device)
                u_for_X_te = torch.rand(X_te_tot.shape[0], 1).to(device)
            LOOs_for_quantile_predictor, NCs_for_cdf = compute_NC_scores_for_Few_Shot_Auxiliary(self.lr_inner, self.inner_iter, full_xi, tr_dataset, X_te_tot, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te_tot, self.num_classes, no_grad=True)
            LOOs_for_quantile_predictor_dict['full_fold_ind_task_'+str(ind_task)] = LOOs_for_quantile_predictor
            NCs_for_cdf_dict['full_fold_ind_task_'+str(ind_task)] = NCs_for_cdf.detach()
        curr_lambda = 0
        delta_lambda = 0.1 # NC from 0 ~ 1 
        init_status = None
        iter_count = 0
        while delta_lambda > 0.0001:
            avg_cdf = 0
            for ind_task in range(self.num_total_tasks_meta_tr_cal):
                avg_cdf += compute_emp_cdf(curr_lambda, predictor_net, LOOs_for_quantile_predictor_dict['full_fold_ind_task_'+str(ind_task)], NCs_for_cdf_dict['full_fold_ind_task_'+str(ind_task)])
            avg_cdf /= (self.num_total_tasks_meta_tr_cal+1)
            if avg_cdf >= 1-alpha:
                curr_lambda -= delta_lambda
                if init_status is None:
                    init_status = 1
                curr_status = 1
            else:
                curr_lambda += delta_lambda
                if init_status is None:
                    init_status = -1
                curr_status = -1
            if init_status * curr_status < 0: # means status changed
                delta_lambda /= 2
                init_status = None
            else:
                pass
            iter_count += 1
            if iter_count > 10000:
                break # too many..
        print('final:!!!', 'curr avg cdf', avg_cdf, 'target', 1-alpha)
        print('corrected lambda', curr_lambda)
        torch.save(curr_lambda, quantile_predictor_net_path + 'correction_lambda.pt')
        torch.save(alpha, quantile_predictor_net_path + 'corrected_alpha_conditional.pt')
        # now conditional lambda correction

        return full_xi, predictor_net, curr_lambda, alpha

def compute_condtional_correction(delta, alpha, num_samples_for_cdf, num_tasks_cal):
    alpha_prime_num_step = 10000
    alpha_prime_step_size = (1-(1-delta)**(1/num_tasks_cal))/(alpha_prime_num_step+1)
    smallest_sqrt_term = 999999999999
    for ind_search_alpha_prime in range(alpha_prime_num_step):
        alpha_prime = alpha_prime_step_size*(ind_search_alpha_prime+1)
        tmp_1 = num_tasks_cal*np.sqrt(np.log(2/alpha_prime)/(2*num_samples_for_cdf))
        tmp_2 = np.log(1-(1-delta)/((1-alpha_prime)**num_tasks_cal))
        inside_sqrt_term = (-2/(num_tasks_cal**2))*tmp_1*tmp_2
        if inside_sqrt_term < smallest_sqrt_term:
            #print('smallest_sqrt_term', smallest_sqrt_term)
            smallest_sqrt_term = inside_sqrt_term
            best_alpha_prime = alpha_prime
        else:
            pass
    if alpha-np.sqrt(smallest_sqrt_term) < 0:
        return 0
    else:
        return alpha-np.sqrt(smallest_sqrt_term)



def compute_emp_cdf(curr_lambda, predictor_net, curr_task_LOOs_for_quantile_predictor, curr_task_NCs_for_cdf):
    est_quantile = predictor_net(curr_task_LOOs_for_quantile_predictor) # no need to compute multiple times but for neatness of code..
    corrected_quantile = est_quantile + curr_lambda
    curr_task_NCs_for_cdf = torch.squeeze(curr_task_NCs_for_cdf)
    # now compute CDF
    emp_cdf = curr_task_NCs_for_cdf.shape[0] - torch.count_nonzero(torch.nn.functional.relu(curr_task_NCs_for_cdf - corrected_quantile))  # (num_te)
    emp_cdf = float(emp_cdf)
    emp_cdf /= curr_task_NCs_for_cdf.shape[0]
    return emp_cdf


def meta_training_few_shot_auxiliary_entire_procedure(args, xi, N):
    N_for_meta_tr = N
    if_soft_inf_val_differentiable = True
    meta_tr_class_few_shot_auxiliary = Meta_Tr_Few_Shot_Auxiliary(args.exp_mode, N_for_meta_tr, args.num_classes, args.dim_x,  args.meta_tr_setting, args.meta_val_setting, args.num_tasks_ratio_meta_cal, args.mb_size_te_samples)
    full_xi, predictor_net, curr_lambda, alpha = meta_tr_class_few_shot_auxiliary.forward(args.folding_num, args.saved_net_path_auxiliary_meta, args.device, args.alpha, xi,  args.NC_mode, args.num_meta_iterations, args.num_meta_iterations_predictor, args.mb_size_task, args.mb_size_tr_set, args.mb_size_te_samples, lr_meta=args.meta_lr, lr_meta_predictor=args.meta_lr_predictor)      
    return full_xi, predictor_net, curr_lambda, alpha


def k_fold(original_dict, k):
    dict_for_k_fold_dict = {}
    dict_for_k_fold_dict['for_scoring_func'] = []
    dict_for_k_fold_dict['for_cdf_predictor'] = [] 
    num_total_tasks = len(original_dict.keys())//2 # gt_beta_i, i 
    print('num total tasks', num_total_tasks)
    if num_total_tasks == 0:
        pass
    else:
        size_fold = num_total_tasks // k
        for ind_fold in range(k):
            for_scoring_func_dict = {}
            ind_task_scoring_dict = 0
            for_cdf_predictor_dict = {}
            ind_task_cdf_dict = 0
            start_task_ind_for_excluding = size_fold*ind_fold
            for ind_task in range(num_total_tasks):
                if start_task_ind_for_excluding <= ind_task <= start_task_ind_for_excluding+size_fold-1: # 0~9, 10~ 19, ...
                    for_cdf_predictor_dict[ind_task_cdf_dict] = original_dict[ind_task]
                    ind_task_cdf_dict += 1
                else:
                    for_scoring_func_dict[ind_task_scoring_dict] = original_dict[ind_task]
                    ind_task_scoring_dict += 1
            dict_for_k_fold_dict['for_scoring_func'].append(for_scoring_func_dict)
            dict_for_k_fold_dict['for_cdf_predictor'].append(for_cdf_predictor_dict)
    return dict_for_k_fold_dict
        
    
def k_fold_OTA(k, num_classes, N, num_tr_te_per_task_meta):
    dict_for_k_fold_dict = {}
    dict_for_k_fold_dict['for_scoring_func'] = []
    dict_for_k_fold_dict['for_cdf_predictor'] = [] 
    for ind_fold in range(k):
        # actually generator not dict!
        for_scoring_func_dict = Data_gen_OTA(N, mode='trtr_fold_'+str(ind_fold), num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta)
        for_cdf_predictor_dict = Data_gen_OTA(N, mode='trval_fold_'+str(ind_fold), num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta)
        dict_for_k_fold_dict['for_scoring_func'].append(for_scoring_func_dict)
        dict_for_k_fold_dict['for_cdf_predictor'].append(for_cdf_predictor_dict)
    return dict_for_k_fold_dict

def k_fold_MINI(k, num_classes, N, num_tr_te_per_task_meta, num_total_tasks_meta_tr_tr):
    dict_for_k_fold_dict = {}
    dict_for_k_fold_dict['for_scoring_func'] = []
    dict_for_k_fold_dict['for_cdf_predictor'] = [] 
    size_fold = num_total_tasks_meta_tr_tr // k # no big meaning here..
    num_tasks_predictor = size_fold
    num_tasks_scoring = num_total_tasks_meta_tr_tr - size_fold
    for ind_fold in range(k):
        # actually generator not dict!
        # generate twice will give rather different data set
        for_scoring_func_dict = Data_gen_miniimagenet(N, mode='train', num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta, num_total_task=num_tasks_scoring, k_fold_ind=ind_fold*2+0) # generate new combination of tasks
        for_cdf_predictor_dict = Data_gen_miniimagenet(N, mode='train', num_classes=num_classes, supp_and_query=num_tr_te_per_task_meta, num_total_task=num_tasks_predictor, k_fold_ind=ind_fold*2+1)
        dict_for_k_fold_dict['for_scoring_func'].append(for_scoring_func_dict)
        dict_for_k_fold_dict['for_cdf_predictor'].append(for_cdf_predictor_dict)
    return dict_for_k_fold_dict


class Predictor_Deep_Sets(nn.Module):
    def __init__(self):
        super(Predictor_Deep_Sets, self).__init__()
        self.enc1 = nn.Linear(1, 256, bias=True)  # gets NC_score
        self.enc2 = nn.Linear(256, 256, bias=True)
        self.dec1 = nn.Linear(256, 256, bias=True)
        self.dec2 = nn.Linear(256, 1, bias=True) # outputs estimated quantile
        self.activ = nn.ReLU()
        
    def forward(self, x_list):
        # x: [NC_y_dict[0]['NC_i'], ..., NC_y_dict[N-1]['NC_i']] (list)
        # deep sets - permutation invariant
        summed_feature = 0
        for x in x_list:
            summed_feature += self.enc2(self.activ(self.enc1(x.detach())))
        estimated_quantile = self.dec2(self.activ(self.dec1(summed_feature)))
        return estimated_quantile



def compute_NC_scores_for_Few_Shot_Auxiliary(lr_inner, inner_iter, xi, tr_dataset, X_te, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te, num_classes, no_grad=True):    
    N = tr_dataset[0].shape[0] #len(tr_dataset)
    softmax = torch.nn.Softmax(dim=1)
    LOOs_for_quantile_predictor = []
    for i in range(N):
        curr_x_tr = tr_dataset[0][i].unsqueeze(dim=0)  # this should be (1, *) , * for input size
        curr_y_tr = tr_dataset[1][i].unsqueeze(dim=0)  # this should be (1, 1) , single label
        LOO_X_tr = torch.cat([tr_dataset[0][:i],tr_dataset[0][i+1:]], dim=0)
        LOO_Y_tr = torch.cat([tr_dataset[1][:i],tr_dataset[1][i+1:]], dim=0)
        LOO_dataset = (LOO_X_tr, LOO_Y_tr) #tr_dataset[:i]+tr_dataset[i+1:]
        if 'maml' in NC_mode: 
            curr_phi_LOO = GD(xi, LOO_dataset, lr_inner, inner_iter, False)
        else:
            raise NotImplementedError
        prob_vec_i_LOO = compute_prob_vec(curr_phi_LOO, curr_x_tr, xi, NC_mode, num_classes)
        if 'adaptive' in NC_mode:
            u_i = u_for_tr_dataset[i].unsqueeze(dim=0) # (1,1)
        else:
            u_i = None
        NC_i = NC_compute(prob_vec_i_LOO, torch.squeeze(curr_y_tr), NC_mode, u_i, None)
        if no_grad:
            LOOs_for_quantile_predictor.append(NC_i.detach())
        else:
            LOOs_for_quantile_predictor.append(NC_i)
    # fit with full N
    FULL_dataset = tr_dataset
    curr_phi_FULL = GD(xi, FULL_dataset, lr_inner, inner_iter, False)
    prob_vec_y_prime_FULL = compute_prob_vec(curr_phi_FULL, X_te, xi, NC_mode, num_classes) # (num_te, num_classes)
    NCs_for_cdf = NC_compute(prob_vec_y_prime_FULL, Y_te, NC_mode, u_for_X_te, None) # (num_te, 1) 
    return LOOs_for_quantile_predictor, NCs_for_cdf
    

def compute_dataset_for_predictor_net(ind_task, curr_fold_dict_for_predictor, N, IF_ADA_DETERMINISTIC, lr_inner, inner_iter, xi, NC_mode, num_classes, device, alpha):
    ## generate data set for predictor
    if isinstance(curr_fold_dict_for_predictor, dict):
        tr_dataset = curr_fold_dict_for_predictor[int(ind_task)]['tr_'+str(int(0))]
        te_dataset = curr_fold_dict_for_predictor[int(ind_task)]['te_'+str(int(0))]
        assert te_dataset is None 
        (tr_dataset, te_dataset) = randomly_divide_tr_te(tr_dataset, N)
    else:
        tr_dataset, te_dataset = curr_fold_dict_for_predictor.gen(N, device)
    X_te_tot = te_dataset[0]
    Y_te_tot = te_dataset[1]
    if IF_ADA_DETERMINISTIC:
        u_for_tr_dataset = torch.ones(N, 1).to(device)
        u_for_X_te = torch.ones(X_te_tot.shape[0], 1).to(device)
    else:
        u_for_tr_dataset = torch.rand(N, 1).to(device)
        u_for_X_te = torch.rand(X_te_tot.shape[0], 1).to(device)
    LOOs_for_quantile_predictor, NCs_for_cdf = compute_NC_scores_for_Few_Shot_Auxiliary(lr_inner, inner_iter, xi, tr_dataset, X_te_tot, NC_mode, u_for_tr_dataset, u_for_X_te, Y_te_tot, num_classes, no_grad=True)
    input_data = LOOs_for_quantile_predictor#.to(device)
    data_for_output = NCs_for_cdf.detach()
    true_emp_quantile = quantile_plus(data_for_output.unsqueeze(dim=0), 1-alpha) # NCs_for_cdf (unsqueezed): (1, num_te, 1)
    return input_data, true_emp_quantile