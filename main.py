import torch
import numpy as np
import matplotlib.pyplot as plt
from compute_empirical_validity_inefficiency.empirical_est import compute_empirical_est, generate_rep_dataset
from meta_train.meta_training import meta_training_entire_procedure
from meta_train.meta_tr_benchmark import meta_training_few_shot_auxiliary_entire_procedure, Predictor_Deep_Sets
from torch.utils.tensorboard import SummaryWriter
from funcs.utils import reset_random_seed
from nets.dnn import Dnn
from nets.dnn_gam import Dnn_GAM
from nets.cnn import Cnn
from nets.vgg16 import VGG16
import scipy.io as sio
import argparse
import os
import pickle
from funcs.utils_for_eval import abbreviate_exp_details, load_xi_from_specified_exp_details, evaluate_with_xi, exp_details_from_args
 

def parse_args():
    parser = argparse.ArgumentParser(description='meta ca')
    parser.add_argument('--alpha', type=float, default=0.1, help='predetermined miscoverage level')
    parser.add_argument('--N', type=int, default=9, help='number of examples in the data set')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes for classification task')
    parser.add_argument('--num_rep_phi_eval', type=int, default=100, help='number of different meta-test tasks when evaluation')
    parser.add_argument('--num_rep_tr_set_eval', type=int, default=1000, help='number of different data sets when evaluation')
    parser.add_argument('--num_rep_te_samples_eval', type=int, default=500, help='number of different test points when evaluation')
    parser.add_argument('--NC_mode', type=str, default='maml_original', help='[maml_original, maml_adaptive, SVC_original, SVC_adaptive]: if original is contained in the string, then use conventional NC score (Def. 2); if adaptive is contained in the string, then use adaptive NC score (Def .4). If maml is contained, then use neural network classifier; while SVC being contained, use support vector classifer]') 
    parser.add_argument('--set_prediction_mode', type=str, default='JK+mm', choices=['JK+mm', 'CV+3mm', 'SC', 'auxiliary_few_shot_meta'], help='choice of the set predictors -- JK+mm and CV+Kmm (please add desired K instead of 3 in the choices) is for XB-CP and meta-XB, SC is for VB-CP, auxiliary_few_shot_meta for meta-VB') 
    parser.add_argument('--num_total_tasks_meta', type=int, default=20000, help='number of total tasks for meta-training (T)')
    parser.add_argument('--num_tr_te_per_task_meta', type=int, default=500, help='number of total examples for each meta-training data set')
    parser.add_argument('--meta_learning_coeff_for_CE_regul', type=float, default=0, help='used for meta-objective function when considering CE loss as regularization (always 0 except for miniimagenet case)')
    parser.add_argument('--meta_lr', type=float, default=0.001, help='learning rate for meta-training (kappa)')
    parser.add_argument('--dim_x', type=int, default=10, help='dimension of x for synthetic data experiment for Sec.5-A')
    parser.add_argument('--mb_size_task', type=int, default=1, help='minibatch size for tasks during meta-training (tilde T)') 
    parser.add_argument('--mb_size_tr_set', type=int, default=1, help='minibatch size for training set during meta-training (tilde M_t)') 
    parser.add_argument('--mb_size_te_samples', type=int, default=160, help='minibatch size for test points for speed up computing the inefficiency') 
    parser.add_argument('--num_meta_iterations', type=int, default=20000, help='number of iterations for meta-training')    
    parser.add_argument('--if_compute_conditional_coverage', dest='if_compute_conditional_coverage', action='store_true', default=False, help='whether compute per-input conditional coverage during meta-training')    
    parser.add_argument('--exp_mode', type=str, default='toy', choices=['toy','toy_vis_gam', 'miniimagenet', 'OTA'], help='mode for experimental set-up') 
    parser.add_argument('--cuda_ind', type=int, default=-1, help='CUDA device index if using GPU (-1 forces to use CPU)')
    parser.add_argument('--c_sigmoid', type=float, default=1.0, help='approximation parameter for sigmoid')
    parser.add_argument('--c_sigmoid_for_ada_NC', type=float, default=1.0, help='approximation parameter for sigmoid only for soft adaptive NC score')
    parser.add_argument('--c_S', type=float, default=1.0, help='approximation parameter for softmin')
    parser.add_argument('--c_Q', type=float, default=1.0, help='approximation parameter for soft quantile using pinball loss')
    parser.add_argument('--sq_mode', type=str, default='pinball', choices=['OT', 'pinball'], help='mode for soft quantile,  pinball is the proposed method, OT is the optimal transport by Cuturi et al. (OT is not used in the paper but left in case it may be useful for someone :) )') # OT reference: https://arxiv.org/abs/1905.11885
    parser.add_argument('--rand_seed', type=int, default=3, help='init rand seed')
    parser.add_argument('--alpha_TN_init', type=float, default=1.0, help='used when using batch normalization layer (only miniimagenet case here) depending on this coefficient, TaskNorm is considered. alpha=1.0: only BN, alpha=0.0 only IN (we assume 1.0 only BN but left this functionality just in case)') # reference for TaskNorm: https://arxiv.org/abs/2003.03284
    # setting for meta-VB (named after 'auxiliary meta' in this code)
    parser.add_argument('--folding_num', type=int, default=3, help='number of folds during meta-training')
    parser.add_argument('--num_meta_iterations_predictor', type=int, default=20000, help='number of training iterations for quantile estimator')
    parser.add_argument('--meta_lr_predictor', type=float, default=0.001, help='learning rate for quantile estimator')
    parser.add_argument('--num_tasks_ratio_meta_cal', type=float, default=0.5, help='ratio for meta-calibration tasks among total meta-training tasks')
    args = parser.parse_args()

    ######## CHECK LIST ###########
    ### inner learning rate and number of inner steps `during meta-training` that defines the training algorithm for meta-XB and meta-VB is controlled directly in /compute_empirical_validity_inefficiency/empirical_est.py (for meta-XB) and in /meta_train/meta_tr_benchmark.py (for meta-VB)
    ### inner learning rate and number of inner steps `during meta-testing` that defines the training algorithm for meta-XB and meta-VB is controlled directly in /compute_empirical_validity_inefficiency/empirical_est.py (for both meta-XB and meta-VB)
    ### note that if one want to consider increased steps during meta-testing as in most meta-learning implementations, need to manually change setting in the file /compute_empirical_validity_inefficiency/empirical_est.py based on the meta-trained hyperparameter vector (xi)
    ### for toy, we consider same setting, while for OTA and miniimagenet, we consider inner step 1 during meta-training and 5 during meta-testing

    if args.exp_mode == 'toy':
        args.alpha = 0.1
        args.dim_x = 10
        args.num_classes = 5
        if args.num_total_tasks_meta == 0:
            args.num_meta_iterations = 0
        else:
            pass
        if args.set_prediction_mode == 'auxiliary_few_shot_meta':
            args.num_meta_iterations = 20000  
            args.num_meta_iterations_predictor = 5000 
        else: 
            args.num_meta_iterations = 20000 
        args.meta_lr = 0.0001 
        args.mb_size_task = 2
        args.mb_size_tr_set = 4
        args.mb_size_te_samples = 20
        args.M_for_conditional_coverage = 1000 # setting for conditional coverage computation (see, e.g., S1.2 https://arxiv.org/pdf/2006.02544.pdf)
        args.num_tr_te_per_task_meta = 50*(args.N+1) ## (N+1)*M = (10)*50 = 500 
    elif args.exp_mode == 'toy_vis_gam':
        args.num_rep_phi_eval = 1
        args.num_rep_tr_set_eval = 2
        args.num_classes = 6 
        args.num_rep_te_samples_eval = args.num_classes*10
        args.N = 9
        args.alpha = 0.1 
        args.dim_x = 2
        if args.num_total_tasks_meta == 0:
            args.num_meta_iterations = 0
        else:
            pass
        if args.set_prediction_mode == 'auxiliary_few_shot_meta':
            args.num_meta_iterations = 20000
        else:
            args.num_meta_iterations = 20000 
        args.meta_lr = 0.001 
        args.mb_size_task = 1 #2
        args.mb_size_tr_set = 1 #4
        args.mb_size_te_samples = 20
        args.M_for_conditional_coverage = 1000 # setting for conditional coverage computation (see, e.g., S1.2 https://arxiv.org/pdf/2006.02544.pdf)
    elif args.exp_mode == 'OTA':
        args.alpha = 0.1
        args.N = 9
        args.num_classes = 2
        if args.set_prediction_mode == 'auxiliary_few_shot_meta':
            args.num_meta_iterations = 20000
            args.num_meta_iterations_predictor = 5000
        else:
            args.num_meta_iterations = 20000 
        args.meta_lr = 0.00005
        args.mb_size_task = 1
        args.mb_size_tr_set = 1
        args.mb_size_te_samples = 160 
        if args.num_total_tasks_meta == 0:
            pass
        else:
            args.num_total_tasks_meta = args.num_meta_iterations*args.mb_size_task*args.mb_size_tr_set  # every time randomly generate task #20000
        args.M_for_conditional_coverage = 2 # not using conditional 
    elif args.exp_mode == 'miniimagenet':
        args.N = 4
        args.alpha = 0.2
        args.num_classes = 2
        if args.num_total_tasks_meta == 0:
            pass
        else:
            args.num_total_tasks_meta = 20000 # standard assumption for m-tr in miniimagenet
            if args.set_prediction_mode == 'auxiliary_few_shot_meta':
                args.num_meta_iterations = 20000 
                args.num_meta_iterations_predictor = 5000
            else:
                args.num_meta_iterations = 20000 
        args.meta_lr = 0.0001
        args.mb_size_task = 1 
        args.mb_size_tr_set = 1 
        args.mb_size_te_samples = 50 
        args.M_for_conditional_coverage = 2 # not using conditional 
    else:
        raise NotImplementedError
    if args.cuda_ind == -1:
        args.device = "cpu"
    else:
        args.device = torch.device("cuda:" + str(args.cuda_ind) if torch.cuda.is_available() else "cpu")

    if 'CV+' in args.set_prediction_mode:
        assert 'mm' in args.set_prediction_mode
        K = int(args.set_prediction_mode[3])
        args.alpha = args.alpha - (1-K/args.N)/(K+1)
        if args.alpha < 0:
            args.alpha = 0
        print('corrrected alpha from K fold: ', args.alpha)
    else:
        pass

    return args

def full_training_procedure(args):
    args.rand_seed_for_datagen_mte = 0 
    args.rand_seed_for_datagen_mval = 777 
    args.rand_seed_for_datagen_mtr = 999
    args.meta_tr_setting = (args.num_total_tasks_meta, args.num_tr_te_per_task_meta, args.rand_seed_for_datagen_mtr)
    args.meta_val_setting = (5, 10, 200, args.rand_seed_for_datagen_mval) #(5, 10, 200, args.rand_seed_for_datagen_mval)
    reset_random_seed(args.rand_seed) #3 done, 
    phi_for_training = None
    print('alpha: ', args.alpha, 'NC_mode: ', args.NC_mode, 'set pred mode', args.set_prediction_mode)
    dir_path = './saved/meta_cb'
    eval_dict = {}
    eval_dict_dir = dir_path + '/results/nips_2022/' + args.exp_mode + '/'
    if os.path.isdir(eval_dict_dir):
        pass
    else:
        os.makedirs(eval_dict_dir)
    eval_dict_dir_with_setting = eval_dict_dir + str(args.set_prediction_mode)  + 'NC_mode' + str(args.NC_mode) + 'num_tasks_' + str(args.num_total_tasks_meta)+ 'meta_learning_coeff_for_CE_regul_' + str(args.meta_learning_coeff_for_CE_regul) + 'alpha_' + str(args.alpha)
    eval_dict_save_path_for_matlab = eval_dict_dir_with_setting + 'toy.mat'
    eval_dict_save_path_for_pkl = eval_dict_dir_with_setting + 'dict.pkl'
    
    N_list = [args.N]
    validity_te_list = torch.zeros(len(N_list))
    ineff_te_list = torch.zeros(len(N_list))

    exp_details = exp_details_from_args(args)
    meta_trained_net_dir = dir_path + '/saved_nets/nips_2022/' + args.exp_mode + '/'  +  exp_details
    if os.path.isdir(meta_trained_net_dir):
        pass
    else:
        os.makedirs(meta_trained_net_dir)
    args.PATH_dir_for_meta_trained_xi = meta_trained_net_dir

    # for auxiliary 
    args.saved_net_path_auxiliary_meta = meta_trained_net_dir #dir_path + '/saved_nets/auxiliary_meta/'

    if (args.exp_mode == 'toy') or (args.exp_mode == 'toy_vis_gam'):
        if args.exp_mode == 'toy':
            xi = Dnn(dim_x=args.dim_x, num_classes=args.num_classes)
        else:
            xi = Dnn_GAM(dim_x=args.dim_x, num_classes=args.num_classes)
        xi.alpha_TN = None # no need for DNN - for normalization layer
        xi = xi.to(args.device)
    elif args.exp_mode == 'OTA':
        xi = VGG16(num_classes=args.num_classes)
        xi.alpha_TN = None
        xi = xi.to(args.device)
    elif args.exp_mode == 'miniimagenet':
        xi = Cnn(num_classes=args.num_classes)
        xi.alpha_TN = torch.tensor([args.alpha_TN_init], requires_grad=False) # fix to the best value reported in the original tasknorm paper
        xi.alpha_TN = xi.alpha_TN.to(args.device)
        xi = xi.to(args.device)
    else:
        raise NotImplementedError
    ind_N = 0
    for N in N_list:
        if args.num_total_tasks_meta > 0:
            if args.set_prediction_mode == 'auxiliary_few_shot_meta':
                full_xi, predictor_net, corr_lambda, alpha = meta_training_few_shot_auxiliary_entire_procedure(args, xi, N)
                args.alpha = alpha
                if args.exp_mode == 'miniimagenet':
                    full_xi.alpha_TN = torch.tensor([args.alpha_TN_init], requires_grad=False)
                    full_xi.alpha_TN = full_xi.alpha_TN.to(args.device)
                else:
                    full_xi.alpha_TN = None
                xi = (full_xi, predictor_net, corr_lambda)
            elif ('JK+' in args.set_prediction_mode) or ('CV+' in args.set_prediction_mode):
                print('start meta-training!')
                xi = meta_training_entire_procedure(args, xi, N)
            elif args.set_prediction_mode == 'SC':
                raise NotImplementedError
                xi = joint_training_conftr_entire_procedure(args, xi, N)
            else:
                raise NotImplementedError
        else:
            if args.set_prediction_mode == 'auxiliary_few_shot_meta':
                full_xi = xi
                predictor_net = Predictor_Deep_Sets()
                corr_lambda = 0
                if args.exp_mode == 'miniimagenet':
                    full_xi.alpha_TN = torch.tensor([args.alpha_TN_init], requires_grad=False)
                    full_xi.alpha_TN = full_xi.alpha_TN.to(args.device)
                else:
                    full_xi.alpha_TN = None
                xi = (full_xi, predictor_net, corr_lambda)
            else:
                pass

        exp_details = exp_details_from_args(args)
        M_for_conditional_coverage = args.M_for_conditional_coverage #1000 #3 # 1000
        delta = 0.1
        actual_test_ratio = 0.0 # since we are choosing x slab by single realization of phi and training set, we can use one whole for choosing x slab
        eval_dict_dir = dir_path + '/results/nips_2022/' + args.exp_mode + '/M_' + str(M_for_conditional_coverage) + '/'        
        evaluate_with_xi(xi, args, dir_path, eval_dict_dir, exp_details, M_for_conditional_coverage, delta, actual_test_ratio)

def visualization_with_saved_net(args):
    args.rand_seed_for_datagen_mte = 0 
    dir_path = './saved/meta_cb'
    args.saved_net_path_auxiliary_meta = meta_trained_net_dir

    if args.num_total_tasks_meta == 0:
        if (args.exp_mode == 'toy') or (args.exp_mode == 'toy_vis_gam'):
            if args.exp_mode == 'toy':
                xi = Dnn(dim_x=args.dim_x, num_classes=args.num_classes)
            else:
                xi = Dnn_GAM(dim_x=args.dim_x, num_classes=args.num_classes)
            xi.alpha_TN = None # no need for DNN - for normalization layer
            xi = xi.to(args.device)
        elif args.exp_mode == 'OTA':
            xi = VGG16(num_classes=args.num_classes)
            xi.alpha_TN = None
            xi = xi.to(args.device)
        elif args.exp_mode == 'miniimagenet':
            xi = Cnn(num_classes=args.num_classes)
            xi = xi.to(args.device)
            xi.alpha_TN = torch.tensor([args.alpha_TN_init], requires_grad=False) # fix to the best value reported in the original tasknorm paper
            xi.alpha_TN = xi.alpha_TN.to(args.device)
        else:
            raise NotImplementedError
        option_for_pretrained = 'None'
    else:
        option_for_pretrained = 'final'
    dir_path = './saved/meta_cb'
    xi, exp_details = load_xi_from_specified_exp_details(args, dir_path, option_for_pretrained)
    M_for_conditional_coverage = args.M_for_conditional_coverage #3 #1000 #3 # 1000
    delta = 0.1
    actual_test_ratio = 0.0 # since we are choosing x slab by single realization of phi and training set, we can use one whole for choosing x slab
    eval_dict_dir = dir_path + '/results/nips_2022/' + args.exp_mode + '/M_' + str(M_for_conditional_coverage) + '/'
    evaluate_with_xi(xi, args, dir_path, eval_dict_dir, exp_details, M_for_conditional_coverage, delta, actual_test_ratio)

if __name__ == '__main__':
    args = parse_args()
    print('args: ', args)
    ## check whether saved net exists
    exp_details = exp_details_from_args(args)
    dir_path = './saved/meta_cb'
    meta_trained_net_dir = dir_path + '/saved_nets/nips_2022/' + args.exp_mode + '/'  +  exp_details
    print('meta_trained_net_dir', meta_trained_net_dir)
    if os.path.isdir(meta_trained_net_dir) is True:
        print('start meta-testing for evaluation purpose!')
        visualization_with_saved_net(args)
    else: # meta-training (optimizing hyperparameter vector xi) for meta-XB and meta-VB
        print('start meta-training for meta-VB or meta-XB!')
        full_training_procedure(args)
