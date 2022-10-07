import torch
import os
from nets.dnn import Dnn
from nets.dnn_gam import Dnn_GAM
from nets.cnn import Cnn
from nets.vgg16 import VGG16
import pickle
from compute_empirical_validity_inefficiency.empirical_est import compute_empirical_est, generate_rep_dataset
from data.data_gen_miniimagenet import Data_gen_miniimagenet
from data.data_gen_ota import Data_gen_OTA
from meta_train.meta_tr_benchmark import Predictor_Deep_Sets


def abbreviate_exp_details(exp_details, args):
    name = ''
    if ('num_total_tasks_meta_0' in exp_details):
        name += 'conv '
    elif 'SC' in exp_details:
        name += 'joint'
    else:
        name += 'meta'
    
    if 'SVC' in exp_details:
        name += '-SVC'
    else:
        name += '-NN '

    if 'adaptive' in exp_details:
        name += '-A'
    else:
        name += '-O'

    if 'JK+' in exp_details:
        if 'mm' in exp_details:
            name += '-JK+mm'
        else:
            name += '-JK+'
    elif 'CV+' in exp_details:
        name += '-CV+'
    elif 'SC' in exp_details:
        name += '-SC'
    elif 'auxiliary_few_shot_meta' in exp_details:
        name += '-auxiliary_meta'
    else:
        pass
    
    name +=  'num_classes_' + str(args.num_classes) + 'coeff_' + str(args.meta_learning_coeff_for_CE_regul) + 'num_tasks_' + str(args.num_total_tasks_meta)
    
    return name

def exp_details_from_args(args):
    ### TOY 
    if args.exp_mode == 'toy':
        return 'CUSTUM_DIR/toy/final/per_N_exp/' + 'sq_mode' + str(args.sq_mode) +  'seed_' + str(args.rand_seed) +'set_prediction_mode' + str(args.set_prediction_mode)   + '/' + str(args.NC_mode) +  '/' +'N_' + str(args.N) + 'num_class_' + str(args.num_classes) +'_num_total_tasks_meta_' + str(args.num_total_tasks_meta) +  '/coeff_' + str(args.meta_learning_coeff_for_CE_regul) +  'c_sigmoid_for_ada_NC_' + str(args.c_sigmoid_for_ada_NC) + 'tau_for_soft_indicator_' + str(args.c_sigmoid) +  '/'
        #return 'CUSTUM_DIR/toy/final/per_task_exp/' + 'sq_mode' + str(args.sq_mode) +  'seed_' + str(args.rand_seed) +'set_prediction_mode' + str(args.set_prediction_mode) + '/' + str(args.NC_mode) +  '/' +'N_' + str(args.N) + 'num_class_' + str(args.num_classes) +'_num_total_tasks_meta_' + str(args.num_total_tasks_meta) +  '/coeff_' + str(args.meta_learning_coeff_for_CE_regul) +  'c_sigmoid_for_ada_NC_' + str(args.c_sigmoid_for_ada_NC) + 'tau_for_soft_indicator_' + str(args.c_sigmoid) +  '/'
    elif args.exp_mode == 'miniimagenet':
        return 'CUSTUM_DIR/mini/FINAL_EXPERIMENTS/inner_4/class_' + str(args.N) + '/alpha_' + str(args.alpha) + '/affine_false/mb_task_' + str(args.mb_size_task) +   '_mb_tr_' + str(args.mb_size_tr_set) + '_mb_te_' + str(args.mb_size_te_samples) + '/inner_0_0001_iter_1/meta_lr' + str(args.meta_lr) +  'shuffle_false/' + 'alpha_TN_' + str(args.alpha_TN_init) +  'sq_mode' + str(args.sq_mode) +  'seed_' + str(args.rand_seed) +'set_prediction_mode' + str(args.set_prediction_mode)  + '/' + str(args.NC_mode) +  '/' +'N_' + str(args.N) + 'num_class_' + str(args.num_classes) +'_num_total_tasks_meta_' + str(args.num_total_tasks_meta) +  '/coeff_' + str(args.meta_learning_coeff_for_CE_regul) +  'c_sigmoid_for_ada_NC_' + str(args.c_sigmoid_for_ada_NC) + 'tau_for_soft_indicator_' + str(args.c_sigmoid) +  '/'
    elif args.exp_mode == 'OTA':
        return 'CUSTUM_DIR/OTA/final/inner_0_1_iter_1_during_mtr_0_1_iter_5_during_mte_T0_1000_Tmul_1/' + 'sq_mode' + str(args.sq_mode) +  'seed_' + str(args.rand_seed) +'set_prediction_mode' + str(args.set_prediction_mode)  + '/' + str(args.NC_mode) +  '/' +'N_' + str(args.N) + 'num_class_' + str(args.num_classes) +'_num_total_tasks_meta_' + str(args.num_total_tasks_meta) +  '/coeff_' + str(args.meta_learning_coeff_for_CE_regul) +  'c_sigmoid_for_ada_NC_' + str(args.c_sigmoid_for_ada_NC) + 'tau_for_soft_indicator_' + str(args.c_sigmoid) +  '/'
    else:
        raise NotImplementedError
def load_xi_from_specified_exp_details(args, dir_path, option_for_pretrained):
    exp_details = exp_details_from_args(args)
    meta_trained_net_dir = dir_path + '/saved_nets/nips_2022/' + args.exp_mode + '/'  +  exp_details
    print('meta_trained_net_dir', meta_trained_net_dir)
    assert os.path.isdir(meta_trained_net_dir) is True
    PATH_for_pretrained_xi = meta_trained_net_dir + option_for_pretrained
    if ('maml' in args.NC_mode) or ('cavia' in args.NC_mode):
        if args.set_prediction_mode == 'auxiliary_few_shot_meta':
            if args.exp_mode in ['toy', 'toy_vis_gam', 'OTA', 'miniimagenet']: 
                PATH_for_quantile_predictor_auxiliary_meta = args.saved_net_path_auxiliary_meta + 'predictor_net/' + 'deep_sets' 
                try:
                    args.alpha = torch.load(PATH_for_quantile_predictor_auxiliary_meta+'corrected_alpha_conditional.pt')
                except FileNotFoundError:
                    raise ValueError('Meta-training has not been successfully finished. Please remove corresponding directory for the current setting, and meta-train again.')
                
                print('effective alpha: ', args.alpha)
                if (args.exp_mode == 'toy') or (args.exp_mode == 'toy_vis_gam'):
                    if args.exp_mode == 'toy':
                        total_xi = Dnn(dim_x=args.dim_x, num_classes=args.num_classes)
                    else:
                        total_xi = Dnn_GAM(dim_x=args.dim_x, num_classes=args.num_classes)
                else:
                    total_xi = VGG16(num_classes=args.num_classes)
                total_xi = total_xi.to(args.device)
                PATH_for_scoring_full_xi_auxiliary_meta = args.saved_net_path_auxiliary_meta + 'full_set/meta_trained_scoring'
                try:
                    total_xi = torch.load(PATH_for_scoring_full_xi_auxiliary_meta)
                except FileNotFoundError:
                    raise ValueError('Meta-training has not been successfully finished. Please remove corresponding directory for the current setting, and meta-train again.')
                
                predictor_net = Predictor_Deep_Sets()
                predictor_net = predictor_net.to(args.device)
                try:
                    predictor_net.load_state_dict(torch.load(PATH_for_quantile_predictor_auxiliary_meta))  
                except FileNotFoundError:
                    raise ValueError('Meta-training has not been successfully finished. Please remove corresponding directory for the current setting, and meta-train again.')
                try:
                    corr_lambda = torch.load(PATH_for_quantile_predictor_auxiliary_meta + 'correction_lambda.pt')
                except FileNotFoundError:
                    raise ValueError('Meta-training has not been successfully finished. Please remove corresponding directory for the current setting, and meta-train again.')
                
                xi = (total_xi, predictor_net, corr_lambda)
            else:
                raise NotImplementedError
        else:
            if (args.exp_mode == 'toy') or (args.exp_mode == 'toy_vis_gam'):
                if args.exp_mode == 'toy':
                    xi = Dnn(dim_x=args.dim_x, num_classes=args.num_classes)
                else:
                    xi = Dnn_GAM(dim_x=args.dim_x, num_classes=args.num_classes)
                xi = xi.to(args.device)
            elif args.exp_mode == 'OTA':
                xi = VGG16(num_classes=args.num_classes)
                xi = xi.to(args.device)
            elif args.exp_mode == 'miniimagenet':
                xi = Cnn(num_classes=args.num_classes)
                xi = xi.to(args.device)
            else:
                raise NotImplementedError
            if option_for_pretrained == 'None':
                print('-------random init----------')
            else:
                print('------load saved-----------')
                print('from PATH: ', PATH_for_pretrained_xi)
                
                try: 
                    xi = torch.load(PATH_for_pretrained_xi)
                except FileNotFoundError:
                    raise ValueError('Meta-training has not been successfully finished. Please remove corresponding directory for the current setting, and meta-train again.')
                    #raise FileNotFoundError
    else:
        if option_for_pretrained == 'None':
            print('-------SVC from scratch (only for cross val)----------')
            xi = None
        else:
            print('------load saved SVC-----------')
            with open(PATH_for_pretrained_xi, 'rb') as f:
                xi = pickle.load(f)
    return xi, exp_details

def evaluate_with_xi(xi, args, dir_path, eval_dict_dir, exp_details, M_for_conditional_coverage, delta, actual_test_ratio):
    if os.path.isdir(eval_dict_dir + exp_details):
        pass
    else:
        os.makedirs(eval_dict_dir + exp_details)
    eval_dict_dir_with_setting = eval_dict_dir + exp_details
    eval_dict_save_path_for_matlab = eval_dict_dir_with_setting + 'toy.mat'
    eval_dict_save_path_for_pkl = eval_dict_dir_with_setting + 'dict.pkl'
    scheme_details = abbreviate_exp_details(exp_details, args)
    N = args.N 
    dir_PATH_for_full_dataset_dict_mte = dir_path + '/saved_dataset/' + str(args.exp_mode) + 'num_class_' + str(args.num_classes) +'N_' + str(N) + str(args.num_rep_phi_eval) + str(args.num_rep_tr_set_eval) + str(args.num_rep_te_samples_eval)
    if 'soft' in args.NC_mode: # during eval. it should be hard ranking!
        raise NotImplementedError
    else:
        pass
    quantile_mode = 'hard'
    if_soft_inf_val_differentiable = False
    if (args.exp_mode == 'toy'):
        if os.path.isdir(dir_PATH_for_full_dataset_dict_mte):
            print('load saved data set for mte')
            with open(dir_PATH_for_full_dataset_dict_mte + '/meta_te.pkl', 'rb') as f:
                full_dataset_dict_mte = pickle.load(f)
        else:
            os.makedirs(dir_PATH_for_full_dataset_dict_mte)
            full_dataset_dict_mte = generate_rep_dataset(args.exp_mode, None, args.dim_x, args.num_classes, N, args.num_rep_phi_eval, args.num_rep_tr_set_eval, args.num_rep_te_samples_eval, args.rand_seed_for_datagen_mte)
            with open(dir_PATH_for_full_dataset_dict_mte + '/meta_te.pkl', 'wb') as f:
                pickle.dump(full_dataset_dict_mte, f)
            print('mte data set generation done!')
    elif (args.exp_mode == 'toy_vis_gam'):
        print('generate mte data set')
        fixed_m = 2 
        if_special_setting = True
        full_dataset_dict_mte = generate_rep_dataset(args.exp_mode, None, args.dim_x, args.num_classes, N, args.num_rep_phi_eval, args.num_rep_tr_set_eval, args.num_rep_te_samples_eval, args.rand_seed_for_datagen_mte, if_special_setting, fixed_m)
    elif args.exp_mode == 'OTA':
        full_dataset_dict_mte = Data_gen_OTA(N, mode='test', num_classes=args.num_classes, supp_and_query=N+args.num_rep_te_samples_eval)
    elif args.exp_mode == 'miniimagenet':
        full_dataset_dict_mte = Data_gen_miniimagenet(N, mode='test', num_classes=args.num_classes, supp_and_query=N+args.num_rep_te_samples_eval, num_total_task=args.num_rep_phi_eval)
    else:
        raise NotImplementedError
    
    if args.set_prediction_mode == 'SC':
        if 'maml' in args.NC_mode:
            # need this since SC requires soft quantiles during meta-testing
            if_soft_inf_val_differentiable = True
            c_sigmoid = args.c_sigmoid
            c_sigmoid_for_ada_NC = args.c_sigmoid_for_ada_NC
        else: # SVC doesn't need soft things
            if_soft_inf_val_differentiable = False
            c_sigmoid = None
            c_sigmoid_for_ada_NC = None
    else:
        # just for safe
        if_soft_inf_val_differentiable = False
        c_sigmoid = None
        c_sigmoid_for_ada_NC = None

    (validity, inefficiency), _, (conditional_validity, conditional_ineff), dict_for_pd = compute_empirical_est(args.exp_mode, args.device, full_dataset_dict_mte, args.alpha, xi, args.num_classes, args.set_prediction_mode, args.NC_mode, quantile_mode, args.num_rep_phi_eval, args.num_rep_tr_set_eval, args.num_rep_te_samples_eval, if_soft_inf_val_differentiable, True, phi=None, iter_ConfTr=None, N=None, mb_size_task = args.num_rep_phi_eval, mb_size_tr_set=args.num_rep_tr_set_eval, mb_size_te_samples=args.num_rep_te_samples_eval, c_sigmoid=c_sigmoid, c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC, delta=delta, M=M_for_conditional_coverage, actual_test_ratio=actual_test_ratio, scheme_details=scheme_details)
    # save dict_for_pd
    with open(eval_dict_save_path_for_pkl, 'wb') as f:
        pickle.dump(dict_for_pd, f)
    print('N:', N, 'coverage: ', validity, 'ineff', inefficiency, 'conditional coverage: ', conditional_validity, 'conditional ineff: ', conditional_ineff)