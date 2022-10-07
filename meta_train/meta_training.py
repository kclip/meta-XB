import torch
import numpy as np
from compute_empirical_validity_inefficiency.empirical_est import generate_rep_dataset, compute_empirical_est
from torch.utils.tensorboard import SummaryWriter
from funcs.utils import randomly_divide_tr_te
from funcs.jk_plus import Jacknife_plus, GD
from funcs.utils_for_set_prediction import soft_indicator, hard_indicator
from data.data_gen_miniimagenet import Data_gen_miniimagenet
from data.data_gen_ota import Data_gen_OTA

## meta-XB implementation

class Meta_Tr:
    def __init__(self, exp_mode,  N, num_classes, dim_x,  meta_tr_setting, meta_val_setting, mb_size_te_samples):
        (num_total_tasks_meta, num_tr_te_per_task_meta, rand_seed_for_datagen_mtr) = meta_tr_setting
        (num_rep_phi_eval, num_rep_tr_set_eval, num_rep_te_samples_eval, rand_seed_for_datagen_mval) = meta_val_setting
        self.num_rep_phi_eval = num_rep_phi_eval # working as meta-val
        self.num_rep_tr_set_eval = num_rep_tr_set_eval # working as meta-val
        self.num_rep_te_samples_eval = num_rep_te_samples_eval # working as meta-val
        if (exp_mode == 'toy') or (exp_mode == 'toy_vis_gam'):
            if exp_mode == 'toy_vis_gam':
                if_special_setting = True #True
                fixed_m = None
                self.full_dataset_dict_meta_te = generate_rep_dataset(exp_mode, None, dim_x, num_classes, N, self.num_rep_phi_eval, self.num_rep_tr_set_eval, self.num_rep_te_samples_eval, rand_seed_for_datagen_mval, if_special_setting, fixed_m)
            else:
                self.full_dataset_dict_meta_te = generate_rep_dataset(exp_mode, None, dim_x, num_classes, N, self.num_rep_phi_eval, self.num_rep_tr_set_eval, self.num_rep_te_samples_eval, rand_seed_for_datagen_mval)
            
            if exp_mode == 'toy_vis_gam':
                if_special_setting = False # True
                fixed_m = None
                self.full_dataset_dict =         generate_rep_dataset(exp_mode, None, dim_x, num_classes, N, num_total_tasks_meta, 4, 1000, rand_seed_for_datagen_mtr, if_special_setting, fixed_m) # random split for meta-tr
            else:
                self.full_dataset_dict =         generate_rep_dataset(exp_mode, None, dim_x, num_classes, num_tr_te_per_task_meta, num_total_tasks_meta, 1, 0, rand_seed_for_datagen_mtr) # random split for meta-tr
        elif exp_mode == 'OTA':
            # not using self.full_dataset_dict_meta_te for model selection!
            self.full_dataset_dict_meta_te = Data_gen_OTA(N, mode='test', num_classes=num_classes, supp_and_query=N+self.num_rep_te_samples_eval)
            self.full_dataset_dict =         Data_gen_OTA(N, mode='train', num_classes=num_classes, supp_and_query=N+mb_size_te_samples)
        elif exp_mode == 'miniimagenet':
            self.full_dataset_dict_meta_te = Data_gen_miniimagenet(N, mode='test', num_classes=num_classes, supp_and_query=N+self.num_rep_te_samples_eval, num_total_task=self.num_rep_phi_eval) # to ensure i.i.d. task sampling diff. with actual mte
            self.full_dataset_dict =         Data_gen_miniimagenet(N, mode='train', num_classes=num_classes, supp_and_query=N+mb_size_te_samples, num_total_task=num_total_tasks_meta)  # to ensure i.i.d. task sampling diff. with actual mte
        else:
            raise NotImplementedError
        self.exp_mode = exp_mode
        self.num_total_tasks_meta = num_total_tasks_meta
        self.num_tr_te_per_task_meta = num_tr_te_per_task_meta
        self.num_classes = num_classes
        self.dim_x = dim_x
        self.N = N
        assert self.num_tr_te_per_task_meta > N 
    def forward(self, device, alpha, xi, set_prediction_mode, NC_mode, num_meta_iterations, mb_size_task, mb_size_tr_set, mb_size_te_samples, if_soft_inf_val_differentiable, lr_meta=0.001, coeff=0.0, c_sigmoid=None, c_sigmoid_for_ada_NC=None,  if_compute_conditional_coverage=False, PATH_dir_for_meta_trained_xi=None, c_S=None, sq_mode=None, c_Q=None):
        quantile_mode_mtr = 'soft'       
        meta_optimizer = torch.optim.Adam(xi.parameters(), lr_meta)
        iter_for_saving_net = 50        
        smallest_meta_val_ineff = 999999999
        smallest_meta_val_validity = 10
        meta_loss = 0
        if_nan_curr = False
        for iter in range(num_meta_iterations): 
            meta_optimizer.zero_grad()
            if iter % iter_for_saving_net == 0:
                # meta-validation
                quantile_mode = 'hard'
                if 'adaptive' in NC_mode:
                    NC_mode = 'maml_adaptive_hard_ada'
                else:
                    NC_mode = 'maml_original_hard_ada'
                (validity, inefficiency), _, conditional_coverage, _ = compute_empirical_est(self.exp_mode, device, self.full_dataset_dict_meta_te, alpha, xi, self.num_classes, set_prediction_mode, NC_mode, quantile_mode, self.num_rep_phi_eval, self.num_rep_tr_set_eval, self.num_rep_te_samples_eval, False, if_compute_conditional_coverage, phi=None, iter_ConfTr=None, N=None, mb_size_task = self.num_rep_phi_eval, mb_size_tr_set=self.num_rep_tr_set_eval, mb_size_te_samples=self.num_rep_te_samples_eval, c_sigmoid=c_sigmoid,c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC, delta=0.1, M=3, actual_test_ratio=0.75, scheme_details=None, c_S=c_S, sq_mode=sq_mode, c_Q=c_Q)
            if iter % iter_for_saving_net == 0:
                print('saving at iter', iter, 'with validity: ', float(validity), 'with inefficiency: ', float(inefficiency)) #, 'with conditional coverage: ', float(conditional_coverage))
                print('iter', iter, 'meta loss', meta_loss)
                torch.save(xi, PATH_dir_for_meta_trained_xi + str(iter))
                if if_nan_curr:
                    pass
                else:
                    torch.save(xi, PATH_dir_for_meta_trained_xi + 'last_net_without_nan') # overwrite
            else:
                pass

            if 'adaptive' in NC_mode:
                NC_mode = 'maml_adaptive_soft_ada'
            else:
                NC_mode = 'maml_original_soft_ada'
            (validity, ineff), ce_loss, _, _ = compute_empirical_est(self.exp_mode, device, self.full_dataset_dict, alpha, xi, self.num_classes, set_prediction_mode, NC_mode, quantile_mode_mtr, self.num_total_tasks_meta, mb_size_tr_set, self.num_tr_te_per_task_meta, if_soft_inf_val_differentiable, False, phi=None, iter_ConfTr=None, N=self.N, mb_size_task = mb_size_task, mb_size_tr_set=mb_size_tr_set, mb_size_te_samples=mb_size_te_samples, c_sigmoid=c_sigmoid, c_sigmoid_for_ada_NC=c_sigmoid_for_ada_NC,  scheme_details=None, c_S=c_S, sq_mode=sq_mode, c_Q=c_Q) # N=self.N makes random division for meta-tr
            meta_loss = (1-coeff) * ineff  + coeff * ce_loss 
            if torch.isnan(meta_loss):
                print('nan at iter: ', iter, 'loading the previous net at iter: ', iter-1, 'and keep meta-training')
                xi = torch.load(PATH_dir_for_meta_trained_xi + 'last_net_without_nan')
                return xi
            else:
                if_nan_curr = False
                if ('maml' in NC_mode) or ('cavia' in NC_mode):
                    meta_loss.backward(retain_graph=False)
                    meta_optimizer.step()   
                else:
                    raise NotImplementedError
        torch.save(xi, PATH_dir_for_meta_trained_xi + 'final')
        return xi

def meta_training_entire_procedure(args, xi, N):
    N_for_meta_tr = N
    if_soft_inf_val_differentiable = True
    meta_tr_class = Meta_Tr(args.exp_mode, N_for_meta_tr, args.num_classes, args.dim_x,  args.meta_tr_setting, args.meta_val_setting, args.mb_size_te_samples)
    xi = meta_tr_class.forward(args.device, args.alpha, xi, args.set_prediction_mode, args.NC_mode, args.num_meta_iterations, args.mb_size_task, args.mb_size_tr_set, args.mb_size_te_samples, if_soft_inf_val_differentiable, lr_meta=args.meta_lr, coeff=args.meta_learning_coeff_for_CE_regul, c_sigmoid=args.c_sigmoid, c_sigmoid_for_ada_NC=args.c_sigmoid_for_ada_NC, if_compute_conditional_coverage=args.if_compute_conditional_coverage, PATH_dir_for_meta_trained_xi=args.PATH_dir_for_meta_trained_xi,  c_S=args.c_S, sq_mode=args.sq_mode, c_Q=args.c_Q)
    print('NC mode after meta-tr:', args.NC_mode)
    return xi


def vanailla_maml_CE_loss(xi, N, full_dataset_dict, mb_size_task, mb_size_tr_set, mb_size_te_samples, set_prediction_mode, lr_inner, inner_iter, device):
    # this is used for meta-VB in meta_tr_benchmark.py 
    # supp: N, query: num_te
    rand_perm_phi = torch.randperm(mb_size_task)
    meta_loss = 0
    inner_loss = 0
    keep_grad = True
    para_list_from_net = list(map(lambda p: p[0], zip(xi.parameters())))
    for ind_rep_phi in rand_perm_phi:
        for _ in range(mb_size_tr_set):
            if isinstance(full_dataset_dict, dict): # toy
                tr_dataset = full_dataset_dict[int(ind_rep_phi)]['tr_'+str(int(0))]
                (tr_dataset, te_dataset) = randomly_divide_tr_te(tr_dataset, N) # N as in conventional approach
            else: # non-toy which is based on the generator
                tr_dataset, te_dataset = full_dataset_dict.gen(N, device)
            #(X_tr, Y_tr) = tr_dataset
            intermediate_updated_para_list = GD(xi, tr_dataset, lr_inner, inner_iter, keep_grad)
            # query
            (X_te, Y_te) = te_dataset
            X_te = X_te[:mb_size_te_samples]
            Y_te = Y_te[:mb_size_te_samples]
            out = xi(X_te, intermediate_updated_para_list, xi.bn_running_stats, xi.alpha_TN)
            meta_loss += torch.nn.functional.cross_entropy(out, torch.squeeze(Y_te))
    return meta_loss, inner_loss
