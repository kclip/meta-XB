import torch
import numpy as np

class Data_gen_GAM:
    def __init__(self, num_classes, P_1=1, radius=1.3, noise_p_in_nbr=0.2, rand_break_tie_p=0.00):
        self.M = num_classes
        self.theta = 1-(np.sqrt(5)-1)/2
        self.P_1 = P_1
        self.radius = radius
        self.noise_p_in_nbr = noise_p_in_nbr
        self.rand_break_tie_p = rand_break_tie_p
        self.dict = {}
        self.init_phase = torch.rand(1)*np.pi*2 # 0 ~ 2pi # working as random task
        for m in range(self.M):
            x = gam_m_to_x(m, self.P_1, self.M, self.theta, self.init_phase)
            self.dict[m] = {}
            self.dict[m]['tx_symb_complex'] = x
            self.dict[m]['tx_symb'] = x
            self.dict[m]['tx_label'] = m
            self.dict[m]['nbrs_symb'] = []
            self.dict[m]['nbrs_label'] = []
            self.dict[m]['outside_nbrs_label'] = []
            self.dict[m]['gt_prob'] = torch.zeros(self.M)
            self.dict[m]['num_nbrs'] = None
            self.dict[m]['num_out_nbrs'] = None
            for m_nbr in range(self.M):
                if m_nbr == m: # only choosing nbr!
                    pass
                else:
                    x_nbr_candidate = gam_m_to_x(m_nbr, self.P_1, self.M, self.theta, self.init_phase)
                    dist = np.abs(x_nbr_candidate-x)
                    if dist <= self.radius:
                        self.dict[m]['nbrs_symb'].append(x_nbr_candidate)
                        self.dict[m]['nbrs_label'].append(m_nbr) # this will be mostly used!
                    else:
                        self.dict[m]['outside_nbrs_label'].append(m_nbr) 
            self.dict[m]['num_nbrs'] = len(self.dict[m]['nbrs_label'])
            self.dict[m]['num_out_nbrs'] = len(self.dict[m]['outside_nbrs_label'])
            assert self.dict[m]['num_nbrs'] + self.dict[m]['num_out_nbrs'] + 1 == self.M
            e_m = torch.rand(self.dict[m]['num_nbrs'])
            e_m *= self.rand_break_tie_p
            self.dict[m]['e_m'] = e_m - torch.sum(e_m)/self.dict[m]['num_nbrs'] # normalize
            # define prob
            sum_added_rand_break_tie_noise = 0
            ind_nbr = 0
            for m_cand in range(self.M):
                if m_cand == m:
                    if self.dict[m]['num_nbrs'] == 0:
                        self.dict[m]['gt_prob'][m_cand] = 1
                    else:
                        self.dict[m]['gt_prob'][m_cand] = 1-self.noise_p_in_nbr
                elif m_cand in self.dict[m]['nbrs_label']:
                    self.dict[m]['gt_prob'][m_cand] = self.noise_p_in_nbr/self.dict[m]['num_nbrs'] + self.dict[m]['e_m'][ind_nbr]
                    ind_nbr += 1
                elif m_cand in self.dict[m]['outside_nbrs_label']:
                    pass
                else:
                    raise NotImplementedError
        self.beta = self.dict
    def gen(self, num_samples, fixed_m, if_test_fix_ordering_for_vis, fixed_training_sample):
        # fixed_training_sample: [ None, None, (15,7) ]
        if num_samples == 0:
            return None
        else:
            with torch.no_grad():
                X = []
                Y = []
                ind_vis_toy = 0
                for ind_sample in range(num_samples):
                    # randomly choose among M
                    if fixed_m is None:
                        curr_constell_ind = int(torch.randperm(self.M)[0])
                    else:
                        curr_constell_ind = fixed_m
                    if if_test_fix_ordering_for_vis:
                        curr_constell_ind = ind_vis_toy % self.M
                    else:
                        pass
                    if fixed_training_sample is None:
                        pass
                    else:
                        if fixed_training_sample[ind_sample] is None:
                            pass
                        else:
                            curr_constell_ind = fixed_training_sample[ind_sample][0]
                    x_complex = self.dict[curr_constell_ind]['tx_symb']
                    x = torch.zeros(2)
                    x[0] = x_complex.real
                    x[1] = x_complex.imag
                    gt_probs = self.dict[curr_constell_ind]['gt_prob']
                    Cate = torch.distributions.categorical.Categorical(gt_probs)
                    y = Cate.sample()
                    if fixed_training_sample is None:
                        pass
                    else:
                        if fixed_training_sample[ind_sample] is None:
                            pass
                        else:
                            y = torch.tensor(fixed_training_sample[ind_sample][1])
                    X.append(x.unsqueeze(dim=0)) # (1,dim_x)
                    Y.append(y.unsqueeze(dim=0)) # (1,1)
                    ind_vis_toy += 1
                X = torch.cat(X, dim=0) # (num_samples, dim_x)
                Y = torch.cat(Y, dim=0) # (num_samples, 1)
                Y = Y.unsqueeze(dim=1)
                return (X, Y)

def gam_m_to_x(m, P_1, M, theta, init_phase):
    m += 1
    a_m = pow((2*P_1*m)/(M+1), 0.5)
    x = a_m*np.exp(1j*2*np.pi*theta*m) * np.exp(1j*init_phase)
    return x

def get_gt_prob_from_X(X, dict):
    M = len(dict.keys())
    gt_prob = torch.zeros(X.shape[0], M)
    for ind_sample in range(X.shape[0]):
        for m_cand in range(M):
            #print(torch.view_as_complex(X[ind_sample]), dict[m_cand]['tx_symb'])
            if torch.view_as_complex(X[ind_sample]) == dict[m_cand]['tx_symb']:
                gt_prob[ind_sample] = dict[m_cand]['gt_prob']
            else:
                pass
    return gt_prob
