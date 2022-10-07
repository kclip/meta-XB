import h5py
import torch
import numpy as np
import math

class Data_gen_OTA:
    def __init__(self, N, mode, num_classes, supp_and_query, path_for_h5py=None, fixed_snr_value=30, num_total_tasks=None):
        self.N = N
        if path_for_h5py is None:
            path_for_h5py = '/DIR_FOR_OTA/radioml_2018/archive/GOLD_XYZ_OSC.0001_1024.hdf5'
        else:
            pass
        self.f = h5py.File(path_for_h5py, "r")
        # mode: 'train', 'test'
        # here: 'train' with modulation index % 3 == 0,1 , 'test' modulation index % 3 == 2
        if mode == 'train':
            self.modulation_indices = [0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22]
        elif mode == 'test':
            self.modulation_indices = [2,5,8,11,14,17,20,23]
        elif mode == 'trtr': # for meta-auxiliary
            self.modulation_indices = [0,3,6,9,12,15,18,21]
        elif mode == 'trcal': # for meta-auxiliary
            self.modulation_indices = [1,4,7,10,13,16,19,22]
        else:
            pass
        if 'fold' in mode:
            # trtr k-fold
            if mode == 'trval_fold_0':
                self.modulation_indices = [0,9,18]
            elif mode == 'trval_fold_1':
                self.modulation_indices = [3,12,21]
            elif mode == 'trval_fold_2':
                self.modulation_indices = [6,15,21]
            elif mode == 'trtr_fold_0':
                self.modulation_indices = [3,6,12,15,21]
            elif mode == 'trtr_fold_1':
                self.modulation_indices = [0,6,9,15,18]
            elif mode == 'trtr_fold_2':
                self.modulation_indices = [0,3,9,12,18]
            else:
                raise NotImplementedError
        else:
            pass
        self.supp_and_query = supp_and_query
        self.fixed_snr_value = fixed_snr_value
        self.num_classes = num_classes
        self.num_total_tasks = num_total_tasks 
        if num_total_tasks is not None:
            raise NotImplementedError # need to define what is the number of tasks here!
    def gen(self, N, device):
        if N is None:
            N = self.N
        else:
            pass
        #num_te_for_gen = self.supp_and_query-N
        # randomly select tasks depending on the mode
        rand_perm = torch.randperm(len(self.modulation_indices))[0:self.num_classes]
        curr_modulation_indices = [self.modulation_indices[i] for i in rand_perm] # e.g., [13,7]
        # task is defined from curr_modulation_indices, e.g., BPSK and 16QAM for 2 classes
        # now generate N samples by randomly sampling modulation and also randomly sampling frames for that modulation type
        dict_for_mapping_actual_modul_to_m_ary = {}
        for ind_class in range(self.num_classes):
            dict_for_mapping_actual_modul_to_m_ary[curr_modulation_indices[ind_class]] = torch.tensor(ind_class) # {13: 0, 7: 1}
        X = []
        Y = []
        for _ in range(self.supp_and_query):
            curr_modul = curr_modulation_indices[torch.randperm(self.num_classes)[0]]
            if self.fixed_snr_value is None:
                curr_snr_db = 2*(torch.randperm(26)[0]-10) # 0,1,...,25 -> -10, -9, ..., 15 -> -20, -18 .., 30
            else:
                curr_snr_db = self.fixed_snr_value # fixed snr value as the given one
            curr_frame = torch.randperm(4096)[0]
            curr_actual_index = self.get_actual_index_given_setting(curr_modul, curr_snr_db, curr_frame)
            x = self.f['X'][curr_actual_index] # (1024, 2)
            x = np.transpose(x) # (2, 1024) (C, L)
            y = dict_for_mapping_actual_modul_to_m_ary[curr_modul] # (1)
            x = torch.from_numpy(x)
            X.append(x.unsqueeze(dim=0)) # (1,2,1024)
            Y.append(y.unsqueeze(dim=0)) # (1,1)
        X = torch.cat(X, dim=0).to(device) # (N+num_te_for_gen, 2, 1024)
        Y = torch.cat(Y, dim=0).to(device) # (N+num_te_for_gen, 1)
        Y = Y.unsqueeze(dim=1)
        tr_dataset = (X[:N], Y[:N]) # (N, 2, 1024), (N, 1) <-> (N, C, L), (N, 1)
        te_dataset = (X[N:], Y[N:]) # (num_te_for_gen, 2, 1024), (num_te_for_gen, 1)
        return tr_dataset, te_dataset
    @staticmethod
    def get_actual_index_given_setting(curr_modulation, curr_snr_db, curr_frame, total_frames=4096, total_snrs=26):
        # num samples for each modulation: total_frames*total_snrs
        total_samples_for_modulation = total_frames*total_snrs
        # change snr db into snr index
        snr_index = int(curr_snr_db/2)+10 # snr db: -20, -18, ...30 -> -10,-9, ..., 14,15 -> 0, 1, ..., 24, 25 # should be int!
        return curr_modulation*total_samples_for_modulation + snr_index*total_frames + curr_frame
