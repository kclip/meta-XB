import torch
import numpy as np


class Data_gen_toy: # for Sec. 5-A
    def __init__(self, dim_x, num_classes):
        self.beta = torch.normal(mean=torch.zeros(dim_x, num_classes), std=1.0) # (dim_x, num_classes)
        self.dim_x = dim_x # p
        self.num_classes = num_classes # 10
    def gen(self, num_samples):
        if num_samples == 0:
            return None
        else:
            with torch.no_grad():
                X = []
                Y = []
                for _ in range(num_samples):
                    x = torch.normal(mean=torch.zeros(self.dim_x), std=1.0) # (dim_x)
                    if torch.rand(1) < 0.2: # w.p. 0.2
                        x[0] = 1
                    else: # w.p. 0.8
                        x[0] = -8
                    gt_probs = torch.zeros(self.num_classes)
                    for ind_class in range(self.num_classes):
                        gt_probs[ind_class] = torch.exp(torch.sum(self.beta[:, ind_class] * x))
                    gt_probs /= torch.sum(gt_probs)
                    Cate = torch.distributions.categorical.Categorical(gt_probs)
                    y = Cate.sample()
                    X.append(x.unsqueeze(dim=0)) # (1,dim_x)
                    Y.append(y.unsqueeze(dim=0)) # (1,1)
                X = torch.cat(X, dim=0) # (num_samples, dim_x)
                Y = torch.cat(Y, dim=0) # (num_samples, 1)
                Y = Y.unsqueeze(dim=1)
                return (X, Y)

def Trans(vector):
    return torch.transpose(vector,0,1)