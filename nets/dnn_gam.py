import torch
import torch.nn as nn

class Dnn_GAM(nn.Module):
    def __init__(self, dim_x, num_classes):
        super(Dnn_GAM, self).__init__()
        self.fc1 = nn.Linear(dim_x, dim_x*6, bias=True) 
        self.fc2 = nn.Linear(dim_x*6, dim_x*6, bias=True)
        self.fc3 = nn.Linear(dim_x*6, num_classes, bias=True)  
        self.activ = nn.ELU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, var, bn_running_stats=None, alpha_TN=None, if_softmax_out=False):
        # bn_running_stats for consistency with cnn -- not used for dnn
        if var is None:
            var = list(map(lambda p: p[0], zip(self.parameters())))
        else:
            pass
        num_layer = len(var)//2 # weight, bias
        for idx_layer in range(num_layer): # 0,1, ..., num_layer-1
            x = torch.nn.functional.linear(x, var[2*idx_layer], var[2*idx_layer + 1])
            if idx_layer < num_layer-1: # 0, 1, ..., num_layer - 2
                x = self.activ(x)
            else:
                pass
        if if_softmax_out:
            x = self.softmax(x)
        else:
            pass
        return x