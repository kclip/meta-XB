import torch
import torch.nn as nn
import torch.nn.functional as F

IF_AFFINE = True

class Cnn(nn.Module):
    def __init__(self, num_classes=None, if_max_pool=True, if_affine=IF_AFFINE, num_filters=32, kernel_size=3, stride=1, padding=1, if_bn_compute_directly=True):
        super(Cnn, self).__init__()
        self.stride = stride
        self.padding = padding
        num_channels = 3
        self.if_max_pool = if_max_pool
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size, stride=stride, padding=padding) # convolutional layer
        #self.bn1 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        if if_affine:
            self.bn1 = nn.Linear(1, 1, bias=True) # weight will be used for gamma, bias will be for beta 
        else:
            # this is dummy
            self.bn1 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride, padding=padding) # convolutional layer
        if if_affine:
            self.bn2 = nn.Linear(1, 1, bias=True) # weight will be used for gamma, bias will be for beta 
        else:
            #self.bn2 = None
            self.bn2 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        #self.bn2 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride, padding=padding) # convolutional layer
        if if_affine:
            self.bn3 = nn.Linear(1, 1, bias=True) # weight will be used for gamma, bias will be for beta 
        else:
            #self.bn3 = None
            self.bn3 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        #self.bn3 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        if self.if_max_pool:
            self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride, padding=padding) # convolutional layer
        else:
            self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size, stride=stride)
        if if_affine:
            self.bn4 = nn.Linear(1, 1, bias=True) # weight will be used for gamma, bias will be for beta 
        else:
            #self.bn4 = None
            self.bn4 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        #self.bn4 = nn.BatchNorm2d(num_filters, affine = if_affine, track_running_stats=True) # batch normalization layer
        self.fc = nn.Linear(5 * 5 * num_filters, num_classes) # fully connected (FC) layer
        self.softmax = nn.Softmax(dim=1)
        self.if_bn_compute_directly = if_bn_compute_directly
    #def forward(self, x, var, bn_running_stats=None, alpha_TN=None, if_softmax_out=False, if_affine=False, if_max_pool=True, G=8, eps=1e-5):
    def forward(self, x, var, bn_running_stats=None, alpha_TN=None, if_softmax_out=False, if_affine=IF_AFFINE, if_max_pool=True, G=8, eps=1e-5):
        if var is None:
            raise NotImplementedError
            var = list(map(lambda p: p[0], zip(self.parameters())))
        else:
            pass
        idx = 0
        stride = self.stride
        padding = self.padding
        while idx < len(var):
            if if_affine:
                gap = 4
            else:
                gap = 2
            if idx > 0:
                x = F.relu(x)
            if idx == 0:
                w1, b1 = var[idx], var[idx + 1] # weight and bias
                if if_affine:
                    gamma1, beta1 = var[idx+2], var[idx+3]
                else:
                    gamma1, beta1 = None, None
                x = F.conv2d(input = x, weight = w1, bias = b1,stride=stride,
                               padding=padding)
                x, batch_stat = task_norm(x, alpha_TN, bn_running_stats[idx//gap], (gamma1, beta1))
                bn_running_stats[idx//gap] = batch_stat
                if if_max_pool:
                    x = F.max_pool2d(x, kernel_size=2)
                idx += gap
            elif idx == gap * 1:
                w2, b2 = var[idx], var[idx + 1]  # weight and bias

                if if_affine:
                    gamma2, beta2 = var[idx+2], var[idx+3]
                else:
                    gamma2, beta2 = None, None

                x = F.conv2d(input=x, weight=w2, bias=b2, stride=stride,
                             padding=padding)
                x, batch_stat = task_norm(x, alpha_TN, bn_running_stats[idx//gap], (gamma2, beta2))
                bn_running_stats[idx//gap] = batch_stat
                if if_max_pool:
                    x = F.max_pool2d(x, kernel_size=2)
                idx += gap
            elif idx == gap * 2:
                w3, b3 = var[idx], var[idx + 1]  # weight and bias

                if if_affine:
                    gamma3, beta3 = var[idx+2], var[idx+3]
                else:
                    gamma3, beta3 = None, None

                x = F.conv2d(input=x, weight=w3, bias=b3, stride=stride,
                             padding=padding)
                x, batch_stat = task_norm(x, alpha_TN, bn_running_stats[idx//gap], (gamma3, beta3))
                bn_running_stats[idx//gap] = batch_stat
                if if_max_pool:
                    x = F.max_pool2d(x, kernel_size=2)
                idx += gap
            elif idx == gap * 3:
                w4, b4 = var[idx], var[idx + 1]  # weight and bias

                if if_affine:
                    gamma4, beta4 = var[idx+2], var[idx+3]
                else:
                    gamma4, beta4 = None, None

                x = F.conv2d(input=x, weight=w4, bias=b4, stride=stride,
                             padding=padding)
                x, batch_stat = task_norm(x, alpha_TN, bn_running_stats[idx//gap], (gamma4, beta4))
                bn_running_stats[idx//gap] = batch_stat
                if if_max_pool:
                    x = F.max_pool2d(x, kernel_size=2)
                idx += gap
            else:
                x = torch.reshape(x, (x.shape[0], -1))
                fc_w, fc_b = var[idx], var[idx + 1]  # weight and bias for fc
                p = F.linear(input = x, weight = fc_w, bias = fc_b)
                idx += 2
        if if_softmax_out:
            p = self.softmax(p)
        else:
            pass
        return p


def custum_BN_old(x, mean, var):
    # x: (N, C, H, W)
    # mean: (C) -> (1,C) -> (1,C,1) -> (1,C,1,1)
    # var: (C)
    return (x-mean)/torch.sqrt(var)

# reference for TaskNorm: https://arxiv.org/abs/2003.03284

def task_norm(x, alpha_TN, bn_running_stats, affine_para, eps=1e-5):
    if bn_running_stats[0] is None:
        # during adaptation!
        # batch norm # (N, C, H, W)
        batch_mean = torch.mean(x, dim=[0,2,3], keepdim=True)
        batch_var = torch.var(x, dim=[0,2,3], unbiased=False, keepdim=True)
    else:
        batch_mean = bn_running_stats[0]
        batch_var = bn_running_stats[1]
    # currently only IN since it is reported to have better results
    instance_mean = torch.mean(x, dim=[2,3], keepdim=True)
    instance_var = torch.var(x, dim=[2,3], unbiased=False, keepdim=True)
    
    task_mean = alpha_TN*batch_mean + (1-alpha_TN)*instance_mean
    task_var = alpha_TN*(batch_var + pow((batch_mean-task_mean),2)) + (1-alpha_TN)*(instance_var+pow((instance_mean-task_mean),2))

    if affine_para[0] is None:
        x = (x-task_mean)/torch.sqrt(task_var+eps)
    else:
        x = ((x-task_mean)/torch.sqrt(task_var+eps))*affine_para[0] + affine_para[1]
    return x, (batch_mean, batch_var)
