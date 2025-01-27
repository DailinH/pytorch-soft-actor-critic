# import torch 
# import torch.nn as nn
# from torchvision.models import resnet18
# import numpy as np

# class AffineCouplingLayer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, mask):
#         super(AffineCouplingLayer, self).__init__()
#         self.mask = mask
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), 
#             nn.LeakyReLU(), 
#             nn.Linear(hidden_dim, hidden_dim), 
#             nn.LeakyReLU(), 
#             nn.Linear(hidden_dim, 2*input_dim)
#         )

#     def forward(self, input, reverse=False):
#         x0 = torch.mul(input, self.mask)
#         st = self.net(x0)
#         # rescale s with tanh and scale factor
#         s, t = torch.chunk(st, 2, dim=1)
#         s = torch.mul(1-self.mask, torch.tanh(s))
#         t = torch.mul(1-self.mask, t)
#         if reverse:
#             # FROM Z TO X
#             tmp = torch.mul(input-t, torch.exp(-s))
#             output = x0 + torch.mul(1-self.mask, tmp)
#             log_det = -s.sum(-1)
#         else: 
#             # FROM X TO Z
#             tmp = torch.mul(input, torch.exp(s)) + t
#             output = x0 + torch.mul(1-self.mask, tmp)
#             log_det = s.sum(-1)
#         return output, log_det 

# class Net(nn.Module):
#     def __init__(self, N, input_dim, hidden_dim, device):
#         super(Net, self).__init__()
#         self.n = 4
#         self.device = device
#         mask_checkerboard = np.indices((1, input_dim)).sum(axis=0)%2

#         mask_checkerboard = np.append(mask_checkerboard,1 - mask_checkerboard,axis=0)
#         mask_checkerboard = np.append(mask_checkerboard, mask_checkerboard, axis=0)
#         mask_checkerboard = np.append(mask_checkerboard, 1 - mask_checkerboard, axis=0)
#         # print("input dim", N, input_dim, mask_checkerboard)

#         self.masks = torch.Tensor(mask_checkerboard).to(self.device)
#         self.layers = nn.ModuleList([AffineCouplingLayer(input_dim=input_dim, hidden_dim=hidden_dim, mask=self.masks[i]) for i in range(self.n)])

#     def forward(self, input, reverse=False):
#         # stack 3 layers with alternating checkboard pattern.
#         log_det_loss = torch.zeros(input.size()[0]).to(self.device)
#         z = input
#         index_range = range(self.n) if not reverse else range(self.n-1, -1 , -1)
#         # print(z, reverse)
#         for idx in index_range:
#             z, log_det = self.layers[idx](z, reverse)
#             log_det_loss += log_det
#         return z, log_det_loss

import numpy as np
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

class RealNVP(nn.Module):
    def __init__(self, nets, nett, masks, prior):
        super(RealNVP, self).__init__()
        
        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])
        
    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J
    
    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp
        
    def sample(self, batchSize): 
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x