# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

# from torchmetrics.functional import pairwise_euclidean_distance
# from torchmetrics.functional import pairwise_manhattan_distance

def pairwise_euclidean_distance(x, y):
    """Calculates the pairwise euclidean distance matrix.
    Args:
        x: tensor of shape ``[N,d]``
        y: tensor of shape ``[M,d]``
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    # xx + yy - 2 * x * yT 
    distance = torch.sparse.addmm(
        mat = (xx + yy).double(), 
        mat1 = x.to_sparse().double(), 
        mat2 = y.T.double(), 
        beta = -2, alpha = 1).half()
    return distance.sqrt()

def pairwise_manhattan_distance(x, y):
    """Calculates the pairwise manhattan similarity matrix.
    Args:
        x: tensor of shape ``[N,d]``
        y: if provided, a tensor of shape ``[M,d]``
    """
    distance = (x.unsqueeze(1) - y.unsqueeze(0).repeat(x.shape[0], 1, 1)).abs().sum(dim=-1)
    return distance

def calculate_entropy(feats, k=1):
    _,C,H,W = feats.size()
    feats = feats.view(-1,C,H*W)
            
    N = C # N number dimension
    K = C//10 # 采用最合适的 k 值
    
    Hs = []
    for feat in feats: # feat: (C,H*W)
        dist = pairwise_euclidean_distance(feat, feat) # (C, C)
        order = torch.argsort(dist, dim=1)
        H = 0
        for n in range(N):
            # ball V
            r_ball = dist[n][order[n][K]]
            H += r_ball
        Hs.append(H)
    Hs = torch.stack(Hs)
    H = torch.sum(Hs)
    return torch.log(H+1)

@weighted_loss
def layer_entropy_loss(value, target):
    return value-target

@LOSSES.register_module()
class EntropyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, net_info):
        var = 0
        for block in net_info:
            delta_entropy = []
            pre_layer_entropy = calculate_entropy(block[2])
            for layer_i in range(2, len(block)-3, 3): # every 3 layers consist a block
                # print(f"--------------------block{layer_i}------------------------------------")
                # print(f"eloss/EntropyLoss/delta: {calculate_entropy(block[layer_i+3])-calculate_entropy(block[layer_i])}")
                delta_entropy.append(calculate_entropy(block[layer_i+3])-pre_layer_entropy)
                pre_layer_entropy = calculate_entropy(block[layer_i])
            delta_entropy = torch.stack(delta_entropy)
            var += torch.var(delta_entropy)        
        eloss_var = self.loss_weight * layer_entropy_loss(var, 0)
        
        # end_entropy = calculate_entropy(net_info[-1][-1])
        # eloss_end = self.loss_weight * layer_entropy_loss(end_entropy, 0)
        return {"eloss_var": [eloss_var]} #, "eloss_end": [eloss_end]
