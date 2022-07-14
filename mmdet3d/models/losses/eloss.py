# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn

from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss

# from torchmetrics.functional import pairwise_euclidean_distance
# from torchmetrics.functional import pairwise_manhattan_distance


def pairwise_euclidean_distance(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * x @ y.t()
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-8).sqrt()
    return dist


def calculate_entropy(feats, k=1):
    _,C,H,W = feats.size()
    feats = feats.view(-1,C,H*W)
            
    N = C # N number dimension
    K = C//10 # 采用最合适的 k 值
    
    H_total = 0
    for feat in feats: # feat: (C,H*W)
        dist = pairwise_euclidean_distance(feat, feat) # (C, C)
        order = torch.argsort(dist, dim=1)
        for n in range(N):
            # ball V
            r_ball = dist[n][order[n][K]]
            H_total += r_ball
    return torch.log(H_total+1)

@weighted_loss
def layer_entropy_loss(value, target):
    return value # - target

@LOSSES.register_module()
class EntropyLoss(nn.Module):
    def __init__(self, 
                 loss_weight=1.0,
                 pick_block_num = 1):
        super(EntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pick_block_num = pick_block_num

    def forward(self, net_info):
        var = 0
        for block in net_info[:self.pick_block_num]: # pick the first block
            delta_entropy = []
            
            pre_layer_entropy = calculate_entropy(block[2])
            for layer_i in range(2, len(block)-3, 3): # every 3 layers
                # print(f"--------------------layer{layer_i}------------------------------------")
                current_layer_entropy = calculate_entropy(block[layer_i+3])
                delta_entropy.append(current_layer_entropy-pre_layer_entropy)
                # print(f"eloss/forward/current_layer_entropy:{current_layer_entropy}")
                pre_layer_entropy = current_layer_entropy
            
            delta_entropy = torch.stack(delta_entropy)
            var += torch.var(delta_entropy)        
        eloss_var = self.loss_weight * layer_entropy_loss(var, 0)

        return {"eloss_var": [eloss_var]}
