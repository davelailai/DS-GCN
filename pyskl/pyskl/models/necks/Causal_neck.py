# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
# from .gread import global_add_pool, global_mean_pool, global_max_pool
# from .gread import GlobalAttention, Set2Set
import scipy.spatial as sp
from .causalnn import *


from ..builder import NECKS, build_loss


@NECKS.register_module()
class CausalNeck(nn.Module):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 in_channels,
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__()

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode
        self.node_type = 5
        self.fc_cls = nn.Linear(in_channels, self.node_type)
        self.cMLP = cMLP(25, lag=9, hidden=[100])


    
    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)
        x_feature = x
        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)
        return x, x_feature

    def node_precost(self, x, node_type):
        N,M,C,T,V = x.shape
        x = x.mean(3).permute(0,1,3,2).reshape(-1,C)
        node_pre = self.fc_cls(x)
        node_type = node_type.unsqueeze(0).unsqueeze(2).repeat(N*M,1,1)
        node_label = torch.zeros(N*M, V, 5).scatter_(2,node_type,1)
        # node_label = node_label.unsqueeze(1).repeat(1,T,1,1).reshape(-1,5).to(x.device)
        node_label = node_label.reshape(-1,5).to(x.device)
        # node_type = node_type.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N,M,1,T,1)
        # node_type = node_type.permute(0,1,3,4,2).reshape(-1,1).to(x.device)
        loss = nn.CrossEntropyLoss(reduce=False)
        node_loss = loss(node_pre, node_label)

        return torch.mean(node_loss)
    def GcCost(self, x, lam_ridge):
        x= x.mean(dim = 1)
        B,C,T,V = x.shape
        x = x.reshape(-1,T,V)
        p = V
        loss_fn = nn.MSELoss(reduction='mean')

        # Calculate smooth error.
        loss = sum([loss_fn(self.cMLP.networks[i](x[:, :-1]), x[:, 9:, i:i+1])
                    for i in range(p)])
        ridge = sum([ridge_regularize(net, lam_ridge) for net in self.cMLP.networks])

        smooth = loss + ridge

        return smooth

        # for it in range(max_iter):
        #     # Take gradient step.
        #     smooth.backward()
        #     for param in cmlp.parameters():
        #         param.data = param - lr * param.grad

        #     # Take prox step.
        #     if lam > 0:
        #         for net in cmlp.networks:
        #             prox_update(net, lam, lr, penalty)

        #     cmlp.zero_grad()

        #     # Calculate loss for next iteration.
        #     loss = sum([loss_fn(cmlp.networks[i](X[:, :-1]), X[:, lag:, i:i+1])
        #                 for i in range(p)])
        #     ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        #     smooth = loss + ridge

        #     # Check progress.
        #     if (it + 1) % check_every == 0:
        #         # Add nonsmooth penalty.
        #         nonsmooth = sum([regularize(net, lam, penalty)
        #                         for net in cmlp.networks])
        #         mean_loss = (smooth + nonsmooth) / p
        #         train_loss_list.append(mean_loss.detach())


        # mask = mask[:,:,0,:,:].reshape(-1,1)

        # return torch.sum(node_loss)/torch.sum(mask)






