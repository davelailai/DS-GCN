# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, normal_init, xavier_init
from .gread import global_add_pool, global_mean_pool, global_max_pool
from .gread import GlobalAttention, Set2Set
import scipy.spatial as sp
from torch_scatter import scatter, scatter_add, scatter_max

from ..builder import NECKS, build_loss


@NECKS.register_module()
class ReadoutNeck(nn.Module):
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
                 read_op,
                 num_position,
                 gamma = 0.1,
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.emb_dim = in_channels
        self.read_op = read_op
        self.num_position = num_position
        self.gamma = gamma

        self.protos = torch.nn.Parameter(torch.zeros(num_position, in_channels), requires_grad=True)
        torch.nn.init.xavier_normal_(self.protos)
        # normal_init(self.protos, std=self.init_std)
    
        if read_op == 'sum':
            self.gread = global_add_pool
        elif read_op == 'mean':
            self.gread = global_mean_pool 
        elif read_op == 'max':
            self.gread = global_max_pool 
        elif read_op == 'attention':
            self.gread = GlobalAttention(gate_nn = torch.nn.Linear(in_channels, 1))
        elif read_op == 'set2set':
            self.gread = Set2Set(in_channels, processing_steps = 2) 
        else:
            raise ValueError("Invalid graph readout type.")

        # self.in_c = in_channels
        # self.rep_dim = in_channels*num_position
        # self.fc_cls = nn.Linear(self.rep_dim, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.protos, std=self.init_std)


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

        assert len(x.shape) == 5
        N, M, C, T, V = x.shape
        # AlineLoss = self.get_aligncost(x)
        x = x.mean(1)

        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        batch = torch.zeros(x.shape[0]).type(torch.int64).to(x.device)
        for i in range(N):
            batch[i*V*T:(i+1)*V*T] =i
        size = N
        A = self.get_alignment(x)
        
        sbatch = self.num_position * batch + torch.max(A, dim=1)[1]
        ssize = self.num_position * (batch.max().item() + 1)
        x = self.gread(x, sbatch, size=ssize)
        x = x.reshape(N, self.num_position, -1)
        # x = x.mean(dim=1)
        x = x.mean(dim=1)
        x = x.reshape(N, -1)

        return x

    def init_protos(self, protos):
        self.protos.data = protos

    def get_alignment(self, x):
        # D = self._compute_distance_matrix(x, self.protos)
        D = 1 - torch.cosine_similarity(x.unsqueeze(1), self.protos.unsqueeze(0), dim=2)
        A = torch.zeros_like(D).scatter_(1, torch.argmin(D, dim=1, keepdim=True), 1.)
        return A 

    def get_aligncost(self, x):
        N, M, C, T, V = x.shape
        # AlineLoss = self.get_aligncost(x)
        x = x.mean(1)

        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        batch = torch.zeros(x.shape[0]).type(torch.int64).to(x.device)
        for i in range(N):
            batch[i*V*T:(i+1)*V*T] =i

        # D = self._compute_distance_matrix(x, self.protos)
        D = 1 - torch.cosine_similarity(x.unsqueeze(1), self.protos.unsqueeze(0), dim=2)
        A = torch.zeros_like(D).scatter_(1, torch.argmin(D, dim=1, keepdim=True), 1.)
        # D = 1 - self.cos(x, self.protos)
        if self.gamma == 0:
            D = torch.min(D, dim=1)[0]
        else:
            D = -self.gamma * torch.log(torch.sum(torch.exp(-D/self.gamma), dim=1) + 1e-12)

        
        N = scatter_add(A, batch, dim=0)
        D = D.unsqueeze(1)*A
        D_loss = scatter_add(D, batch, dim=0)
        return torch.mean(D_loss/(N + 1e-12))
        # N = torch.zeros(D.shape[0], batch.max().item() + 1, device=D.device).scatter(1,)
        # N = torch.zeros(D.shape[0], batch.max().item() + 1, device=D.device).scatter_(1, batch.unsqueeze(dim=1), 1.)
        # N /= N.sum(dim=0, keepdim=True)
        # return torch.mean(D / N.sum(dim=1), dim=0)
        

    def _compute_distance_matrix(self, h, p):
        h_ = torch.pow(torch.pow(h, 2).sum(1, keepdim=True), 0.5)
        p_ = torch.pow(torch.pow(p, 2).sum(1, keepdim=True), 0.5)
        hp_ = torch.matmul(h_, p_.transpose(0, 1))
        hp = torch.matmul(h, p.transpose(0, 1))
        return 1 - hp / (hp_ + 1e-12)
