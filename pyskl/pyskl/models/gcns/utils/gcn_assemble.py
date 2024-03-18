from ast import IsNot
from locale import ABDAY_1
from math import ceil
from shutil import ExecError
from ssl import ALERT_DESCRIPTION_CERTIFICATE_REVOKED
from tkinter import N
from turtle import screensize
from xmlrpc.client import Fault
from xmlrpc.server import resolve_dotted_attribute
from matplotlib.pyplot import axes, axis
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer

from .init_func import bn_init, conv_branch_init, conv_init

from .sparse_mosules import SparseConv2d, SparseConv1d
from .gcn_sparse import unit_aagcn_sparse, unit_ctrgcn_sparse, dggcn_sparse
from .tcn_sparse import unit_tcn_sparse,mstcn_sparse,dgmstcn_sparse

EPS = 1e-4


class Assemble_model(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_list,
                 A,
                 sparse_ratio,
                #  set=[0,2,4,6,8],
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels =in_channels
        self.out_channels = out_channels
        



class dggcn_sparse_assemble(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model_list,
                 A,
                 sparse_ratio,
                #  set=[0,2,4,6,8],
                 ratio=0.25,
                 ctr='T',
                 ada='T',
                 subset_wise=False,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels
        self.convs = nn.ModuleList()
        for i, sparse in enumerate(sparse_ratio):
            self.convs.append(dggcn_sparse_unit(in_channels, mid_channels, A[set[i]:set[i+1]-1], sparse))
        
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
    def forward(self, x):
        for i in range(self.num_subset):
            
            z = self.convs[i](x, A=None,sparse=0)
            y = z + y if y is not None else z
    
