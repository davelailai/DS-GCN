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

EPS = 1e-4


class unit_gcn_sparse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='init',
                 conv_pos='pre',
                 with_res=False,
                 sparse_ratio=0,
                 norm='BN',
                 act='ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.act = build_activation_layer(self.act_cfg)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = SparseConv2d(in_channels, out_channels * A.size(0), 1, sparse_ratio)
        elif self.conv_pos == 'post':
            self.conv = SparseConv2d(A.size(0) * in_channels, out_channels, 1, sparse_ratio)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, sparse=0, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x,sparse) if self.with_res else 0
        if A is not None:
            self.A = A
        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x,sparse)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x,sparse)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass

class unit_aagcn_sparse(nn.Module):
    def __init__(self, in_channels, out_channels, A, sparse_ratio=0, coff_embedding=4, adaptive=True, attention=True):
        super(unit_aagcn_sparse, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention

        num_joints = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(SparseConv2d(in_channels, out_channels , 1, sparse_ratio))
            # self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(SparseConv2d(in_channels, inter_channels, 1, sparse_ratio))
                self.conv_b.append(SparseConv2d(in_channels, inter_channels, 1, sparse_ratio))
                # self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                # self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.register_buffer('A', A)

        if self.attention:
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            # s attention
            ker_joint = num_joints if num_joints % 2 else num_joints - 1
            pad = (ker_joint - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_joint, padding=pad)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.down = lambda x: x
        if in_channels != out_channels:
            self.down = nn.Sequential(
                SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         bn_init(m, 1)
        # bn_init(self.bn, 1e-6)
        # for i in range(self.num_subset):
        #     conv_branch_init(self.conv_d[i], self.num_subset)

        # if self.attention:
        #     nn.init.constant_(self.conv_ta.weight, 0)
        #     nn.init.constant_(self.conv_ta.bias, 0)

        #     nn.init.xavier_normal_(self.conv_sa.weight)
        #     nn.init.constant_(self.conv_sa.bias, 0)

        #     nn.init.kaiming_normal_(self.fc1c.weight)
        #     nn.init.constant_(self.fc1c.bias, 0)
        #     nn.init.constant_(self.fc2c.weight, 0)
        #     nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x,sparse=0):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x,sparse).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x,sparse).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V),sparse)
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V),sparse)
                y = z + y if y is not None else z

        res = x
        try:
            len(self.down)
        except:
            try:
                index = self.down(res,sparse)
            except:
                res = self.down(res)
            else:
                res = self.down(res,sparse)
        else:
            for item in self.down:
                try:
                    index = item(res,sparse)
                except:
                    res = item(res)
                else:
                    res = item(res,sparse)

        y = self.relu(self.bn(y) + res)

        if self.attention:
            # spatial attention first
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))  # N 1 V
            y = y * se1.unsqueeze(-2) + y
            # then temporal attention
            se = y.mean(-1)  # N C T
            se1 = self.sigmoid(self.conv_ta(se))  # N 1 T
            y = y * se1.unsqueeze(-1) + y
            # then spatial temporal attention ??
            se = y.mean(-1).mean(-1)  # N C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))  # N C
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # A little bit weird
        return y


class CTRGC_sparse(nn.Module):
    def __init__(self, in_channels, out_channels, sparse_ratio=0, rel_reduction=8):
        super(CTRGC_sparse, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = SparseConv2d(self.in_channels, self.rel_channels, 1, sparse_ratio)
        self.conv2 = SparseConv2d(self.in_channels, self.rel_channels, 1, sparse_ratio)
        self.conv3 = SparseConv2d(self.in_channels, self.out_channels, 1, sparse_ratio)
        self.conv4 = SparseConv2d(self.rel_channels, self.out_channels, 1, sparse_ratio)
        # self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        # self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, threhold=0, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x,threhold).mean(-2), self.conv2(x,threhold).mean(-2), self.conv3(x,threhold)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1,threhold) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         conv_init(m)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         bn_init(m, 1)

class unit_ctrgcn_sparse(nn.Module):
    def __init__(self, in_channels, out_channels, A,sparse_ratio=0):

        super(unit_ctrgcn_sparse, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC_sparse(in_channels, out_channels,sparse_ratio))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, sparse=0):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, sparse, self.A[i], self.alpha)
            y = z + y if y is not None else z
        
        res = x
        try:
            len(self.down)
        except:
            try:
                index = self.down(res,sparse)
            except:
                res = self.down(res)
            else:
                res = self.down(res,sparse)
        else:
            for item in self.down:
                try:
                    index = item(res,sparse)
                except:
                    res = item(res)
                else:
                    res = item(res,sparse)
        y +=res
        return self.relu(y)

    def init_weights(self):
        pass


class unit_sgn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, A):
        # x: N, C, T, V; A: N, T, V, V
        x1 = x.permute(0, 2, 3, 1).contiguous()
        x1 = A.matmul(x1).permute(0, 3, 1, 2).contiguous()
        return self.relu(self.bn(self.conv(x1) + self.residual(x)))


class dggcn_sparse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 sparse_ratio = 0,
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
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())

        # Introduce non-linear
        self.pre = nn.Sequential(
            SparseConv2d(in_channels, mid_channels * num_subsets, 1, sparse_ratio),
            # nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        # self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)
        self.post = SparseConv2d(mid_channels * num_subsets, out_channels, 1, sparse_ratio)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = SparseConv2d(in_channels, mid_channels * num_subsets, 1, sparse_ratio)
            self.conv2 = SparseConv2d(in_channels, mid_channels * num_subsets, 1, sparse_ratio)
            # self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            # self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None,sparse=0):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        # res = self.down(x)
        res = x
        try:
            len(self.down)
        except:
            try:
                index = self.down(res,sparse)
            except:
                res = self.down(res)
            else:
                res = self.down(res,sparse)
        else:
            for item in self.down:
                try:
                    index = item(res,sparse)
                except:
                    res = item(res)
                else:
                    res = item(res,sparse)

        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        pre_x = x
        for item in self.pre:
            try:
                index = item(pre_x,sparse)
            except:
                pre_x= item(pre_x)
            else:
                pre_x = item(pre_x,sparse)
        pre_x = pre_x.reshape(n, self.num_subsets, self.mid_channels, t, v)
        # pre_x = self.pre(x,sparse).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x,sparse).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x,sparse).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            ada_graph = getattr(self, self.ada_act)(ada_graph)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()

        x = x.reshape(n, -1, t, v)
        x = self.post(x,sparse)

        return self.act(self.bn(x) + res)


class dgphgcn1_sparse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 ratio=0.25,
                 sparse_ratio=0,
                 decompose = False,
                 ctr='T',
                 ada='T',  
                 node_attention = False,
                 edge_attention = False, 
                 ada_attention = False,
                 target_specific = False,
                 add_type = False,
                 sub_att = True,
                 stage = True,
                 num_types=5,    
                 edge_num=15,
                 subset_wise=True,
                 ada_act='softmax',
                 ctr_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ctr = ctr
        self.ada = ada
        self.ada_act = ada_act
        self.ctr_act = ctr_act
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.target_specific = target_specific
        self.ada_attention = ada_attention
        self.num_types = num_types
        self.edge_num = edge_num
        self.edge_type = edge_type
        self.node_type = node_type
        self.add_type = add_type
        self.decompose = decompose
        self.subset_wise = subset_wise
        self.sub_att = sub_att
        if stage == False:
            self.node_attention = False
            self.edge_attention = False
            self.target_specific = False
            self.decompose = decompose = False
            self.subset_wise = False
        assert ada_act in ['tanh', 'relu', 'sigmoid', 'softmax']
        assert ctr_act in ['tanh', 'relu', 'sigmoid', 'softmax']

        # self.subset_wise = subset_wise

        assert self.ctr in [None, 'NA', 'T']
        assert self.ada in [None, 'NA', 'T']

        if ratio is None:
            ratio = 1 / self.num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = build_activation_layer(self.act_cfg)

        self.A = nn.Parameter(A.clone())
        
        if decompose:
            self.semantic_num = ceil(self.num_subsets/3)
            self.norm_num = self.num_subsets - self.semantic_num
        else:
            self.semantic_num = 0
            self.norm_num = self.num_subsets


        # Introduce non-linear
        if self.target_specific & self.decompose:
            self.nodeconv= nn.Sequential(
                SparseConv2d(in_channels, mid_channels * num_subsets, 1, sparse_ratio),
                # nn.Conv2d(in_channels, self.semantic_num*num_types*mid_channels, 1),
                build_norm_layer(self.norm_cfg, self.semantic_num*num_types*mid_channels)[1], self.act)
            self.pre = nn.Sequential(
                SparseConv2d(in_channels, self.norm_num * mid_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, self.norm_num * mid_channels, 1),
                build_norm_layer(self.norm_cfg, mid_channels * self.norm_num)[1], self.act)
        else:
            self.pre = nn.Sequential(
                SparseConv2d(in_channels, mid_channels * num_subsets, 1, sparse_ratio),
                # nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
                build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)

        self.post = SparseConv2d(mid_channels * num_subsets, out_channels, 1, sparse_ratio)
        # self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        if sub_att:
            self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
            self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        else:
            self.alpha = nn.Parameter(torch.zeros(3))
            self.beta = nn.Parameter(torch.zeros(3))
        
            
      
        if self.ada or self.ctr:
            if self.decompose!=0:
                if self.node_attention:
                    self.conv1_se  = SparseConv2d(in_channels, self.semantic_num*mid_channels*num_types, 1, sparse_ratio)
                    self.conv2_se  = SparseConv2d(in_channels, self.semantic_num*mid_channels*num_types, 1, sparse_ratio)
                    # self.conv1_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                    # self.conv2_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                else:
                    self.conv1_se  = SparseConv2d(in_channels, self.semantic_num*mid_channels, 1, sparse_ratio)
                    self.conv2_se  = SparseConv2d(in_channels, self.semantic_num*mid_channels, 1, sparse_ratio)
                    # self.conv1_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels, kernel_size=1)
                    # self.conv2_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels, kernel_size=1)

            self.conv1 = SparseConv2d(in_channels, self.norm_num*mid_channels, 1, sparse_ratio)
            self.conv2 = SparseConv2d(in_channels, self.norm_num*mid_channels, 1, sparse_ratio)
            # self.conv1 = nn.Conv2d(in_channels, self.norm_num*mid_channels, 1)
            # self.conv2 = nn.Conv2d(in_channels, self.norm_num*mid_channels, 1)
        
        if self.edge_attention:
            if self.decompose!=0:
                self.edge_linears = nn.Sequential(
                    SparseConv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels, 1, sparse_ratio),
                    # nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1),
                    # build_norm_layer(self.norm_cfg, edge_num*self.semantic_num*mid_channels)[1], self.act
                    )
                # self.edge_linears = nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1)

        
        if self.ada_attention:
            self.ada_linears =SparseConv2d(num_subsets,edge_num*num_subsets, 1, sparse_ratio),
            # self.ada_linears =nn.Conv2d(num_subsets,edge_num*num_subsets,1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                SparseConv2d(in_channels, out_channels, 1, sparse_ratio),
                # nn.Conv2d(in_channels, out_channels, 1),
                build_norm_layer(self.norm_cfg, out_channels)[1])
        else:
            self.down = lambda x: x
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape

        res = self.down(x)
        A = self.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        

        if self.target_specific & self.decompose:
            # x_node = pre_x_ori[:,self.semantic_num:self.norm_num,:].reshape(n, -1, t, v)
            x_node = self.nodeconv(x)
            x_node = x_node.view(n, self.semantic_num,self.num_types, self.mid_channels, t, v)
            x_node = torch.diagonal(x_node[:,:,self.node_type,:,:,:],dim1=2,dim2=-1) 
            x_norm = self.pre(x).reshape(n, self.norm_num, self.mid_channels, t, v)
            pre_x = torch.cat((x_node,x_norm),axis=1)
        else:
            pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)

        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)
            
            x1_norm = self.conv1(tmp_x).reshape(n, self.norm_num, self.mid_channels, -1, v)
            x2_norm = self.conv2(tmp_x).reshape(n, self.norm_num, self.mid_channels, -1, v)
            x1_sem = None
            x2_sem = None
            if self.decompose:
                x1_sem = self.conv1_se(tmp_x)
                x2_sem = self.conv1_se(tmp_x)
                if self.node_attention:   
                    x1_sem = x1_sem.view(n, self.semantic_num, self.mid_channels, self.num_types, -1, v)
                    x2_sem = x2_sem.view(n, self.semantic_num, self.mid_channels, self.num_types, -1, v)
                    x1_sem = torch.diagonal(x1_sem[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                    x2_sem = torch.diagonal(x2_sem[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                else:
                    x1_sem = x1_sem.reshape(n, self.semantic_num, self.mid_channels, -1, v)
                    x2_sem = x2_sem.reshape(n, self.semantic_num, self.mid_channels, -1, v)

            # else:
            #     x1 = x1.reshape(n, self.num_subsets, self.mid_channels, -1, v)
            #     x2 = x2.reshape(n, self.num_subsets, self.mid_channels, -1, v)
            if x1_sem == None:
                x1 = x1_norm
                x2 = x2_norm
            else:
                x1 = torch.cat((x1_norm,x1_sem),axis=1)
                x2 = torch.cat((x2_norm,x1_sem),axis=1)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            if self.decompose:
                if self.edge_attention:
                    # diff_sem = x1[:,self.semantic_num:self.norm_num,:].unsqueeze(-1) - x2[:,self.semantic_num:self.norm_num,:].unsqueeze(-2)
                    diff_sem = x1[:,self.norm_num-self.semantic_num:self.norm_num,:].unsqueeze(-1) - x2[:,self.norm_num-self.semantic_num:self.norm_num,:].unsqueeze(-2)
                    edge_speci = self.edge_linears(diff_sem.view(n,-1, v,v)).view(n,self.semantic_num,self.edge_num,self.mid_channels,v,v)
                    edge_select= self.edge_type.int().reshape(-1)
                    select = torch.zeros(len(edge_select),dtype=int)
                    for i in range(len(edge_select)):
                            select[i] = self.edge_num*i + edge_select[i]        
                    edge_speci = edge_speci.permute(0,1,3,4,5,2).reshape(n,self.semantic_num,self.mid_channels,self.edge_num*v*v)
                    edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
                    edge_att = edge_att.reshape(n, self.semantic_num, self.mid_channels,v,v)
                    ada_graph = edge_att.unsqueeze(-3)
                else:
                    ada_graph = x1[:,self.semantic_num:self.norm_num,:].unsqueeze(-1) - x2[:,self.semantic_num:self.norm_num,:].unsqueeze(-2)

                diff_norm = x1[:,0:self.norm_num-self.semantic_num,:].unsqueeze(-1) - x2[:,0:self.norm_num-self.semantic_num,:].unsqueeze(-2)
                diff_node = x1[:,self.norm_num:,:].unsqueeze(-1) - x2[:,self.norm_num:,:].unsqueeze(-2)
                ada_graph = torch.cat((diff_norm,ada_graph,diff_node),axis=1)
            else:
                ada_graph = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        
            ada_graph = getattr(self, self.ctr_act)(ada_graph)
            
            # ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                if self.num_subsets == len(self.alpha):
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
                else:
                    alpha = torch.repeat_interleave(self.alpha, ceil(self.num_subsets/3)) 
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, alpha[2*self.semantic_num-self.norm_num:])
            else:
                ada_graph = ada_graph * self.alpha[0]
            A = ada_graph + A

        if self.ada is not None:
            # * The shape of ada_graph is N, K, 1, T[1], V, V
            ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
            if self.ada_attention:
                ada_speci = self.ada_linears(ada_graph.squeeze()).view(n,self.num_subsets,self.edge_num,-1,v,v)
                edge_select= self.edge_type.int().reshape(-1)
                select = torch.zeros(len(edge_select),dtype=int)
                for i in range(len(edge_select)):
                        select[i] = self.edge_num*i + edge_select[i]    
                ada_speci = ada_speci.permute(0,1,3,4,5,2).reshape(n,self.num_subsets,-1,self.edge_num*v*v)
                ada_graph = torch.index_select(ada_speci, -1, select.to(ada_speci.device))
                ada_graph = ada_graph.reshape(n, self.num_subsets, -1,v,v)
                ada_graph = ada_graph.unsqueeze(-3)

            ada_graph = getattr(self, self.ada_act)(ada_graph)

            if self.subset_wise:
                if self.num_subsets == len(self.beta):
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
                else:
                    beta = torch.repeat_interleave(self.beta, ceil(self.num_subsets/3)) 
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, beta[2*self.semantic_num-self.norm_num:])
                # ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
            else:
                ada_graph = ada_graph * self.beta[0]
            A = ada_graph + A

       

        if self.ctr is not None or self.ada is not None:
            assert len(A.shape) == 6
            # * C, T can be 1
            if A.shape[2] == 1 and A.shape[3] == 1:
                A = A.squeeze(2).squeeze(2)
                x = torch.einsum('nkctv,nkvw->nkctw', pre_x, A).contiguous()
            elif A.shape[2] == 1:
                A = A.squeeze(2)
                x = torch.einsum('nkctv,nktvw->nkctw', pre_x, A).contiguous()
            elif A.shape[3] == 1:
                A = A.squeeze(3)
                x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
            else:
                x = torch.einsum('nkctv,nkctvw->nkctw', pre_x, A).contiguous()
        else:
            # * The graph shape is K, V, V
            A = A.squeeze()
            assert len(A.shape) in [2, 3] and A.shape[-2] == A.shape[-1]
            if len(A.shape) == 2:
                A = A[None]
            x = torch.einsum('nkctv,kvw->nkctw', pre_x, A).contiguous()
        
        x = x.reshape(n, -1, t, v) 
        x = self.post(x)    
        return self.act(self.bn(x) + res)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)