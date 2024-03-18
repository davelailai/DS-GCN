from ast import IsNot
from locale import ABDAY_1
from math import ceil
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

# from .sparse_mosules import SparseConv2d, SparseConv1d

EPS = 1e-4


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='init',
                 conv_pos='pre',
                 with_res=False,
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
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        if A is not None:
            self.A = A
        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass

# class unit_gcn(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  A,
#                  adaptive='importance',
#                  conv_pos='pre',
#                  with_res=False,
#                  norm='BN',
#                  act='ReLU'):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_subsets = A.size(0)

#         assert adaptive in [None, 'init', 'offset', 'importance', 'causal', 'GC']
#         self.adaptive = adaptive
#         assert conv_pos in ['pre', 'post']
#         self.conv_pos = conv_pos
#         self.with_res = with_res

#         self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
#         self.act_cfg = act if isinstance(act, dict) else dict(type=act)
#         self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
#         self.act = build_activation_layer(self.act_cfg)

#         if self.adaptive == 'init':
#             self.A = nn.Parameter(A.clone())
#         else:
#             self.register_buffer('A', A)

#         if self.adaptive in ['offset', 'importance']:
#             self.PA = nn.Parameter(A.clone())
#             if self.adaptive == 'offset':
#                 nn.init.uniform_(self.PA, -1e-6, 1e-6)
#             elif self.adaptive == 'importance':
#                 nn.init.constant_(self.PA, 1)
#         if self.adaptive in ['causal']:
#             self.PA = nn.Parameter(A.clone())
#             nn.init.uniform_(self.PA, -1e-6, 1e-6)

#         if self.conv_pos == 'pre':
#             self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
#         elif self.conv_pos == 'post':
#             self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)
#         if self.adaptive =='GC':
#             if self.conv_pos == 'pre':
#                 self.conv = nn.Conv2d(in_channels, out_channels * (A.size(0)+1), 1)
#             elif self.conv_pos == 'post':
#                 self.conv = nn.Conv2d((A.size(0)+1) * in_channels, out_channels, 1)

#         if self.with_res:
#             if in_channels != out_channels:
#                 self.down = nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels, 1),
#                     build_norm_layer(self.norm_cfg, out_channels)[1])
#             else:
#                 self.down = lambda x: x

#     def forward(self, x, A=None):
#         """Defines the computation performed at every call."""
#         n, c, t, v = x.shape
#         res = self.down(x) if self.with_res else 0
#         if A is None:
#             A_switch = {None: self.A, 'init': self.A}
#             if hasattr(self, 'PA'):
#                 A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA, 'causal':self.A + self.PA})
#             A = A_switch[self.adaptive]
#         else:
#             A = A.unsqueeze(1).float()
#             B = A.size(0)
#             A_link = self.A.unsqueeze(0).repeat(B,1,1,1)
#             A = torch.cat((A,A_link),dim=1)
#             if B != n:
#                 index = A.repeat(int(n/B),1,1,1)
#                 index[0:-1:2]=A
#                 index[1::2]=A
#                 A = index
#         if self.adaptive == 'causal':
#             predic = torch.einsum('nctv,kvw->nkctw', (x, self.PA)).contiguous()
#             predic = predic.sum(dim=1)[:,:,0:-1,:].mean(dim=2)
#             original = x[:,:,-1,:].squeeze()
#             noise = torch.norm(original-predic,2,dim=1).mean()
#             # causal = torch.norm(self.PA, 2)
#         if self.conv_pos == 'pre':
#             x = self.conv(x)    
#             if self.adaptive == 'GC':
#                 x = x.view(n, self.num_subsets + 1, -1, t, v)
#                 x = self.GC_graph_pre(x, A)
#                 # print(x.shape)
#                 # for i in range(self.num_subsets + 1):
#                 #     y = torch.einsum('nctv,nvw->nctw', x[:,i,:], A[:,i,:]).contiguous()
#                 #     x = torch.einsum('nkctv,nkvw->nctw', (x, A)).contiguous()
#             else:
#                 x = x.view(n, self.num_subsets, -1, t, v)
#                 x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
#         elif self.conv_pos == 'post':
#             if self.adaptive == 'GC':
#                 x = self.GC_graph_pro(x, A)
#                 # x = torch.einsum('nctv,nkvw->nkctw', (x, A)).contiguous()
#             else:
#                 x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
#             x = x.view(n, -1, t, v)
#             x = self.conv(x)
#         if self.adaptive == 'causal':
#             return self.act(self.bn(x) + res), noise
#         else:
#             return self.act(self.bn(x) + res)
            
#     def init_weights(self):
#         pass

#     def GC_graph_pre(self, x, A):
#         y = None
#         for i in range(self.num_subsets + 1):
#             x1 = torch.einsum('nctv,nvw->nctw', x[:,i,:], A[:,i,:]).contiguous()
#             y = x1 + y if y is not None else x1
#         return x1
#     def GC_graph_pro(self, x, A):
#         y = None
#         for i in range(self.num_subsets + 1):
#             x1 = torch.einsum('nctv,nvw->nctw', x, A[:,i,:]).contiguous()
#             y = x1 + y if y is not None else x1
#         return x1



class unit_gcnedge(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='importance',
                 conv_pos='pre',
                 with_res=False,
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
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
            self.edge_conv = nn.Conv2d(in_channels*3, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)
            self.edge_conv = nn.Conv2d(A.size(0) * in_channels*3, out_channels * A.size(0), 1)

        self.edge_conv_T = nn.Conv2d(in_channels, out_channels,1)
        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x

    def forward(self, x, edge_rep, edge_rep_T,  A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        node_rep = x
        # edge_rep = x.unsqueeze(-1).repeat(1,1,1,1,v)
        # edge_rep = edge_rep - edge_rep.permute(0,1,2,4,3)
        # edge_rep = 

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]
        # node_with_edge = self.node_update(node_rep, edge_rep, A) # node representation
        edge_with_node = self.node_edge_node(node_rep, edge_rep) # edge representation
        if self.conv_pos == 'pre':

            node_rep = self.conv(node_rep) 
            edge_rep = self.edge_conv(edge_with_node) 

            node_rep = node_rep.view(n, self.num_subsets, -1, t, v)
            edge_rep = edge_rep.view(n, self.num_subsets, -1, t, v, v)
            node_only = torch.einsum('nkctv,kvw->nctw', (node_rep, A)).contiguous()
            edge_only = edge_rep * A.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            edge_only = edge_only.sum(1).sum(-1)    
            x = node_only + edge_only
            
            # edge_rep = self.node_edge_node(node_rep, edge_rep).sum(1)

        elif self.conv_pos == 'post':
            node_only = torch.einsum('nctv,kvw->nkctw', (node_rep, A)).contiguous()
            edge_only = edge_rep.unsqueeze(1)* A.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            edge_only = edge_rep * A.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            x = node_only + edge_only
            x = x.view(n, -1, t, v)
            x = self.conv(x)

            edge_rep = self.node_edge_node(node_rep, edge_rep)
            edge_rep = self.edge_conv(edge_rep)

        edge_rep = edge_rep.sum(1).squeeze(1).reshape(n,self.out_channels,t,-1)

        if edge_rep_T.size(1)!=self.out_channels:
            edge_rep_T = self.edge_conv_T(edge_rep_T)
        return self.act(self.bn(x) + res), self.act(self.bn(edge_rep)), self.act(self.bn(edge_rep_T))

    def init_weights(self):
        pass
    
    def node_edge_node(self, node_rep, edge_rep):
        B, C, T, V = node_rep.shape
        edge_rep = edge_rep.reshape(B,C,T,V,V)
        index1 = node_rep.unsqueeze(-1).repeat(1,1,1,1,node_rep.size(-1))
        node_cat = torch.cat((index1,index1.permute(0,1,2,4,3)),axis=1)
        edge_cat = torch.cat((node_cat, edge_rep),axis=1)
        return edge_cat.reshape(B, 3*C, T,-1)

    def node_update(self, node_rep, edge_rep, A):
        edge_only = edge_rep.unsqueeze(1) * A.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        edge_only = edge_only.sum(1).sum(-1)
        node_rep = node_rep + edge_only
        return node_rep




class unit_aagcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, attention=True):
        super(unit_aagcn, self).__init__()
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
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.num_subset):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
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
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        if self.attention:
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = self.A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

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

class unit_aahgcn(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels,
                A, 
                edge_type, 
                node_type,
                node_att=False, 
                edge_att=False, 
                num_types=5, 
                edge_num=15,
                coff_embedding=4, 
                adaptive=True, 
                attention=True):
        super(unit_aahgcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention
        self.node_att = node_att
        self.edge_att = edge_att
        self.node_num = num_types
        self.edge_num = edge_num
        self.node_type = node_type
        self.edge_type = edge_type

        num_joints = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if self.adaptive:
            self.A = nn.Parameter(A)

            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            if self.node_att:
                for i in range(self.num_subset):
                    self.conv_a.append(nn.Conv2d(in_channels, inter_channels*num_types, 1))
                    self.conv_b.append(nn.Conv2d(in_channels, inter_channels*num_types, 1))
            else:
                for i in range(self.num_subset):
                    self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                    self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            if self.edge_att:
                self.conv_edge = nn.ModuleList()
                for i in range(self.num_subset):
                    self.conv_edge.append(nn.Conv2d(1, edge_num, 1))
            
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
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

        if self.attention:
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)

            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)

            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            for i in range(self.num_subset):
                if self.node_att:
                    # A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                    # A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                    A1 = self.conv_a[i](x)
                    A2 = self.conv_b[i](x)
                    A1 = A1.view(N, self.inter_c, self.node_num, T, V) #B  C node_sube T,V
                    A2 = A2.view(N, self.inter_c, self.node_num, T, V) #B  C node_sube T,V
                    A1 = torch.diagonal(A1[:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                    A2 = torch.diagonal(A2[:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                    A1 = A1.permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                    A2 = A2.reshape(N, self.inter_c * T, V)
                    A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                    if self.edge_att:
                        A1 = A1.unsqueeze(1)
                        edge_speci = self.conv_edge[i](A1).view(N, self.edge_num, -1,V,V)
                        # index = self.edge_att_conv(diff).view(N, self.edge_num, -1,V,V)
                        edge_select= self.edge_type.int().reshape(-1)
                        select = torch.zeros(len(edge_select),dtype=int)
                        for j in range(len(edge_select)):
                                select[j] = self.edge_num*j + edge_select[j]
                        edge_speci = edge_speci.permute(0,2,3,4,1).reshape(N,-1,self.edge_num*V*V)
                        edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
                        edge_att = edge_att.reshape(N, -1 ,V,V) 
                        A1= edge_att.squeeze()

                    A1 = self.A[i] + A1 * self.alpha

 

                else:
                    A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                    A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                    A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                    A1 = self.A[i] + A1 * self.alpha

                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            for i in range(self.num_subset):
                A1 = self.A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.relu(self.bn(y) + self.down(x))

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

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # N, R, V, V
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

class CTRHGC(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                rel_reduction=8, 
                node_attention = True,
                edge_attention = False, 
                target_specific = False,
                full_channels = False,
                add_type = False,
                ada = False,
                num_types=5, 
                edge_num=15,
                semantic_index = False):
        super(CTRHGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.target_specific = target_specific
        self.full_channels = full_channels
        self.num_types = num_types
        self.edge_num = edge_num
        self.add_type = add_type
        self.ada = ada
        self.semantic_index = semantic_index
        if self.ada:
            self.beta = nn.Parameter(torch.zeros(1))
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        if self.node_attention & self.semantic_index:
            self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels*num_types, kernel_size=1)
            self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels*num_types, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
            self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)

        if self.edge_attention & self.semantic_index:
            if self.full_channels:
                self.edge_att_conv = nn.Conv2d(self.rel_channels, edge_num*self.out_channels,1)
            else:
                self.edge_att_conv = nn.Conv2d(self.rel_channels, edge_num*self.rel_channels,1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        
        if self.target_specific & self.semantic_index:
            self.nodeconv = nn.Conv2d(self.in_channels, num_types*self.out_channels, kernel_size=1)
        # else:
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        self.tanh = nn.Tanh()
        self.init_weights()

    def forward(self, x, node_type, edge_type, A=None, alpha=1):
        # Input: N, C, T, V
        N,C,T,V = x.size()
        x1, x2, x3 = self.conv1(x), self.conv2(x), self.conv3(x)
        if self.node_attention & self.semantic_index:
            x1 = x1.view(N, self.rel_channels, self.num_types, T, V) #B  C node_sube T,V
            x2 = x2.view(N, self.rel_channels, self.num_types, T, V) #B  C node_sube T,V
            x1 = torch.diagonal(x1[:,:,node_type,:,:],dim1=-3,dim2=-1).mean(-2)
            x2 = torch.diagonal(x2[:,:,node_type,:,:],dim1=-3,dim2=-1).mean(-2)
        else:
            x1 = x1.mean(-2)
            x2 = x2.mean(-2)

        diff = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        if self.edge_attention & self.semantic_index:
            edge_speci = self.edge_att_conv(diff).view(N, self.edge_num, -1,V,V)
            # index = self.edge_att_conv(diff).view(N, self.edge_num, -1,V,V)
            edge_select= edge_type.int().reshape(-1)
            select = torch.zeros(len(edge_select),dtype=int)
            for i in range(len(edge_select)):
                    select[i] = self.edge_num*i + edge_select[i]
            edge_speci = edge_speci.permute(0,2,3,4,1).reshape(N,-1,self.edge_num*V*V)
            edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
            edge_att = edge_att.reshape(N, -1 ,V,V)  
            if not self.full_channels:
                edge_att = self.conv4(edge_att) 
            if self.add_type:
                edge_att = edge_att + self.conv4(diff)
        else:
            edge_att = self.conv4(diff)

        
        
        A = (edge_att * alpha + (A[None,None] if A is not None else 0))

        if self.ada:

            ada_graph = torch.einsum('ncv,ncw->nvw', x1, x2)[:, None]
            A = ada_graph * self.beta + A

        if self.target_specific & self.semantic_index:
            x_node = self.nodeconv(x)
            x_node = x_node.view(N, self.num_types, self.out_channels, T, V)
            x_node = torch.diagonal(x_node[:,node_type,:,:,:],dim1=1,dim2=-1)  
            x3 = x3 + x_node  
        # N, R, V, V
        x1 = torch.einsum('ncuv,nctu->nctv', A, x3)
        return x1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

class unit_ctrhgcn(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                A,
                edge_type, 
                node_type, 
                semantic_index = False,
                rel_reduction = 8,
                node_attention = False,
                edge_attention = False, 
                target_specific = False,
                full_channels = False,
                add_type = False,
                ada = False,
                num_types=5,    
                edge_num=15):

        super(unit_ctrhgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.edge_type = edge_type
        self.node_type = node_type

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            if i==0:
                node_attention = False
                self.convs.append(CTRHGC(in_channels, out_channels,
                                        rel_reduction=rel_reduction, 
                                        node_attention=node_attention,
                                        edge_attention=edge_attention, 
                                        target_specific=target_specific,
                                        full_channels=full_channels,
                                        add_type=add_type,
                                        ada = ada,
                                        num_types=num_types, 
                                        edge_num=edge_num,
                                        semantic_index=semantic_index))
            if i==1:
                edge_attention = False
                self.convs.append(CTRHGC(in_channels, out_channels,
                                        rel_reduction=rel_reduction, 
                                        node_attention=node_attention,
                                        edge_attention=edge_attention, 
                                        target_specific=target_specific,
                                        full_channels=full_channels,
                                        add_type=add_type,
                                        ada = ada,
                                        num_types=num_types, 
                                        edge_num=edge_num,
                                        semantic_index=semantic_index))
            if i==2:
                edge_attention = False
                node_attention = False
                self.convs.append(CTRHGC(in_channels, out_channels,
                                        rel_reduction=rel_reduction, 
                                        node_attention=node_attention,
                                        edge_attention=edge_attention, 
                                        target_specific=target_specific,
                                        full_channels=full_channels,
                                        add_type=add_type,
                                        ada = ada,
                                        num_types=num_types, 
                                        edge_num=edge_num,
                                        semantic_index=semantic_index))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(self.A.size(0)))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        for i in range(self.num_subset):
            z = self.convs[i](x, self.node_type, self.edge_type, self.A[i], self.alpha[i])
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

class unit_ctrgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A):

        super(unit_ctrgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None

        for i in range(self.num_subset):
            z = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)


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


class GTNGC(nn.Module):   
    def __init__(self, 
                in_dim, 
                out_dim, 
                A,  
                edge_attention = False, 
                adaptive='importance',
                num_types=5, 
                reduce=8, 
                edge_num=15,
                *kwargs):
        super().__init__()
        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        if self.adaptive == 'init':
            self.A = nn.Parameter(A[0].clone())
        else:
            self.register_buffer('A', A[0])

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A[0].clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        self.dim = in_dim
        self.out_channels = out_dim
        self.num_types = num_types
        self.num_node = A.size(1)

        # self.PA = nn.Parameter(torch.FloatTensor(A.size(1), A.size(1)))
        # torch.nn.init.uniform_(self.PA, -1e-6, 1e-6)
        if in_dim <= 16:
            self.inter_channels = 8
        else:
            self.inter_channels = in_dim // reduce

        self.inter_channels = out_dim//reduce
        self.k_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)
        self.q_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)
        self.v_linears = nn.Conv2d(in_dim, out_dim*num_types, kernel_size=1)
        self.soft = nn.Softmax(-2)

        self.edge_attention = edge_attention
        if self.edge_attention:
            self.edge_transfor = nn.Parameter(torch.FloatTensor(edge_num, self.out_channels))
            torch.nn.init.uniform_(self.edge_transfor, -1e-6, 1e-6)

        self.soft = nn.Softmax(-2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
      
    def forward(self, x, node_type, edge_type, A=None, mask=None, alpha=1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) (B,C,T,V)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        '''
            Create Attention and Message tensor beforehand.
        '''
        A_switch = {None: A, 'init': A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': A + self.PA, 'importance': A * self.PA})
        A = A_switch[self.adaptive]

        select = torch.zeros(self.num_node).int()
        B,C,T,V = x.size()
        att_all_k = self.k_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T) #B V C T
        att_all_q = self.q_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T)# B V C T
        for i in range(self.num_node):    
            select[i] = self.num_node*node_type[i] + i
        res_all = self.v_linears(x).view(B,self.out_channels,self.num_types,T,V).permute(0,1,3,2,4).reshape(B,self.out_channels,T,-1)
        att_msg_k = torch.index_select(att_all_k, 1 , select.cuda()).view(B,V,-1)  # B V C T
        att_msg_q = torch.index_select(att_all_q, 1, select.cuda()).permute(0,2,3,1).view(B,-1,V)

        res_msg = torch.index_select(res_all, 3 , select.cuda())
        res_att = self.soft(torch.matmul(att_msg_k, att_msg_q) / att_msg_k.size(-1)) # B V V

        ## adjancet combination
        x1 = (res_att * alpha + (A.unsqueeze(0) if A is not None else 0))
        
        if self.edge_attention:
        ## edge attention generation according to edge type 
            edge_select = edge_type.int().view(-1)
            edge_att = torch.index_select(self.edge_transfor, 0, edge_select.cuda()).reshape(A.size(1),A.size(1),-1)    
            ## edge attention projection
            x1 = x1.unsqueeze(-1) * edge_att.unsqueeze(0) #B V V C
            # x1 = self.soft(x1)
            # x1 = x1.mean(-1)
            x1 = x1.permute(0, 3, 1, 2)
            B,C,T,V = res_msg.size()
            x1 = torch.einsum('ncuv,nctv->nctu', x1, res_msg).reshape(B,C,T,V)    
        else:   
            B,C,T,V = res_msg.size()
            x1 = torch.einsum('nuv,nctv->nctu', x1, res_msg).reshape(B,C,T,V)  
        return x1

class GTGC(nn.Module):   
    def __init__(self, 
                in_dim, 
                out_dim, 
                A,  
                edge_attention = False, 
                target_specific = False,
                num_types=5, 
                reduce=8, 
                edge_num=15,
                global_attention= True,
                norm='BN',
                *kwargs):
        super().__init__()
        self.dim = in_dim
        self.out_channels = out_dim
        self.num_types = num_types
        self.num_node = A.size(1)
        self.edge_num = edge_num
        self.num_set = A.shape[0]

        # self.PA = nn.Parameter(torch.FloatTensor(A.size(1), A.size(1)))
        # torch.nn.init.uniform_(self.PA, -1e-6, 1e-6)
        if in_dim <= 16:
            self.inter_channels = 8
        else:
            self.inter_channels = in_dim // reduce

        self.inter_channels = out_dim//reduce
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_dim, self.num_set*self.inter_channels, 1),
            build_norm_layer(self.norm_cfg, self.num_set*self.inter_channels)[1], 
            nn.ReLU(inplace=True))
        # self.k_linears = nn.ModuleList()
        # self.q_linears = nn.ModuleList()
        # for i in range(num_types):
        #     self.k_linears.append(nn.Conv2d(in_dim, self.inter_channels, kernel_size=1))
        #     self.q_linears.append(nn.Conv2d(in_dim, self.inter_channels, kernel_size=1))
        self.k_linears = nn.Conv2d(in_dim, self.num_set*self.inter_channels*num_types, kernel_size=1)
        self.q_linears = nn.Conv2d(in_dim, self.num_set*self.inter_channels*num_types, kernel_size=1)
        # self.v_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)    
        self.soft = nn.Softmax(-1)


        self.edge_attention = edge_attention
        self.target_specific = target_specific

        self.alpha = nn.Parameter(torch.zeros(1))

        self.global_attention=global_attention
        if self.global_attention:
            self.beta = nn.Parameter(torch.zeros(1))

        self.active = nn.ReLU()

        if self.edge_attention:
            self.edge_linears = nn.Conv2d(self.num_set*self.inter_channels,self.num_set*edge_num*self.inter_channels,1)
            # self.alpha = nn.Parameter(torch.zeros(1))
        if self.target_specific:
            self.out_linears = nn.Conv2d(self.inter_channels*self.num_set, num_types*out_dim, kernel_size=1)
        else:
            self.out_linears = nn.Conv2d(self.inter_channels*self.num_set, out_dim, 1)

        # self.init_weights()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             conv_init(m)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             bn_init(m, 1)
        # bn_init(self.bn, 1e-6)
      
    def forward(self, x, node_type, edge_type, A=None, mask=None, alpha=1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) (B,C,T,V)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        '''
            Create Attention and Message tensor beforehand.
        '''
        # A = A[None, :, None, None]
        B,C,T,V = x.size()

        pre_x = self.pre_conv(x).reshape(B, self.num_set,self.inter_channels,T,V)
        # select = torch.zeros(self.num_node).int()
        # tem_x = x.mean(dim=-1, keepdim=True)
        att_all_k = self.k_linears(x).view(B, self.num_set, self.inter_channels, self.num_types, T, V) #B a_set C node_sube T,V
        att_all_q = self.q_linears(x).view(B, self.num_set, self.inter_channels, self.num_types, T, V) #B a_set C node_sube T,V
        att_msg_k = torch.diagonal(att_all_k[:,:,:,node_type,:,:],dim1=-3,dim2=-1).mean(-2)
    
        att_msg_q = torch.diagonal(att_all_q[:,:,:,node_type,:,:],dim1=-3,dim2=-1).mean(-2)
        # att_msg_q = torch.diagonal(att_msg_q,dim1=-2,dim2=-1).view(B, self.num_set, self.inter_channels,T,V).mean(-2)
        # # att_all_v = self.v_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T)# B V C T
        # node_mask = torch.zeros(self.num_node,self.num_types).scatter_(1,node_type.reshape(-1,1),1).permute(1,0)
        # node_mask = node_mask[None,None,None,:,None].repeat(B,self.num_set,self.inter_channels, 1,T,1).to(att_all_k.device)
        # att_msg_k = (att_all_k*node_mask).sum(-2).mean(-2) # B C V
        # att_msg_q = (att_all_q*node_mask).sum(-2).mean(-2) # B C V
        # for i in range(self.num_node):
        #     select[i] = self.num_types*i + node_type[i]
        # att_msg_k = torch.index_select(att_all_k, 1 , select.cuda()).mean(-1).permute(0,2,1).contiguous()# B V C
        # att_msg_q = torch.index_select(att_all_q, 1, select.cuda()).mean(-1).permute(0,2,1).contiguous() # B V C
        diff = att_msg_k.unsqueeze(-1) - att_msg_q.unsqueeze(-2)
        # # att_msg_v = torch.index_select(att_all_v, 1, select.cuda()) # B V C T
        # out = self.a_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T)     
        # x1 = torch.index_select(out, 1 , select.cuda()).permute(0,2,3,1)


        '''
                        Step 1: Heterogeneous Mutual Attention
        '''
        # self.edge_attention = False
        if self.edge_attention:
            edge_speci = self.edge_linears(diff.view(B,-1, V,V)).view(B,self.num_set,self.edge_num,self.inter_channels,V,V)
            index = self.edge_linears(diff.view(B,-1, V,V)).view(B,self.num_set,self.edge_num,self.inter_channels,V,V)
            edge_select= edge_type.int().reshape(-1)
            select = torch.zeros(len(edge_select),dtype=int)
            for i in range(len(edge_select)):
                    select[i] = self.edge_num*i + edge_select[i]
            
            edge_speci = edge_speci.permute(0,1,3,4,5,2).reshape(B,self.num_set,self.inter_channels,self.edge_num*V*V)
            edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
            edge_att = edge_att.reshape(B, self.num_set, self.inter_channels,V,V)
            

            # edge_att = edge_att[:,:,0,:,:,:].squeeze()

            # edge_select= edge_type.long().reshape(-1)
            # edge_att = torch.diagonal(edge_att[:,:,edge_select,:],dim1=-3,dim2=-1).view(B,self.num_set,self.inter_channels,V,V)
            # edge_mask = torch.zeros(edge_select.shape[0],self.edge_num).scatter_(1,edge_select,1)
            # edge_mask = edge_mask.reshape(V,V,-1).permute(2,0,1)
            # edge_mask = edge_mask[None,None,:,None].repeat(B,self.num_set,1, self.inter_channels,1,1).to(edge_speci.device)

            # edge_att = (edge_speci * edge_mask).sum(2)
            #  edge_att = torch.masked_select(edge_speci, mask.to(edge_speci.device))
            # edge_att = edge_att.reshape(B,self.inter_channels,V,V)

            # edge_select = edge_type.long().unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B,1,self.inter_channels,1,1).to(edge_speci.device)
            # edge_att= torch.zeros_like(edge_select).type(edge_speci.type())
            # edge_att = edge_att.scatter_(dim=1, index=edge_select, src=edge_speci)
        else:
            edge_att = self.active(diff) # B V V

        '''
                        Step 2: Heterogeneous Message Passing
        '''
        # res_att = (res_att * (A.unsqueeze(0) if A is not None else 0))
        A = (edge_att * self.alpha + (A[None,:,None] if A is not None else 0))

        if self.global_attention:
            glo_att= torch.einsum('nkcv,nkcw->nkvw',att_msg_k,att_msg_q)
            A = A + (glo_att[:,:,None]*self.beta)

        # res_att = self.soft(res_att)
        # self.target_specific=False
        if self.target_specific:
            x1 = torch.einsum('nkctu,nkcuv->nkctv', pre_x, A)
            x1 = self.out_linears(x1.reshape(B, self.num_set*self.inter_channels,-1, V)).view(B, self.num_types, self.out_channels, T, V)
            x1 = torch.diagonal(x1[:,node_type,:,:,:],dim1=1,dim2=-1)
            # node_mask = torch.zeros(self.num_node,self.num_types).scatter_(1,node_type.reshape(-1,1),1).permute(1,0)
            # node_mask = node_mask[None,:,None,None].repeat(B,1,self.out_channels,T,1).to(x1.device)
            # x1 = (x1*node_mask).sum(1)
        else:
            x1 = torch.einsum('nkctu,nkcuv->nkctv', pre_x, A)
            x1 = self.out_linears(x1.reshape(B, self.num_set*self.inter_channels,-1, V))
            # return x1
        return x1

class unit_gtgcn(nn.Module):
    def __init__(self, in_channels, 
                       out_channels, 
                       A, 
                       kernel_size, 
                       edge_type, 
                       node_type, 
                       residual=True,
                       edge_attention = False, 
                       target_specific = False,
                       global_attention =False,
                       adaptive='init',
                       num_types=5, 
                       reduce=8, 
                       edge_num=15,
                       norm='BN'):
        super(unit_gtgcn, self).__init__()

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A[0].clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.size(0)
        # self.convs = nn.ModuleList()
        # for i in range(self.num_subset):
        self.convs=GTGC(in_channels, out_channels, A, edge_attention, target_specific, num_types, reduce, edge_num, global_attention)
        
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    build_norm_layer(self.norm_cfg, out_channels)[1])
            else:
                self.down = lambda x: x  
        else:
            self.down = lambda x: 0
        self.node_type = node_type
        self.edge_type = edge_type

        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
        A = A_switch[self.adaptive]
        y = self.convs(x, self.node_type, self.edge_type, A)
        # y = None
        # A_new = []
        # for i in range(A.shape[0]):
        #     z = self.convs[i](x, self.node_type, self.edge_type, A[i])
        #     y = z + y if y is not None else z
        #     # A_new.append(res_att)
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             conv_init(m)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             bn_init(m, 1)
    #     bn_init(self.bn, 1e-6)

class GTNGCH(nn.Module):   
    def __init__(self, 
                in_dim, 
                out_dim, 
                A,  
                edge_attention = False, 
                target_specific = False,
                adaptive='importance',
                num_types=5, 
                reduce=8, 
                edge_num=15,
                *kwargs):
        super().__init__()
        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        if self.adaptive == 'init':
            self.A = nn.Parameter(A[0].clone())
        else:
            self.register_buffer('A', A[0])

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A[0].clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        self.dim = in_dim
        self.out_channels = out_dim
        self.num_types = num_types
        self.num_node = A.size(1)

        # self.PA = nn.Parameter(torch.FloatTensor(A.size(1), A.size(1)))
        # torch.nn.init.uniform_(self.PA, -1e-6, 1e-6)
        if in_dim <= 16:
            self.inter_channels = 8
        else:
            self.inter_channels = in_dim // reduce

        self.inter_channels = out_dim//reduce
        self.k_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)
        self.q_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)
        # self.v_linears = nn.Conv2d(in_dim, self.inter_channels*num_types, kernel_size=1)
        
        self.soft = nn.Softmax(-1)

        self.edge_attention = edge_attention
        self.target_specific = target_specific

        self.alpha = nn.Parameter(torch.zeros(1))


        if self.edge_attention:
            self.edge_transfor = nn.Parameter(torch.FloatTensor(edge_num, self.out_channels))
            torch.nn.init.uniform_(self.edge_transfor, -1e-6, 1e-6)
            self.relation_pri   = nn.Parameter(torch.ones(edge_num))
            self.relation_att   = nn.Parameter(torch.Tensor(edge_num, self.inter_channels,self.inter_channels))
            torch.nn.init.uniform_(self.relation_att, -1e-6, 1e-6)
            # self.alpha = nn.Parameter(torch.zeros(1))

        if self.target_specific:
            self.relation_msg   = nn.Parameter(torch.Tensor(edge_num, in_dim, self.inter_channels))
            torch.nn.init.uniform_(self.relation_msg, -1e-6, 1e-6)
            self.a_linears = nn.Conv2d(self.inter_channels, out_dim*num_types, kernel_size=1)
        else:
            self.a_linears = nn.Conv2d(in_dim, out_dim*num_types, kernel_size=1)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)
      
    def forward(self, x, node_type, edge_type, A=None, mask=None, alpha=1):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) (B,C,T,V)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        '''
            Create Attention and Message tensor beforehand.
        '''
        A_switch = {None: A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': A + self.PA, 'importance': A * self.PA})
        A = A_switch[self.adaptive]

        select = torch.zeros(self.num_node).int()
        B,C,T,V = x.size()
        att_all_k = self.k_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T) #B V C T
        att_all_q = self.q_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T)# B V C T
        # att_all_v = self.v_linears(x).permute(0, 3, 1, 2).view(B, V, self.num_types, self.inter_channels, T).reshape(B, -1, self.inter_channels, T)# B V C T
        for i in range(self.num_node):    
            select[i] = self.num_types*i + node_type[i]    
        att_msg_k = torch.index_select(att_all_k, 1 , select.cuda()).mean(-1) # B V C
        att_msg_q = torch.index_select(att_all_q, 1, select.cuda()).mean(-1) # B V C
        # att_msg_v = torch.index_select(att_all_v, 1, select.cuda()) # B V C T
        
        '''
                        Step 1: Heterogeneous Mutual Attention
        '''
        if self.edge_attention:
        ## edge attention generation according to edge type 
            edge_select = edge_type.int().view(-1)
            edge_att = torch.index_select(self.relation_att, 0, edge_select.cuda()).reshape(A.size(1),A.size(1), self.inter_channels, self.inter_channels)   # V V C C 
            edge_pri = torch.index_select(self.relation_pri, 0, edge_select.cuda()).reshape(A.size(1),A.size(1),-1) # V, V 1
            ## meta path generation
            res_att = torch.einsum('bvc,vuca,bua->bvu',att_msg_k, edge_att,att_msg_q)
            ## meta path attention since not all meta path have the same attribution to graph
            # x1 = (res_att * alpha + (A.unsqueeze(0) if A is not None else 0))
            res_att = res_att * edge_pri.unsqueeze(0).squeeze(-1)   
            res_att = self.soft(res_att)
        else:   
            res_att = self.soft(torch.matmul(att_msg_k, att_msg_q.permute(0,2,1)) / att_msg_k.size(-1)) # B V V
        
        '''
                        Step 2: Heterogeneous Message Passing
        '''
        # res_att = (res_att * (A.unsqueeze(0) if A is not None else 0))
        res_att = (res_att * self.alpha + (A.unsqueeze(0) if A is not None else 0))
        # res_att = self.soft(res_att)
        if self.target_specific:
            edge_select = edge_type.int().view(-1)
            relation_trans = torch.index_select(self.relation_msg, 0, edge_select.cuda()).reshape(A.size(1),A.size(1), self.dim, self.inter_channels)   # V V C C
            res_msg = torch.einsum('bvct,vuca->bvuat',x.permute(0,3,1,2) ,relation_trans).permute(0,3,4,1,2) #B C T U V
            # res_msg = torch.einsum('bvct,vuca->bvuct',att_msg_v ,relation_trans).permute(0,3,4,1,2) #B C T U V
            res = (res_msg * res_att.unsqueeze(1).unsqueeze(1)).sum(-1)
            
            out = self.a_linears(res).permute(0, 3, 1, 2).view(B, V, self.num_types, self.out_channels, T).reshape(B, -1, self.out_channels, T)     
            x1 = torch.index_select(out, 1 , select.cuda()).permute(0,2,3,1)
            # return x1
        else:
            x1 = torch.einsum('nctu,nuv->nctv', x, res_att)
            out = self.a_linears(x1).permute(0, 3, 1, 2).view(B, V, self.num_types, self.out_channels, T).reshape(B, -1, self.out_channels, T)     
            x1 = torch.index_select(out, 1 , select.cuda()).permute(0,2,3,1)
            # return x1
        return x1, res_att

class dggcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
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
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
            self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
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
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
            x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

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
        x = self.post(x)
        return self.act(self.bn(x) + res)

class dghgcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 ratio=0.25,
                 ctr='T',
                 ada='T',  
                 node_attention = False,
                 edge_attention = False, 
                 ada_attention = False,
                 target_specific = False,
                 add_type = False,
                 num_types=5,    
                 edge_num=15,
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
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.target_specific = target_specific
        self.ada_attention = ada_attention
        self.num_types = num_types
        self.edge_num = edge_num
        self.edge_type = edge_type
        self.node_type = node_type
        self.add_type = add_type
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
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        if self.target_specific:
            self.nodeconv = nn.Conv2d(mid_channels * num_subsets, num_types*out_channels, 1)
        
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))

        if self.ada or self.ctr:
            if self.node_attention:
                self.conv1 = nn.Conv2d(in_channels, num_subsets*mid_channels*num_types, kernel_size=1)
                self.conv2 = nn.Conv2d(in_channels, num_subsets*mid_channels*num_types, kernel_size=1)
            else:
                self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
                self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        
        if self.edge_attention:
            self.edge_linears = nn.Conv2d(num_subsets*mid_channels,edge_num*num_subsets*mid_channels,1)
        
        if self.ada_attention:
            self.ada_linears =nn.Conv2d(num_subsets,edge_num*num_subsets,1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
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
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)

            x1 = self.conv1(tmp_x)
            x2 = self.conv2(tmp_x)
            if self.node_attention:
                x1 = x1.view(n, self.num_subsets, self.mid_channels, self.num_types, -1, v)
                x2 = x2.view(n, self.num_subsets, self.mid_channels, self.num_types, -1, v)
                x1 = torch.diagonal(x1[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                x2 = torch.diagonal(x2[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)
            else:
                x1 = x1.reshape(n, self.num_subsets, self.mid_channels, -1, v)
                x2 = x2.reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
            if self.edge_attention:
                edge_speci = self.edge_linears(diff.view(n,-1, v,v)).view(n,self.num_subsets,self.edge_num,self.mid_channels,v,v)
                edge_select= self.edge_type.int().reshape(-1)
                select = torch.zeros(len(edge_select),dtype=int)
                for i in range(len(edge_select)):
                        select[i] = self.edge_num*i + edge_select[i]        
                edge_speci = edge_speci.permute(0,1,3,4,5,2).reshape(n,self.num_subsets,self.mid_channels,self.edge_num*v*v)
                edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
                edge_att = edge_att.reshape(n, self.num_subsets, self.mid_channels,v,v)
                ada_graph = edge_att.unsqueeze(-3)
                if self.add_type:
                    ada_graph = diff + ada_graph
            else:
                ada_graph = diff

            ada_graph = getattr(self, self.ctr_act)(ada_graph)
            
            # ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
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
        if self.target_specific:
            x_node = self.nodeconv(x)
            x_node = x_node.view(n, self.num_types, self.out_channels, t, v)
            x_node = torch.diagonal(x_node[:,self.node_type,:,:,:],dim1=1,dim2=-1)  
            x = self.post(x)+ x_node
        else:
            x = self.post(x)    
        return self.act(self.bn(x) + res)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

class dgphgcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 ratio=0.25,
                 part_ratio = 0.4,
                 ctr='T',
                 ada='T',  
                 node_attention = False,
                 edge_attention = False, 
                 ada_attention = False,
                 target_specific = False,
                 add_type = False,
                 num_types=5,    
                 edge_num=15,
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
        self.node_attention = node_attention
        self.edge_attention = edge_attention
        self.target_specific = target_specific
        self.ada_attention = ada_attention
        self.num_types = num_types
        self.edge_num = edge_num
        self.edge_type = edge_type
        self.node_type = node_type
        self.add_type = add_type
        self.part_ratio = part_ratio

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
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        if self.target_specific:
            self.nodeconv = nn.Conv2d(mid_channels * num_subsets, num_types*out_channels, 1)
        
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)

        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.semantic_num = int(self.num_subsets * self.part_ratio)
        self.norm_num = self.num_subsets - self.semantic_num
        if self.ada or self.ctr:
            if self.node_attention & self.part_ratio!=0:
                self.conv1_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                self.conv2_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                if self.part_ratio!=1:
                    self.conv1 = nn.Conv2d(in_channels, mid_channels * self.norm_num, 1)
                    self.conv2 = nn.Conv2d(in_channels, mid_channels * self.norm_num, 1)
            else:
                self.conv1 = nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1)
                self.conv2 = nn.Conv2d(in_channels, mid_channels * self.num_subsets, 1)


        
        if self.edge_attention:
            if self.part_ratio!=0 & self.edge_attention:
                # self.edge_linears = nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1)
                self.edge_linears = nn.Sequential(
                    nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1),
                    # build_norm_layer(self.norm_cfg, edge_num*self.semantic_num*mid_channels)[1], self.act
                )
        
        if self.ada_attention:
            self.ada_linears =nn.Conv2d(num_subsets,edge_num*num_subsets,1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
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
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if self.ctr is not None or self.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (self.ctr == 'NA' or self.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)
            x1_norm=None
            x2_norm=None
            x1_sem = None
            x2_sem = None
            if self.part_ratio!=0:
                if self.node_attention:
                    x1_sem = self.conv1_se(tmp_x)
                    x2_sem = self.conv1_se(tmp_x)
                    x1_sem = x1_sem.view(n, self.semantic_num, self.mid_channels, self.num_types, -1, v)
                    x2_sem = x2_sem.view(n, self.semantic_num, self.mid_channels, self.num_types, -1, v)
                    x1_sem = torch.diagonal(x1_sem[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)
                    x2_sem = torch.diagonal(x2_sem[:,:,:,self.node_type,:,:],dim1=-3,dim2=-1)  
                    if self.part_ratio!=1:
                        x1_norm = self.conv1(tmp_x).reshape(n, self.norm_num, self.mid_channels, -1, v)
                        x2_norm = self.conv2(tmp_x).reshape(n, self.norm_num, self.mid_channels, -1, v)  
                else:
                    x1_norm = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
                    x2_norm = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
                    
                if x1_sem == None:
                    x1 = x1_norm
                    x2 = x2_norm
                elif x1_norm ==None:
                    x1 = x1_sem
                    x2 = x2_sem
                else:
                    x1 = torch.cat((x1_norm,x1_sem),axis=1)
                    x2 = torch.cat((x2_norm,x1_sem),axis=1)
            else:
                x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
                x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)

        if self.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            diff_sem = None
            diff_norm = None
            if self.part_ratio!=0:
                diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
                if self.edge_attention:
                    diff_sem = diff[:,self.norm_num:,:]
                    edge_speci = self.edge_linears(diff_sem.view(n,-1, v,v)).view(n,self.semantic_num,self.edge_num,self.mid_channels,v,v)
                    edge_select= self.edge_type.int().reshape(-1)
                    select = torch.zeros(len(edge_select),dtype=int)
                    for i in range(len(edge_select)):
                            select[i] = self.edge_num*i + edge_select[i]        
                    edge_speci = edge_speci.permute(0,1,3,4,5,2).reshape(n,self.semantic_num,self.mid_channels,self.edge_num*v*v)
                    edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
                    edge_att = edge_att.reshape(n, self.semantic_num, self.mid_channels,v,v)
                    ada_graph = edge_att.unsqueeze(-3)
                    if self.part_ratio!=1:
                        diff_norm = diff[:,:self.norm_num,:]
                    if diff_norm != None:
                        ada_graph = torch.cat((diff_norm,ada_graph),axis=1)
                else:
                    ada_graph = diff
            else:
                ada_graph = x1.unsqueeze(-1) - x2.unsqueeze(-2)
                # if self.add_type:
                #     ada_graph = diff + ada_graph
                # ada_graph = torch.cat((diff_norm,ada_graph),axis=1)
            ada_graph = getattr(self, self.ctr_act)(ada_graph)
            
            # ada_graph = getattr(self, self.ctr_act)(diff)

            if self.subset_wise:
                ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.alpha)
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
        if self.target_specific:
            x_node = self.nodeconv(x)
            x_node = x_node.view(n, self.num_types, self.out_channels, t, v)
            x_node = torch.diagonal(x_node[:,self.node_type,:,:,:],dim1=1,dim2=-1)  
            x = self.post(x)+ x_node
        else:
            x = self.post(x)    
        return self.act(self.bn(x) + res)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

class dgphgcn1(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 ratio=0.25,
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
                nn.Conv2d(in_channels, self.semantic_num*num_types*mid_channels, 1),
                build_norm_layer(self.norm_cfg, self.semantic_num*num_types*mid_channels)[1], self.act)
            self.pre = nn.Sequential(
                nn.Conv2d(in_channels, self.norm_num * mid_channels, 1),
                build_norm_layer(self.norm_cfg, mid_channels * self.norm_num)[1], self.act)
        else:
            self.pre = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
                build_norm_layer(self.norm_cfg, mid_channels * num_subsets)[1], self.act)
        
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)

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
                    self.conv1_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                    self.conv2_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels*num_types, kernel_size=1)
                else:
                    self.conv1_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels, kernel_size=1)
                    self.conv2_se = nn.Conv2d(in_channels, self.semantic_num*mid_channels, kernel_size=1)


            self.conv1 = nn.Conv2d(in_channels, self.norm_num*mid_channels, 1)
            self.conv2 = nn.Conv2d(in_channels, self.norm_num*mid_channels, 1)
        
        if self.edge_attention:
            if self.decompose!=0:
                # self.edge_linears = nn.Sequential(
                #     nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1),
                #     # build_norm_layer(self.norm_cfg, edge_num*self.semantic_num*mid_channels)[1], self.act
                #     )
                self.edge_linears = nn.Conv2d(self.semantic_num*mid_channels,edge_num*self.semantic_num*mid_channels,1)

        
        if self.ada_attention:
            self.ada_linears =nn.Conv2d(num_subsets,edge_num*num_subsets,1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
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