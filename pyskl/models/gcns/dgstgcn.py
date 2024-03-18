import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import dggcn, dgmstcn, unit_tcn, mstcn, unit_tcn, dghgcn, dgphgcn, dgphgcn1,dgmsmlp
EPS = 1e-4


class DGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, edge_type, node_type,stride=1, residual=True, **kwargs):
        super().__init__()
        # prepare kwargs for gcn and tcn
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'dgmstcn','dgmsmlp']
        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type =='dgmstcn':
            self.tcn = dgmstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type =='dgmsmlp':
            self.tcn = dgmsmlp(out_channels, out_channels, stride=stride, **tcn_kwargs)
        
        gcn_type = gcn_kwargs.pop('type', 'dghgcn')
        assert gcn_type in ['dghgcn', 'dgphgcn', 'dgphgcn1','dggcn']
        if gcn_type == 'dggcn':
            self.gcn = dggcn(in_channels, out_channels, A, **gcn_kwargs)
        if gcn_type =='dghgcn':
            self.gcn = dghgcn(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)
        if gcn_type =='dgphgcn':
            self.gcn = dgphgcn(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)
        if gcn_type =='dgphgcn1':
            self.gcn = dgphgcn1(in_channels, out_channels, A, edge_type, node_type, **gcn_kwargs)
        # self.tcn = dgmstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)
        
    def init_weights(self):
        pass
        # self.tcn.init_weights()
        # self.gcn.init_weights()


@BACKBONES.register_module()
class DGSTGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        node_type = torch.tensor(self.graph.node_type, requires_grad=False)
        edge_type = torch.tensor(
            self.graph.edge_type, dtype=torch.float32, requires_grad=False)

        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)
        if 'gcn_stage' in self.kwargs:
            for i in range(num_stages):
                if i in self.kwargs['gcn_stage']:
                    lw_kwargs[i]['gcn_stage'] = True
                else:
                    lw_kwargs[i]['gcn_stage'] = False


        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [DGBlock(in_channels, base_channels, A.clone(), edge_type, node_type, 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(DGBlock(in_channels, out_channels, A.clone(), edge_type, node_type, stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        # x = x.float()
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x