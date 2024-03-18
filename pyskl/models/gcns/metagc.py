import torch
import torch.nn as nn
import copy as cp

from ...utils import Graph
from ..builder import BACKBONES
from .utils import gcmlp, unitmlp


class GClock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs
                 ):
        super(GClock, self).__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn', 'unit_gcnedge']

        self.tcn = gcmlp(in_channels, out_channels, stride=stride, **tcn_kwargs)
        self.tcn1 = gcmlp(in_channels, out_channels, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 =self.tcn(x)
        x2 = self.tcn1(x)
        # feature, loss, ridge = self.gcn1(x)
        # y = self.relu(feature.clone())
        # y = self.relu(self.gcn1(x))
        return x

    def init_weights(self):
        # self.tcn1.init_weights()
        self.tcn.init_weights()


@BACKBONES.register_module()
class METAGC(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 inflate_stages=[2, 4],
                 down_stages=[2, 4],
                 pretrained=None,
                 num_person=2,
                 **kwargs):
        super(METAGC, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.num_person = num_person
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        # lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        # for k, v in kwargs.items():
        #     if isinstance(v, tuple) and len(v) == num_stages:
        #         for i in range(num_stages):
        #             lw_kwargs[i][k] = v[i]
        modules = [GClock(in_channels, base_channels, A.clone(), residual=False, **kwargs0)]
        # for i in range(2, num_stages + 1):
        #     in_channels = base_channels
        #     out_channels = base_channels * (1 + (i in inflate_stages))
        #     stride = 1 + (i in down_stages)
        #     modules.append(GClock(base_channels, out_channels, A.clone(), stride=stride, **kwargs))
        #     base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

    def forward(self, x):
        losses = []
        ridges = []
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for gcn in self.net:
            x ,loss, ridge= gcn(x)
            loss = torch.stack(loss)
            ridge = torch.stack(ridge).sum(dim=0)
            losses.append(loss)
            ridges.append(ridge)

        x = x.reshape((N, M) + x.shape[1:])
        return x, losses, ridges
