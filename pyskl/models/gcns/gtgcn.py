import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import mstcn, unit_gtgcn, unit_tcn, unitmlp, msmlp


class GTGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 **kwargs):
        super(GTGCNBlock, self).__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'unitmlp', 'msmlp']
        gcn_type = gcn_kwargs.pop('type', 'unit_gtgcn')
        assert gcn_type in ['unit_gtgcn']
        
        self.gcn1 = unit_gtgcn(in_channels, out_channels, A, kernel_size[1], edge_type, node_type, **gcn_kwargs)
    
        
        if tcn_type == 'unit_tcn':
            self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size[0], stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn1 = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type == 'unitmlp':
            self.tcn1 = unitmlp(out_channels, out_channels, kernel_size[0], stride=stride, **tcn_kwargs)
        elif tcn_type == 'msmlp':
            self.tcn1 = msmlp(out_channels, out_channels, stride=stride, **tcn_kwargs)


        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A):
        spatial= self.gcn1(x, A)
        y = self.relu(self.tcn1(spatial)+ self.residual(x))
        # y = self.relu(self.tcn1(self.gcn1(x, A)) + self.residual(x))
        return y

    # def init_weights(self):
    #     self.tcn1.init_weights()
    #     self.gcn1.init_weights()


@BACKBONES.register_module()
class GTGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 num_person=2,
                 **kwargs):
        super(GTGCN, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        node_type = torch.tensor(self.graph.node_type, requires_grad=False)
        edge_type = torch.tensor(
            self.graph.edge_type, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.num_person = num_person
        self.base_channels = base_channels
        self.pretrained = pretrained

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        modules = [GTGCNBlock(in_channels, base_channels, A, edge_type, node_type, kernel_size=kernel_size, residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            modules.append(GTGCNBlock(base_channels, out_channels, A, edge_type, node_type, stride=stride, kernel_size=kernel_size, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)
    # def init_weights(self):
    #     for module in self.net:
    #         module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.float()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for gcn in self.net:
            x= gcn(x, self.A)

        x = x.reshape((N, M) + x.shape[1:])
        return x
