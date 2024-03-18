import torch
import torch.nn as nn

from ...utils import Graph
from ..builder import BACKBONES
from .utils import MSTCN, unit_ctrgcn, unit_tcn, unit_ctrhgcn, msmlp


class CTRGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 edge_type,
                 node_type,
                 semantic_index = False,
                 stride=1,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2],
                 tcn_dropout=0,
                 **kwargs):
        super(CTRGCNBlock, self).__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'mstcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp']
        gcn_type = gcn_kwargs.pop('type', 'unit_ctrhgcn')
        assert gcn_type in ['unit_ctrgcn', 'unit_ctrhgcn']

        
        if gcn_type =='unit_ctrgcn':
            self.gcn1 = unit_ctrgcn(in_channels, out_channels, A, **gcn_kwargs)
        elif gcn_type =='unit_ctrhgcn':
            self.gcn1 =unit_ctrhgcn(in_channels, out_channels, A, edge_type, node_type, semantic_index, **gcn_kwargs)
        if tcn_type == 'mstcn':
            self.tcn1 = MSTCN(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilations=dilations,
                residual=False,
                tcn_dropout=tcn_dropout)
        elif tcn_type == 'msmlp':
            self.tcn1 = msmlp(out_channels, out_channels, stride=stride, **tcn_kwargs) 
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

    def init_weights(self):
        pass
        # self.tcn1.init_weights()
        # self.gcn1.init_weights()


@BACKBONES.register_module()
class CTRGCN(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 semantic_stage=range(1,11),
                 pretrained=None,
                 num_person=2,
                 **kwargs):
        super(CTRGCN, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        node_type = torch.tensor(self.graph.node_type, requires_grad=False)
        edge_type = torch.tensor(
            self.graph.edge_type, dtype=torch.float32, requires_grad=False)

        self.num_person = num_person
        self.base_channels = base_channels

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        semantic_index = 1 in semantic_stage
        modules = [CTRGCNBlock(in_channels, base_channels, A.clone(), edge_type, node_type, semantic_index,residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            semantic_index = i in semantic_stage
            modules.append(CTRGCNBlock(base_channels, out_channels, A.clone(), edge_type, node_type, semantic_index, stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for gcn in self.net:
            x = gcn(x)

        x = x.reshape((N, M) + x.shape[1:])
        return x
