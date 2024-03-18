import torch
import torch.nn as nn

from ..builder import BACKBONES
from .utils import MSTCN, unit_gcgcn, unit_tcn, gc_sparse, gc_component



@BACKBONES.register_module()
class GCGCN_component(nn.Module):
    def __init__(self,
                #  graph_cfg,
                in_channels=3,
                causal_channel = 100,
                feature_update = [64,128,1],
                feature_hidden= [100, 10, 1],
                time_len = 9,
                time_serious=25,
                bias: bool = True,
                init_mode = 'kaiming_uniform',
                init_scale = 1.0):
        super(GCGCN_component, self).__init__()

        self.net = gc_component(in_channels,
                                causal_channel, 
                                feature_update,
                                feature_hidden,
                                time_len,
                                time_serious,
                                bias,
                                init_mode,
                                init_scale)   
        # self.net = gc_component()
    def init_weights(self):
        self.net.init_weights()
        self.net.pool_init()
        # for module in self.net:
        #     module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        gc, predic_loss, panelty, regularize = self.net(x)
        gc= gc.reshape(N,M,V,V)

        return gc, predic_loss, panelty, regularize
