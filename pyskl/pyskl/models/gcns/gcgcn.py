import torch
import torch.nn as nn

from ..builder import BACKBONES
from .utils import MSTCN, unit_gcgcn, unit_tcn, gc_sparse



@BACKBONES.register_module()
class GCGCN(nn.Module):
    def __init__(self,
                #  graph_cfg,
                 in_channels=3,
                 num_person=2,
                 mid_channels=50, 
                 stride=1, 
                 feature_hidden= [10, 100, 10, 1],
                 causal_hidden = [100],
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4)],
                 time_serious=9,
                 **kwargs):
        super(GCGCN, self).__init__()

        self.net = gc_sparse(in_channels, 
                             mid_channels, 
                             feature_hidden,
                             causal_hidden ,
                             ms_cfg,
                             time_serious)   

    def init_weights(self):
        self.net.init_weights()
        # for module in self.net:
        #     module.init_weights()

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        predic_loss, gc, ridge = self.net(x)
        gc= gc.reshape(N,M,V,V)

        return predic_loss, gc, ridge
