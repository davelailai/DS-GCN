from calendar import EPOCH
import copy as cp
from gc import get_threshold
from cv2 import threshold

import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

# from pyskl.models.gcns.utils.gcn import unit_gcn, unit_gcnedge

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import mstcn, unit_gcn, unit_tcn, unit_gcnedge, unit_tcnedge, unitmlp,msmlp, unit_gcn_sparse, get_sparsity,unit_tcn_sparse

EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp','unit_tcn_sparse']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn', 'unit_gcnedge','unit_gcn_sparse']

        
        if gcn_type =='unit_gcn':
            self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)
        elif gcn_type =='unit_gcnedge':
            self.gcn =unit_gcnedge(in_channels, out_channels, A, **gcn_kwargs)
        elif gcn_type =='unit_gcn_sparse':
            self.gcn =unit_gcn_sparse(in_channels, out_channels, A, **gcn_kwargs)
        


        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type =='unit_tcnedge':
            self.tcn = unit_tcnedge(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'unitmlp':
            self.tcn = unitmlp(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'msmlp':
            self.tcn = msmlp(out_channels, out_channels, stride=stride, **tcn_kwargs)
        elif tcn_type == 'unit_tcn_sparse':
            self.tcn = unit_tcn_sparse(out_channels, out_channels, 9, stride=stride, conv_sparsity=tcn_kwargs['sparse_ratio'])
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_sparse(in_channels, out_channels, kernel_size=1, stride=stride, conv_sparsity=tcn_kwargs['sparse_ratio'])

    def forward(self, x, sparsity=0, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x,sparsity, A),sparsity) + res
        return self.relu(x)
    
    def init_weights(self):
        # pass
        self.tcn.init_weights()
        self.gcn.init_weights()

    


@BACKBONES.register_module()
class STGCN_sparse(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 sparse_decay=False,
                 linear_sparsity=0,
                 warm_up=0,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs
        self.sparse_decay = sparse_decay
        self.linear_sparsity = linear_sparsity
        self.warm_up = warm_up

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

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x, current_epoch, max_epoch):
        if current_epoch < self.warm_up:
            sparsity = 0
        else:
            if self.sparse_decay:
                if current_epoch<(max_epoch/2.0):
                    sparsity=get_sparsity(self.linear_sparsity,current_epoch,0,max_epoch/2)
                else:
                    sparsity=self.linear_sparsity
            else:
                sparsity=self.linear_sparsity
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.float()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # threshold = self.get_threshold(sparsity,epoch=current_epoch)

        for i in range(self.num_stages):

            threshold = self.get_threshold(self.gcn,sparsity,epoch=current_epoch)
            # threshold = self.get_threshold(self.gcn[i],sparsity,epoch=current_epoch)
            x = self.gcn[i](x,threshold)

        x = x.reshape((N, M) + x.shape[1:])
        return x
    
    def get_threshold(self, model, sparsity,epoch=None):
        local=[]
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold= self.percentile(local,sparsity*100)
        
        return threshold
 
    def get_mask(self, model, current_epoch, max_epoch):
        # if current_epoch < self.warm_up:
        #     sparsity = 0
        # else:
        #     if self.sparse_decay:
        #         if current_epoch<(max_epoch/2.0):
        #             sparsity=get_sparsity(self.linear_sparsity,current_epoch,0,max_epoch/2)
        #         else:
        #             sparsity=self.linear_sparsity
        #     else:
        #         sparsity=self.linear_sparsity
        sparsity=self.linear_sparsity
        threshold = self.get_threshold(model,sparsity)
        mask=[]
        weight = []
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==self.linear_sparsity:
                mask.append(p.detach().flatten())
            if hasattr(p, 'is_mask') and p.is_mask:
                weight.append(p.detach().flatten())
        mask = torch.cat(mask)
        weight = torch.cat(weight)
        mask =mask<=threshold
        weight = weight*mask

        return weight

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
    
    def regularize(self, lam, penalty, current_epoch, max_epoch):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W =[]
        for i in range(self.num_stages):
            # index = self.get_mask(self.gcn[i], current_epoch, max_epoch)

            W.append(self.get_mask(self.gcn[i], current_epoch, max_epoch))
        # W = gc
        # W = network.layers[0].weight
        # W = torch.stack(W)
        # W = torch.cat(W)
        # b,hidden, p, p1, lag = weight.shape
        panelty = []
        if penalty == 'GL':
            W = torch.cat(W)
            return lam * torch.sum(torch.norm(W, dim=0))
        elif penalty == 'GSGL':
            panelty = []
            for weight_layer in W:
                index = torch.sum(torch.norm(weight_layer, dim=0))
                panelty.append(index)
            panelty = torch.stack(panelty)
            return lam*torch.sum(panelty)
            # return panelty
            # return lam * (torch.sum(torch.norm(W, dim=(1, -1)))
            #             + torch.sum(torch.norm(W, dim=1)))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            # return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 1)))
            #                 for i in range(lag)])
            pass
        else:
            raise ValueError('unsupported penalty: %s' % penalty)
