import torch
import torch.nn as nn

from ...utils import Graph
from ..builder import BACKBONES
from .utils import MSTCN, mstcn_sparse, unit_ctrgcn, unit_tcn_sparse,unit_tcn, unit_ctrhgcn, unit_ctrgcn_sparse,get_sparsity


class CTRGCNBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
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

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp','mstcn_sparse']
        gcn_type = gcn_kwargs.pop('type', 'unit_ctrhgcn')
        assert gcn_type in ['unit_ctrgcn', 'unit_ctrhgcn','unit_ctrgcn_sparse']

        
        if gcn_type =='unit_ctrgcn':
            self.gcn1 = unit_ctrgcn(in_channels, out_channels, A, **gcn_kwargs)
        elif gcn_type =='unit_ctrgcn_sparse':
            self.gcn1 =unit_ctrgcn_sparse(in_channels, out_channels, A, **gcn_kwargs)

        self.tcn1 = mstcn_sparse(out_channels,out_channels,stride=stride,dropout=tcn_dropout,**tcn_kwargs)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn_sparse(in_channels, out_channels, kernel_size=1, stride=stride, conv_sparsity=tcn_kwargs['sparse_ratio'])

    def forward(self, x,sparsity=0):
        y = self.gcn1(x,sparsity)
        y = self.tcn1(y,sparsity)
        try:
            index = self.residual(x,sparsity)
        except:
            res = self.residual(x)
        else:
            res = self.residual(x,sparsity)
        # res = self.residual(x,sparsity)
        y = y+res
        # y = self.relu(self.tcn1(self.gcn1(x,sparsity),sparsity) + self.residual(x,sparsity))
        return self.relu(y)

    def init_weights(self):
        self.tcn1.init_weights()
        self.gcn1.init_weights()


@BACKBONES.register_module()
class CTRGCN_sparse(nn.Module):
    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 semantic_stage=range(1,11),
                 pretrained=None,
                 sparse_decay=False,
                 linear_sparsity=0,
                 warm_up=0,
                 num_person=2,
                 **kwargs):
        super(CTRGCN_sparse, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        node_type = torch.tensor(self.graph.node_type, requires_grad=False)
        edge_type = torch.tensor(
            self.graph.edge_type, dtype=torch.float32, requires_grad=False)

        self.num_person = num_person
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.sparse_decay = sparse_decay
        self.linear_sparsity = linear_sparsity
        self.warm_up = warm_up

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        semantic_index = 1 in semantic_stage
        modules = [CTRGCNBlock(in_channels, base_channels, A.clone(), residual=False, **kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            semantic_index = i in semantic_stage
            modules.append(CTRGCNBlock(base_channels, out_channels, A.clone(), stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

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
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):

            threshold = self.get_threshold(self.net[i],sparsity,epoch=current_epoch)
            x = self.net[i](x,threshold)

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

    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
        # for gcn in self.net:
        #     x = gcn(x)

        # x = x.reshape((N, M) + x.shape[1:])
        # return x
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

            W.append(self.get_mask(self.net[i], current_epoch, max_epoch))
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
