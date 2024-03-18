import torch
import torch.nn as nn

from pyskl.models.gcns.utils import gcn_sparse, tcn_sparse

from ...utils import Graph
from ..builder import BACKBONES
from .utils import MSTCN, mstcn_sparse, unit_ctrgcn, unit_tcn_sparse,unit_tcn, unit_ctrhgcn, unit_ctrgcn_sparse,get_sparsity
from .ctrgcn_sparse import CTRGCNBlock
from .aagcn_sparse import AAGCNBlock
from .stgcn_sparse import STGCNBlock
from .dggcn_sparse import DGBlock

class AssembleBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 model,
                 sparse_ratio,
                 A,
                 edge_type=None,
                 node_type=None,
                 stride=1,
                 sparse_decay=False,
                 warm_up=0,
                 residual=True,
                 kernel_size=5,
                 dilations=[1, 2],
                 tcn_dropout=0,
                 **kwargs):
        super(AssembleBlock, self).__init__()


        # assert model in ['ST-GCN', 'AA-GCN', 'CTR-GCN', 'DG-GCN']
        self.model_len = len(model)
        self.sparse_ratio = sparse_ratio
        self.warm_up = warm_up
        self.sparse_decay = sparse_decay
        models = []
        A = A.reshape((len(model),int(A.shape[0]/len(model)))+A.shape[1:])
        for i,(model_unit, sparse_unit) in enumerate(zip(model, sparse_ratio)): 
            # tcn_sparse_ratio=sparse_unit
            # gcn_sparse_ratio=sparse_unit
            A_unit = A[i,:]
            assert model_unit in ['ST-GCN', 'AA-GCN', 'CTR-GCN', 'DG-GCN']   
            if model_unit =='ST-GCN':
                ST_kwargs = kwargs['ST_kwargs']
                models.append(STGCNBlock(in_channels,out_channels,A_unit,stride=stride, 
                                         tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**ST_kwargs))
            if model_unit =='AA-GCN':
                AA_kwargs = kwargs['AA_kwargs']
                models.append(AAGCNBlock(in_channels,out_channels,A_unit,edge_type,node_type,
                                         stride=stride,tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**AA_kwargs))
            if model_unit =='CTR-GCN':
                CTR_kwargs = kwargs['CTR_kwargs']
                models.append(CTRGCNBlock(in_channels,out_channels,A_unit,
                                          stride=stride,tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**CTR_kwargs))
            if model_unit =='DG-GCN':
                DG_kwargs = kwargs['DG_kwargs']
                models.append(DGBlock(in_channels,out_channels,A_unit,edge_type, node_type,
                                      stride=stride,tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**DG_kwargs))
        self.net = nn.ModuleList(models)
        
    def forward(self,x, current_epoch, max_epoch):
        x_Assemble = []
        for i in range(self.model_len):
            if current_epoch < self.warm_up:
                sparsity = 0
            else:
                if self.sparse_decay:
                    if current_epoch<(max_epoch/2.0):
                        sparsity=get_sparsity(self.sparse_ratio[i],current_epoch,0,max_epoch/2)
                    else:
                        sparsity=self.sparse_ratio[i]
                else:
                    sparsity=self.sparse_ratio[i]
            threshold = self.get_threshold(self.net[i],sparsity,self.sparse_ratio[i])
            x_unit = self.net[i](x[i],threshold)
            x_Assemble.append(x_unit)

        return x_Assemble

    def get_threshold(self, model, sparsity,linear_sparsity):
        local=[]
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold= self.percentile(local,sparsity*100)
        return threshold
    
    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
        
        
    def init_weights(self):
        for module in self.net:
            module.init_weights()

@BACKBONES.register_module()
class Assemble_sparse(nn.Module):
    def __init__(self,
                 graph_cfg,
                 model_list,
                 sparse_ratio,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 semantic_stage=range(1,11),
                 pretrained=None,
                 sparse_decay=False,
                #  linear_sparsity=0,
                 warm_up=0,
                 num_person=2,
                 **kwargs):
        super(Assemble_sparse, self).__init__()

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
        # self.linear_sparsity = linear_sparsity
        self.warm_up = warm_up
        self.model_list = model_list
        self.sparse_ratio = sparse_ratio

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        semantic_index = 1 in semantic_stage
        modules = [AssembleBlock(in_channels, base_channels, model_list,sparse_ratio, A.clone(), residual=False, sparse_decay=False,warm_up=self.warm_up,**kwargs0)]
        for i in range(2, num_stages + 1):
            in_channels = base_channels
            out_channels = base_channels * (1 + (i in inflate_stages))
            stride = 1 + (i in down_stages)
            semantic_index = i in semantic_stage
            modules.append(AssembleBlock(base_channels, out_channels, model_list,sparse_ratio, A.clone(),sparse_decay=False, warm_up=self.warm_up, stride=stride, **kwargs))
            base_channels = out_channels
        self.net = nn.ModuleList(modules)

    def init_weights(self):
        for module in self.net:
            module.init_weights()

    def forward(self, x, current_epoch, max_epoch):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = self.data_bn(x.view(N, M * V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x_assemble = []
        for i in range(len(self.model_list)):
            x_assemble.append(x)

        for i in range(self.num_stages):
            x_assemble = self.net[i](x_assemble, current_epoch, max_epoch)

        # x_unit = x_assemble[0]
        
        # x_unit = x_unit.reshape((N, M) + x_unit.shape[1:])
        x_assemble = torch.stack(x_assemble)
        x_assemble = x_assemble.reshape((x_assemble.shape[0],N, M) + x_assemble.shape[2:])

        return x_assemble
      
        
    def get_threshold(self, model, sparsity,linear_sparsity):
        local=[]
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold= self.percentile(local,sparsity*100)
        return threshold
    
    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
    
    def get_mask(self, model, current_epoch, max_epoch,linear_sparsity):
        # if current_epoch < self.warm_up:
        #     sparsity = 0
        # else:
        #     if self.sparse_decay:
        #         if current_epoch<(max_epoch/2.0):
        #             sparsity=get_sparsity(linear_sparsity,current_epoch,0,max_epoch/2)
        #         else:
        #             sparsity=linear_sparsity
        #     else:
        #         sparsity=linear_sparsity
        sparsity=linear_sparsity
        threshold = self.get_threshold(model,sparsity,linear_sparsity)
        mask=[]
        weight = []
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
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
        for j in range(len(self.model_list)):
            for i in range(self.num_stages):
                W.append(self.get_mask(self.net[i].net[j], current_epoch, max_epoch,self.sparse_ratio[j]))
                # W
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
