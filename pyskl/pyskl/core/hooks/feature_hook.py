import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, OptimizerHook
from torch.nn.utils import clip_grad
import logging
from math import ceil

@HOOKS.register_module()
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, x, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        x = x[0]
        n, c, t, v = x.shape
        res = module.down(x)
        A = module.A

        # 1 (N), K, 1 (C), 1 (T), V, V
        A = A[None, :, None, None]
        

        if module.target_specific & module.decompose:
            # x_node = pre_x_ori[:,self.semantic_num:self.norm_num,:].reshape(n, -1, t, v)
            x_node = module.nodeconv(x)
            x_node = x_node.view(n, module.semantic_num,module.num_types, module.mid_channels, t, v)
            x_node = torch.diagonal(x_node[:,:,module.node_type,:,:,:],dim1=2,dim2=-1) 
            x_norm = module.pre(x).reshape(n, module.norm_num, module.mid_channels, t, v)
            pre_x = torch.cat((x_node,x_norm),axis=1)
        else:
            pre_x = module.pre(x).reshape(n, module.num_subsets, module.mid_channels, t, v)

        # * The shape of pre_x is N, K, C, T, V

        x1, x2 = None, None
        if module.ctr is not None or module.ada is not None:
            # The shape of tmp_x is N, C, T or 1, V
            tmp_x = x

            if not (module.ctr == 'NA' or module.ada == 'NA'):
                tmp_x = tmp_x.mean(dim=-2, keepdim=True)
            
            x1_norm = module.conv1(tmp_x).reshape(n, module.norm_num, module.mid_channels, -1, v)
            x2_norm = module.conv2(tmp_x).reshape(n, module.norm_num, module.mid_channels, -1, v)
            x1_sem = None
            x2_sem = None
            if module.decompose:
                x1_sem = module.conv1_se(tmp_x)
                x2_sem = module.conv1_se(tmp_x)
                if module.node_attention:   
                    x1_sem = x1_sem.view(n, module.semantic_num, module.mid_channels, module.num_types, -1, v)
                    x2_sem = x2_sem.view(n, module.semantic_num, module.mid_channels, module.num_types, -1, v)
                    x1_sem = torch.diagonal(x1_sem[:,:,:,module.node_type,:,:],dim1=-3,dim2=-1)
                    x2_sem = torch.diagonal(x2_sem[:,:,:,module.node_type,:,:],dim1=-3,dim2=-1)
                else:
                    x1_sem = x1_sem.reshape(n, module.semantic_num, module.mid_channels, -1, v)
                    x2_sem = x2_sem.reshape(n, module.semantic_num, module.mid_channels, -1, v)

            # else:
            #     x1 = x1.reshape(n, self.num_subsets, self.mid_channels, -1, v)
            #     x2 = x2.reshape(n, self.num_subsets, self.mid_channels, -1, v)
            if x1_sem == None:
                x1 = x1_norm
                x2 = x2_norm
            else:
                x1 = torch.cat((x1_norm,x1_sem),axis=1)
                x2 = torch.cat((x2_norm,x1_sem),axis=1)

        if module.ctr is not None:
            # * The shape of ada_graph is N, K, C[1], T or 1, V, V
            if module.decompose:
                if module.edge_attention:
                    # diff_sem = x1[:,self.semantic_num:self.norm_num,:].unsqueeze(-1) - x2[:,self.semantic_num:self.norm_num,:].unsqueeze(-2)
                    diff_sem = x1[:,module.norm_num-module.semantic_num:module.norm_num,:].unsqueeze(-1) - x2[:,module.norm_num-module.semantic_num:module.norm_num,:].unsqueeze(-2)
                    edge_speci = module.edge_linears(diff_sem.view(n,-1, v,v)).view(n,module.semantic_num,module.edge_num,module.mid_channels,v,v)
                    edge_select= module.edge_type.int().reshape(-1)
                    select = torch.zeros(len(edge_select),dtype=int)
                    for i in range(len(edge_select)):
                            select[i] = module.edge_num*i + edge_select[i]        
                    edge_speci = edge_speci.permute(0,1,3,4,5,2).reshape(n,module.semantic_num,module.mid_channels,module.edge_num*v*v)
                    edge_att = torch.index_select(edge_speci, -1, select.to(edge_speci.device))
                    edge_att = edge_att.reshape(n, module.semantic_num, module.mid_channels,v,v)
                    ada_graph = edge_att.unsqueeze(-3)
                else:
                    ada_graph = x1[:,module.semantic_num:module.norm_num,:].unsqueeze(-1) - x2[:,module.semantic_num:module.norm_num,:].unsqueeze(-2)

                diff_norm = x1[:,0:module.norm_num-module.semantic_num,:].unsqueeze(-1) - x2[:,0:module.norm_num-module.semantic_num,:].unsqueeze(-2)
                diff_node = x1[:,module.norm_num:,:].unsqueeze(-1) - x2[:,module.norm_num:,:].unsqueeze(-2)
                ada_graph = torch.cat((diff_norm,ada_graph,diff_node),axis=1)
            else:
                ada_graph = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        
            ada_graph = getattr(module, module.ctr_act)(ada_graph)
            
            # ada_graph = getattr(self, self.ctr_act)(diff)

            if module.subset_wise:
                if module.num_subsets == len(module.alpha):
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, module.alpha)
                else:
                    alpha = torch.repeat_interleave(module.alpha, ceil(module.num_subsets/3)) 
                    ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, alpha[2*module.semantic_num-module.norm_num:])
            else:
                ada_graph = ada_graph * module.alpha[0]
            A = ada_graph + A

        # if module.ada is not None:
        #     # * The shape of ada_graph is N, K, 1, T[1], V, V
        #     ada_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        #     if module.ada_attention:
        #         ada_speci = module.ada_linears(ada_graph.squeeze()).view(n,module.num_subsets,module.edge_num,-1,v,v)
        #         edge_select= module.edge_type.int().reshape(-1)
        #         select = torch.zeros(len(edge_select),dtype=int)
        #         for i in range(len(edge_select)):
        #                 select[i] = module.edge_num*i + edge_select[i]    
        #         ada_speci = ada_speci.permute(0,1,3,4,5,2).reshape(n,module.num_subsets,-1,module.edge_num*v*v)
        #         ada_graph = torch.index_select(ada_speci, -1, select.to(ada_speci.device))
        #         ada_graph = ada_graph.reshape(n, module.num_subsets, -1,v,v)
        #         ada_graph = ada_graph.unsqueeze(-3)

        #     ada_graph = getattr(module, module.ada_act)(ada_graph)

        #     if module.subset_wise:
        #         if module.num_subsets == len(module.beta):
        #             ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, module.beta)
        #         else:
        #             beta = torch.repeat_interleave(module.beta, ceil(module.num_subsets/3)) 
        #             ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, beta[2*module.semantic_num-module.norm_num:])
        #         # ada_graph = torch.einsum('nkctuv,k->nkctuv', ada_graph, self.beta)
        #     else:
        #         ada_graph = ada_graph * module.beta[0]
        #     A = ada_graph + A

       
        self.fea = fea_out
        self.A = A
        self.ori_A = module.A

    def hook_repre(self, module, fea_in, fea_out):
        '''
        注意用于处理feature的hook函数必须包含三个参数[module, fea_in, fea_out]，参数的名字可以自己起，但其意义是
        固定的，第一个参数表示torch里的一个子module，比如Linear,Conv2d等，第二个参数是该module的输入，其类型是
        tuple；第三个参数是该module的输出，其类型是tensor。注意输入和输出的类型是不一样的，切记。
        '''
        self.fea = fea_out

def get_feas_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    for module in model.module.backbone.gcn._modules:
        m = model.module.backbone.gcn._modules[module].gcn
    # m = model.module.backbone.gcn._modules['9'].gcn
   
        cur_hook = HookTool()
        m.register_forward_hook(cur_hook.hook_fun)
        fea_hooks.append(cur_hook)

    return fea_hooks

def get_rep_by_hook(model):
    """
    提取Conv2d后的feature，我们需要遍历模型的module，然后找到Conv2d，把hook函数注册到这个module上；
    这就相当于告诉模型，我要在Conv2d这一层，用hook_fun处理该层输出的feature.
    由于一个模型中可能有多个Conv2d，所以我们要用hook_feas存储下来每一个Conv2d后的feature
    """
    fea_hooks = []
    m = model.module.backbone
   
    cur_hook = HookTool()
    m.register_forward_hook(cur_hook.hook_repre)
    fea_hooks.append(cur_hook)

    return fea_hooks