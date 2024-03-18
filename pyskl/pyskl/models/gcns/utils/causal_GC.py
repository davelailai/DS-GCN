from ssl import ALERT_DESCRIPTION_CERTIFICATE_REVOKED
from tkinter import N
from xmlrpc.server import resolve_dotted_attribute
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from .tcn import unit_tcn
import math

from .init_func import bn_init, conv_branch_init, conv_init

EPS = 1e-4
def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act

class GCCGC(nn.Module): #GC channel wise
    def __init__(self, in_channels, out_channels, stride=1, rel_reduction=8, time_step=9):
        super(GCCGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.time_step = time_step
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.in_channels*time_step, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()
        self.rel_reduction = rel_reduction

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        B, C, T, V = x.shape
        x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x = x.permute(0,1,3,2).reshape(-1,T).unsqueeze(0) # BCV T
        # x = x.reshape(-1, T, V).permute(0,2,1)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)
        x1 = x1.reshape(B, -1, self.time_step, V, V).permute(0,1,3,4,2).reshape(-1, V, V, self.time_step)
        x = F.pad(x,(self.time_step-1,0))
        z= [F.conv1d(x,  x1[:,:,i,:].squeeze(), groups=B*C, stride = self.stride) for i in range(V)]
        z= torch.stack(z) # 
        T = z.shape[-1]
        z = z.squeeze().permute(1,2,0).reshape(B,C,T,V)
        z = self.conv3(z)
        # for i in range(self.rel_reduction):
        #     weight = x1[:,:,i,:].squeeze()
        #     z = F.conv1d(x,  weight, groups=B*C)
        # N, R, V, V
        # x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        # x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return z

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    
    # def gc_gcn(self, weight, )

    

class GCGC(nn.Module): # GC sample wise
    def __init__(self, in_channels, out_channels, stride=1, rel_reduction=8, time_step=9, lam=0.1):
        super(GCGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.time_step = time_step
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_channels, self.time_step, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.time_step, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.conv4 = nn.Conv2d(self.rel_channels, time_step, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()
        self.rel_reduction = rel_reduction
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        B, C, T, V = x.shape
        x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        # x = x.permute(0,1,3,2).reshape(-1,T).unsqueeze(0) # BCV T
        x = x.permute(1,0,3,2).reshape(C,-1, T) #C BV T
        # x = x.reshape(-1, T, V).permute(0,2,1)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        ridge = self.ridge_regularize(x1)
        x1 = x1 * alpha + (A[None, None] if A is not None else 0)
        x1 = x1.reshape(B, -1, self.time_step, V, V).permute(0,1,3,4,2).reshape(-1, V, V, self.time_step)
        x = F.pad(x,(self.time_step-1,0))
        z= [F.conv1d(x,  x1[:,:,i,:].squeeze(), groups=B, stride = self.stride) for i in range(V)]
        z= torch.stack(z) # 
        loss = self.prodection(z, x)
        # ridge = self.ridge_regularize(x1)
        T = z.shape[-1]
        z = z.permute(2,1,3,0)
        z = self.conv3(z)
        # for i in range(self.rel_reduction):
        #     weight = x1[:,:,i,:].squeeze()
        #     z = F.conv1d(x,  weight, groups=B*C)
        # N, R, V, V
        # x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        # x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return z, loss, ridge

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    def prodection(self, z, x):
        V, C, B, T = z.shape
        z = z.permute(2,0,1,3).reshape(B,V, C*T)
        x = x[:,:,self.time_step-1:]
        x = x[:,:,::self.stride]
        x = x.permute(1,0,2).reshape(B, V, C*T)
        loss = self.loss_fn(z,x)
        return loss
        # ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        # smooth = loss + ridge
    def ridge_regularize(self, x1):
        '''Apply ridge penalty at all subsequent layers.'''
        B,V,V,T = x1.shape
        x1 = x1.reshape(B,-1)
        ridge = torch.sum(x1**2,dim=-1)
        return ridge
        # return lam * sum([torch.sum(fc.weight ** 2))

    
    # def gc_gcn(self, weight, )
class GCGC_T(nn.Module): # GC sample wise
    def __init__(self, in_channels, out_channels, stride=1, rel_reduction=8, time_step=9, lam=0.1):
        super(GCGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.time_step = time_step
        self.stride = stride
        self.conv1 = unit_tcn(self.in_channels, self.time_step, kernel_size=time_step)
        self.conv2 = nn.Conv2d(self.in_channels, self.time_step, kernel_size=time_step)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        # self.conv4 = nn.Conv2d(self.rel_channels, time_step, kernel_size=1)
        self.tanh = nn.Tanh()
        self.init_weights()
        self.rel_reduction = rel_reduction
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        B, C, T, V = x.shape
        x1, x2 = self.conv1(x).mean(1), self.conv2(x).mean(1)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        # x = x.permute(0,1,3,2).reshape(-1,T).unsqueeze(0) # BCV T
        x = x.permute(1,0,3,2).reshape(C,-1, T) #C BV T
        # x = x.reshape(-1, T, V).permute(0,2,1)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        ridge = self.ridge_regularize(x1)
        x1 = x1 * alpha + (A[None, None] if A is not None else 0)
        x1 = x1.reshape(B, -1, self.time_step, V, V).permute(0,1,3,4,2).reshape(-1, V, V, self.time_step)
        x = F.pad(x,(self.time_step-1,0))
        z= [F.conv1d(x,  x1[:,:,i,:].squeeze(), groups=B, stride = self.stride) for i in range(V)]
        z= torch.stack(z) # 
        loss = self.prodection(z, x)
        # ridge = self.ridge_regularize(x1)
        T = z.shape[-1]
        z = z.permute(2,1,3,0)
        z = self.conv3(z)
        # for i in range(self.rel_reduction):
        #     weight = x1[:,:,i,:].squeeze()
        #     z = F.conv1d(x,  weight, groups=B*C)
        # N, R, V, V
        # x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        # x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return z, loss, ridge

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    def prodection(self, z, x):
        V, C, B, T = z.shape
        z = z.permute(2,0,1,3).reshape(B,V, C*T)
        x = x[:,:,self.time_step-1:]
        x = x[:,:,::self.stride]
        x = x.permute(1,0,2).reshape(B, V, C*T)
        loss = self.loss_fn(z,x)
        return loss
        # ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        # smooth = loss + ridge
    def ridge_regularize(self, x1):
        '''Apply ridge penalty at all subsequent layers.'''
        B,V,V,T = x1.shape
        x1 = x1.reshape(B,-1)
        ridge = torch.sum(x1**2,dim=-1)
        return ridge

class unit_gcgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride):

        super(unit_gcgcn, self).__init__()
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels

        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()

        for i in range(self.num_subset):
            self.convs.append(GCGC(in_channels, out_channels, stride))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        
        if stride !=1:
            self.down = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

        self.A = nn.Parameter(A.clone())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = None
        losses = []
        ridges = []

        for i in range(self.num_subset):
            z, loss,ridge = self.convs[i](x, self.A[i], self.alpha)
            y = z + y if y is not None else z
            losses.append(loss)
            ridges.append(ridge)


        y = self.bn(y)
        y = y + self.down(x)
        return self.relu(y), losses, ridges

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

class gc_gcn(nn.Module): #GC channel wise
    def __init__(self, 
                in_channels, 
                out_channels, 
                stride=1, 
                rel_reduction=8, 
                ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4)],
                time_step=9):
        super(gc_gcn, self).__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if in_channels <= 16:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction
        self.time_step = time_step
        self.stride = stride

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        
        branches_1 = []
        branches_2 = []

        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch_1 = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), 
                nn.BatchNorm2d(branch_c), 
                self.act,
                nn.Conv1d(branch_c, branch_c, kernel_size=cfg[0],stride=stride,dilation=cfg[1]),
                nn.Conv2d(branch_c, branch_c, kernel_size=1))
            branch_2 = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), 
                nn.BatchNorm2d(branch_c), 
                self.act,
                nn.Conv1d(branch_c, branch_c, kernel_size=cfg[0],stride=stride,dilation=cfg[1]),
                nn.Conv2d(branch_c, branch_c, kernel_size=1))
                # unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1], norm=None))
            branches_1.append(branch_1)
            branches_2.append(branch_2)

        self.branches_1 = nn.ModuleList(branches_1)
        self.branches_2 = nn.ModuleList(branches_2)
        
        self.tanh = nn.Tanh()
        self.init_weights()
        self.rel_reduction = rel_reduction

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        B, C, T, V = x.shape
        for temconv1, temconv2 in self.branches_1, self.branches_2:
            out1 = temconv1(x)
            out2 = temconv1(x)
        x1, x2 = self.conv1(x).mean(-2), self.conv2(x).mean(-2)
        # X1, X2: N, R, V
        # N, R, V, 1 - N, R, 1, V
        x = x.permute(0,1,3,2).reshape(-1,T).unsqueeze(0) # BCV T
        # x = x.reshape(-1, T, V).permute(0,2,1)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)
        x1 = x1.reshape(B, -1, self.time_step, V, V).permute(0,1,3,4,2).reshape(-1, V, V, self.time_step)
        x = F.pad(x,(self.time_step-1,0))
        z= [F.conv1d(x,  x1[:,:,i,:].squeeze(), groups=B*C, stride = self.stride) for i in range(V)]
        z= torch.stack(z) # 
        T = z.shape[-1]
        z = z.squeeze().permute(1,2,0).reshape(B,C,T,V)
        z = self.conv3(z)
        # for i in range(self.rel_reduction):
        #     weight = x1[:,:,i,:].squeeze()
        #     z = F.conv1d(x,  weight, groups=B*C)
        # N, R, V, V
        # x1 = self.conv4(x1) * alpha + (A[None, None] if A is not None else 0)  # N,C,V,V
        # x1 = torch.einsum('ncuv,nctu->nctv', x1, x3)
        return z

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    
    # def gc_gcn(self, weight, )

class gc_sparse(nn.Module): #GC channel wise
    def __init__(self, 
                in_channels, 
                mid_channels,  
                feature_hidden= [10, 100, 10, 1],
                causal_hidden = [100],
                ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4)],
                time_serious=25,
                stride=1):
        super(gc_sparse, self).__init__()
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        out_channels = mid_channels*num_branches
        self.out_channels = out_channels
        self.act = nn.ReLU()

        self.time_step = time_serious
        self.stride = stride

        self.loss_fn = nn.MSELoss(reduction='none')

        # if mid_channels is None:
        #     mid_channels = out_channels // num_branches
        #     rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        # else:
        #     assert isinstance(mid_channels, float) and mid_channels > 0
        #     mid_channels = int(out_channels * mid_channels)
        #     rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        # self.rem_mid_channels = rem_mid_channels
        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch = nn.ModuleList([nn.Conv1d(time_serious, mid_channels, kernel_size=cfg[0],stride=stride,dilation=cfg[1]) for _ in range(time_serious)])
            branches.append(branch)
        
        self.branches = nn.ModuleList(branches)
       

        self.branches_follow = nn.ModuleList([nn.Conv1d(out_channels, 1, kernel_size=1) for _ in range(time_serious)])
        
        feature_branches = []
        for i, out_channel in enumerate(feature_hidden):
            if i==0:
                branch = nn.Sequential(nn.Conv2d(in_channels, out_channel, kernel_size=1),
                                       nn.BatchNorm2d(out_channel), 
                                       self.act
                                       ) 
            else:
                branch = nn.Sequential(nn.Conv2d(feature_hidden[i-1], out_channel, kernel_size=1),
                                       nn.BatchNorm2d(out_channel), 
                                       self.act
                                       ) 
            feature_branches.append(branch)
        self.feature_branches = nn.ModuleList(feature_branches)
        
        self.pool_init()

        Causal_branches = []
        for i, out_channel in enumerate(causal_hidden):
            if i==0:
                branch = nn.Sequential(nn.Conv1d(time_serious, out_channel, kernel_size=1),
                                       nn.BatchNorm1d(out_channel), 
                                       self.act)
            else:
                branch = nn.Sequential(nn.Conv1d(causal_hidden[i-1], out_channel, kernel_size=1),
                                       nn.BatchNorm1d(out_channel), 
                                       self.act)
            Causal_branches.append(branch)

        Causal_branches.append(nn.Sequential(nn.Conv1d(causal_hidden[-1], out_channels, kernel_size=1),nn.BatchNorm1d(out_channels), self.act))
        self.Causal_branches = nn.ModuleList(Causal_branches)

    def forward(self, x, A=None, alpha=1):
        # Input: N, C, T, V
        B, C, T, V = x.shape
        x_temp = x

        #feature update
        # x_temp = x
        # for feature_conv in self.feature_branches:
        #     x_temp = feature_conv(x_temp)
        
        # Granger causal aggregatation
        x_temp = x_temp.reshape(-1, T, V).permute(0,2,1)
        predic = []
        for i, networks in enumerate(self.branches):
            cfg = self.ms_cfg[i]
            pad = (cfg[0] + (cfg[0] - 1) * (cfg[1] - 1) - 1) 
            x_temp1 = F.pad(x_temp,(pad,0))
            tempx = torch.cat([network(x_temp1) for network in networks], dim=2)
            predic.append(tempx)
        predic = torch.stack(predic)

        n, B, c, _ = predic.shape

        predic = predic.permute(1,0,2,3).reshape(B,n*c,-1).reshape(B,-1, T,V).reshape(int(B/C),C,-1,T,V)
        
        # granger causal SE selection
        x_temp2 = x
        for feature_conv in self.feature_branches:
            x_temp2 = feature_conv(x_temp2)
        x_temp2= x_temp2.reshape(-1, T, V).permute(0,2,1)
        x_temp2 = x_temp2.mean(-1, keepdim=True)
        for i, causal_net in enumerate(self.Causal_branches):
            x_temp2 = causal_net(x_temp2)
        x_temp2 = x_temp2.unsqueeze(-1)
        
        # causal selection
        predic = predic*x_temp2.unsqueeze(1)
        
        # channel reduction
        predic = predic.reshape(B,-1, T, V)
        predic = self.act(predic)    
        predic_final = []
        for i, network in enumerate(self.branches_follow):
             predic_tem = predic[:,:,:,i].squeeze()
             predic_tem = network(predic_tem)
             predic_final.append(predic_tem)
        predic_final = torch.stack(predic_final).squeeze().permute(1,0,2)
        
        # Granger loss
        predic_loss = self.prediction(predic_final, x_temp)
        
        # x_temp2 = x_temp.mean(-1, keepdim=True)
        # for i, causal_net in enumerate(self.Causal_branches):
        #     x_temp2 = causal_net(x_temp2)
        # x_temp2 = x_temp2.unsqueeze(-1)

        # Granger causality matrix
        gc_pool = self.GC_pool()
        gc = gc_pool.unsqueeze(0) * x_temp2.unsqueeze(-1)

        regulize = self.regularize(gc, 1e-2,penalty='GSGL')

        # gc = torch.norm(gc, dim=(1, -1))
        # gc = gc.max(1)
        gc = torch.max(gc,dim=1)[0]
        gc = torch.max(gc,dim=-1)[0]

        return predic_loss, gc, regulize



    def prediction(self, predic, x_temp):
        B,V,T = predic.shape
        x = x_temp[:,:,1:]
        z = predic[:,:,0:-1]
        loss = self.loss_fn(z,x)
        return loss

    def pool_init(self):
        for network in self.branches:
            for net in network:
                net.weight.is_pool = True
                net.bias.is_pool = True
        for net in self.branches_follow:
            net.weight.is_pool = True
            net.bias.is_pool = True
        # for network in self.feature_branches:
        #     for net in network:
        #         if isinstance(net, nn.Conv2d):
        #             net.weight.is_pool = True
        #             net.bias.is_pool = True
        #         if isinstance(net,nn.BatchNorm2d):
        #              net.weight.is_pool = True
        #              net.bias.is_pool = True


            
    def GC_pool(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        # for net in self.branches:
        #     for name, parameters  in net.named_parameters():
        #             print(name, parameters.size())
        #         #  print(torch.norm(parameters, dim=(0, 2)))
        GCs = []
        for network in self.branches:
            GC = [net.weight for net in network]
            GC = torch.stack(GC)
            GCs.append(GC)

        GCs = torch.stack(GCs)
        n,v,c,_,s = GCs.shape
        GCs = GCs.permute(0,2,1,3,4).reshape(n*c,v,v,s)
        return GCs
    
    def regularize(self, gc, lam, penalty):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W = gc
        # W = network.layers[0].weight
        b,hidden, p, p1, lag = W.shape
        if penalty == 'GL':
            return lam * torch.sum(torch.norm(W, dim=(1, -1)))
        elif penalty == 'GSGL':
            return lam * (torch.sum(torch.norm(W, dim=(1, -1)))
                        + torch.sum(torch.norm(W, dim=1)))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 1)))
                            for i in range(lag)])
        else:
            raise ValueError('unsupported penalty: %s' % penalty)


    def ridge_regularize(self, lam=1e-2):
        '''Apply ridge penalty at all subsequent layers.'''

        return lam * sum([torch.sum(fc.weight ** 2) for fc in self.branches_follow])
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
            
class gc_component(nn.Module): #GC channel wise
    def __init__(self, 
                in_channels=3, 
                causal_channel=100, 
                feature_update=[64,128,1],
                feature_hidden=[100,10,1],
                time_len=9,
                time_serious=25,
                bias: bool = True,
                init_mode = 'kaiming_uniform',
                init_scale = 1.0):
        super(gc_component, self).__init__()
        self.in_ch = in_channels
        self.feature_update = feature_update
        self.causal_channel = causal_channel
        self.feature_hidden = feature_hidden
        self.time_len = time_len
        self.time_serious = time_serious
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.bias_flag = bias

        self.function = F.conv1d
        self.act = nn.ReLU()

        self.tanh = nn.Tanh()

        self.loss_fn = nn.MSELoss(reduction='mean')
        

        ## VAR parameters
        self.weight = nn.Parameter(torch.ones(self.causal_channel,self.time_serious,self.time_len))
        self.weight.is_pool = True
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)
        with torch.no_grad():
            self.weight_norm = torch.norm(self.weight,dim=0).detach()

        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(self.causal_channel))
        else:
            self.bias = None
        ## feature update before VAR
        if self.feature_update is not None:
            feature_branches = []
            for i, out_channel in enumerate(self.feature_update):
                if i==0:
                    branch = nn.Sequential(nn.Conv2d(in_channels, out_channel, kernel_size=1),
                                        nn.BatchNorm2d(out_channel),
                                        # self.act
                                        ) 
                else:
                    branch = nn.Sequential(nn.Conv2d(feature_update[i-1], out_channel, kernel_size=1),
                                        nn.BatchNorm2d(out_channel)
                                        # self.act
                                        ) 
                feature_branches.append(branch)
            self.feature_branches = nn.ModuleList(feature_branches)

            self.conv1 = nn.Conv2d(self.feature_update[-1], self.causal_channel, kernel_size=1)
            self.conv2 = nn.Conv2d(self.feature_update[-1], self.causal_channel, kernel_size=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, causal_channel, kernel_size=1)
            self.conv2 = nn.Conv2d(in_channels, causal_channel, kernel_size=1)

        ## feature aggragation after VAR
        follow_branches = []
        for i, out_channel in enumerate(self.feature_hidden):
            if i==0:
                # weight = nn.Parameter(torch.ones(out_channel,causal_channel,1))
                branch = nn.Conv1d(time_serious*causal_channel, time_serious*out_channel, kernel_size=1, groups=time_serious)
                                       
            else:
                # weight = nn.Parameter(torch.ones(out_channel,feature_hidden[i-1],1))
                branch = nn.Conv1d(time_serious*feature_hidden[i-1], time_serious*out_channel, kernel_size=1, groups=time_serious)
                                       
            # follow_branches.append(weight)
            follow_branches.append(branch)
        self.follow_branches = nn.ModuleList(follow_branches)

    def forward(self, x):
        B, C, T, V = x.shape
        x_temp = x
 
        ## feature update
        if self.feature_update is not None:
            for feature_conv in self.feature_branches:
                x_temp = self.act(feature_conv(x_temp))
            x1 = self.conv1(x_temp).mean(-2)
            x2 = self.conv2(x_temp).mean(-2)
        else:
            x_temp = x.mean(1,keepdim=True)
            x1 = self.conv1(x).mean(-2)
            x2 = self.conv2(x).mean(-2)
        causal_matrix = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        # Granger causal aggregatation
        x_temp = x_temp.reshape(-1, T, V).permute(0,2,1).reshape(-1,T).unsqueeze(0)
        predic = []
        for i in range(self.time_serious):
            index = causal_matrix[:,:,:,i].unsqueeze(-1)
            weight_gene = self.weight.unsqueeze(0)
            weight_gene = weight_gene/self.weight_norm.to(self.weight.device).unsqueeze(0).unsqueeze(0)
            weight_refine = index * weight_gene
            weight_refine = weight_refine.reshape(-1,self.time_serious,self.time_len)
            ret = self.function(x_temp, weight_refine, groups=index.shape[0])
            # for j, weight in enumerate(self.follow_branches):
            #     weight_follow = weight.unsqueeze(0)
            #     weight_refine = index * weight_follow
            #     weight_refine = weight_refine.reshape(-1,self.time_serious,1)

            predic.append(ret)
        predics = torch.stack(predic)
        predics = predics.reshape(V,B,self.causal_channel,-1).permute(1,0,2,3).reshape(B,V*self.causal_channel,-1)
        # predics = predics.reshape(V,B,self.causal_channel,-1).reshape(V*B,self.causal_channel,-1)
        for j, conv_f in enumerate(self.follow_branches):
            predics = self.act(conv_f(predics))
        
        
        panelty = self.regularize(causal_matrix,1e-4,'GSGL')
        regularize = self.ridge_regularize(lam=1e-4)
        predic_loss = self.predic_loss(predics,x_temp)
        # print(panelty,regularize,predic_loss)
        GCs = self.GC(causal_matrix)
        # prediction = predics.reshape(V,B,self.feature_hidden[-1],-1).permute(1,2,3,0)
        # x_comp = x_temp.reshape(-1,B,V,T).permute(1,0,3,2)
        return GCs, predic_loss, panelty, regularize

    def GC(self, causal_matrix, ignore_lag=True, threshold=False):
        B,C,N,M = causal_matrix.shape
        causal_matrix = causal_matrix.unsqueeze(-1)
        GC = causal_matrix*self.weight.unsqueeze(-2).unsqueeze(0)

        if ignore_lag:
            GC = torch.norm(torch.norm(GC, dim=(1)),dim=(-1))
        else:
            GC = torch.norm(GC, dim=(1))
        # GCs = GC.reshape(B,-1)
        if threshold:
            return (GC > 0).int()
        else:
            return GC

    def predic_loss(self, prediction, x_comp):
        B,V,_ = prediction.shape
        prediction = prediction
        x_comp = x_comp.reshape(B,V, -1)
        loss = self.loss_fn(x_comp[:,:,self.time_len:],prediction[:,:,:-1])

        return loss

    def regularize(self, W, lam, penalty):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        b, hidden, p, p1 = W.shape
        if penalty == 'GL':
            return lam * torch.sum(torch.norm(torch.norm(W, dim=(1)),dim=1))
        elif penalty == 'GSGL':
            return lam * (torch.sum(torch.norm(torch.norm(W, dim=(1)),dim=1))
                        + torch.sum(torch.norm(W, dim=1)))
        # elif penalty == 'H':
        #     # Lowest indices along third axis touch most lagged values.
        #     return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 1)))
        #                     for i in range(lag)])
        else:
            raise ValueError('unsupported penalty: %s' % penalty)


    def ridge_regularize(self, lam=1e-3):
        '''Apply ridge penalty at all subsequent layers.'''
        return lam * sum([torch.sum(fc.weight ** 2) for fc in self.follow_branches])

    def init_param_(self, param, init_mode=None, scale=None):
        if init_mode == 'kaiming_normal':
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            param.data *= scale
        elif init_mode == 'uniform':
            nn.init.uniform_(param, a=-1, b=1)
            param.data *= scale
        elif init_mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            param.data *= scale
        elif init_mode == 'signed_constant':
            # From github.com/allenai/hidden-networks
            fan = nn.init._calculate_correct_fan(param, 'fan_in')
            gain = nn.init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            nn.init.kaiming_normal_(param)    # use only its sign
            param.data = param.data.sign() * std
            param.data *= scale
        else:
            raise NotImplementedError   
    def pool_init(self):
        if self.feature_update is not None:
            for network in self.feature_branches:
                for net in network:
                    net.weight.is_pool = False
                    net.bias.is_pool = False

        self.conv1.weight.is_pool = False
        self.conv1.bias.is_pool = False
        self.conv2.weight.is_pool = False
        self.conv2.bias.is_pool = False

        for net in self.follow_branches:
            net.weight.is_pool = True
            net.bias.is_pool = True


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)

class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC