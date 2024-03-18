from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple


EPSILON = np.finfo(float).eps
def concrete_neuron(logit_p, train=False, temp=1.0 , **kwargs):
    '''
    Use concrete distribution to approximate binary output. Here input is logit(keep_prob).
    '''
    if train is False:
        result = logit_p.data.new().resize_as_(logit_p.data).fill_(1.)
        result[logit_p.data < 0.] = 0.
        return result

    # Note that p is the retain probability here
    p = torch.sigmoid(logit_p)
    unif_noise = logit_p.data.new().resize_as_(logit_p.data).uniform_()

    approx = (
        torch.log(1. - p + EPSILON)
        - torch.log(p + EPSILON)
        + torch.log(unif_noise + EPSILON)
        - torch.log(1. - unif_noise + EPSILON)
    )
    drop_prob = torch.sigmoid(approx / temp)
    keep_prob=(1. - drop_prob)
    mask=keep_prob.clone().detach()>0.5
    #mask=p.clone().detach()>0.5
    mask=mask.float()
    out=mask-keep_prob.detach()+keep_prob
    return out


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, threshold, zeros, ones):
        #k_val = percentile(scores, sparsity*100)
        #if glob:
        out = torch.where(scores < threshold, zeros.to(scores.device), ones.to(scores.device))
        #else:
        #    k_val = percentile(scores, threshold*100)
        #    out = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))            
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class SparseModule(nn.Module):
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

    def rerandomize_(self, param, mask, mode=None, la=None, mu=None,
                     init_mode=None, scale=None, param_twin=None,param_score=None):
        if param_twin is None:
            raise NotImplementedError
        else:
            param_twin = param_twin.to(param.device)

        with torch.no_grad():
            if mode == 'bernoulli':
                assert (la is not None) and (mu is None)
                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                ones = torch.ones(param.size()).to(param.device)
                b = torch.bernoulli(ones * la)
                

                t1 = param.data * mask
                t2 = param.data * (1 - mask) * (1 - b)
                t3 = rnd.data * (1 - mask) * b

                param.data = t1 + t2 + t3
                #for score
                #b1=torch.bernoulli(ones*la*0.5)
                #param_score.data=param_score.data*mask
            elif mode == 'manual':
                assert (la is not None) and (mu is not None)

                t1 = param.data * (1 - mask)
                t2 = param.data * mask

                rnd = param_twin
                self.init_param_(rnd, init_mode=init_mode, scale=scale)
                rnd *= (1 - mask)

                param.data = (t1*la + rnd.data*mu) + t2
            else:
                raise NotImplementedError


class SparseConv2d(SparseModule):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size: _size_2_t,
                conv_sparsity = 0,
                stride:_size_2_t = 1,
                padding: Union[str, _size_2_t] = 0,
                dilation: _size_2_t = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',  
                init_mode = 'kaiming_uniform',
                init_mode_mask = 'kaiming_uniform',
                init_scale = 1.0,
                init_scale_score = 1.0,
                rerand_rate=0.5,
                device=None,
                dtype=None):
        super().__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        # if kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.group = groups
        self.bias_flag = bias
        self.padding_mode = padding_mode

        self.sparsity = conv_sparsity
        self.init_mode = init_mode
        self.init_mode_mask = init_mode_mask

        self.init_scale = init_scale
        self.init_scale_score = init_scale_score
        self.rerand_rate = rerand_rate
        self.function = F.conv2d

        self.initialize_weights(2)

    def initialize_weights(self, convdim=None):
        if convdim == 1:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size))
        elif convdim == 2:
            self.weight = nn.Parameter(torch.ones(self.out_ch, self.in_ch, self.kernel_size[0], self.kernel_size[1]))

        else:
            raise NotImplementedError
        self.weight.is_mask=True
        
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        if self.bias_flag:
            self.bias = nn.Parameter(torch.zeros(self.out_ch))
        else:
            self.bias = None

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

    def get_subnet(self, weight_score=None, threshold=0):
        if weight_score is None:
            weight_score = self.weight_score
        
        subnet = GetSubnet.apply(self.weight_score, threshold,
                                 self.weight_zeros, self.weight_ones)
        return subnet

    def forward(self, input, threshold):
        subnet = self.get_subnet(self.weight_score, threshold)
        pruned_weight = self.weight * subnet
        ret = self.function(
            input, pruned_weight, self.bias, self.stride, 
            self.padding, self.dilation, self.group
        )
        return ret

    def rerandomize(self, mode, la, mu):
        rate = self.rerand_rate
        mask = GetSubnet.apply(self.weight_score, self.sparsity * rate,
                               self.weight_zeros, self.weight_ones)
        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin)

class SparseConv1d(SparseConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.function = F.conv1d
        self.initialize_weights(1)


class SparseLinear(SparseModule):
    def __init__(self, in_ch, out_ch, bias=False, args=None):
        super().__init__()

        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        if args.init_mode_linear is not None:
            self.init_mode = args.init_mode_linear
        else:
            self.init_mode = args.init_mode

        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(out_ch, in_ch))
        self.weight_score = nn.Parameter(torch.ones(self.weight.size()))
        #self.mask=torch.ones(list(self.weight.size())+[2]).cuda()
        #self.mask[:,:,0]=0

        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch))
        else:
            self.bias = None

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False


    def forward(self, x, threshold,manual_mask=None):
        weight_score=self.weight_score
        if self.weight_score.sparsity!=self.sparsity:
            subnet = GetSubnet.apply(weight_score, self.weight_score.sparsity,self.weight_zeros, self.weight_ones)
            pruned_weight = self.weight * subnet
        else:
            if manual_mask is None:
                subnet = GetSubnet.apply(weight_score, threshold,self.weight_zeros, self.weight_ones)
                pruned_weight = self.weight * subnet
            else:
                pruned_weight = self.weight * manual_mask

        ret = F.linear(x, pruned_weight, self.bias)
        return ret

    def rerandomize(self, mode, la, mu, sparsity, manual_mask=None):
        if sparsity ==None:
            sparsity=self.sparsity
        if manual_mask is None:
            rate = self.rerand_rate
            mask = GetSubnet.apply(torch.sigmoid(self.weight_score), sparsity * rate,self.weight_zeros, self.weight_ones)

        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin,param_score=self.weight_score)


class SparseParameter(SparseModule):
    def __init__(self, heads, out_channels, args=None):
        super().__init__()

        if args.linear_sparsity is not None:
            self.sparsity = args.linear_sparsity
        else:
            self.sparsity = args.conv_sparsity

        if args.init_mode_linear is not None:
            self.init_mode = args.init_mode_linear
        else:
            self.init_mode = args.init_mode

        self.init_mode_mask = args.init_mode_mask
        self.init_scale = args.init_scale
        self.init_scale_score = args.init_scale_score

        self.weight = nn.Parameter(torch.ones(1,heads, out_channels))
        self.weight_score = nn.Parameter(torch.ones((1,heads,out_channels)))
        self.weight_score.is_score = True
        self.weight_score.sparsity = self.sparsity
        #self.mask=torch.ones(list(self.weight.size())+[2]).cuda()

        self.weight_twin = torch.zeros(self.weight.size())
        self.weight_twin.requires_grad = False

        self.init_param_(self.weight_score, init_mode=self.init_mode_mask, scale=self.init_scale_score)
        self.init_param_(self.weight, init_mode=self.init_mode, scale=self.init_scale)

        self.weight_zeros = torch.zeros(self.weight_score.size())
        self.weight_ones = torch.ones(self.weight_score.size())
        self.weight_zeros.requires_grad = False
        self.weight_ones.requires_grad = False

    def forward(self, threshold,manual_mask=None):
        weight_score=self.weight_score
        if manual_mask is None:
            subnet = GetSubnet.apply(weight_score, threshold,self.weight_zeros, self.weight_ones)
            pruned_weight = self.weight * subnet
        else:
            pruned_weight = self.weight * manual_mask
        
        return pruned_weight

    def rerandomize(self, mode, la, mu,sparsity, manual_mask=None):
        if sparsity==None:
            sparsity=self.sparsity
        
        if manual_mask is None:
            rate = self.rerand_rate
            mask = GetSubnet.apply(torch.sigmoid(self.weight_score), sparsity * rate,self.weight_zeros, self.weight_ones)
        else:
            mask = manual_mask

        scale = self.init_scale
        self.rerandomize_(self.weight, mask, mode, la, mu,
                self.init_mode, scale, self.weight_twin,param_score=self.weight_score)
