from xml.etree.ElementInclude import include
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer

from .init_func import bn_init, conv_init
from torch.nn import functional as F

from .sparse_mosules import SparseConv2d


class unit_tcn_sparse(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, conv_sparsity=0, stride=1, dilation=1, norm='BN', dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = SparseConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            conv_sparsity=conv_sparsity,
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = build_norm_layer(self.norm_cfg, out_channels)[1] if norm is not None else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x, sparsity=0):
        return self.drop(self.bn(self.conv(x, sparsity)))

    def init_weights(self):
        pass
        # conv_init(self.conv)
        # bn_init(self.bn, 1)

class mstcn_sparse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 sparse_ratio=0,
                 mid_channels=None,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU()

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(SparseConv2d(in_channels, branch_c, kernel_size=1, conv_sparsity=sparse_ratio, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        SparseConv2d(in_channels, branch_c, kernel_size=1, conv_sparsity=sparse_ratio),
                        # nn.Conv2d(in_channels, branch_c, kernel_size=1), 
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                SparseConv2d(in_channels, branch_c, kernel_size=1,conv_sparsity=sparse_ratio),
                # nn.Conv2d(in_channels, branch_c, kernel_size=1), 
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn_sparse(branch_c, branch_c, kernel_size=cfg[0],conv_sparsity=sparse_ratio,stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, 
            SparseConv2d(tin_channels, out_channels, kernel_size=1,conv_sparsity=sparse_ratio),
            # nn.Conv2d(tin_channels, out_channels, kernel_size=1)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x, sparsity=0):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            try:
                len(tempconv)
            except:
                try:
                    index = tempconv(x,sparsity)
                except:
                    res = tempconv(x)
                else:
                    res = tempconv(x,sparsity)
            else:
                res = x
                for item in tempconv:
                    try:
                        index = item(res,sparsity)
                    except:
                        res= item(res)
                    else:
                        res = item(res,sparsity)
            branch_outs.append(res)

        feat = torch.cat(branch_outs, dim=1)
        for item in self.transform:
            try:
                index = item(feat,sparsity)
            except:
                feat= item(feat)
            else:
                feat = item(feat,sparsity)
        # feat = self.transform(feat, sparsity)
        return feat

    def forward(self, x, sparsity=0):
        out = self.inner_forward(x, sparsity)
        out = self.bn(out)
        return self.drop(out)

    def init_weights(self):
        pass

class dgmstcn_sparse(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 sparse_ratio=0,
                 mid_channels=None,
                 num_joints=25,
                 dropout=0.,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act = nn.ReLU()
        self.num_joints = num_joints
        # the size of add_coeff can be smaller than the actual num_joints
        self.add_coeff = nn.Parameter(torch.zeros(self.num_joints))

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(
                    SparseConv2d(in_channels, branch_c, kernel_size=1,conv_sparsity=sparse_ratio, stride=(stride, 1)),
                    # nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1))
                    )
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        SparseConv2d(in_channels, branch_c, kernel_size=1,conv_sparsity=sparse_ratio),
                        # nn.Conv2d(in_channels, branch_c, kernel_size=1), 
                        nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                SparseConv2d(in_channels, branch_c, kernel_size=1,conv_sparsity=sparse_ratio),
                # nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                unit_tcn_sparse(branch_c, branch_c, kernel_size=cfg[0], conv_sparsity=sparse_ratio, stride=stride, dilation=cfg[1], norm=None))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, 
            # nn.Conv2d(tin_channels, out_channels, kernel_size=1),
            SparseConv2d(tin_channels, out_channels, kernel_size=1, conv_sparsity=sparse_ratio)
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def inner_forward(self, x):
        N, C, T, V = x.shape
        x = torch.cat([x, x.mean(-1, keepdim=True)], -1)

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        local_feat = out[..., :V]
        global_feat = out[..., V]
        global_feat = torch.einsum('nct,v->nctv', global_feat, self.add_coeff[:V])
        feat = local_feat + global_feat

        feat = self.transform(feat)
        return feat

    def forward(self, x):
        out = self.inner_forward(x)
        out = self.bn(out)
        return self.drop(out)
    
    def init_weights(self):
        pass
