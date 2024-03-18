# Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
# import pickle
# import shutil
# import tempfile
# import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

import mmcv
from ..hooks import get_feas_by_hook, get_rep_by_hook
# from mmcv.runner import get_dist_info
# from mmcv.runner import get_fe


def single_gpu_test_feature(model: nn.Module, data_loader: DataLoader) -> list:
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    features = []
    label = []
  
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    fea_hooks = get_feas_by_hook(model)
    # for matrix in fea_hooks:
    #     original_A.append(matrix.ori_A)

    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)

        results.extend(result)

        feature = []
        original_A = []
        label.extend(data['label'])
        for matrix in fea_hooks:
            x = matrix.A
            b,n,k,t,v,c = data['keypoint'].shape
            x = x.squeeze()
            _,num_set,c_out,_,_ = x.shape
            x = torch.norm(x,dim=(2))
            # x = x.mean(2)

            # x = x.reshape(b,n,k,num_set,1,v,v)
            # x = x.mean(1).mean(1).mean(2)
            # pool = nn.AdaptiveAvgPool2d(1)
            # N, M, C, T, V = x.shape
            # x = x.reshape(N * M, C, T, V)
            # x = pool(x)
            # x = x.reshape(N, M, C)
            # x = x.mean(dim=1).cpu().numpy()
            feature.append(x)
            original_A.append(matrix.ori_A)

        original_A = torch.stack(original_A)    
        feature = torch.stack(feature)
        features.append(feature)

        # x = fea_hooks[0].fea
        # pool = nn.AdaptiveAvgPool2d(1)
        # N, M, C, T, V = x.shape
        # x = x.reshape(N * M, C, T, V)
        # x = pool(x)
        # x = x.reshape(N, M, C)
        # x = x.mean(dim=1).cpu().numpy()
        # features.extend(x)
        # # label = []
        # original_A = []

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    # features=torch.stack(features[])
    return results, features, label, original_A


