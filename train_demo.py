# CUDA_VISIBLE_DEVICES = 1

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import time

import mmcv
import torch

import torch.distributed as dist
# from mmcv import Config
from mmengine import Config
# from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmengine.runner import set_random_seed
from mmengine.dist import get_dist_info,init_dist
from mmengine.utils import get_git_hash
# from mmcv.utils import get_git_hash

from pyskl import __version__
from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger, mc_off, mc_on, test_port



cfg = Config.fromfile('configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/j.py')
# /home/jbridge/Jianyang/mmaction2/configs/skeleton/GTGCN/GTGCN_80e_ntu60_xsub_keypoint.py
# /home/jbridge/Jianyang/mmaction2/configs/skeleton/CTRGCN/ctrgcn_80e_ntu60_xsub_keypoint.py
# /home/jbridge/Jianyang/pyskl/configs/posec3d/slowonly_r50_463_k400/joint.py

# cfg.model.cls_head['type']='ClsHead'
# # cfg.model.neck = cfg.model.cls_head5
# cfg.model.neck=dict(type='ReadoutNeck', in_channels=256, read_op='sum', num_position=5)
# cfg.model.cls_head['in_channels']=256

cfg.seed = 0
cfg.gpu_ids = range(0,1)
set_random_seed(0, deterministic=False)
cfg.setdefault('omnisource', False)
# cfg.lr_config = dict(policy='Cyclic', target_ratio=(1e5, 1e-4), cyclic_times=10)
# cfg.lr_config = dict(policy='OneCycle', max_lr=[0.5], div_factor = 100)
# cfg.lr_config = dict(policy='Cyclic', by_epoch=False, target_ratio=(50,1e-3), cyclic_times=20)

# cfg.optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0005, nesterov=True)
# cfg.model.backbone.type = 'GTGCN_var'
# cfg.optimizer = dict(
#     type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
# cfg.optimizer_config = dict(type='MultiModuleHook', grad_clip=dict(max_norm=45, norm_type=2))
# cfg.data.videos_per_gpu=16

# cfg.ann_file_train = '/home/jbridge/Jianyang/mmaction2/data/ntu/nturgb+d_skeletons_60_3d_nmtvc/xsub/train.pkl'
# cfg.ann_file_val = '/home/jbridge/Jianyang/mmaction2/data/ntu/nturgb+d_skeletons_60_3d_nmtvc/xsub/val.pkl'
# cfg.workdir = '/home/jbridge/Jianyang/mmaction2/work_dirs/stgcn_80e_ntu60_xsub_keypoint_3d/'
cfg.data.videos_per_gpu=8
cfg.work_dir = './work_dirs_test/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j'
# cfg.load_from ='/home/jbridge/Jianyang/pyskl/work_dirs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/j/epoch_80.pth'
cfg.log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])
print(f'Config:\n{cfg.pretty_text}')

cfg.workflow = [('train', 1)]
datasets = [build_dataset(cfg.data.train)]

model = build_model(cfg.model)
print(model)



mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

train_model(model, datasets, cfg, distributed=False)

