# Copyright (c) OpenMMLab. All rights reserved.
from unittest import result
import numpy as np
from scipy.stats import mode as get_mode

from ..builder import PIPELINES
from .compose import Compose
from .formatting import Rename

from mmdet.models import SparseRCNN
from mmdet.models import ResNet
import os
import numpy as np
import pickle
from mmdet.apis import init_detector, inference_detector
import mmcv
from mmcv import Config

EPS = 1e-4

@PIPELINES.register_module()
class ObjectDetection:
    """Normalize the range of keypoint values. """

    def __init__(self, video_file, label_map):
        self.video_file = video_file
        self.label_map = label_map
        self.config_file = 'configs/detection/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
        self.model_checkpoint = 'sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20201223_024605-9fe92701.pth'
        self.model = init_detector(self.config_file, self.model_checkpoint)
    def __call__(self, results): 
        with open(self.label_map, "r") as f:  
            label_map = f.readlines()
        index_fram = results['frame_dir']
        label = results['label']
        class_folder = label_map[label].strip('\n')
        video_path = os.path.join(self.video_file, class_folder,index_fram+'.avi')
        video = mmcv.VideoReader(video_path)
        W_,H_ = results['original_shape']
        results['bboxes'] = []
        results['box_labels'] = []
        for frame in video:
            object_result = inference_detector(self.model, frame)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(object_result)]
            labels = np.concatenate(labels)
            bboxes = np.vstack(object_result)
            w,h,_ = frame.shape
            ratio1, ratio2 = W_/w,H_/h
            bboxes[:,0]= bboxes[:,0]*ratio1
            bboxes[:,2]= bboxes[:,2]*ratio1
            bboxes[:,1]= bboxes[:,1]*ratio2
            bboxes[:,3]= bboxes[:,3]*ratio2
            scores = bboxes[:, -1]
            inds = scores > 0.3
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            results['bboxes'].append(bboxes)
            results['box_labels'].append(labels)
        return results