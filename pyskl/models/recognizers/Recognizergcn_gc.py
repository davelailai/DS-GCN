import numpy as np
# from sklearn.metrics import completeness_score
import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
import torch.nn as nn
import torch.nn.functional as F
# import sys 
# sys.path.append("/home/jbridge/Jianyang/pyskl/pyskl/utils") 
# from visualize import Vis3DPose



@RECOGNIZERS.register_module()
class RecognizerGCN_GC(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        # A = kwargs['causal']
        keypoint = keypoint[:, 0]
        node_type = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2])
        losses = dict()
        # predic_loss, gc, regulize = self.extract_feat(keypoint)
        GCs, predic_loss, panelty, regularize = self.extract_feat(keypoint)
        # mask = torch.zeros_like(predic_loss)
        # total_frame = kwargs['total_frames']
        # clip_len = kwargs['clip_len']
        # for i, index in enumerate(total_frame):
        #     if index<clip_len[i]:
        #         mask[6*i:6*(i+1)-1,:,0:index-1]=1
        # predic_loss = torch.mean(predic_loss*mask)
        # loss_node = self.neck.node_precost(self, x, node_type)
        if self.with_neck:
            # loss_node = self.neck.node_precost(x, node_type)
            x = self.neck(x)
        cls_score = self.cls_head(GCs)
        gt_label = label.squeeze(-1)
        
        loss = self.cls_head.loss(cls_score, gt_label)
        # loss['node_loss']=loss_node
        # loss = []
        # loss.pop('loss_cls')
        loss['predic_loss'] = predic_loss
        loss['panelty_loss'] = panelty
        loss['ridge_loss'] = regularize
        # loss['regulize_loss'] = regulize

       
        # stucture_loss = self.structure_similarity(x, gt_label)
        # loss = self.cls_head.loss(cls_score, gt_label,x)
        # loss['simarity_loss'] =stucture_loss
        losses.update(loss)

        return losses

    def forward_test(self, keypoint, **kwargs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert self.with_cls_head or self.feat_ext
        bs, nc = keypoint.shape[:2]
        keypoint = keypoint.reshape((bs * nc, ) + keypoint.shape[2:])
        # vis = Vis3DPose()
        # vis.vis

        x, _, _ = self.extract_feat(keypoint)
        if self.with_neck:
            x = self.neck(x)
        feat_ext = self.test_cfg.get('feat_ext', False)
        pool_opt = self.test_cfg.get('pool_opt', 'all')
        score_ext = self.test_cfg.get('score_ext', False)
        if feat_ext or score_ext:
            assert bs == 1
            assert isinstance(pool_opt, str)
            dim_idx = dict(n=0, m=1, t=3, v=4)

            if pool_opt == 'all':
                pool_opt == 'nmtv'
            if pool_opt != 'none':
                for digit in pool_opt:
                    assert digit in dim_idx

            if isinstance(x, tuple) or isinstance(x, list):
                x = torch.cat(x, dim=2)
            assert len(x.shape) == 5, 'The shape is N, M, C, T, V'
            if pool_opt != 'none':
                for d in pool_opt:
                    x = x.mean(dim_idx[d], keepdim=True)

            if score_ext:
                w = self.cls_head.fc_cls.weight
                b = self.cls_head.fc_cls.bias
                x = torch.einsum('nmctv,oc->nmotv', x, w)
                if b is not None:
                    x = x + b[..., None, None]
                x = x[None]
            return x.data.cpu().numpy().astype(np.float16)

        
        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(bs, nc, cls_score.shape[-1])
        # harmless patch
        if 'average_clips' not in self.test_cfg:
            self.test_cfg['average_clips'] = 'prob'

        cls_score = self.average_clip(cls_score)
        if isinstance(cls_score, tuple) or isinstance(cls_score, list):
            cls_score = [x.data.cpu().numpy() for x in cls_score]
            return [[x[i] for x in cls_score] for i in range(bs)]

        return cls_score.data.cpu().numpy()

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        return self.backbone(keypoint)
    
    def structure_similarity(self, x, gt_label):
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        x = F.normalize(x)  
        structure_similarity = torch.mm(x, x.transpose(1,0))
        label_similarity =torch.zeros(N,N).cuda()

        for i in range(N):
            for j in range(N):
               label_similarity[i,j]= gt_label[i]==gt_label[j] 

        structure_loss = torch.mean((structure_similarity-label_similarity)**2)
        return structure_loss
        
