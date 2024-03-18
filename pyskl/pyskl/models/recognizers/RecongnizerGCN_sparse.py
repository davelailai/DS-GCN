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
class RecognizerGCN_sparse(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """
    # if self.train_cfg is None:
    #     train_cfg = dict()
    # if self.test_cfg is None:
    #         test_cfg = dict()
    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        # A = kwargs['causal']
        keypoint = keypoint[:, 0]
        node_type = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2])
        losses = dict()
        x= self.extract_feat(keypoint,**kwargs)
            # loss_node = self.neck.node_precost(self, x, node_type)
        if self.with_neck:
            if self.neck == 'SemanticNeck':
                index = x.sum(-1).sum(-1).sum(-1)
                x = self.neck(x,index)
            else:
            # loss_node = self.neck.node_precost(x, node_type)
                x = self.neck(x)
        cls_score = self.cls_head(x)
        gt_label = label.squeeze(-1)
        
        loss = self.cls_head.loss(cls_score, gt_label)

        if kwargs['current_epoch']<=self.backbone.warm_up:

            if self.train_cfg is None or self.train_cfg['panelty']==None:
                pass
            elif self.train_cfg['lam']=='gradual':
                lam_min = torch.tensor(0)
                lam_max = torch.tensor(1)
                lam = lam_min+lam_max*kwargs['current_epoch']/self.backbone.warm_up
                # lam = lam_min+torch.tensor(kwargs['current_iter']/10).floor()*2e-5
                if lam>lam_max:
                    lam = lam_max
                panelty = self.backbone.regularize(lam, self.train_cfg['panelty'], kwargs['current_epoch'], kwargs['total_epoch'])
                loss['panelty_loss']=panelty
            else:
                lam = self.train_cfg['lam']
                panelty = self.backbone.regularize(lam, self.train_cfg['panelty'], kwargs['current_epoch'], kwargs['total_epoch'])
                loss['panelty_loss']=panelty

        # loss['node_loss']=loss_node
            
       
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
        if len(self.test_cfg)==0:
            kwargs['current_epoch'] = 101
            kwargs['total_epoch'] = 100
        else:
            kwargs['current_epoch'] = self.test_cfg['current_epoch']
            kwargs['total_epoch'] = self.test_cfg['max_epoch']
        
        x = self.extract_feat(keypoint,**kwargs)
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
        # return_loss=False
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(keypoint, label, **kwargs)

        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, keypoint, **kwargs):
        """Extract features through a backbone.

        Args:
            keypoint (torch.Tensor): The input keypoints.

        Returns:
            torch.tensor: The extracted features.
        """
        current_epoch = kwargs['current_epoch']
        total_epoch = kwargs['total_epoch']
        return self.backbone(keypoint,current_epoch,total_epoch)
    
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
        
