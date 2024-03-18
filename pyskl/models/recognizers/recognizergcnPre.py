import copy
import numpy as np
# from sklearn.metrics import completeness_score
import torch
from yaml import KeyToken

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
import torch.nn as nn
import torch.nn.functional as F
import random
# import sys 
# sys.path.append("/home/jbridge/Jianyang/pyskl/pyskl/utils") 
# from visualize import Vis3DPose



@RECOGNIZERS.register_module()
class RecognizerGCNPre(BaseRecognizer):
    """GCN-based recognizer for skeleton-based action recognition. """

    def forward_train(self, keypoint, label, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1
        keypoint = keypoint[:, 0]
        
        # NCE loss within a graph
        keypoint_mask = copy.deepcopy(keypoint)
        N, M, T, V, C = keypoint_mask.shape
        sample_size = int(0.5 * V)
        node_type = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2])
        # keypoint_mask = keypoint_mask.reshape(N*M,T*V,C)
        mask_node_index = torch.stack([torch.LongTensor(random.sample(range(V), sample_size)) for i in range(N*M)], dim=0).to(keypoint_mask.device)
        mask = torch.ones(N*M, V).to(keypoint_mask.device)
        mask.scatter_(dim=1, index=mask_node_index, value=0.0)
        mask = mask.unsqueeze(1).unsqueeze(-1).repeat(1,T,1,1).reshape(N, M, T, V, -1)
        keypoint_mask = keypoint_mask*mask
        keypoint_mask[keypoint_mask==0]=1.0

        # select = keypoint_mask.sum(1)
        # select = select.sum(1)

        # node_num = T * V
        # sample_size = int(0.3 * node_num)
        # mask_node_index = torch.stack([torch.LongTensor(random.sample(range(T*V), sample_size)) for i in range(N*M)], dim=0).to(keypoint_mask.device)
        # for i in range(N*M):
        #     keypoint_mask[i,mask_node_index[i,:],:] = 0

        # keypoint_mask = keypoint_mask.reshape(N,M,T,V,C)

        x_modify = self.extract_feat(keypoint_mask) # N,M,C,T,V
        loss = dict()
        
        # loss['loss_cls'] = loss['node_loss']
        x = self.extract_feat(keypoint)
        
        # loss['node_loss'] = self.neck.get_intracost(x, x_modify)
        # if kwargs['optimizer_current']=='module':
        #     # x = self.extract_feat(keypoint)
        #     fea_agg = self.neck(x)
        #     cls_score = self.cls_head(fea_agg)
        #     gt_label = label.squeeze(-1)
            
        #     loss = self.cls_head.loss(cls_score, gt_label)
        # else:
        #     loss = dict() 

        # _, loss['neck_loss'] = self.neck.get_aligncost(x.clone())
            
        loss['node_loss'] = self.neck.node_precost(x_modify, node_type, mask)
        loss['graph_loss'] = self.neck.get_intercost(x, x_modify)
        # stucture_loss = self.structure_similarity(x, gt_label)
        # loss['simarity_loss'] =stucture_loss
        loss['loss_cls'] = loss['graph_loss'] + loss['node_loss']
        losses = dict()
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

        x = self.extract_feat(keypoint)
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
        
        x = self.neck(x)
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
        
