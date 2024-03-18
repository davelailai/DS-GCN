import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
# from .gread import global_add_pool, global_mean_pool, global_max_pool
# from .gread import GlobalAttention, Set2Set
from ...core import top_k_accuracy
import torch.nn.functional as F



@HEADS.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score


@HEADS.register_module()
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)


@HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HEADS.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


@HEADS.register_module()
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)


@HEADS.register_module()
class HGTHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 pose_type = 'coco',
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)
        self.node_cls = nn.Linear(self.in_c, 5)
        if pose_type == 'nturgb+d':
            self.node_label = torch.tensor([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,0,1,1,2,2])
        if pose_type == 'coco':
            self.node_label = torch.tensor([0,0,0,0,0,1,2,1,2,1,2,3,4,3,4,3,4])

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
 
        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        ## Graph Classification   
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x1 = pool(x)
        x1 = x1.reshape(N, M, C)
        x1 = x1.mean(dim=1)
        assert x1.shape[1] == self.in_c
        if self.dropout is not None:
            x1 = self.dropout(x1)
        cls_score = self.fc_cls(x1)

        # Node Type Classification
        x2 = x.mean(dim=-2)
        x2 = x2.reshape(N, M, C, V)
        x2 = x2.mean(dim=1)
        x2 = x2.permute(0, 2, 1).reshape(N*V, C)        
        assert x2.shape[1] == self.in_c
        if self.dropout is not None:
            x2 = self.dropout(x2)
        node_cls_score = self.node_cls(x2)
        label = self.node_label.repeat(N, 1).to(x.device).reshape(N*V)
        loss = nn.CrossEntropyLoss()
        node_cls_loss = loss(node_cls_score, label)
        return ['cls_score',cls_score], ['node_cls_loss', node_cls_loss]

@HEADS.register_module()
class ClsHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

@HEADS.register_module()
class GCHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        N, M, V, V = x.shape

        x = x.reshape(N, M, -1)
        x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score


@HEADS.register_module()
class Assemble_GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 head=4,
                 head_ada=False,
                 KL_div=False,
                 each_loss=True,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)
        self.model_len = head
        self.head_ada = head_ada
        module = [nn.Linear(self.in_c, num_classes)]
        self.head = head
        self.KL_div= KL_div
        self.each_loss = each_loss
        if self.head_ada:
            self.alpha = nn.Parameter(torch.ones(self.head))

        if head >1 and each_loss:
            for i in range(head):
                module.append(nn.Linear(self.in_c, num_classes))
        self.fc_cls = nn.ModuleList(module)

    def single_forward(self, x, i):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x = self.single_process(x)
        cls_score = self.fc_cls[i](x)
        return cls_score
    def single_process(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        # cls_score = self.fc_cls[i](x)
        return x
    def forward(self,x):
        # x_mean = (x * self.alpha).mean(0)
        if self.head_ada:
            x_mean = torch.einsum('hnkctu,h->hnkctu', x, self.alpha)
            x_mean = x_mean.mean(0)
        else:
            x_mean = x.mean(0)
        cls_score = [self.single_forward(x_mean,0)]
        if self.head >1 and self.each_loss:
            for i in range(x.shape[0]):
                x_single = x[i,:]
                cls_score.append(self.single_forward(x_single,i+1))
        

        # cls_score = []
        # for i in range(x.shape[0]):
        #     x_single = x[i,:]
        # # for x_single in x:
        #     cls_score.append(self.single_forward(x_single))
        # # x_mean = torch.stack(x)
        # x_mean = x.mean(0)
        # # if self.model_len==0:
        # #     x_all = x
        # cls_score.append(self.single_forward(x_mean)) 

        return cls_score
    def loss(self, feature, cls_score, label, **kwargs):

        losses = self.single_loss(cls_score[0], label, **kwargs)
        if self.head>1 and self.each_loss:
            for i in range(self.head):
                cls_score_single = cls_score[i+1]
                loss = self.single_loss(cls_score_single, label, **kwargs)
                
                losses['single_'+str(i+1)+'_loss'] = loss['loss_cls']
        if self.KL_div:
            for i in range(self.head):
                if i==0:
                    cls_feature_single = [self.single_process(feature[i])]
                else:
                    cls_feature_single.append(self.single_process(feature[i]))

            kl_loss = self.KL_loss(cls_feature_single)
            losses['KL_loss']=kl_loss
        
        # loss = 0
        # for cls_score_single in cls_score:
        #     loss = self.single_loss(cls_score_single, label, **kwargs)
        #     loss
        #     losses.append(self.single_loss(cls_score_single, label, **kwargs))
        return losses

    
    def single_loss(self, cls_score, label, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``label``.
        Args:
            cls_score (torch.Tensor): The output of the model.
            label (torch.Tensor): The target output of the model.
        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if label.shape == torch.Size([]):
            label = label.unsqueeze(0)
        elif label.dim() == 1 and label.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            label = label.unsqueeze(0)

        if not self.multi_class and cls_score.size() != label.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       label.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            label = ((1 - self.label_smooth_eps) * label + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, label, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses
    def KL_loss(self,cls_score):

        kl_loss = []
        for i in range(len(cls_score)):
            p=cls_score[i]
            for q in cls_score[(i+1):]:
                kl_loss.append( F.kl_div(F.log_softmax(p, dim=0), F.softmax(q, dim=0), reduction='sum'))

        # for i in cls_score[]
        # for i in range(len(cls_score)):
        #     for j in range(len(cls_score)):
        #         p = cls_score[i]
        #         q = cls_score[j]
        #         kl_loss = F.kl_div(F.log_softmax(p, dim=0), F.softmax(q, dim=0), reduction='sum')
        KL_loss = torch.stack(kl_loss)
        KL_loss = KL_loss.mean()
        return KL_loss

    
            
