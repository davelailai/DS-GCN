# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import build_optimizer


def sparse_optimizers(model,cfgs, mode='score_only'):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    optimizers = {}
    if mode == 'score_only':
        params = [param for param in model.parameters()
                            if hasattr(param, 'is_score') and param.is_score]
    elif mode == 'normal':
        params = [param for param in model.parameters() if not (hasattr(param, 'is_score') and param.is_score)]
    cfgs['constructor'] = 'SparseOptimizerConstructor'
    return build_optimizer(params, cfgs)