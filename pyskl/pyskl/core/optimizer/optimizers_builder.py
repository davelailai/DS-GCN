# Copyright (c) OpenMMLab. All rights reserved.
# from typing import ParamSpec
from mmcv.runner import build_optimizer


def build_optimizers(model, cfgs):
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
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        if not isinstance(cfg, dict):
            is_dict_of_dict = False
    if is_dict_of_dict:
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            if hasattr(cfg_,'sparse'):    
                if cfg_['sparse'] == 'score_only':
                    params = [param for param in model.parameters()
                            if hasattr(param, 'is_score') and param.is_score]
                elif cfg_['sparse']== 'normal':
                    params = [param for param in model.parameters() if not (hasattr(param, 'is_score') and not param.is_score)]
                cfg_['constructor'] = 'SparseOptimizerConstructor'
                cfg_.pop('sparse')
                optimizers[key] = build_optimizer(params, cfg_)
            elif hasattr(cfg_,'Causal'):
                if cfg_['Causal'] == 'pool':
                    params = [param for param in model.parameters()
                            if hasattr(param, 'is_pool') and param.is_pool]
                elif cfg_['Causal']== 'SE':
                    params = [param for param in model.parameters() if not (hasattr(param, 'is_pool') and not param.is_pool)]
                cfg_['constructor'] = 'SparseOptimizerConstructor'
                cfg_.pop('Causal')
                optimizers[key] = build_optimizer(params, cfg_)
                
            else:
                if key=='module':
                    optimizers[key] = build_optimizer(model, cfg_)
                else:
                    module = getattr(model, key)
                    optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    if hasattr(cfgs,'sparse'):
        if cfgs['sparse'] == 'score_only':
            params = [param for param in model.parameters()
                            if hasattr(param, 'is_score') and param.is_score]
        elif cfgs['sparse']== 'normal':
            params = [param for param in model.parameters() if not (hasattr(param, 'is_score') and not param.is_score)]
        cfgs['constructor'] = 'SparseOptimizerConstructor'
        cfgs.pop('sparse')

        return build_optimizer(params, cfgs)
    


    return build_optimizer(model, cfgs)