import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook, OptimizerHook
from torch.nn.utils import clip_grad
import logging


@HOOKS.register_module()
class MultiModuleHook(OptimizerHook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """

    def __init__(self, grad_clip=None, detect_anomalous_params=False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer[runner.optimizer_current].zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        if runner.optimizer_current=='module':
            runner.outputs['losses']['loss_cls'].backward(retain_graph=True)
        if runner.optimizer_current=='neck':
            runner.outputs['losses']['neck_loss'].backward(retain_graph=True)

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                        runner.outputs['num_samples'])
        runner.optimizer[runner.optimizer_current].step()

    def detect_anomalous_parameters(self, loss, runner):
        logger = runner.logger
        parameters_in_graph = set()
        visited = set()

        def traverse(grad_fn):
            if grad_fn is None:
                return
            if grad_fn not in visited:
                visited.add(grad_fn)
                if hasattr(grad_fn, 'variable'):
                    parameters_in_graph.add(grad_fn.variable)
                parents = grad_fn.next_functions
                if parents is not None:
                    for parent in parents:
                        grad_fn = parent[0]
                        traverse(grad_fn)

        traverse(loss.grad_fn)
        for n, p in runner.model.named_parameters():
            if p not in parameters_in_graph and p.requires_grad:
                logger.log(
                    level=logging.ERROR,
                    msg=f'{n} with shape {p.size()} is not '
                    f'in the computational graph \n')