import torch
import torch.nn as nn

from pyskl.models.gcns.utils import gcn_sparse, tcn_sparse
from torch.distributions.normal import Normal

from ...utils import Graph
from ..builder import BACKBONES
from .utils import MSTCN, mstcn_sparse, unit_ctrgcn, unit_tcn_sparse,unit_tcn, unit_ctrhgcn, unit_ctrgcn_sparse,get_sparsity
from .ctrgcn_sparse import CTRGCNBlock,CTRGCN_sparse
from .aagcn_sparse import AAGCNBlock,AAGCN_sparse
from .stgcn_sparse import STGCNBlock,STGCN_sparse
from .dggcn_sparse import DGBlock,DGSTGCN_sparse



class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

@BACKBONES.register_module()
class SMoEAssemble_sparse(nn.Module):
    def __init__(self,
                 graph_cfg,
                 model_list,
                 sparse_ratio,
                 in_channels=3,
                 base_channels=64,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 semantic_stage=range(1,11),
                 pretrained=None,
                 sparse_decay=False,
                #  linear_sparsity=0,
                 warm_up=0, 
                 num_person=2,
                 out_channel=256,
                 noisy_gating=True, 
                 k_num=1,
                 **kwargs):
        super(SMoEAssemble_sparse, self).__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.num_experts=len(model_list)-1
        self.w_gate = nn.Parameter(torch.zeros(out_channel, self.num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(out_channel, self.num_experts), requires_grad=True)
        self.num_person = num_person
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.sparse_decay = sparse_decay
        # self.linear_sparsity = linear_sparsity
        self.warm_up = warm_up
        self.model_list = model_list
        self.sparse_ratio = sparse_ratio
        self.k = k_num
        self.noisy_gating = noisy_gating

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'tcn_dropout'}
        semantic_index = 1 in semantic_stage
        models=[]
        for  i,(model_unit, sparse_unit) in enumerate(zip(model_list, sparse_ratio)): 
            assert model_unit in ['ST-GCN', 'AA-GCN', 'CTR-GCN', 'DG-GCN']
            if model_unit =='ST-GCN':
                ST_kwargs = kwargs['ST_kwargs']
                models.append(STGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                           tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**ST_kwargs))
            if model_unit =='AA-GCN':
                AA_kwargs = kwargs['AA_kwargs']
                models.append(AAGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                           tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**AA_kwargs))
            if model_unit =='CTR-GCN':
                CTR_kwargs = kwargs['CTR_kwargs']
                models.append(CTRGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                            tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**CTR_kwargs))
            if model_unit =='DG-GCN':
                DG_kwargs = kwargs['DG_kwargs']
                models.append(DGSTGCN_sparse(graph_cfg,in_channels=in_channels,linear_sparsity=sparse_unit,warm_up=warm_up,
                                             tcn_sparse_ratio=sparse_unit,gcn_sparse_ratio=sparse_unit,**DG_kwargs))

        self.experts = nn.ModuleList(models)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def init_weights(self):
        for module in self.experts:
            module.init_weights()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, current_epoch, max_epoch,loss_coef=1e-2):
        # N, M, T, V, C = x.size()
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = self.data_bn(x.view(N, M * V * C, T))
        # x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x_base=self.experts[-1](x, current_epoch, max_epoch)
        x_base = self.GCN_feature(x_base)

        gates, load = self.noisy_top_k_gating(x_base, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
    

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs=[]
        for i in range(self.num_experts):
            if expert_inputs[i].shape[0]==0:
                pass
            else:
                expert_outputs.append(self.experts[i](expert_inputs[i],current_epoch, max_epoch))
        # expert_outputs = [self.experts[i](expert_inputs[i],current_epoch, max_epoch) for i in range(self.num_experts)]
        # _, _, C, T, V = expert_inputs[0].shape
        # N,M = x.shape[:2]
        expert_outputs=[self.GCN_feature(index) for index in expert_outputs]
        # x = x.reshape(N * M, C*T*V)
        # expert_outputs=[index.reshape(-1, C*T*V) for index in expert_outputs]
        y = dispatcher.combine(expert_outputs)
        # y=y.view(N,M,C,T,V)
        
        return y,loss

    def GCN_feature(self, x):
        pool = nn.AdaptiveAvgPool2d(1)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)

        x = pool(x)
        # x = pool(x)
        x = x.reshape(N, M, C)
        x = x.mean(dim=1)
        # x = x.reshape(N*M, C)
        # x = x.mean(dim=1)
        return x
    
    def get_threshold(self, model, sparsity,linear_sparsity):
        local=[]
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                local.append(p.detach().flatten())
        local=torch.cat(local)
        threshold= self.percentile(local,sparsity*100)
        return threshold
    
    def percentile(self, t, q):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        return t.view(-1).kthvalue(k).values.item()
    
    def get_mask(self, model, current_epoch, max_epoch,linear_sparsity):
        # if current_epoch < self.warm_up:
        #     sparsity = 0
        # else:
        #     if self.sparse_decay:
        #         if current_epoch<(max_epoch/2.0):
        #             sparsity=get_sparsity(linear_sparsity,current_epoch,0,max_epoch/2)
        #         else:
        #             sparsity=linear_sparsity
        #     else:
        #         sparsity=linear_sparsity
        sparsity=linear_sparsity
        threshold = self.get_threshold(model,sparsity,linear_sparsity)
        mask=[]
        weight = []
        for name, p in model.named_parameters():
            if hasattr(p, 'is_score') and p.is_score and p.sparsity==linear_sparsity:
                mask.append(p.detach().flatten())
            if hasattr(p, 'is_mask') and p.is_mask:
                weight.append(p.detach().flatten())
        mask = torch.cat(mask)
        weight = torch.cat(weight)
        mask =mask<=threshold
        weight = weight*mask

        return weight

    def regularize(self, lam, penalty, current_epoch, max_epoch):
        '''
        Calculate regularization term for first layer weight matrix.

        Args:
        network: MLP network.
        penalty: one of GL (group lasso), GSGL (group sparse group lasso),
            H (hierarchical).
        '''
        W =[]
        for j in range(len(self.model_list)):
            for i in range(self.num_stages):
                try:
                    W.append(self.get_mask(self.experts[j].gcn[i], current_epoch, max_epoch,self.sparse_ratio[j]))
                except:
                    W.append(self.get_mask(self.experts[j].net[i], current_epoch, max_epoch,self.sparse_ratio[j]))   
                else:
                    W.append(self.get_mask(self.experts[j].gcn[i], current_epoch, max_epoch,self.sparse_ratio[j]))
                   
                # W
        # W = gc
        # W = network.layers[0].weight
        # W = torch.stack(W)
        # W = torch.cat(W)
        # b,hidden, p, p1, lag = weight.shape
        panelty = []
        if penalty == 'GL':
            W = torch.cat(W)
            return lam * torch.sum(torch.norm(W, dim=0))
        elif penalty == 'GSGL':
            panelty = []
            for weight_layer in W:
                index = torch.sum(torch.norm(weight_layer, dim=0))
                panelty.append(index)
            panelty = torch.stack(panelty)
            return lam*torch.sum(panelty)
            # return panelty
            # return lam * (torch.sum(torch.norm(W, dim=(1, -1)))
            #             + torch.sum(torch.norm(W, dim=1)))
        elif penalty == 'H':
            # Lowest indices along third axis touch most lagged values.
            # return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 1)))
            #                 for i in range(lag)])
            pass
        else:
            raise ValueError('unsupported penalty: %s' % penalty)
