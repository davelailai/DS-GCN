import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy



class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation):
        super(MLP, self).__init__()
        self.activation = activation_helper(activation)

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag)
        modules = [layer]

        for d_in, d_out in zip(hidden, hidden[1:] + [1]):
            layer = nn.Conv1d(d_in, d_out, 1)
            modules.append(layer)

        # Register parameters.
        self.layers = nn.ModuleList(modules)

    def forward(self, X):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                X = self.activation(X)
            X = fc(X)

        return X.transpose(2, 1)


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu'):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            for net in self.networks:
                for name, parameters  in net.named_parameters():
                     print(name, parameters.size())
                     print(torch.norm(parameters, dim=(0, 2)))
            # GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
            #       for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0).int()
        else:
            return GC
    loss_fn = nn.MSELoss(reduction='mean')

def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])

def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def activation_helper(activation, dim=None):
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'leakyrelu':
        act = nn.LeakyReLU()
    elif activation is None:
        def act(x):
            return x
    else:
        raise ValueError('unsupported activation: %s' % activation)
    return act