from functools import partial
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import MSELoss
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
import torch
import time
import math
import dgl
import numpy as np
import torch as th
from dgl.base import DGLError
from torch.nn import init

class MyModel(th.nn.Module):
    def __init__(self, g, dropout, n_features):
        '''

        :param g:
        :param dropout:

        c_hat = ReLU(f1*c+f2*(Q C)+b) = (nX1)
        Q= nXn
        C = nX1
        Q*C = nX1
        so dimmension of  input is [n,2], output [n,1], Linear layer  [2,1]
        '''
        super(MyModel, self).__init__()
        self.g = g
        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(n_features, 2))
        self.layers.append(th.nn.ReLU(inplace=True))
        self.dropout = th.nn.Dropout(p=dropout)

    def forward(self, features):
        h = features.float()
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(h)
        return h

class GCN(th.nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = th.nn.ModuleList()
        # input layer
        self.layers.append(MyGraphConv(in_feats, n_hidden, 'none',activation=activation))
        # n_layers hidden layer
        for i in range(n_layers - 1):
            self.layers.append(MyGraphConv(n_hidden, n_hidden, 'none',activation=activation))
        # 1 output layer, use softmax as activation on this layer
        self.layers.append(MyGraphConv(n_hidden, n_classes,'none',activation=partial(F.softmax,dim=1)))
        #try softmax as activation
        #self.layers.append(GraphConv(n_hidden, n_classes, activation=activation))
        self.dropout = th.nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.g, h)
        #output is f(x) through a softmax filter
        return h


class MyGraphConv(GraphConv):
    r"""
        adjusted GraphConv layer with weights setting to 1 initially

    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = th.nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = th.nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.

        """
        if self.weight is not None:
            init.eye_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
