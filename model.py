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
        self.layers.append(GraphConv(in_feats, n_hidden, activation=None))
        # n_layers hidden layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=None))
        # 1 output layer, use softmax as activation on this layer
        self.layers.append(GraphConv(n_hidden, n_classes,activation=partial(F.softmax,dim=1)))
        self.dropout = th.nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.g, h)
        #output is f(x) through a softmax filter
        return h