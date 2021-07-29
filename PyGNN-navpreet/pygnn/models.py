import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, InnerProduct
from utils import norm_embed


class GNN(nn.Module):
    def __init__(self, nfeat, nhid, ndim, dropout):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.embeddings = GraphConvolution(nhid, ndim)
        self.reconstructions = InnerProduct(ndim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.embeddings(x, adj)
        x = norm_embed(x)
        x = self.reconstructions(x)
        return x
