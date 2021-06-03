import torch
import time
import math
import dgl
import numpy as np
import torch.nn as nn
import torch as th
from dgl.data import citation_graph as citegrh
from dgl.data import CoraGraphDataset
from dgl import DGLGraph
import dgl.function as fn
import networkx as nx
import torch.nn.functional as F
from dgl.data import RedditDataset,KarateClubDataset
from dgl.nn import GraphConv


class MylossFunc(nn.Module):
    def __init__(self, deta):
        super(MylossFunc, self).__init__()
        self.deta = deta

    def forward(self, out, label):
        out = torch.nn.functional.softmax(out, dim=1)
        m = torch.max(out, 1)[0]
        penalty = self.deta * torch.ones(m.size())
        loss = torch.where(m > 0.5, m, penalty)
        loss = torch.sum(loss)
        loss = Variable(loss, requires_grad=True)
        return

class GCN(nn.Module):
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
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # output layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.g, h)
        return h

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def Q2(G1:dgl.DGLGraph):
    #calculate matrix Q with diag set to 0
    #A=np.array(nx.adjacency_matrix(G1).todense())
    G1=dgl.to_networkx(G1)
    A=np.array(nx.adjacency_matrix(G1).todense())
    T=A.sum(axis=(0,1))
    Q=A*0
    w_in=A.sum(axis=1)
    w_out=w_in.reshape(w_in.shape[0],1)
    K=w_in*w_out/T
    Q=(A-K)/T
    #set Qii to zero for every i
    for i in range(Q.shape[0]):
        Q[i][i]=0
    return Q

if __name__=="__main__":

    dropout=0.5
    gpu=-1
    lr=0.01
    n_epochs=200
    n_hidden=16  # 隐藏层节点的数量
    n_layers=2  # 输入层 + 输出层的数量
    weight_decay=5e-4  # 权重衰减
    self_loop=True  # 自循环

    # cora 数据集
    '''
    data = citegrh.load_cora()
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    train_mask = torch.BoolTensor(data.train_mask)
    val_mask = torch.BoolTensor(data.val_mask)
    test_mask = torch.BoolTensor(data.test_mask)

    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    '''
    # load reddit data
    #data = RedditDataset(self_loop=False)
    data =KarateClubDataset()
    n_classes = data.num_classes

    g = data[0]
    n_edges=g.number_of_edges()
    n=len(g.ndata['label'])
    labels=g.ndata['label']
    #construct features, train,val,test masks
    g.ndata['feat']= th.eye(n)
    in_feats=g.ndata['feat'].shape[1]
    features=torch.FloatTensor(g.ndata['feat'])
    masks=np.array([True]*n)
    train_mask = masks
    val_mask = masks
    test_mask = masks


    if self_loop:
        g=dgl.remove_self_loop(g)


    if gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()


    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)

    if cuda:
        model.cuda()

    # 采用交叉熵损失函数和 Adam 优化器
    # not self_defined entropy loss
    loss_fcn = torch.nn.CrossEntropyLoss()

    #use modularity as loss function
    Q=Q2(g)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr)

    # 训练，并评估
    dur = []
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        # forward
        logits = model(features)
        #modularity as loss function

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dur.append(time.time() - t0)

        if epoch % 10 == 0:
            acc = evaluate(model, features, labels, val_mask)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
                  "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                                 acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))