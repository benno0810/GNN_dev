import torch
import time
import math
import dgl
import numpy as np
import torch as th
from dgl.data import citation_graph as citegrh
from dgl.data import CoraBinary
from dgl.data import CoraGraphDataset
from dgl import DGLGraph
import dgl.function as fn
import networkx as nx
import torch.nn.functional as F
from dgl.data import RedditDataset, KarateClubDataset
from dgl.nn import GraphConv
from torch.nn import MSELoss
from losses import compute_loss_multiclass


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


class MylossFunc(th.nn.Module):
    def __init__(self,C):
        super(MylossFunc, self).__init__()
        ## define C as parameter
        #self.params = th.nn.ParameterList([C])
        self.params=th.nn.Parameter(data=C.float(),requires_grad=True)
        print()

    def forward(self,Q):
        # -tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(C),Q),C))
        C=self.params
        C=th.sigmoid(C)
        Q=Q.float()
        temp = th.matmul(th.matmul(C.t(), Q), C)
        loss = -temp.trace()

        return loss


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
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # output layer
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = th.nn.Dropout(p=dropout)

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


'''
class CrossEntropyLoss(th.nn.Module):
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
'''


def Q2(G1: dgl.DGLGraph):
    # calculate matrix Q with diag set to 0
    # A=np.array(nx.adjacency_matrix(G1).todense())
    G1 = dgl.to_networkx(G1)
    A = np.array(nx.adjacency_matrix(G1).todense())
    T = A.sum(axis=(0, 1))
    Q = A * 0
    w_in = A.sum(axis=1)
    w_out = w_in.reshape(w_in.shape[0], 1)
    K = w_in * w_out / T
    Q = (A - K) / T
    # set Qii to zero for every i
    for i in range(Q.shape[0]):
        Q[i][i] = 0
    return Q


# a utility function to convert a scipy.coo_matrix to torch.SparseFloat
def sparse2th(mat):
    value = mat.data
    indices = th.LongTensor([mat.row, mat.col])
    # tensor = th.FloatTensor(th.from_numpy(value).float())
    tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), mat.shape)
    return tensor.to_dense()


# network visualization utility function

def visualize(labels, g):
    pos = nx.spring_layout(g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)


if __name__ == "__main__":

    dropout = 0.5
    gpu = -1
    lr = 1e-2
    n_epochs = 200000
    n_hidden = 16  # 隐藏层节点的数量
    n_layers = 2  # 输入层 + 输出层的数量
    weight_decay = 5e-4  # 权重衰减
    self_loop = True  # 自循环

    # cora_binary
    data = CoraBinary()
    g, features, labels = data[1]
    n_edges = g.number_of_edges()
    labels = th.LongTensor(labels)
    n_classes = 2


    n = len(labels)
    train_mask = [True] * n
    val_mask = train_mask
    test_mask = train_mask

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
    g = DGLGraph(data.graph)
    '''

    # load reddit data
    # data = RedditDataset(self_loop=False)
    '''
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
    '''

    Q = Q2(g)
    print(np.max(Q))

    C_init = Q[0:2] * 0
    C_init[0] = np.random.randint(2, size=(1, Q.shape[0]))
    C_init[1] = 1 - C_init[0]
    C = th.tensor(data=C_init.T, requires_grad=True)
    C=C.float()
    Q = th.from_numpy(Q)
    Q=Q.float()
    Q_C = th.matmul(Q,C)
    print(C)
    features = torch.cat([C.detach(),Q_C.detach()], dim=1)
    in_feats = n_classes * 2  # column number of [C,QC]
    # if self_loop:
    #    g=dgl.remove_self_loop(g)

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
    '''
        model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)
    '''

    model = MyModel(g, dropout, in_feats)
    for p in model.parameters():
        print(p)

    if cuda:
        model.cuda()

    # use crossentropyLoss as loss, must consider the permutations,
    # loss_fcn = torch.th.nn.CrossEntropyLoss()
    loss_fcn = MylossFunc(C)

    for p in loss_fcn.parameters():
        print(p)

    loss_fcn2 = th.nn.MSELoss()
    optimizer_l = torch.optim.SGD(loss_fcn.parameters(),
                                  lr=lr)
    optimizer_f = torch.optim.Adam(model.parameters(),
                                   lr=lr)

    # 训练，并评估
    dur = []
    for epoch in range(n_epochs):
        t0 = time.time()

        loss_fcn.train()
        loss_l = loss_fcn(Q)
        optimizer_l.zero_grad()
        loss_l.backward()
        optimizer_l.step()


        model.train()

        # forward
        C_hat = model(features)
        loss_f = loss_fcn2(C_hat, C.detach())
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        # step以后求trace(CQC)最小-<-loss function, tensor 是C_i

        dur.append(time.time() - t0)

        if epoch % 1000 == 0:
            #print f0,f1,f2
            for p in model.parameters():
                print(p)
            #print C
            print(C_hat)
            # calculate accuracy
            acc_1 = evaluate(model, features, labels, val_mask)
            acc_2 = evaluate(model, features, 1 - labels, val_mask)
            acc = max(acc_1, acc_2)
            print("Epoch {} | Time(s) {} | Modularity {} | Accuracy {} | "
                  "ETputs(KTEPS) {}".format(epoch, np.mean(dur), -loss_l,
                                                acc, n_edges / np.mean(dur) / 1000))

    print()
    acc_1 = evaluate(model, features, labels, test_mask)
    acc_2 = evaluate(model, features, 1 - labels, test_mask)
    acc = max(acc_1, acc_2)
    print("Test accuracy {:.2%}".format(acc))