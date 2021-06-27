import matplotlib.pyplot as plt
from dgl.data import RedditDataset, KarateClubDataset
import networkx as nx
import torch.nn.functional as F
from dgl.data import citation_graph as citegrh
from dgl.data import CoraBinary
from dgl.data import CoraGraphDataset
from dgl import DGLGraph
import torch
import time
import math
import dgl
import numpy as np
import torch as th

def Q2(G1: dgl.DGLGraph,mask):
    # calculate modularity matrix Q
    # A=np.array(nx.adjacency_matrix(G1).todense())
    G1 = dgl.to_networkx(G1)
    A = np.array(nx.adjacency_matrix(G1).todense())
    A_row_cut = A[mask==True,:]
    A_col_cut = A_row_cut[:,mask.T==True]
    A=A_col_cut
    T = A.sum(axis=(0, 1))
    Q = A * 0
    w_in = A.sum(axis=1)
    w_in = w_in.reshape(w_in.shape[0], 1)
    #w_out = w_in.reshape(w_in.shape[0], 1)
    w_out=A.sum(axis=0)

    K = w_in * w_out / T
    Q = (A - K) / T
    # set Qii to zero for every i, try not setting this
    #for i in range(Q.shape[0]):
    #    Q[i][i] = 0
    return Q

def load_citation_graph():
    data =CoraGraphDataset()
    n_classes = data.num_classes
    g = data[0]
    n_edges=g.number_of_edges()
    n=len(g.ndata['label'])
    labels=g.ndata['label']
    #construct features, train,val,test masks
    features=g.ndata['feat']
    #features=sparse2th(features)
    in_feats = features.shape[1]
    return g, features, n_classes, in_feats, n_edges, labels

# a utility function to convert a scipy.coo_matrix to torch.SparseFloat
def sparse2th(mat):
    value = mat.data
    indices = th.LongTensor([mat.row, mat.col])
    # tensor = th.FloatTensor(th.from_numpy(value).float())
    tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), mat.shape)
    return tensor.to_dense()


# network visualization utility function
def visualize(labels, g):
    nx_g=g.to_networkx()
    pos = nx.spring_layout(nx_g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(nx_g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.show()

def load_cora_binary():
    data = CoraBinary()
    g,features,labels=data[1]
    n_edges=g.number_of_edges()
    features=sparse2th(features)
    labels=th.LongTensor(labels)
    in_feats=features.shape[1]
    n_classes=2
    print(th.max(features))

    return g,features,n_classes,in_feats,n_edges,labels

def load_kara():
    data =KarateClubDataset()
    n_classes = data.num_classes
    g = data[0]
    n_edges=g.number_of_edges()
    n=len(g.ndata['label'])
    labels=g.ndata['label']
    #construct features, train,val,test masks
    g.ndata['feat']=g.adj().to_dense()
    #test=g.adj()
    in_feats=g.adj().shape[1]
    #this features is not efficient
    features=g.ndata['feat']

    return g,features,n_classes,in_feats,n_edges,labels



def load_les_miserables():
    nx_data=nx.generators.les_miserables_graph()
    data=dgl.from_networkx(nx_data)
    n_classes = getattr(data,'num_classes',3)
    g = data
    n_edges=g.number_of_edges()
    n=len(nx_data)
    default_labels=torch.LongTensor([1]*n)
    if 'label' not in g.ndata:
        g.ndata['label']=default_labels
    labels=g.ndata['label']
    #construct features, train,val,test masks
    g.ndata['feat']=g.adj().to_dense()
    #test=g.adj()
    in_feats=g.adj().shape[1]
    #this features is not efficient
    features=g.ndata['feat']

    return g, features, n_classes, in_feats, n_edges, labels

def kernal_weights_analysis(model):
    weight_keys=model.state_dict().keys()
    for key in weight_keys:
        weight_t = model.state_dict()[key].numpy()
        weight_shape=weight_t.shape
        weight_mean=weight_t.mean()
        weight_std=weight_t.std(ddof=1)
        weight_min=weight_t.min()
        weight_max=weight_t.max()
        print(weight_shape,weight_mean,weight_max,weight_min,weight_std)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def C_construction(model,features,mask):
    #construct binary allocaton of C
    model.eval()
    with torch.no_grad():
        logits = model(features)
        _, indices = torch.max(logits[mask], dim=1)
        C= logits[mask]*0
        for i,community in enumerate(indices):
            C[i][community]=1
    return C

def evaluate_M(C,Q,cuda):
    Q= Q.float()
    if cuda:
        C = C.cuda()
        Q = Q.cuda()
    with torch.no_grad():
        score=th.matmul(th.matmul(C.t(), Q), C).trace()
    return score