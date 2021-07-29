import os
from scipy import io
import pycombo
import pandas as pd
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
from sklearn.metrics.cluster import normalized_mutual_info_score
from karateclub import LabelPropagation
import community as community_louvain
import random
def print_parameter(model):
    for p in model.parameters():
        print(p)
    print(model)

def Q2(G1: dgl.DGLGraph,mask):
    """

    :param G1: input graph
    :param mask: batch mask, for graph has nodes size of N, [1,2,3,4...N], the mask is size of N array [True, True, False...]
    :return: modularity matrix Q, where modularity=tr(C*Q*C.T)
    """
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
    """
    :return: model input for dataset, Cora, citeseer
    https://docs.dgl.ai/en/0.6.x/_modules/dgl/data/citation_graph.html
    """
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

#
def sparse2th(mat):
    """
    a utility function to convert a scipy.coo_matrix to torch.SparseFloat
    :param mat: scipy sparce matrix
    :return:
    """
    value = mat.data
    indices = th.LongTensor([mat.row, mat.col])
    # tensor = th.FloatTensor(th.from_numpy(value).float())
    tensor = th.sparse.FloatTensor(indices, th.from_numpy(value).float(), mat.shape)
    return tensor.to_dense()


def visualize(labels, g):
    """
    network visualization utility function
    :param labels: labels of graph nodes
    :param g: graph, nxnetwork class
    :return:
    """
    nx_g=g.to_networkx()
    pos = nx.spring_layout(nx_g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(nx_g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.show()

def load_cora_binary():
    """
    One subset of cora datasets which contains only 2 classes out of 7,
    see https://docs.dgl.ai/en/0.6.x/tutorials/models/1_gnn/6_line_graph.html?highlight=corabinary
    :return:
    """
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
    """
    karaclub dataset
    :return:
    """
    data =KarateClubDataset()
    n_classes = data.num_classes
    g = data[0]
    n_edges=g.number_of_edges()
    #g.ndata['label']=g.ndata['label']*0
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
    """
    track kernal weight to find possible issues
    :param model:
    :return:
    """
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
    """
    accuracy evaluation with given ground truth labels
    :param model: GCN model
    :param features: input features
    :param labels: given labels, size of N
    :param mask:  dataset mask, size of N
    :return:
    """
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def NMI(C_out,C_init):
    """
    comparison between input features and model outputs,the less the better (tune a lot) if tradient direction is right
    :param C_out:
    :param C_init:
    :return:
    """
    _, out_indices = torch.max(C_out, dim=1)
    _, in_indices = torch.max(C_init, dim=1)
    return normalized_mutual_info_score(out_indices,in_indices)



def C_construction(model,features,mask):
    """
    deprecated, use evaluate_M(), construct strict binary attachment for every community from a relaxation attachment
    :param model:
    :param features:
    :param mask:
    :return:
    """
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
    """
    construct strict binary attachment then evaluate the modularity score of this attachment
    :param C:
    :param Q:
    :param cuda:
    :return:
    """
    Q= Q.float()
    C=C.float()
    if cuda:
        C = C.cuda()
        Q = Q.cuda()
    with torch.no_grad():
        if abs(C.max().item())>0:
            #C_softmax= F.softmax(C,dim=1)
            C_softmax = C
            labels=C_softmax.argmax(1)
            temp=C*0
            C = temp.scatter(1,labels.unsqueeze(1),1)
    score=th.matmul(th.matmul(C.t(), Q), C).trace()
    if cuda:
        return C, score.cpu()
    else:
        return C, score

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)


def loadNetworkMat(filename,
                   path='/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/ProcessedMat/'):
    A = io.loadmat(path + filename)
    if check_symmetric(A['net']):
        G = nx.from_numpy_matrix(A['net'])
    else:
        G = nx.from_numpy_matrix(A['net'], create_using=nx.DiGraph)
    return G


def getNewComboPartition(G, maxcom=-1, suppressCppOutput=False):
    # https://pypi.org/project/pycombo/
    partition, modularity = pycombo.execute(G, return_modularity=True, max_communities=maxcom, random_seed=42)
    return modularity, partition


def getNewComboSeries(G, maxcom, tries=5, verbose=0):
    part = [None] * tries
    M = np.zeros(tries)
    for i in range(tries):
        M[i], part[i] = getNewComboPartition(G, maxcom)
        # M[i] = modularity(G,part[i])
        if verbose > 0:
            print('Combo try {},mod={:.6f}'.format(i + 1, M[i]))
    return part[np.argmax(M)]


def generate_model_input(nx_g,cuda):
    """
    generate initial input community attachment from some classic methods. add zero columns according to scale ratios
    this is for creating input for arbitrary community numbers prior
    tune a little bit from classic method to prevent over-optimized input
    :param nx_g:
    :param cuda:
    :return:
    """
    # load cora_binary, train_masks,val_masks,test_masks are used for future accuracy comparement with supervised algorithm
    # g, features, n_classes, in_feats, n_edges,labels = load_cora_binary()
    # g, features, n_classes, in_feats, n_edges, labels = load_kara()
    # g, features, n_classes, in_feats, n_edges, labels=load_les_miserables()
    # g, features, n_classes, in_feats, n_edges, labels = load_citation_graph()

    g = dgl.from_networkx(nx_g)
    n_classes = getattr(g, 'num_classes', 3)

    n_edges = g.number_of_edges()
    n = len(nx_g)
    default_labels = torch.LongTensor([1] * n)
    if 'label' not in g.ndata:
        g.ndata['label'] = default_labels
    labels = g.ndata['label']

    # construct initial features, train,val,test masks
    g.ndata['feat'] = g.adj().to_dense()
    in_feats = g.adj().shape[1]
    features = g.ndata['feat']
    mask = [True] * n

    # graph visualization
    # visualize(labels,g)

    # calculate matrix Q, initial community attachment C (with overlap)
    # overwrite n_classes
    n = len(g)
    mask = [True] * n
    mask = th.BoolTensor(mask)
    Q = {}
    Q = Q2(g, mask)
    Q = th.from_numpy(Q)

    # generate input features (community binary partition vector) based on louvain partition
    if nx.classes.function.is_directed(nx_g):
        model = LabelPropagation()
        model.fit(nx_g)
        partition = model.get_memberships()
        # modularity,partition=getNewComboPartition(nx_g)
    else:
        partition = community_louvain.best_partition(nx_g)

    n_classes = np.max(list(partition.values())) + 1
    C_init = Q[0:n_classes] * 0

    C_init = C_init.T
    for node in partition.keys():
        C_init[node][partition[node]] = 0.8
        #create a trivial noise to make it worse than classic method
        C_init[node][random.randint(0,partition[node])]+=0.1
        C_init[node][random.randint(0, partition[node])] += 0.1
    # squezz for matrix that are all zeros
    C_init = C_init.T
    C_init = C_init[~(C_init == 0).all(1)]
    C_init = C_init.T
    C_out, modularity_classic = evaluate_M(C_init, Q, cuda)
    # add zero columns to create possibility ratio set to 100%
    scale_ratio = 2
    # C_init=th.cat((C_init,C_init),dim=1)

    features = C_init.float()
    # scale_up the features
    # original dim of feature,[X,Y], new dim of feature: [1,1,X,Y]
    features = features.view(1, 1, features.shape[0], features.shape[1])
    shape = (features.shape[-2], features.shape[-1] * scale_ratio)
    features = th.nn.functional.interpolate(features, size=shape)
    # features = features.view(shape)/scale_ratio
    # reduce the dim of features to [X,Y]
    features = features.view(shape)
    temp = features * 0
    # find largest position of features [x,y]for every row(x), then add a new dim to let the label size=[x,1]
    feature_labels = features.argmax(1).unsqueeze(1)
    features = temp.scatter(1, feature_labels, 1)

    # features=th.matmul(features,features.t())
    n_classes = features.shape[1]
    in_feats = features.shape[1]

    return g, features, n_classes, in_feats, n_edges, labels, Q, mask,modularity_classic

def save_result(data_name, graph_type, modularity_scores_gcn,
                modularity_scores_combo, modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir):
    """
    :param modularity:  dict, modularity score
    :param C: dict, binary attachment of communities for every nodes
    :param dict, model_structure: GCN layer dims
    :param file_name: str filename
    :return:
    """
    file_d = data_dir + "/result1"
    if not os.path.exists(file_d):
        os.mkdir(file_d)
    # save_modularity,C,model_structure
    result_path = file_d + "/result.csv"
    #graph_type_df = pd.DataFrame.from_dict(graph_type, orient='index', columns='graph_type')
    graph_type_df = pd.DataFrame(list(graph_type.values()),index = list(graph_type.keys()),columns=['graph_type'])
    modularity_gcn_df = pd.DataFrame(list(modularity_scores_gcn.values()),  index = list(modularity_scores_gcn.keys()),columns=['modularity_gcn'])
    modularity_combo_df = pd.DataFrame(list(modularity_scores_combo.values()), index=list(modularity_scores_combo.keys()),
                                     columns=['modularity_combo'])
    modularity_classic_df = pd.DataFrame(list(modularity_scores_classic.values()), index=list(modularity_scores_classic.keys()),
                                     columns=['modularity_classic'])
    nmi_gcn_df = pd.DataFrame(list(nmi_gcn.values()),  index = list(nmi_gcn.keys()),columns=['NMI_gcn'])
    nmi_df=pd.DataFrame(list(nmi.values()),  index = list(nmi.keys()),columns=['NMI'])
    model_df = pd.DataFrame(list(model_parameter.values()), index = list(model_parameter.keys()),columns=['model_parameter'])
    result_metrics = pd.concat([graph_type_df, modularity_classic_df,modularity_gcn_df,modularity_combo_df,nmi_gcn_df,nmi_df, model_df],axis=1)
    result_metrics.to_csv(result_path, encoding='utf-8')
    """
    modularity_path=file_d+"/"+file_name+"_modularity"+".txt"
    with open(modularity_path,'w+',encoding='utf-8') as f:
        f.write(str(modularity))
    f.close()
    C_path=file_d+"/"+file_name+"_C"+".txt"
    np.savetxt(C_path,C)
    model_structure_path=file_d+"/"+file_name+"_model_structure"+".txt"
    #np.savetxt(model_structure_path,model_structure)
    with open(model_structure_path,'w+',encoding='utf-8') as f:
        f.write(model_structure)
    f.close()
    """

def partition_to_binary_attachment(partition):
    """
    deprecated
    :param partition:
    :return:
    """
    row_length = len(partition.keys())
    column_length = np.max(list(partition.values())) + 1
    C_init = np.zeros((row_length, column_length), dtype=float)
    for node in partition.keys():
        C_init[node][partition[node]] = 1
    # try C*C.T
    # squezz for matrix that are all zeros
    C_init = C_init.T
    C_init = C_init[~(C_init == 0).all(1)]
    C_init = C_init.T

    return th.LongTensor(C_init)