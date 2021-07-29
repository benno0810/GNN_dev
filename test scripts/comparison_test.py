import pycombo
from scipy import io
import numpy as np
import networkx as nx
import os
import community as community_louvain
import pandas as pd
import torch
import time
import math
import dgl as dgl
from dgl.nn import pytorch
import numpy as np
import torch as th
import dgl.function as fn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
from GraphSAGE.losses import compute_loss_multiclass
from utils import *
from model import *
from loss import *
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from karateclub import LabelPropagation
import glob
from sklearn.preprocessing import OneHotEncoder
from data.unsupervisedmodularitybasedGNNdata import utils as dataset_utils

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


def tile_array(a, b0, b1):
    pass


def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda, nn_model):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    lr = 0.005
    n_epochs = 100
    n_hidden = features.shape[1]  # number of hidden nodes
    n_layers = 0  # number of hidden layers
    weight_decay = 5e-4  #
    self_loop = True  #
    last_score = 0
    step_size=int(n_epochs/10)

    if self_loop:
        g = dgl.add_self_loop(g)
    # run single train of some model
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        g = g.to('cuda:0')
        mask = mask.cuda()

    if cuda:
        norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)

    if nn_model == 'GCN':
        model = eval(nn_model)(g,
                               in_feats,
                               n_hidden,
                               n_classes,
                               n_layers,
                               F.elu,
                               dropout)
    if nn_model == 'RelGraphConv':
        model = eval(nn_model)(in_feat=in_feats,
                               out_feat=n_classes,
                               num_rels=3,
                               regularizer='basis',
                               num_bases=None,
                               bias=True,
                               activation=None,
                               self_loop=True,
                               low_mem=False,
                               dropout=0.0,
                               layer_norm=False
                               )

    for p in model.parameters():
        print(p)

    print(model)
    # kernal_weights_analysis(model)

    # use crossentropyLoss as loss, must consider the permutations,
    # loss_fcn = torch.th.nn.CrossEntropyLoss()
    loss_fcn = ModularityScore(n_classes, cuda)
    if cuda:
        model.cuda()

    for p in loss_fcn.parameters():
        print(p)

    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #apply weight_decay scheduler
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.65)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # train and evaluate (with modularity score and labels)
    dur = []
    M = []
    P = [[1], [2], [3], [4], [5], [6]]


    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        C_hat = model(features)
        # use train_mask to train
        loss = loss_fcn(C_hat[mask], Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        StepLR.step()
        # C_out=C_construction(model,features)

        # use eval_mask to see overfitting
        # modularity_score = evaluate_M(C_hat[train_mask], Q, cuda)
        dur.append(time.time() - t0)
        #if epoch % step_size == 0:
        if epoch % 1 == 0:
            # record modularity
            # for i,p in enumerate(model.parameters()):
            #    print(p)
            #    P[i].append(np.mean(np.abs(p.grad.cpu().detach().numpy())))
            C_out,eval_loss = evaluate_M(C_hat,Q,cuda)
            print("Epoch {} | Time(s) {}  Train_Modularity {} | True_Modularity {}"
                  "ETputs(KTEPS) {}".format(epoch, np.mean(dur), eval_loss, -loss,
                                            n_edges / np.mean(dur) / 1000))
            if abs((loss - last_score) / last_score) < 0.000005:
                break
            last_score = loss

    C_out = C_construction(model, features, mask)
    #print(C_out)
    modularity_init = evaluate_M(features, Q, cuda)
    print('initial modularity is', modularity_init)
    C_hat,modularity_score = evaluate_M(C_hat, Q, cuda)
    print(C_hat)
    # with open('modularity_history.txt', 'w') as f:
    #     for line in M:
    #         f.write(line + '\n')
    # f.close()
    return modularity_score.cpu().detach().numpy(), C_out.cpu(), model.__str__(),features.cpu()


def generate_model_input(nx_g):
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
    # test=g.adj()
    in_feats = g.adj().shape[1]
    # this features is not efficient
    features = g.ndata['feat']
    mask = [True] * n

    # graph visualization
    # visualize(labels,g)
    return g, features, n_classes, in_feats, n_edges, labels


def main(nx_g, nn_model):
    # note g is a networkx class
    gpu = 0
    if gpu < 0:
        cuda = False
    else:
        cuda = True
    # prepare training data, set hyperparameters

    g, features, n_classes, in_feats, n_edges, labels = generate_model_input(nx_g)

    # calculate matrix Q, initial community attachment C (with overlap)
    # overwrite n_classes
    n = len(g)
    mask = [True] * n
    mask = th.BoolTensor(mask)
    Q = {}
    Q = Q2(g, mask)
    Q = th.from_numpy(Q)

    # generate input features (community binary partition vector) based on louvain partition
    if nx.classes.function.is_directed(G):
        model = LabelPropagation()
        model.fit(G)
        partition = model.get_memberships()
        # modularity,partition=getNewComboPartition(G)
    else:
        partition = community_louvain.best_partition(G)

    n_classes = np.max(list(partition.values())) + 1
    C_init = Q[0:n_classes] * 0

    C_init = C_init.T
    for node in partition.keys():
        C_init[node][partition[node]] = 1
    # try C*C.T
    # squezz for matrix that are all zeros
    C_init = C_init.T
    C_init = C_init[~(C_init == 0).all(1)]
    C_init = C_init.T
    C_out,modularity_classic = evaluate_M(C_init,Q,cuda)
    # add zero columns to create possibility ratio set to 100%
    scale_ratio = 2
    # C_init=th.cat((C_init,C_init),dim=1)

    features = C_init.float()
    # scale_up the features
    features = features.view(1, 1, features.shape[0], features.shape[1])
    shape = (features.shape[-2], features.shape[-1] * scale_ratio)
    features = th.nn.functional.interpolate(features, size=shape)
    features = features.view(shape)/scale_ratio
    # features=th.matmul(features,features.t())
    n_classes = features.shape[1]
    in_feats = features.shape[1]



    return train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda, nn_model),modularity_classic


def partition_to_binary_attachment(partition):
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

    return


def save_result_combo(data_name, graph_type, modularity, C, data_dir):
    file_d = data_dir + "/result1"
    if not os.path.exists(file_d):
        os.mkdir(file_d)
    # save_modularity,C,model_structure
    modularity_path = file_d + "/" + file_name + "_modularity_combo" + ".txt"
    with open(modularity_path, 'w+', encoding='utf-8') as f:
        f.write(str(modularity))
    f.close()
    C_path = file_d + "/" + file_name + "_C_combo" + ".txt"
    np.savetxt(C_path, C)
    return


if __name__ == "__main__":
    test_number = 10
    work_dir = os.getcwd()
    nn_model = 'GCN'
    graphs_path ='data/unsupervisedmodularitybasedGNNdata/graphs/associative_n=400_k=5/test/'

    data_dir = os.path.join(work_dir, graphs_path)
    # G = loadNetworkMat('karate_34.mat',data_dir)
    num_test_graphs = len(glob.glob(os.path.join(graphs_path, "adj-*.npy")))
    label_encoder = OneHotEncoder(categories='auto', sparse=False)
    print(f"Number of graphs found: {num_test_graphs}")
    modularity_scores_gcn = {}
    nmi_gcn={}
    nmi={}
    C_init={}
    C_out = {}
    C_out_combo = {}
    graph_type = {}
    modularity_scores_combo = {}
    model_parameter = {}
    data_name = []
    modularity_scores_classic={}
    for idx in range(num_test_graphs):
        if idx and idx % 10 == 0:
            print(idx, time.time())
        file = idx
        labels_path = os.path.join(graphs_path, f"labels-{idx}.npy")
        adj_path = os.path.join(graphs_path, f"adj-{idx}.npy")
        labels = np.load(labels_path, allow_pickle=True)
        true_labels = label_encoder.fit_transform(labels.reshape(-1, 1))
        adjacency = np.load(adj_path, allow_pickle=True).tolist()
        G = dataset_utils.graph_from_adjacency(adjacency, labels)
        if nx.classes.function.is_directed(G):
            graph_type[file] = 'directed'
        else:
            graph_type[file] = 'undirected'
        # need to figure out it is weighted or not
        modularity_scores_combo[file], partition = getNewComboPartition(G)
        C_out_combo[file] = partition_to_binary_attachment(partition)
        [modularity_scores_gcn[file], C_out[file], model_parameter[file], C_init[file]], modularity_scores_classic[
            file] = main(G, nn_model)

        nmi_gcn[file] = NMI(C_out[file], C_init[file])
        nmi[file] = NMI(C_out_combo[file], C_init[file])


    save_result(data_name, graph_type, modularity_scores_gcn, modularity_scores_combo, modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)

    ##save log

    #save_result_combo(data_name, graph_type, modularity_scores_combo[file], C_outs_combo[file], data_dir)

    print('something')

