import pycombo
from scipy import io
import numpy as np
import networkx as nx
import os
import community as community_louvain

import torch
import time
import math
import dgl
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

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)
def loadNetworkMat(filename, path = '/Users/stanislav/Desktop/NYU/NYURESEARCH/STRATEGIC_RESEARCH/ModularityMaximum/SampleNetworks/ProcessedMat/'):
    A = io.loadmat(path + filename)
    if check_symmetric(A['net']):
        G = nx.from_numpy_matrix(A['net'])
    else:
        G = nx.from_numpy_matrix(A['net'], create_using=nx.DiGraph)
    return G

def getNewComboPartition(G, maxcom=-1, suppressCppOutput = False):
    #https://pypi.org/project/pycombo/
    partition, modularity = pycombo.execute(G, return_modularity=True, max_communities = maxcom, random_seed=42)
    return modularity, partition

def getNewComboSeries(G,maxcom,tries=5,verbose=0):
    part = [None]*tries
    M = np.zeros(tries)
    for i in range(tries):
        M[i], part[i]=getNewComboPartition(G,maxcom)
        #M[i] = modularity(G,part[i])
        if verbose>0:
            print('Combo try {},mod={:.6f}'.format(i+1,M[i]))
    return part[np.argmax(M)]





def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    lr = 5e-2
    n_epochs = 10000
    n_hidden = features.shape[1]*1  # number of hidden nodes
    n_layers = 1  # number of hidden layers
    weight_decay = 5e-4  #
    self_loop = True  #
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
        mask=mask.cuda()

    if cuda:
        norm = norm.cuda()
    # g.ndata['norm'] = norm.unsqueeze(1)

    model = GCN(g,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                F.relu,
                dropout)

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
    # train and evaluate (with modularity score and labels)
    dur = []
    M = []
    # P= [[1],[2],[3],[4],[5],[6]]
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        C_hat = model(features)
        # use train_mask to train
        loss = loss_fcn(C_hat[mask], Q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # C_out=C_construction(model,features)

        # use eval_mask to see overfitting
        #modularity_score = evaluate_M(C_hat[train_mask], Q, cuda)
        dur.append(time.time() - t0)
        if epoch % 1000 == 0:
            # record modularity
            #for i,p in enumerate(model.parameters()):
            #    print(p)
            #    P[i].append(np.mean(np.abs(p.grad.cpu().detach().numpy())))
            M.append(str(-loss.item()))
            # acc=0.5
            print("Epoch {} | Time(s) {}  Train_Modularity {} | "
                  "ETputs(KTEPS) {}".format(epoch, np.mean(dur),  -loss,
                                             n_edges / np.mean(dur) / 1000))

    C_out = C_construction(model, features, mask)
    print(C_out)
    modularity_score = evaluate_M(C_out, Q, cuda)
    # with open('modularity_history.txt', 'w') as f:
    #     for line in M:
    #         f.write(line + '\n')
    # f.close()
    return modularity_score.cpu().detach().numpy(),C_out.cpu().detach().numpy(),model.__str__()

def generate_model_input(nx_g):
    # load cora_binary, train_masks,val_masks,test_masks are used for future accuracy comparement with supervised algorithm
    # g, features, n_classes, in_feats, n_edges,labels = load_cora_binary()
    # g, features, n_classes, in_feats, n_edges, labels = load_kara()
    # g, features, n_classes, in_feats, n_edges, labels=load_les_miserables()
    # g, features, n_classes, in_feats, n_edges, labels = load_citation_graph()

    g=dgl.from_networkx(nx_g)
    n_classes = getattr(g,'num_classes',3)

    n_edges=g.number_of_edges()
    n=len(nx_g)
    default_labels=torch.LongTensor([1]*n)
    if 'label' not in g.ndata:
        g.ndata['label']=default_labels
    labels=g.ndata['label']
    #construct initial features, train,val,test masks
    g.ndata['feat']=g.adj().to_dense()
    #test=g.adj()
    in_feats=g.adj().shape[1]
    #this features is not efficient
    features=g.ndata['feat']
    mask = [True] * n



    # graph visualization
    # visualize(labels,g)
    return g, features, n_classes, in_feats, n_edges, labels

def main(nx_g):
    #note g is a networkx class
    gpu = 0
    if gpu < 0:
        cuda = False
    else:
        cuda = True
    # prepare training data, set hyperparameters


    g, features, n_classes, in_feats, n_edges, labels = generate_model_input(nx_g)


    # calculate matrix Q, initial community attachment C (with overlap)
    # overwrite n_classes
    n=len(g)
    mask = [True]*n
    mask=th.BoolTensor(mask)
    Q = {}
    Q = Q2(g, mask)
    Q = th.from_numpy(Q)

    # generate input features (community binary partition vector) based on louvain partition
    if nx.classes.function.is_directed(G):
        model = LabelPropagation()
        model.fit(G)
        partition = model.get_memberships()
        #modularity,partition=getNewComboPartition(G)
    else:
        partition =community_louvain.best_partition(G)

    n_classes = np.max(list(partition.values())) + 1
    C_init = Q[0:n_classes] * 0

    C_init = C_init.T
    for node in partition.keys():
        C_init[node][partition[node]] = 1
    # try C*C.T
    #squezz for matrix that are all zeros
    C_init = C_init.T
    C_init =C_init[~(C_init==0).all(1)]
    C_init=C_init.T
    features = C_init.float()
    # features=th.matmul(features,features.t())
    n_classes=features.shape[1]
    in_feats = features.shape[1]



    print('initial modularity is', evaluate_M(features, Q, cuda))

    return train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda)

def partition_to_binary_attachment(partition):
    row_length = len(partition.keys())
    column_length = np.max(list(partition.values())) + 1
    C_init = np.zeros((row_length,column_length),dtype=float)
    for node in partition.keys():
        C_init[node][partition[node]] = 1
    # try C*C.T
    #squezz for matrix that are all zeros
    C_init = C_init.T
    C_init =C_init[~(C_init==0).all(1)]
    C_init=C_init.T

    return C_init

def save_result(modularity,C,model_structure,file_name,data_dir):

    """
    :param modularity:  float, modularity score
    :param C: binary attachment of communities for every nodes
    :param model_structure: GCN layer dims
    :param file_name: str filename
    :return:
    """
    file_d=data_dir+"/result1"
    if not os.path.exists(file_d):
        os.mkdir(file_d)
    #save_modularity,C,model_structure
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
    return

def save_result_combo(modularity,C,file_name,data_dir):
    file_d=data_dir+"/result1"
    if not os.path.exists(file_d):
        os.mkdir(file_d)
    #save_modularity,C,model_structure
    modularity_path=file_d+"/"+file_name+"_modularity_combo"+".txt"
    with open(modularity_path,'w+',encoding='utf-8') as f:
        f.write(str(modularity))
    f.close()
    C_path=file_d+"/"+file_name+"_C_combo"+".txt"
    np.savetxt(C_path,C)
    return


if __name__=="__main__":
    test_number=10
    work_dir=os.getcwd()
    data_dir=os.path.join(work_dir,'data/ComboSampleData/')
    #G = loadNetworkMat('karate_34.mat',data_dir)
    G = loadNetworkMat('celeganmetabolic_453.mat',data_dir)
    modularity_scores_gcn={}
    C_outs={}
    C_outs_combo={}
    graph_type={}
    modularity_scores_combo={}
    models={}
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            #print(file[-3:])
            if file[-3:]=='mat':
                G = loadNetworkMat(file, data_dir)
                if nx.classes.function.is_directed(G):
                    graph_type[file]='directed'
                else:
                    graph_type[file] = 'undirected'
                print(file,graph_type[file])
                #need to figure out it is weighted or not
                # modularity_scores_combo[file], partition = getNewComboPartition(G)
                # C_outs_combo[file] = partition_to_binary_attachment(partition)
                # modularity,C_out,models[file]=main(G)
                # modularity_scores_gcn[file]=modularity
                # C_outs[file] = C_out
                #
                #
                # ##save log
                # save_result(modularity_scores_gcn[file],C_outs[file],models[file],file,data_dir)
                # save_result_combo(modularity_scores_combo[file],C_outs_combo[file],file,data_dir)

    print('something')


    # plt.subplot(111)
    # nx.draw(G)
    # plt.show()
    #
    # modularity_score_combo,part1=getNewComboPartition(G)
    # main(G)
    # for i in range(test_number):
    #     continue


