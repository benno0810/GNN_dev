import numpy as np
import networkx as nx
import os
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
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from train import main,train





if __name__ == "__main__":
    test_number = 10
    work_dir = os.getcwd()
    nn_model = 'GCN'
    data_dir = os.path.join(work_dir, 'data/ComboSampleData/')
    # G = loadNetworkMat('karate_34.mat',data_dir)
    G = loadNetworkMat('celeganmetabolic_453.mat', data_dir)
    modularity_scores_gcn = {}
    nmi_gcn={}
    nmi={}
    C_init={}
    C_out = {}
    C_out_combo = {}
    graph_type = {}
    n_communities={}
    initial_partition_approach={}
    modularity_scores_combo = {}
    modularity_scores_combo_restricted={}
    loss={}
    model_parameter = {}
    data_name = []
    modularity_scores_classic={}
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # print(file[-3:])
            # if 'celegansneural_297' not in file:
            #     continue
            if file[-3:] == 'mat':
                #append dataset name list
                if 'USairports_1858' not in file:
                    continue
                data_name.append(file)
                G = loadNetworkMat(file, data_dir)
                if nx.classes.function.is_directed(G):
                    graph_type[file] = 'directed'
                    initial_partition_approach[file]='LPA'
                else:
                    graph_type[file] = 'undirected'
                    initial_partition_approach[file]='Louvain'
                print(file, graph_type[file])

                # need to figure out it is weighted or not
                modularity_scores_combo[file], partition = getNewComboPartition(G)

                C_out_combo[file] = partition_to_binary_attachment(partition)

                #modularity_score, C_hat, model_structure, features, n_classes

                modularity_scores_gcn[file], loss[file],C_out[file], model_parameter[file],C_init[file],modularity_scores_classic[file], n_communities[file] = main(G, nn_model,grad_direction=-1,data_dir=data_dir,dataset=file[:-4])
                modularity_scores_combo_restricted[file], partition = getNewComboPartition(G, maxcom=n_communities[file])
                nmi_gcn[file]=NMI(C_out[file],C_init[file])
                nmi[file] = NMI(C_out_combo[file], C_init[file])



    ##save log
    save_result(data_name, graph_type, modularity_scores_gcn, modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)
    #<network>,<initial partition approach>,<number of communities>,<initial modularity>,<modularity after fune-tuning>,<COMBO modularity >,<COMBO modularity without restricting the number of communities and the optimal number of communities it returns >
    save_result_for_report(data_name, initial_partition_approach,n_communities, modularity_scores_gcn, loss,modularity_scores_combo, modularity_scores_combo_restricted,modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)

    print('something')
