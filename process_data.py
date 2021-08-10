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



def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda, nn_model):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    lr = 0.0005
    early_stop_rate=0.000005
    loss_direction=-1 #
    n_epochs = 10000
    n_hidden = features.shape[1]  # number of hidden nodes
    n_layers = 1  # number of hidden layers
    weight_decay_gamma = 0.65 #
    self_loop = True  #
    early_stop = True
    visualize_model=False
    last_score = 0
    step_size=int(n_epochs/100)

    # step_size = 1
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
                               F.relu,
                               dropout)
        if cuda:
            model.cuda()
    else:
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
        if cuda:
            model.cuda()

    loss_fcn = ModularityScore(n_classes, cuda,loss_direction)


    if visualize_model:
        print_parameter(model)
        print_parameter(loss_fcn)

    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #apply weight_decay scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=weight_decay_gamma)

    # train and evaluate (with modularity score and labels)
    dur = []



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

        dur.append(time.time() - t0)
        if epoch % step_size == 0:
        #if epoch % 1 == 0:
            if visualize_model:
                print_parameter(model)
                print_parameter(loss_fcn)
            C_out,eval_loss = evaluate_M(C_hat,Q,cuda)
            print("Epoch {} | Time(s) {}  Train_Modularity {} | True_Modularity {}"
                  "ETputs(KTEPS) {}".format(epoch, np.mean(dur), eval_loss, abs(loss),
                                             n_edges / np.mean(dur) / 1000))
        if early_stop:
            if abs((loss - last_score) / last_score) < 0.000005 or torch.isnan(loss).sum()>0:
                loss=last_score
                C_hat=last_C_hat
                C_out, eval_loss = evaluate_M(C_hat, Q, cuda)
                print("Epoch {} | Time(s) {}  Train_Modularity {} | True_Modularity {}"
                      "ETputs(KTEPS) {}".format(epoch, np.mean(dur), eval_loss, abs(loss),
                                                n_edges / np.mean(dur) / 1000))
                break

        last_score = loss
        last_C_hat=C_hat

    #C_out = C_construction(model, features, mask)

    C_init,modularity_init = evaluate_M(features, Q, cuda)
    print('initial modularity is', modularity_init)
    C_hat,modularity_score = evaluate_M(C_hat, Q, cuda)
    if torch.isnan(modularity_score):
        modularity_score=loss
    print(C_hat)
    return modularity_score.cpu().detach().numpy(), C_hat.cpu(), model.__str__(),features.cpu()


def main(nx_g, nn_model):
    # note g is a networkx class
    gpu = 0
    if gpu < 0:
        cuda = False
    else:
        cuda = True
    # prepare training data, set hyperparameters
    g, features, n_classes, in_feats, n_edges, labels,Q,mask,modularity_classic = generate_model_input(nx_g,cuda)
    return train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, cuda, nn_model),modularity_classic


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
    modularity_scores_combo = {}
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
                # if file !='karate_34.mat':
                #     continue
                data_name.append(file)
                G = loadNetworkMat(file, data_dir)
                if nx.classes.function.is_directed(G):
                    graph_type[file] = 'directed'
                else:
                    graph_type[file] = 'undirected'
                print(file, graph_type[file])

                # need to figure out it is weighted or not
                modularity_scores_combo[file], partition = getNewComboPartition(G)
                C_out_combo[file] = partition_to_binary_attachment(partition)
                [modularity_scores_gcn[file], C_out[file], model_parameter[file],C_init[file]],modularity_scores_classic[file] = main(G, nn_model)

                nmi_gcn[file]=NMI(C_out[file],C_init[file])
                nmi[file] = NMI(C_out_combo[file], C_init[file])



    ##save log
    save_result(data_name, graph_type, modularity_scores_gcn, modularity_scores_combo, modularity_scores_classic,nmi_gcn,nmi,model_parameter,
                data_dir)

    print('something')
