import numpy as np
import networkx as nx
import os
import pandas as pd
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
from config import  InitLearningRate

def train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, args):
    # sethyperparameter
    dropout = 0.0
    gpu = 0
    early_stop_rate = 0.000005
    loss_direction = 1
    n_epochs = 100
    n_hidden = features.shape[1]  # number of hidden nodes
    n_layers = 0  # number of hidden layers
    weight_decay_gamma = 0.65  #
    self_loop = True  #
    early_stop = False
    visualize_model = False
    last_score = 0

    grad_direction = args['grad_direction']
    lr = args['lr']
    cuda = args['cuda']
    nn_model = args['nn_model']
    cache_middle_result=args['cache_middle_result']
    if 'early_stop' in args.keys():
        early_stop=args['early_stop']
    else:
        early_stop=False
    #early_stop=False


    # step_size=int(n_epochs/100)

    step_size = 1
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

    loss_fcn = ModularityScore(n_classes, cuda, loss_direction)

    if visualize_model:
        print_parameter(model)
        print_parameter(loss_fcn)

    # optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # apply weight_decay scheduler
    # use self written D method
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=weight_decay_gamma)
    optimizer = mySGD(model.parameters(), lr=lr, batch_size=features.shape[0], grad_direction=grad_direction)

    # train and evaluate (with modularity score and labels)
    dur = []
    M=[]

    print("initial inputs \n", features)
    print("#####################")

    for epoch in range(n_epochs):

        model.train()
        t0 = time.time()
        C_hat = model(features)
        # print('#############WX###################')
        # print(C_hat)
        # use train_mask to train
        loss = loss_fcn(C_hat[mask], Q)
        # if loss jump check parameters
        if epoch == 0:
            print("initial output WX : \n", C_hat)
            print('initial parameters and gradients')
            for i, p in enumerate(list(model.parameters())):
                print("param {}".format(i), p)
                print("param {} grad".format(i), p.grad)
            print('#####################')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # StepLR.step()

        dur.append(time.time() - t0)
        if epoch % step_size == 0:
            # if epoch % 1 == 0:
            # if visualize_model:
            #     print_parameter(model)
            #     print_parameter(loss_fcn)
            C_out, eval_loss = evaluate_M(C_hat, Q, cuda)
        if epoch == 0:
            # print('calculate the modularity from classic method')
            ground_truth_modularity = eval_loss
        print(
            "Epoch {} | Time(s) {} |  True_Modularity {} | Ground_Truth_Modulairty {} | ETputs(KTEPS) {}".format(epoch,
                                                                                                                 np.mean(
                                                                                                                     dur),
                                                                                                                 (loss),
                                                                                                                 ground_truth_modularity,
                                                                                                                 n_edges / np.mean(
                                                                                                                     dur) / 1000))
        if epoch>0 and early_stop:
            if ((loss - last_score) / last_score) < -1e-3 :
                loss = last_score
                C_hat = last_C_hat
                C_out, eval_loss = evaluate_M(C_hat, Q, cuda)

                print(
                    "Epoch {} | Time(s) {} |  True_Modularity {} | Ground_Truth_Modulairty {} | ETputs(KTEPS) {}".format(
                        epoch,
                        np.mean(
                            dur),
                        (loss),
                        ground_truth_modularity,
                        n_edges / np.mean(
                            dur) / 1000))

                break
        if cache_middle_result:
            M.append(loss.item())



        last_score = loss
        last_C_hat = C_hat

    # C_out = C_construction(model, features, mask)

    C_init, modularity_init = evaluate_M(features, Q, cuda)
    print('initial modularity is', modularity_init)
    C_hat, modularity_score = evaluate_M(C_hat, Q, cuda)
    if torch.isnan(modularity_score):
        modularity_score = loss
    print(C_hat)
    return modularity_score.cpu().detach().numpy(), loss.item(),C_hat.cpu(), model.__str__(), features.cpu(),M


def main(nx_g, nn_model, grad_direction,data_dir,dataset):
    # note g is a networkx class
    init_lr = InitLearningRate()
    cache_middle_result=True
    middle_result = {}
    lr_mode = 'scanning'
    lr = np.logspace(-5, 1, num=14)
    print(lr)
    gpu = 0

    if gpu < 0:
        cuda = False
    else:
        cuda = True
    # prepare training data, set hyperparameters
    g, features, n_classes, in_feats, n_edges, labels, Q, mask, modularity_classic = generate_model_input(nx_g, cuda)
    if lr_mode == 'scanning':
        print('learning_rate scanning mode for {} intervals from {} to {}'.format(len(lr), np.min(lr), np.max(lr)))
        for i, learning_rate in enumerate(lr):
            args = {
                'lr': learning_rate,
                'grad_direction': grad_direction,
                'nn_model': nn_model,
                'cuda': cuda,
                'cache_middle_result':cache_middle_result
            }
            modularity_score,loss ,C_hat, model_structure, features, middle_result[args['lr']] = train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, args)
            if i == len(lr) - 1:
                save_middle_result(middle_result,data_dir,dataset)
                return modularity_score,loss, C_hat, model_structure, features, modularity_classic, n_classes
    if lr_mode =='training':
        print('all initial rating tuned, now start to learn')
        for i, learning_rate in enumerate(lr):
            args = {
                'lr': init_lr.get_init_lr(dataset),
                'grad_direction': grad_direction,
                'nn_model': nn_model,
                'cuda': cuda,
                'cache_middle_result':cache_middle_result,
                'early_stop':True
            }
            #modularity_score.cpu().detach().numpy(), C_hat.cpu(), model.__str__(), features.cpu(),M
            modularity_score,loss ,C_hat, model_structure, features, middle_result[args['lr']]= train(g, features, n_classes, in_feats, n_edges, labels, mask, Q, args)
            return modularity_score,loss, C_hat, model_structure, features, modularity_classic, n_classes
