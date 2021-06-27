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

def train(g, features, n_classes, in_feats, n_edges, labels,train_mask,val_mask,test_mask,Q,cuda):
    #sethyperparameter
    dropout = 0.2
    gpu = 0
    lr = 5e-2
    n_epochs = 800000
    n_hidden =128  # 隐藏层节点的数量
    n_layers = 2 # 输入层 + 输出层的数量
    weight_decay = 5e-4  # 权重衰减
    self_loop = True  # 自循环


    #run single train of some model
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0

    if cuda:
        torch.cuda.set_device(gpu)
        features = features.cuda()
        labels = labels.cuda()
        g=g.to('cuda:0')
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()

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

    optimizer = torch.optim.SGD(model.parameters(),
                                  lr=lr)

    # train and evaluate (with modularity score and labels)
    dur = []
    M=[]
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        C_hat = model(features)
        #use train_mask to train
        loss = loss_fcn(C_hat[val_mask],Q['val'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #C_out=C_construction(model,features)

        #use eval_mask to see overfitting
        modularity_score=evaluate_M(C_hat[train_mask],Q['train'],cuda)
        dur.append(time.time() - t0)
        if epoch % 1000 == 0:
            #record modularity
            M.append(str(-loss.item()))
            acc_1 = evaluate(model, features, labels, val_mask)
            acc_2 = evaluate(model, features, 1 - labels, val_mask)
            acc = max(acc_1, acc_2)
            #acc=0.5
            print("Epoch {} | Time(s) {} | Eval_Modularity {} | Train_Modularity {} | Eval_Accuracy {} | "
                  "ETputs(KTEPS) {}".format(epoch, np.mean(dur),modularity_score, -loss,
                                                acc, n_edges / np.mean(dur) / 1000))

    C_out=C_construction(model,features,test_mask)
    modularity_score=evaluate_M(C_out,Q['test'],cuda)
    with open('modularity_history.txt','w') as f:
        for line in M:
            f.write(line+'\n')
    f.close()

def main():
    gpu=0
    if gpu < 0:
        cuda = False
    else:
        cuda = True
    #prepare training data, set hyperparameters
    # load cora_binary, train_masks,val_masks,test_masks are used for future accuracy comparement with supervised algorithm
    #g, features, n_classes, in_feats, n_edges,labels = load_cora_binary()
    #g, features, n_classes, in_feats, n_edges, labels = load_kara()
    #g, features, n_classes, in_feats, n_edges, labels=load_les_miserables()
    g, features, n_classes, in_feats, n_edges, labels = load_citation_graph()

    #graph visualization

    #visualize(labels,g)
    n = len(labels)
    if 'train_mask' not in g.ndata:
        train_mask = [True] * n
        train_mask=th.BoolTensor(train_mask)
    else:
        train_mask=g.ndata['train_mask']
    if 'val_mask' not in g.ndata:
        val_mask = [True] * n
        val_mask=th.BoolTensor(train_mask)
    else:
        val_mask=g.ndata['val_mask']
    if 'test_mask' not in g.ndata:
        test_mask = [True] * n
        test_mask=th.BoolTensor(test_mask)
    else:
        test_mask=g.ndata['test_mask']

    #calculate matrix Q, initial community attachment C (with overlap)

    #overwrite n_classes
    n_classes=7
    #construct Q['train'], Q['eval'],Q['test'] seperately
    Q={}
    Q['train'] = Q2(g,train_mask)
    Q['train'] = th.from_numpy(Q['train'])
    Q['val']= Q2(g,val_mask)
    Q['val'] = th.from_numpy(Q['val'])
    Q['test'] = Q2(g, test_mask)
    Q['test'] = th.from_numpy(Q['test'])
    #C_init = Q['train'][0:n_classes] * 0
    #for i in range(0,n_classes):
    #    C_init[i] = np.random.randint(2, size=(1, Q.shape[0]))
    #C = th.tensor(data=C_init.T, requires_grad=True)
    #C=C.float()

    #Q=Q.float()
    #Q_C = th.matmul(Q,C)




    train(g, features, n_classes, in_feats, n_edges, labels,train_mask,val_mask,test_mask,Q,cuda)









if __name__ == "__main__":
    main()




