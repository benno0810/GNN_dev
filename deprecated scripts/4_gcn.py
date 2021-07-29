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
import pycombo
def train(g, features, n_classes, in_feats, n_edges, labels,train_mask,val_mask,test_mask,all_mask,Q,cuda,dropout,gpu,lr,n_epochs,n_hidden,n_layers,weight_decay,self_loop):



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

    #optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    optimizer = torch.optim.SGD(model.parameters(),lr = lr)
    # train and evaluate (with modularity score and labels)
    dur = []
    M=[]
    P= [[1],[2],[3],[4],[5],[6]]
    for epoch in range(n_epochs):
        model.train()
        t0 = time.time()
        C_hat = model(features[train_mask])
        #use train_mask to train
        loss = loss_fcn(C_hat[train_mask],Q['train'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #C_out=C_construction(model,features)

        #use eval_mask to see overfitting
        modularity_score=evaluate_M(C_hat[train_mask],Q['val'],cuda)
        dur.append(time.time() - t0)
        if epoch % 100 == 0:
            #record modularity
            for i,p in enumerate(model.parameters()):
                #print(p)
                P[i].append(np.mean(np.abs(p.grad.cpu().detach().numpy())))
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

"""
    #sethyperparameter
    dropout = 0.0
    gpu = 0
    lr = 5e-1
    n_epochs = 2000
    n_hidden =8  # 隐藏层节点的数量
    n_layers = 0 # 输入层 + 输出层的数量
    weight_decay = 5e-4  # 权重衰减
    self_loop = True  # 自循环
"""
def main(dropout=0.0,
         gpu=0,
         lr=5e-2,
         n_epochs=10000,
         n_hidden=8,
         n_layers=0,
         weight_decay=5e-4,
         self_loop=True):
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
        val_mask=th.BoolTensor(val_mask)
    else:
        val_mask=g.ndata['val_mask']
    if 'test_mask' not in g.ndata:
        test_mask = [True] * n
        test_mask=th.BoolTensor(test_mask)
    else:
        test_mask=g.ndata['test_mask']
    if 'all_mask' not in g.ndata:
        all_mask = [True] * n
        all_mask=th.BoolTensor(all_mask)
    else:
        all_mask=g.ndata['all_mask']

    #calculate matrix Q, initial community attachment C (with overlap)
    #construct Q['train'], Q['eval'],Q['test'] seperately
    Q={}
    Q['train'] = Q2(g,train_mask)
    Q['train'] = th.from_numpy(Q['train'])
    Q['val']= Q2(g,val_mask)
    Q['val'] = th.from_numpy(Q['val'])
    Q['test'] = Q2(g, test_mask)
    Q['test'] = th.from_numpy(Q['test'])
    Q['all']= Q2(g,all_mask)
    Q['all'] = th.from_numpy(Q['all'])

    #generate random input features
    C_init = Q['train'][0:n_classes] * 0
    for i in range(0,n_classes):
        C_init[i] = th.FloatTensor(np.random.randint(2, size=(1, Q['train'].shape[0])))
    C_init=C_init.t()
    features=C_init
    #C = th.tensor(data=C_init.T, requires_grad=True)
    #C=C.float()

    #Q=Q.float()
    #Q_C = th.matmul(Q,C)

    #generate input feature using louvain algorithm
    '''
    nx_g =  nx.karate_club_graph()
    partition = community_louvain.best_partition(nx_g)
    n_classes=np.max(list(partition.values()))+1
    C_init = Q['train'][0:n_classes] * 0
    C_init = C_init.T
    for node in partition.keys():
        C_init[node][partition[node]]=1
    features=C_init

    print(C_init)
    '''
    '''
    
    '''
    #nx_g =  nx.karate_club_graph()
    #partition = community_louvain.best_partition(nx_g)
    nx_g = dgl.to_networkx(g)
    nx_g=nx_g.to_undirected()
    partition = community_louvain.best_partition(nx_g)
    modularity = community_louvain.modularity(partition,nx_g)
    #[partition,modularity] = pycombo.execute(nx_g)
    print('modularity score of input is {}'.format(modularity))
    n_classes=np.max(list(partition.values()))+1
    C_init = Q['all'][0:n_classes] * 0
    C_init = C_init.T
    index=0
    for node in partition.keys():
        if train_mask[node]==True:
            C_init[index][partition[node]]=1
            index+=1

    #try C*C.T
    features=C_init.float()
    g.ndata['h']=features
    #hidden_noes equal to in_fests
    hidden_nodes=n_classes
    #features=th.matmul(features,features.t())
    in_feats=features.shape[1]

    print(C_init)
    train(g, features, n_classes, in_feats, n_edges, labels,train_mask,val_mask,test_mask,all_mask,Q,cuda,dropout,gpu,lr,n_epochs,n_hidden,n_layers,weight_decay,self_loop)









if __name__ == "__main__":
    main()




