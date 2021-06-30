# -*- codeing = utf-8 -*-
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
from torch.nn import MSELoss


def Q1(comm, G):
    # 边的个数

    edges = G.edges()

    m = len(edges)

    # print 'm',m

    # 每个节点的度

    du = G.degree()

    # print 'du',du

    # 通过节点对（同一个社区内的节点对）计算

    ret = 0.0

    for c in comm:

        for x in c:

            for y in c:

                # 边都是前小后大的

                # 不能交换x，y，因为都是循环变量

                if x <= y:

                    if (x, y) in edges:

                        aij = 1.0

                    else:

                        aij = 0.0

                else:

                    if (y, x) in edges:

                        aij = 1.0

                    else:

                        aij = 0

                # print x,' ',y,' ',aij

                tmp = aij - du[x] * du[y] * 1.0 / (2 * m)

                # print du[x],' ',du[y]

                # print tmp

                ret = ret + tmp

                # print ret

                # print ' '

    ret = ret * 1.0 / (2 * m)

    # print 'ret ',ret

    return ret

def Q2(G1,C):
    # calculate matrix Q with diag set to 0
    # A=np.array(nx.adjacency_matrix(G1).todense())
    A = np.array(nx.adjacency_matrix(G1).todense())
    T = A.sum(axis=(0, 1))
    Q = A * 0
    w_in = A.sum(axis=1)
    w_out = w_in.reshape(w_in.shape[0], 1)
    K = w_in * w_out / T
    Q = (A - K) / T
    # set Qii to zero for every i
    #for i in range(Q.shape[0]):
    #    Q[i][i] = 0
    score = np.trace(np.matmul(np.matmul(C.T,Q),C))
    return score

def getModularityMatrix(G, symmetrize = False, correctLoops = False):  # build mobilarity matrix and return as numpy
    A = np.array(nx.adjacency_matrix(G).todense(), dtype = float)
    if correctLoops and not isinstance(G,nx.DiGraph):
        A += np.diag(np.diag(A))
    wout = A.sum(axis=1)
    win = A.sum(axis=0)
    T = wout.sum()
    Q = A / T - np.matmul(wout.reshape(-1, 1), win.reshape(1, -1)) / (T ** 2)
    if symmetrize:
        Q = (Q + Q.transpose()) / 2
    return Q
def modularity(G, partition, correctLoops = False): #modularity of the networkx graph given partition dictionary
    Q = getModularityMatrix(G, correctLoops = correctLoops)
    C = np.array([partition[n] for n in G.nodes()]) #could there be an indexing mismatch between Q and C
    return (Q * (C.reshape(-1,1) == C.reshape(1,-1))).sum()

if __name__=="__main__":

    #C=np.loadtxt(r"C:\Users\benno\OneDrive\Documents\GitHub\GNN_dev\__py_debug_temp_var_585734174.csv", dtype=np.float,delimiter=",")
    '''
    print(C)
    com={}
    G= nx.karate_club_graph()
    flag=0
    for row_index,row in enumerate(C):
        indices=np.argmax(row)
        
        for k in com:
            if indices in k:
                flag=1
        if flag==0:
            com.append([row+1])
        flag=0
        continue

        
        if indices not in com.keys():
            com[indices]=[row_index]
        else:
            com[indices].append(row_index)
    M_score = nx.algorithms.community.quality.modularity(G, com.values(), weight='weight')
    M_score_2 = Q1(com.values(),G)
    M_score_3 = Q2(G,C)
    M_score_4 = modularity(G,list(com.values()))
    print(com)

    print(M_score)
    '''


    G = nx.karate_club_graph()
    #print(G.nodes[:]['club'])
    G.nodes[5]["club"]
    #for node in G.nodes:
    #    print(node['club'])