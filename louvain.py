import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils import *

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
# load the karate club graph
G = nx.karate_club_graph()

#first compute the best partition
partition = community_louvain.best_partition(G)

# compute the best partition
partition = community_louvain.best_partition(G)

# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
M_score = community_louvain.modularity(partition, G, weight='weight')
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)

#create binary partition matrix baed on partition
n_classes = np.max(list(partition.values())) + 1
C_init = np.zeros([34,4],dtype=float)
for node in partition.keys():
    C_init[node][partition[node]] = 1

print(C_init)
com={}
for row_index, row in enumerate(C_init):
    indices = np.argmax(row)
    '''
    for k in com:
        if indices in k:
            flag=1
    if flag==0:
        com.append([row+1])
    flag=0
    continue

    '''
    if indices not in com.keys():
        com[indices] = [row_index]
    else:
        com[indices].append(row_index)
M_score = nx.algorithms.community.quality.modularity(G, com.values(), weight='weight')
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
M_score_2 = Q1(com.values(), G)
M_score_3 = Q2(G, C_init)
#M_score_4 = modularity(G, list(com.values()))
plt.show()