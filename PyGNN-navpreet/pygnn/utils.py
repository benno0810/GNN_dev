import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
import networkx as nx

def load_data():

    graph = pkl.load(open("data/Taxi2017_total_net.pkl", "rb"))
    adj = np.array(nx.adjacency_matrix(graph[0]).todense(), dtype=float)
    adj = adj[graph[1],:][:,graph[1]]

    return adj


def normalize(adj):

    adj = adj + np.identity(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    adj_norm = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    adj_norm = torch.FloatTensor(np.array(adj_norm))

    return adj_norm

def norm_embed(embed):

    embedx,embedy = torch.chunk(embed,chunks=2,dim=1)
    ES = (embedx ** 2).sum(axis=0) / (embedy ** 2).sum(axis=0)
    embedx = embedx / (ES ** 0.25)
    embedy = embedy * (ES ** 0.25)
    embed_norm = torch.cat((embedx,embedy),dim=1)
    return embed_norm

def toy_data():

    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_edge(1, 2, weight=10)
    graph.add_edge(1, 5, weight=57)
    graph.add_edge(2, 1, weight=8)
    graph.add_edge(2, 4, weight=34)
    graph.add_edge(2, 5, weight=75)
    graph.add_edge(4, 1, weight=24)
    graph.add_edge(5, 4, weight=14)
    graph.add_edge(5, 1, weight=73)
    graph.add_edge(5, 2, weight=48)

    adj = np.array(nx.adjacency_matrix(graph).todense(), dtype=float)

    return adj
