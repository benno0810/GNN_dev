# PyGNN



In order to run on any graph, load_data() function must return the adjacency matrix of the network.

For example:

```
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
```



Run the following command to start the training

```
python train.py --epochs --lr --hidden --ndim
```

You can also look at the embeddings after training by printing embedx and embedy