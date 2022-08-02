'''
input:
    - number_of_subgraphs
    - average_number_of_nodes_per_subgraph
    - change_window: arr of size num_of_subgraphs
    - number_of_features
    - params: (mean, variance): arr of size num_of_subgraphs
    - max_mean_change
'''
import random
from itertools import combinations, groupby
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Graph:
    def __init__(self, num_subgraphs: int, node_count_range_per_subgraph: Tuple[int, int], num_features: int, dist_params: List[Tuple[float, float]], change_window: float, change_bounds: float, steps: int = 1000):
        self.num_subgraphs = num_subgraphs
        # args = [dist_params]
        # assert all(isinstance(arg, list) and len(
        #     arg) == num_subgraphs for arg in args), 'Invalid class attributes.'
        self.node_count_range_per_subgraph = node_count_range_per_subgraph
        self.num_features = num_features
        self.dist_params = dist_params
        self.change_window = change_window
        self.change_bounds = change_bounds
        self.G = nx.Graph()
        self.__build()

    def __build(self):
        node_id = 0  # start node_id
        # possible number of nodes per subgraph
        node_count_arr = range(*self.node_count_range_per_subgraph)
        subgraph_idx = range(self.num_subgraphs)
        subgraphs = []
        for _ in subgraph_idx:
            # choose a number of nodes
            node_count = random.choice(node_count_arr)
            # generate nodes
            nodes = list(range(node_id, node_id + node_count))
            node_id += node_count
            # generate edges s.t. there are no isolated nodes
            edges = self.__build_connected_graph(nodes)
            subgraphs += [(nodes, edges)]  # store subgraph
        # connect subgraphs randomly s.t. there are no isolated subgraph
        subgraph_edges = [(random.choice(subgraphs[i][0]), random.choice(subgraphs[j][0]))
                          for i, j in self.__build_connected_graph(subgraph_idx)]  # select nodes randomly from each subgraph to be subgraph connectors
        nodes, edges = zip(*subgraphs)
        nodes = [n for node_arr in nodes for n in node_arr]
        edges = [e for edge_arr in edges for e in edge_arr]
        edges.extend(subgraph_edges)
        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def __build_connected_graph(self, nodes):
        edges = combinations(nodes, 2)
        groups = groupby(edges, key=lambda x: x[0])
        return [random.choice(list(e)) for _, e in groups]

    def draw(self):
        nx.draw(self.G)
        plt.show()


def test():
    graph = Graph(num_subgraphs=3,
                  node_count_range_per_subgraph=[10, 20],
                  num_features=3,
                  dist_params=[(0, 1), (10, 5), (-2, 3)],
                  change_window=50,
                  change_bounds=15,
                  steps=1000)
    graph.draw()


if __name__ == '__main__':
    test()
