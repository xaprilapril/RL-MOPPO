import numpy as np
import networkx as nx


def compute_k_anonymity_reward(G, k=2):
    """k-anonymity penalty: sum of max(0, k - |equiv_class(deg)|) across nodes, normalized."""
    degree_counts = {}
    for _, d in G.degree():
        degree_counts[d] = degree_counts.get(d, 0) + 1
    penalty = sum(max(0, k - degree_counts[d]) for _, d in G.degree())
    return -penalty / max(G.number_of_nodes(), 1)


def compute_utility_reward(G, G0):
    """Negative absolute deviation of clustering coefficient from original graph."""
    c_t = nx.average_clustering(G)
    c_0 = nx.average_clustering(G0)
    return -abs(c_t - c_0)


def graph_to_flat_adj(G, n):
    """Return flattened adjacency matrix as float32 numpy array."""
    return nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32).flatten()
