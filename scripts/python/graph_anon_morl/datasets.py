"""Dataset loaders for graph anonymization experiments."""
import gzip
import urllib.request
from pathlib import Path

import networkx as nx

FACEBOOK_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"


def load_facebook_ego(n_nodes: int = 100, seed: int = 42, data_dir: str = "data") -> nx.Graph:
    """
    Load ego-Facebook (SNAP). Downloads once and caches locally.
    Returns a connected BFS subgraph of n_nodes nodes rooted at the highest-degree node.
    Full graph: 4039 nodes, 88234 edges.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    gz_path = data_path / "facebook_combined.txt.gz"

    if not gz_path.exists():
        print(f"Downloading ego-Facebook from SNAP -> {gz_path} ...")
        urllib.request.urlretrieve(FACEBOOK_URL, gz_path)
        print("Download complete.")

    with gzip.open(gz_path, "rt") as f:
        G_full = nx.read_edgelist(f, nodetype=int)

    # BFS from highest-degree node gives a dense, realistic subgraph
    seed_node = max(G_full.degree, key=lambda x: x[1])[0]
    bfs_order = list(nx.bfs_tree(G_full, seed_node).nodes())
    sub_nodes = bfs_order[:n_nodes]
    sub = G_full.subgraph(sub_nodes).copy()
    sub = nx.convert_node_labels_to_integers(sub)
    return sub
