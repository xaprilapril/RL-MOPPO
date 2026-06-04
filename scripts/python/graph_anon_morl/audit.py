import numpy as np
import networkx as nx


def degree_quartile_audit(G0: nx.Graph, G_final: nx.Graph):
    """
    Measure edge-loss rate by degree quartile.
    Returns dict keyed by quartile label with per-group stats.
    """
    degrees = dict(G0.degree())
    deg_vals = list(degrees.values())
    q1 = np.percentile(deg_vals, 25)
    q3 = np.percentile(deg_vals, 75)

    def quartile(d):
        if d <= q1:
            return "Q1_peripheral"
        if d >= q3:
            return "Q4_hub"
        return "Q2Q3_middle"

    groups = {n: quartile(d) for n, d in degrees.items()}

    edge_changes = {n: 0 for n in G0.nodes()}
    for u, v in G0.edges():
        if not G_final.has_edge(u, v):
            edge_changes[u] += 1
            edge_changes[v] += 1
    for u, v in G_final.edges():
        if not G0.has_edge(u, v):
            edge_changes[u] += 1
            edge_changes[v] += 1

    stats = {}
    for label in ("Q1_peripheral", "Q2Q3_middle", "Q4_hub"):
        nodes = [n for n, g in groups.items() if g == label]
        if not nodes:
            continue
        avg_deg = float(np.mean([degrees[n] for n in nodes]))
        avg_loss = float(np.mean([edge_changes[n] for n in nodes]))
        stats[label] = {
            "n_nodes": len(nodes),
            "avg_degree_G0": avg_deg,
            "avg_edge_loss": avg_loss,
            "normalized_loss": avg_loss / max(avg_deg, 1e-6),
        }
    return stats


def fairness_disparity_index(audit_stats: dict) -> float:
    """
    Q1 normalized loss / Q4 normalized loss.
    > 1 means peripheral nodes bear disproportionate anonymization cost.
    """
    q1 = audit_stats.get("Q1_peripheral", {}).get("normalized_loss", 0.0)
    q4 = audit_stats.get("Q4_hub", {}).get("normalized_loss", 1e-6)
    return q1 / max(q4, 1e-6)


def trex_trajectory_cluster(trajectories, n_clusters: int = 3):
    """
    Cluster trajectories by cumulative [r_priv, r_util] profile (TREX-style).
    trajectories: list of lists of 2-element reward vectors.
    Returns (labels, features, cluster_centers).
    """
    from sklearn.cluster import KMeans

    features = np.array([
        [np.sum([step[0] for step in traj]), np.sum([step[1] for step in traj])]
        for traj in trajectories
    ])
    k = min(n_clusters, len(features))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(features)
    return labels, features, km.cluster_centers_
