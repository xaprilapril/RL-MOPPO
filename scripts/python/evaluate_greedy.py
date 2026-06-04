"""Evalua el baseline greedy de Liu & Terzi y guarda resultados en el mismo
formato que evaluate.py para integrarse con make_comparison_figs.py."""
import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from graph_anon_morl.baselines import greedy_k_anonymize
from graph_anon_morl.datasets import load_facebook_ego
from graph_anon_morl.evaluation import compute_pareto_front, hypervolume_indicator
from graph_anon_morl.utils import compute_k_anonymity_reward, compute_utility_reward
from graph_anon_morl.audit import degree_quartile_audit, fairness_disparity_index


def make_graph(name, n_nodes, seed):
    if name == "karate":
        return nx.karate_club_graph()
    if name == "barbell":
        return nx.barbell_graph(n_nodes // 2, 1)
    if name == "facebook":
        return load_facebook_ego(n_nodes=n_nodes, seed=seed)
    return nx.erdos_renyi_graph(n_nodes, 0.3, seed=seed)


def main():
    p = argparse.ArgumentParser(description="Evalua greedy baseline de k-anonimato")
    p.add_argument("--graph",   default="karate",
                   choices=["karate", "barbell", "facebook", "random"])
    p.add_argument("--n_nodes", type=int, default=20)
    p.add_argument("--k",       type=int, default=2)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out_dir", default="outputs/eval_greedy")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G0 = make_graph(args.graph, args.n_nodes, args.seed)
    print(f"Grafo '{args.graph}': {G0.number_of_nodes()} nodos, "
          f"{G0.number_of_edges()} aristas, k={args.k}")

    G_anon = greedy_k_anonymize(G0, args.k)

    r_priv = compute_k_anonymity_reward(G_anon, args.k)
    r_util = compute_utility_reward(G_anon, G0)

    flips = sum(1 for u, v in G0.edges() if not G_anon.has_edge(u, v))
    flips += sum(1 for u, v in G_anon.edges() if not G0.has_edge(u, v))

    print(f"r_priv = {r_priv:.4f}  (0.0 = k-anonimo)")
    print(f"r_util = {r_util:.4f}")
    print(f"flips  = {flips}")

    # El greedy produce un punto fijo (no depende de alpha).
    # Lo replicamos para el sweep completo para que sea comparable con los demas.
    results = [(float(a), float(r_priv), float(r_util))
               for a in np.linspace(0.0, 1.0, 21)]

    reward_vecs = [(r[1], r[2]) for r in results]
    pareto_idx  = compute_pareto_front(reward_vecs)
    pareto_pts  = [reward_vecs[i] for i in pareto_idx]
    hv          = hypervolume_indicator(pareto_pts)

    audit_stats = degree_quartile_audit(G0, G_anon)
    disparity   = fairness_disparity_index(audit_stats)

    print(f"HV     = {hv:.4f}")
    print(f"Delta  = {disparity:.3f}")

    eval_results = {
        "method":            "greedy_liu_terzi",
        "hypervolume":        hv,
        "pareto_front_size":  len(pareto_pts),
        "pareto_points":      pareto_pts,
        "all_results":        results,
        "fairness_audit":     {k: {sk: float(sv) for sk, sv in v.items()}
                               for k, v in audit_stats.items()},
        "disparity_index":    float(disparity),
        "flips":              flips,
    }

    out_path = out_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
