"""Evaluate MOPPO checkpoint: Pareto sweep + fairness audit + TREX clustering."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from graph_anon_morl.audit import degree_quartile_audit, fairness_disparity_index
from graph_anon_morl.datasets import load_facebook_ego
from graph_anon_morl.env import GraphAnonEnv
from graph_anon_morl.evaluation import (
    compute_pareto_front,
    evaluate_policy,
    hypervolume_indicator,
)
from graph_anon_morl.models import MOPPOActorCritic
from graph_anon_morl.trex import run_trex_analysis, plot_trex_results


def make_graph(name, n_nodes, seed):
    if name == "karate":
        return nx.karate_club_graph()
    if name == "barbell":
        return nx.barbell_graph(n_nodes // 2, 1)
    if name == "facebook":
        return load_facebook_ego(n_nodes=n_nodes, seed=seed)
    return nx.erdos_renyi_graph(n_nodes, 0.3, seed=seed)


def load_model(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device)
    model = MOPPOActorCritic(
        obs_dim=ck["obs_dim"],
        n_actions=ck["n_actions"],
        hidden_dim=ck["args"].get("hidden_dim", 256),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck["args"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/moppo/moppo_checkpoint.pt")
    p.add_argument("--n_weights",  type=int, default=11)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--trex_episodes", type=int, default=30,
                   help="Episodes for TREX trajectory clustering")
    p.add_argument("--trex_clusters",  type=int, default=3)
    p.add_argument("--out_dir", default="outputs/eval")
    args = p.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, train_args = load_model(args.checkpoint, device)
    G0  = make_graph(train_args["graph"], train_args["n_nodes"], train_args["seed"])
    env = GraphAnonEnv(G0, k=train_args["k"], T=train_args["episode_len"])

    # ── 1. Pareto frontier sweep ─────────────────────────────────────────────
    print("=" * 50)
    print("1/3  Pareto frontier sweep...")
    results     = evaluate_policy(env, model, args.n_weights, args.n_episodes, str(device))
    reward_vecs = [(r[1], r[2]) for r in results]
    pareto_idx  = compute_pareto_front(reward_vecs)
    pareto_pts  = [reward_vecs[i] for i in pareto_idx]
    hv          = hypervolume_indicator(pareto_pts)
    print(f"     Hypervolume: {hv:.4f}  |  Pareto front size: {len(pareto_pts)}")

    alphas  = [r[0] for r in results]
    r_privs = [r[1] for r in results]
    r_utils = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(r_privs, r_utils, c=alphas, cmap="viridis", s=80, zorder=3)
    if pareto_pts:
        px, py = zip(*pareto_pts)
        ax.scatter(px, py, marker="*", s=200, color="red", label="Pareto front", zorder=4)
    plt.colorbar(sc, ax=ax, label="alpha (privacy weight)")
    ax.set_xlabel("r_priv (k-anonymity reward)")
    ax.set_ylabel("r_util (clustering coeff. reward)")
    ax.set_title("MOPPO -- Pareto Frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_frontier.png", dpi=150)
    plt.close()

    # ── 2. Fairness audit at alpha=0.5, stochastic, averaged over 5 episodes ──
    print("2/3  Fairness audit (alpha=0.5, stochastic, n=5)...")
    all_audit_stats = []
    all_disparities = []
    with torch.no_grad():
        for _ in range(5):
            obs, _ = env.reset(weight=[0.5, 0.5])
            done   = False
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
                action, _, _, _ = model.get_action(obs_t, deterministic=False, mask=mask_t)
                obs, _, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
            ep_stats  = degree_quartile_audit(env.G0, env.G)
            ep_disp   = fairness_disparity_index(ep_stats)
            all_audit_stats.append(ep_stats)
            all_disparities.append(ep_disp)

    # Average normalized_loss across episodes per quartile
    import numpy as _np
    groups = list(all_audit_stats[0].keys())
    audit_stats = {}
    for g in groups:
        audit_stats[g] = {
            k: float(_np.mean([s[g][k] for s in all_audit_stats]))
            for k in all_audit_stats[0][g]
        }
    disparity = float(_np.mean(all_disparities))
    print(f"     Disparity index Delta (Q1/Q4): {disparity:.3f}  (mean over 5 episodes)")
    for group, s in audit_stats.items():
        print(f"     {group}: norm_loss={s['normalized_loss']:.3f}")

    # ── 3. TREX trajectory clustering ────────────────────────────────────────
    print(f"3/3  TREX analysis ({args.trex_episodes} episodes, k={args.trex_clusters})...")
    trex = run_trex_analysis(
        env, model,
        n_episodes=args.trex_episodes,
        n_clusters=args.trex_clusters,
        device=str(device),
        seed=42,
    )
    plot_trex_results(trex, out_path=str(out_dir / "trex_analysis.png"))

    print("\n  TREX cluster summary:")
    for cs in trex["cluster_stats"]:
        fair = "OK" if cs["mean_disparity"] <= 1.5 else "SESGO"
        print(f"    [{cs['semantic']:20s}]  "
              f"n={cs['n_episodes']:2d}  "
              f"mean_alpha={cs['mean_alpha']:.2f}  "
              f"r_priv={cs['mean_r_priv']:.3f}  "
              f"r_util={cs['mean_r_util']:.3f}  "
              f"Delta={cs['mean_disparity']:.2f} [{fair}]")

    # ── Save all results ─────────────────────────────────────────────────────
    eval_results = {
        "hypervolume":        hv,
        "pareto_front_size":  len(pareto_pts),
        "pareto_points":      pareto_pts,
        "all_results":        results,
        "fairness_audit":     {k: {sk: float(sv) for sk, sv in v.items()}
                               for k, v in audit_stats.items()},
        "disparity_index":    float(disparity),
        "trex": {
            "n_episodes": trex["n_episodes"],
            "n_clusters": trex["n_clusters"],
            "cluster_stats": [
                {k: v for k, v in cs.items()
                 if k not in ("alphas", "r_privs", "r_utils", "disparities")}
                for cs in trex["cluster_stats"]
            ],
        },
    }
    json_path = out_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nOutputs saved to {out_dir}/")
    print(f"  pareto_frontier.png")
    print(f"  trex_analysis.png")
    print(f"  eval_results.json")


if __name__ == "__main__":
    main()
