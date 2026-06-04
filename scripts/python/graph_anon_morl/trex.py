"""
TREX — Trajectory Explanations for MORL policies.

Inspired by: Rajapakse et al. (2026) arXiv:2603.21988

Pipeline:
  1. Run N episodes with random alpha ~ U(0,1)
  2. For each episode: record cumulative [r_priv, r_util] at every step
  3. K-means cluster the trajectory profiles (k=3)
  4. Per cluster: compute mean alpha, final rewards, fairness disparity Delta

Expected cluster semantics (with k=3):
  - Cluster 0: privacy-dominant  (high alpha -> many edge flips for anonymity)
  - Cluster 1: balanced          (mid alpha)
  - Cluster 2: utility-dominant  (low alpha -> few changes to preserve structure)

If clusters do NOT separate by alpha, the policy lacks weight-conditioning quality.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import compute_k_anonymity_reward, compute_utility_reward
from .audit import degree_quartile_audit, fairness_disparity_index


def run_trex_analysis(env, model, n_episodes=30, n_clusters=3, device="cpu", seed=42):
    """
    Run TREX trajectory clustering on a trained MOPPO policy.

    Parameters
    ----------
    env        : GraphAnonEnv instance
    model      : trained MOPPOActorCritic
    n_episodes : number of random-alpha episodes to run (paper: 30)
    n_clusters : k for k-means (paper: 3)
    device     : torch device string
    seed       : random seed for k-means reproducibility

    Returns
    -------
    dict with keys:
        trajectories   : list of per-episode dicts
        cluster_labels : np.ndarray shape (n_episodes,)
        cluster_stats  : list of per-cluster summary dicts
        feature_matrix : np.ndarray shape (n_episodes, T*2)
    """
    rng = np.random.default_rng(seed)
    model.eval()
    trajectories = []

    with torch.no_grad():
        for ep in range(n_episodes):
            # Sample random alpha for this episode
            alpha = float(rng.uniform(0.0, 1.0))
            obs, _ = env.reset(weight=[alpha, 1.0 - alpha])
            done = False

            cumulative_r_priv = 0.0
            cumulative_r_util = 0.0
            step_profile = []   # list of (cum_r_priv, cum_r_util) at each step

            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(env.get_action_mask()).unsqueeze(0).to(device)
                action, _, _, _ = model.get_action(obs_t, deterministic=False, mask=mask_t)
                obs, _, terminated, truncated, info = env.step(action.item())
                done = terminated or truncated

                rv = info.get("reward_vec", None)
                if rv is not None:
                    cumulative_r_priv += float(rv[0])
                    cumulative_r_util += float(rv[1])
                step_profile.append((cumulative_r_priv, cumulative_r_util))

            # Final graph metrics
            final_r_priv = compute_k_anonymity_reward(env.G, env.k)
            final_r_util = compute_utility_reward(env.G, env.G0)
            audit_stats  = degree_quartile_audit(env.G0, env.G)
            disparity    = fairness_disparity_index(audit_stats)

            trajectories.append({
                "ep":           ep,
                "alpha":        alpha,
                "profile":      step_profile,          # list of T (cum_rp, cum_ru) tuples
                "final_r_priv": final_r_priv,
                "final_r_util": final_r_util,
                "disparity":    disparity,
                "audit_stats":  audit_stats,
            })

    # ── Feature matrix: flatten cumulative profile ──────────────────────────
    # Each episode's profile has T rows; flatten to 1D feature vector (T*2,)
    T = len(trajectories[0]["profile"])
    feature_matrix = np.zeros((n_episodes, T * 2), dtype=np.float32)
    for i, traj in enumerate(trajectories):
        flat = []
        for (rp, ru) in traj["profile"]:
            flat.extend([rp, ru])
        feature_matrix[i] = flat

    # Normalize before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_matrix)

    # ── K-means clustering ───────────────────────────────────────────────────
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)

    # ── Per-cluster statistics ───────────────────────────────────────────────
    cluster_stats = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        members = [trajectories[i] for i in range(n_episodes) if mask[i]]
        if not members:
            continue

        alphas      = [m["alpha"]        for m in members]
        r_privs     = [m["final_r_priv"] for m in members]
        r_utils     = [m["final_r_util"] for m in members]
        disparities = [m["disparity"]    for m in members]

        cluster_stats.append({
            "cluster":       c,
            "n_episodes":    len(members),
            "mean_alpha":    float(np.mean(alphas)),
            "std_alpha":     float(np.std(alphas)),
            "mean_r_priv":   float(np.mean(r_privs)),
            "mean_r_util":   float(np.mean(r_utils)),
            "mean_disparity": float(np.mean(disparities)),
            "std_disparity": float(np.std(disparities)),
            "alphas":        alphas,        # kept for plotting
            "r_privs":       r_privs,
            "r_utils":       r_utils,
            "disparities":   disparities,
        })

    # Sort clusters by mean_alpha (ascending) for interpretability
    cluster_stats.sort(key=lambda x: x["mean_alpha"])
    semantic_labels = ["utility-dominant", "balanced", "privacy-dominant"]
    for i, cs in enumerate(cluster_stats):
        cs["semantic"] = semantic_labels[i] if i < len(semantic_labels) else f"cluster-{i}"

    return {
        "trajectories":   trajectories,
        "cluster_labels": cluster_labels,
        "cluster_stats":  cluster_stats,
        "feature_matrix": feature_matrix,
        "n_episodes":     n_episodes,
        "n_clusters":     n_clusters,
    }


def plot_trex_results(trex_result, out_path=None):
    """
    Generate a 3-panel TREX figure:
      Panel 1: Pareto scatter colored by cluster
      Panel 2: Alpha distribution per cluster (box plots)
      Panel 3: Disparity Delta per cluster (bar chart with threshold line)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    cluster_stats  = trex_result["cluster_stats"]
    trajectories   = trex_result["trajectories"]
    cluster_labels = trex_result["cluster_labels"]
    n_clusters     = trex_result["n_clusters"]

    colors = ["#43A047", "#1E88E5", "#E53935"]   # green=utility, blue=balanced, red=privacy
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Pareto scatter ──────────────────────────────────────────────
    ax = axes[0]
    for i, (traj, label) in enumerate(zip(trajectories, cluster_labels)):
        cs = next(c for c in cluster_stats if c["cluster"] == label)
        idx = cluster_stats.index(cs)
        ax.scatter(traj["final_r_priv"], traj["final_r_util"],
                   c=colors[idx % len(colors)], s=60, alpha=0.75, zorder=3)
        ax.annotate(f"α={traj['alpha']:.2f}", (traj["final_r_priv"], traj["final_r_util"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    patches = [mpatches.Patch(color=colors[i % len(colors)],
                               label=f"{cs['semantic']}  (n={cs['n_episodes']})")
               for i, cs in enumerate(cluster_stats)]
    ax.legend(handles=patches, fontsize=8)
    ax.set_xlabel("r_priv (k-anonimato final)", fontsize=10)
    ax.set_ylabel("r_util (clustering coeff final)", fontsize=10)
    ax.set_title("TREX: Trayectorias por Cluster", fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Alpha distribution per cluster ──────────────────────────────
    ax = axes[1]
    alpha_data = [cs["alphas"] for cs in cluster_stats]
    bp = ax.boxplot(alpha_data, patch_artist=True,
                    labels=[cs["semantic"].replace("-", "\n") for cs in cluster_stats])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Valor de alpha (preferencia)", fontsize=10)
    ax.set_title("Distribucion de alpha por Cluster", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Disparity Delta per cluster ────────────────────────────────
    ax = axes[2]
    means = [cs["mean_disparity"] for cs in cluster_stats]
    stds  = [cs["std_disparity"]  for cs in cluster_stats]
    labels = [cs["semantic"].replace("-", "\n") for cs in cluster_stats]
    bar_colors = [colors[min(i, len(colors)-1)] for i in range(len(means))]
    bars = ax.bar(labels, means, yerr=stds, color=bar_colors, alpha=0.8,
                  capsize=5, edgecolor="white")
    ax.axhline(y=1.0, color="green",  linestyle="--", linewidth=1.5, label="Ideal (Δ=1.0)")
    ax.axhline(y=1.5, color="orange", linestyle="--", linewidth=1.5, label="Umbral Rawls (Δ=1.5)")
    ax.set_ylabel("Disparity Index Δ (Q1/Q4)", fontsize=10)
    ax.set_title("Equidad por Cluster (Rawls)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("TREX — Analisis de Trayectorias MOPPO", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"TREX figure -> {out_path}")
    return fig
