"""
figures.py — Generate all paper figures for MOPPO Graph Anonymization.

Produces:
  fig1_pareto_frontier.pdf/.png   — Pareto frontier (Karate + Facebook)
  fig2_training_curve.pdf/.png    — Learning curves (both datasets)
  fig3_trex_analysis.pdf/.png     — TREX trajectory clustering
  fig4_fairness_audit.pdf/.png    — Fairness audit by degree quartile

Usage:
  python figures.py --eval_fb outputs/eval/eval_results.json \
                    --eval_kc outputs/eval_karate/eval_results.json \
                    --metrics_fb outputs/moppo/train_metrics.json \
                    --metrics_kc outputs/moppo_karate/train_metrics.json \
                    --out_dir outputs/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Shared style ────────────────────────────────────────────────────────────
PALETTE = {
    "moppo":      "#1E88E5",   # blue
    "pareto":     "#E53935",   # red
    "utility":    "#43A047",   # green
    "balanced":   "#FB8C00",   # orange
    "privacy":    "#8E24AA",   # purple
    "neutral":    "#78909C",   # grey
    "threshold":  "#FF7043",   # deep orange (Rawls line)
    "ideal":      "#26A69A",   # teal (ideal line)
}
FONT = {"family": "DejaVu Sans", "size": 13}
matplotlib.rc("font", **FONT)
matplotlib.rc("axes", titlesize=15, labelsize=13, linewidth=1.2)
matplotlib.rc("xtick", labelsize=11)
matplotlib.rc("ytick", labelsize=11)
matplotlib.rc("legend", fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
matplotlib.rc("figure", dpi=150)


def load_json(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Figure 1: Pareto Frontier ───────────────────────────────────────────────
def fig1_pareto(eval_fb, eval_kc, out_dir):
    has_kc = eval_kc is not None
    ncols  = 2 if has_kc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    datasets = [("ego-Facebook (n=100)", eval_fb, axes[0])]
    if has_kc:
        datasets.append(("Karate Club (n=34, ablacion)", eval_kc, axes[1]))

    for title, data, ax in datasets:
        if data is None:
            ax.set_visible(False)
            continue

        results = data["all_results"]
        alphas  = [r[0] for r in results]
        r_privs = [r[1] for r in results]
        r_utils = [r[2] for r in results]

        # Background scatter (all points, colored by alpha)
        sc = ax.scatter(r_privs, r_utils,
                        c=alphas, cmap="coolwarm_r", s=120,
                        vmin=0, vmax=1, zorder=3, alpha=0.85,
                        edgecolors="white", linewidths=0.5)

        # Alpha annotations
        for a, rp, ru in zip(alphas, r_privs, r_utils):
            if a in (0.0, 0.5, 1.0):
                ax.annotate(f"α={a:.1f}", (rp, ru),
                            textcoords="offset points", xytext=(6, 5),
                            fontsize=10, color="#333333",
                            arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                            lw=0.8))

        # Pareto front
        pareto = data.get("pareto_points", [])
        if pareto:
            px, py = zip(*pareto)
            px_s = sorted(set(zip(px, py)), key=lambda p: p[0])
            ax.plot([p[0] for p in px_s], [p[1] for p in px_s],
                    "o--", color=PALETTE["pareto"], linewidth=2,
                    markersize=10, zorder=5, label="Pareto front")

        # Baseline: original graph (no modification)
        ax.axvline(x=-0.09, color=PALETTE["neutral"], linestyle=":",
                   linewidth=1.5, label="Baseline (sin modificar, r_priv=−0.09)")

        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label("α (peso privacidad)", fontsize=11)
        cb.ax.tick_params(labelsize=10)

        hv = data.get("hypervolume", 0)
        ax.set_xlabel("r_priv  (k-anonimato final)  ↑ mejor", fontsize=12)
        ax.set_ylabel("r_util  (coef. clustering)  ↑ mejor", fontsize=12)
        ax.set_title(f"Frente de Pareto MOPPO\n{title}  |  HV={hv:.3f}",
                     fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

    fig.suptitle("Figura 1 — Frente de Pareto: Privacidad vs. Utilidad",
                 fontsize=16, fontweight="bold", y=1.02)
    _save(fig, out_dir, "fig1_pareto_frontier")


# ── Figure 2: Training Curves ───────────────────────────────────────────────
def fig2_training(metrics_fb, metrics_kc, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    datasets = [
        ("ego-Facebook (n=100)", metrics_fb, PALETTE["moppo"]),
        ("Karate Club (n=34)", metrics_kc, PALETTE["utility"]),
    ]

    for (label, data, color), ax in zip(datasets, axes):
        if data is None:
            ax.set_visible(False)
            continue

        iters   = [d["iter"] for d in data]
        rewards = [d["mean_reward"] for d in data]

        # Smoothed curve (rolling mean window=20)
        window  = min(20, len(rewards) // 5)
        smooth  = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_smooth = np.arange(window - 1, len(rewards))

        ax.plot(iters, rewards, alpha=0.25, color=color, linewidth=1)
        ax.plot(x_smooth, smooth, color=color, linewidth=2.5,
                label=f"Suavizado (ventana={window})")

        ax.axhline(y=0, color="#AAAAAA", linestyle="--", linewidth=1)
        ax.set_xlabel("Iteración de entrenamiento", fontsize=12)
        ax.set_ylabel("Recompensa escalar media  ↑ mejor", fontsize=12)
        ax.set_title(f"Curva de Aprendizaje — {label}", fontsize=14,
                     fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#FAFAFA")

        # Annotate final value
        ax.annotate(f"Final: {rewards[-1]:.4f}",
                    xy=(iters[-1], rewards[-1]),
                    xytext=(-80, 15), textcoords="offset points",
                    fontsize=10, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    fig.suptitle("Figura 2 — Curvas de Aprendizaje MOPPO",
                 fontsize=16, fontweight="bold")
    _save(fig, out_dir, "fig2_training_curve")


# ── Figure 3: TREX Analysis ─────────────────────────────────────────────────
def fig3_trex(eval_fb, out_dir):
    if eval_fb is None or "trex" not in eval_fb:
        print("  [fig3] No TREX data — skipping")
        return

    trex  = eval_fb["trex"]
    stats = trex["cluster_stats"]

    cluster_colors = [PALETTE["utility"], PALETTE["balanced"], PALETTE["privacy"]]
    semantic_map   = {"utility-dominant": 0, "balanced": 1, "privacy-dominant": 2}
    labels_es      = ["Dominante\nutilidad", "Balanceado", "Dominante\nprivacidad"]

    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Panel A: mean alpha per cluster + std (bar)
    ax0 = fig.add_subplot(gs[0])
    alphas_mean = [s["mean_alpha"] for s in stats]
    alphas_std  = [s["std_alpha"]  for s in stats]
    bars = ax0.bar(labels_es, alphas_mean, yerr=alphas_std,
                   color=cluster_colors, alpha=0.85, capsize=7,
                   edgecolor="white", linewidth=1.5, width=0.5)
    ax0.set_ylabel("Valor de α (peso de privacidad)", fontsize=12)
    ax0.set_title("A — Distribución de α\npor cluster", fontsize=13,
                  fontweight="bold")
    ax0.set_ylim(0, 1)
    ax0.axhline(0.5, color="#AAAAAA", linestyle="--", linewidth=1,
                label="α=0.5 (balance)")
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3, axis="y")
    ax0.set_facecolor("#FAFAFA")

    # Panel B: r_priv vs r_util scatter per cluster
    ax1 = fig.add_subplot(gs[1])
    for cs in stats:
        idx   = semantic_map.get(cs["semantic"], 0)
        color = cluster_colors[idx]
        label = labels_es[idx]
        ax1.scatter(cs["mean_r_priv"], cs["mean_r_util"],
                    c=color, s=300, zorder=4, edgecolors="white",
                    linewidths=1.5, label=label)
        ax1.annotate(f"Δ={cs['mean_disparity']:.1f}",
                     (cs["mean_r_priv"], cs["mean_r_util"]),
                     textcoords="offset points", xytext=(8, 6),
                     fontsize=10, color=color, fontweight="bold")

    ax1.set_xlabel("r_priv final (k-anonimato)  ↑ mejor", fontsize=12)
    ax1.set_ylabel("r_util final (clustering)  ↑ mejor", fontsize=12)
    ax1.set_title("B — Posición en espacio\nobjetivo por cluster",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#FAFAFA")

    # Panel C: Disparity Delta per cluster
    ax2 = fig.add_subplot(gs[2])
    deltas      = [s["mean_disparity"] for s in stats]
    deltas_std  = [s["std_disparity"]  for s in stats]
    bar_colors  = [cluster_colors[semantic_map.get(s["semantic"], 0)] for s in stats]

    bars2 = ax2.bar(labels_es, deltas, yerr=deltas_std,
                    color=bar_colors, alpha=0.85, capsize=7,
                    edgecolor="white", linewidth=1.5, width=0.5)
    ax2.axhline(1.0, color=PALETTE["ideal"],     linestyle="--",
                linewidth=2, label="Ideal (Δ=1.0)")
    ax2.axhline(1.5, color=PALETTE["threshold"], linestyle="--",
                linewidth=2, label="Umbral Rawls (Δ=1.5)")
    ax2.set_ylabel("Índice de Disparidad Δ = Q1/Q4", fontsize=12)
    ax2.set_title("C — Equidad Rawlsiana\npor cluster",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_facecolor("#FAFAFA")

    # Color zones on panel C
    ax2.axhspan(0, 1.5, alpha=0.05, color=PALETTE["ideal"])
    ax2.axhspan(1.5, max(deltas) * 1.2 + 2, alpha=0.05, color=PALETTE["pareto"])

    fig.suptitle(
        "Figura 3 — TREX: Análisis de Trayectorias y Auditoría Ética\n"
        "ego-Facebook (n=100), 30 episodios, k=3 clusters",
        fontsize=15, fontweight="bold")
    _save(fig, out_dir, "fig3_trex_analysis")


# ── Figure 4: Fairness Audit ────────────────────────────────────────────────
def fig4_fairness(eval_fb, out_dir):
    if eval_fb is None:
        return

    audit   = eval_fb.get("fairness_audit", {})
    trex    = eval_fb.get("trex", {})
    c_stats = trex.get("cluster_stats", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Panel A: Normalized loss per quartile (static audit)
    ax = axes[0]
    groups   = list(audit.keys())
    nloss    = [audit[g]["normalized_loss"] for g in groups]
    avg_degs = [audit[g]["avg_degree_G0"]   for g in groups]
    labels   = [f"Q1 Periférico\n(deg≈{avg_degs[0]:.1f})",
                f"Q2Q3 Intermedio\n(deg≈{avg_degs[1]:.1f})",
                f"Q4 Hub\n(deg≈{avg_degs[2]:.1f})"]
    q_colors = [PALETTE["privacy"], PALETTE["balanced"], PALETTE["utility"]]

    bars = ax.bar(labels, nloss, color=q_colors, alpha=0.85,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars, nloss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Pérdida de aristas normalizada (loss/deg)", fontsize=12)
    ax.set_title("A — Auditoría Estática (α=0.5)\nPérdida por cuartil de grado",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#FAFAFA")
    ax.set_ylim(0, max(nloss) * 1.4 + 0.01)

    disp = eval_fb.get("disparity_index", 0)
    ax.text(0.97, 0.95, f"Δ = {disp:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=PALETTE["ideal"] if disp <= 1.5 else PALETTE["pareto"],
                      linewidth=2))

    # Panel B: Delta per TREX cluster + threshold
    ax = axes[1]
    if c_stats:
        semantic_order = ["utility-dominant", "balanced", "privacy-dominant"]
        labels_es      = ["Dominante\nutilidad", "Balanceado", "Dominante\nprivacidad"]
        cluster_colors = [PALETTE["utility"], PALETTE["balanced"], PALETTE["privacy"]]

        ordered = sorted(c_stats, key=lambda x: semantic_order.index(x["semantic"])
                         if x["semantic"] in semantic_order else 99)
        deltas  = [s["mean_disparity"] for s in ordered]
        stds    = [s["std_disparity"]  for s in ordered]
        ns      = [s["n_episodes"]     for s in ordered]

        bars2 = ax.bar(labels_es, deltas, yerr=stds,
                       color=cluster_colors, alpha=0.85, capsize=7,
                       edgecolor="white", linewidth=1.5, width=0.5)

        for bar, val, n in zip(bars2, deltas, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + stds[bars2.index(bar)] + 0.3,
                    f"Δ={val:.1f}\n(n={n})",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.axhline(1.0, color=PALETTE["ideal"], linestyle="--",
                   linewidth=2.5, label="Ideal (Δ=1.0)")
        ax.axhline(1.5, color=PALETTE["threshold"], linestyle="--",
                   linewidth=2.5, label="Umbral Rawlsiano (Δ=1.5)")

        # Zone shading
        ax.axhspan(0, 1.5, alpha=0.06, color=PALETTE["ideal"],
                   label="Zona equitativa")
        ax.axhspan(1.5, max(deltas) * 1.3 + 2, alpha=0.06,
                   color=PALETTE["pareto"], label="Zona de sesgo estructural")

        ax.set_ylabel("Índice de Disparidad Δ (Q1/Q4)", fontsize=12)
        ax.set_title("B — Auditoría TREX (30 episodios estocásticos)\nDisparidad por modo conductual",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Figura 4 — Auditoría de Equidad Rawlsiana\n"
        "MOPPO: nodos periféricos (Q1) vs. hubs (Q4)",
        fontsize=15, fontweight="bold")
    _save(fig, out_dir, "fig4_fairness_audit")


# ── Helper ──────────────────────────────────────────────────────────────────
def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_fb",    default="outputs/eval/eval_results.json")
    ap.add_argument("--eval_kc",    default="outputs/eval_karate/eval_results.json")
    ap.add_argument("--metrics_fb", default="outputs/moppo/train_metrics.json")
    ap.add_argument("--metrics_kc", default="outputs/moppo_karate/train_metrics.json")
    ap.add_argument("--out_dir",    default="outputs/figures")
    args = ap.parse_args()

    eval_fb    = load_json(args.eval_fb)
    eval_kc    = load_json(args.eval_kc)
    metrics_fb = load_json(args.metrics_fb)
    metrics_kc = load_json(args.metrics_kc)

    print("Generando figuras del paper...")
    print("\n[Fig 1] Pareto Frontier")
    fig1_pareto(eval_fb, eval_kc, args.out_dir)
    print("\n[Fig 2] Training Curves")
    fig2_training(metrics_fb, metrics_kc, args.out_dir)
    print("\n[Fig 3] TREX Analysis")
    fig3_trex(eval_fb, args.out_dir)
    print("\n[Fig 4] Fairness Audit")
    fig4_fairness(eval_fb, args.out_dir)
    print(f"\nListo. Figuras guardadas en: {args.out_dir}/")


if __name__ == "__main__":
    main()
