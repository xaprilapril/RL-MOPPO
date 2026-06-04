"""Genera figuras comparativas MOPPO vs baselines."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "..", "..")

def load_eval(rel):
    with open(os.path.join(BASE, rel)) as f:
        return json.load(f)

METHODS = ["PPO a=0", "PPO a=0.5", "PPO a=1.0", "MOPPO"]
COLORS  = ["#78909C", "#FF8F00", "#6A1B9A", "#D32F2F"]
MARKERS = ["o", "s", "^", "*"]
SIZES   = [70, 70, 70, 180]
HATCH   = ["////", "\\\\\\\\", "....", ""]

K2 = [
    "outputs/eval_baseline_k2_a0/eval_results.json",
    "outputs/eval_baseline_k2_a05/eval_results.json",
    "outputs/eval_baseline_k2_a1/eval_results.json",
    "outputs/eval_curriculum/eval_results.json",
]
K4 = [
    "outputs/eval_baseline_k4_a0/eval_results.json",
    "outputs/eval_baseline_k4_a05/eval_results.json",
    "outputs/eval_baseline_k4_a1/eval_results.json",
    "outputs/eval_k4/eval_results.json",
]

FB_PATHS = [
    "outputs/eval_baseline_fb_a0/eval_results.json",
    "outputs/eval_baseline_fb_a05/eval_results.json",
    "outputs/eval_baseline_fb_a1/eval_results.json",
    "outputs/eval_fb/eval_results.json",
]

def fb_available():
    return all(os.path.exists(os.path.join(BASE, p)) for p in FB_PATHS)


def fig_pareto_comparison():
    graphs = [("Karate k=2", K2), ("Karate k=4", K4)]
    if fb_available():
        graphs.append(("Facebook n=100", FB_PATHS))

    ncols = len(graphs)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5))
    if ncols == 1:
        axes = [axes]

    for ax, (title, paths) in zip(axes, graphs):
        for path, label, marker, color, ms in zip(paths, METHODS, MARKERS, COLORS, SIZES):
            r = load_eval(path)
            pts = r["all_results"]
            rp = [p[1] for p in pts]
            ru = [p[2] for p in pts]
            ax.scatter(rp, ru, marker=marker, color=color, s=ms,
                       alpha=0.70, label=label, zorder=3)
            pareto = r["pareto_points"]
            if pareto:
                px, py = zip(*pareto)
                ax.scatter(px, py, marker=marker, color=color,
                           s=ms * 2, edgecolors="black", linewidths=1.4, zorder=5)

        ax.set_xlabel("r_priv (k-anonimato)", fontsize=11)
        ax.set_ylabel("r_util (clustering coeff.)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(True, alpha=0.2)

    plt.suptitle("MOPPO vs. baselines PPO alpha fijo", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(BASE, "outputs/figures/fig_pareto_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("OK:", out)


def fig_hv_bars():
    all_sets = [("k=2", K2), ("k=4", K4)]
    if fb_available():
        all_sets.append(("Facebook", FB_PATHS))

    ncols = len(all_sets) + 1  # +1 para panel de Pareto points
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    x = np.arange(len(METHODS))

    ylims = {"k=2": (0.930, 0.985), "k=4": (0.33, 0.55), "Facebook": (0.55, 1.0)}

    hv_data = {}
    pt_data = {}
    for label, paths in all_sets:
        hv_data[label] = [load_eval(p)["hypervolume"] for p in paths]
        pt_data[label]  = [load_eval(p)["pareto_front_size"] for p in paths]

    for i, (label, paths) in enumerate(all_sets):
        ax = axes[i]
        hvs = hv_data[label]
        ylim = ylims.get(label, (min(hvs)*0.98, max(hvs)*1.02))
        bars = ax.bar(x, hvs, color=COLORS, alpha=0.85, edgecolor="white",
                      hatch=HATCH, width=0.6)
        for bar, v in zip(bars, hvs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (ylim[1]-ylim[0]) * 0.012,
                    f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=9, rotation=15)
        ax.set_ylabel("Hipervolumen", fontsize=10)
        ax.set_title(f"HV — {label}", fontsize=11, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.2, axis="y")

    ax_pts = axes[-1]
    width = 0.22
    offsets = np.linspace(-width, width, len(all_sets))
    for offset, (label, _) in zip(offsets, all_sets):
        pts = pt_data[label]
        bars = ax_pts.bar(x + offset, pts, width=width * 0.9,
                          color=COLORS, alpha=0.75, edgecolor="white",
                          label=label)
        for bar, v in zip(bars, pts):
            ax_pts.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", fontsize=9, fontweight="bold")

    ax_pts.set_xticks(x)
    ax_pts.set_xticklabels(METHODS, fontsize=9, rotation=15)
    ax_pts.set_ylabel("Puntos Pareto no dominados", fontsize=10)
    ax_pts.set_title("Cobertura del frente de Pareto", fontsize=11, fontweight="bold")
    ax_pts.legend(fontsize=9)
    ax_pts.grid(True, alpha=0.2, axis="y")
    ax_pts.set_ylim(0, 7)

    plt.suptitle("Comparacion MOPPO vs. baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(BASE, "outputs/figures/fig_hv_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("OK:", out)


def fig_fairness_comparison():
    all_sets = [("Karate k=2", K2), ("Karate k=4", K4)]
    if fb_available():
        all_sets.append(("Facebook n=100", FB_PATHS))

    ncols = len(all_sets)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    x = np.arange(len(METHODS))

    for ax, (title, paths) in zip(axes, all_sets):
        deltas = [load_eval(p)["disparity_index"] for p in paths]
        bars = ax.bar(x, deltas, color=COLORS, alpha=0.85, edgecolor="white", width=0.55)
        for bar, v in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.axhline(1.0, color="green",  linestyle="--", linewidth=1.5, label="Ideal Delta=1")
        ax.axhline(1.5, color="orange", linestyle="--", linewidth=1.5, label="Umbral Rawls")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=10)
        ax.set_ylabel("Indice de disparidad Delta (Q1/Q4)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(0, max(deltas) * 1.2)

    plt.suptitle("Sesgo de equidad por metodo", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(BASE, "outputs/figures/fig_fairness_comparison.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print("OK:", out)


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE, "outputs/figures"), exist_ok=True)
    fig_pareto_comparison()
    fig_hv_bars()
    fig_fairness_comparison()
    print("Todas las figuras generadas.")
