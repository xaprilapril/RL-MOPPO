"""Calcula media +/- std de HV y Delta sobre multiples seeds."""
import json, os, sys, statistics
from pathlib import Path

BASE = Path(__file__).parent.parent.parent

CONFIGS = [
    ("moppo_k2", "karate",   2),
    ("moppo_k4", "karate",   4),
    ("moppo_fb", "facebook", 2),
]

SEEDS = [42, 1, 2, 3]


def eval_path(tag, seed):
    if seed == 42:
        name = {"moppo_k2": "eval_curriculum",
                "moppo_k4": "eval_k4",
                "moppo_fb": "eval_fb"}[tag]
        return BASE / "outputs" / name / "eval_results.json"
    return BASE / "outputs" / f"{tag}_s{seed}" / "eval_results.json"


def load_if_exists(p):
    if Path(p).exists():
        with open(p) as f:
            return json.load(f)
    return None


def stats(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    print("=" * 60)
    all_complete = True
    results = {}

    for tag, graph, k in CONFIGS:
        hvs, deltas, pts = [], [], []
        missing = []
        for seed in SEEDS:
            r = load_if_exists(eval_path(tag, seed))
            if r is None:
                missing.append(seed)
                all_complete = False
            else:
                hvs.append(r["hypervolume"])
                deltas.append(r["disparity_index"])
                pts.append(r["pareto_front_size"])

        label = f"{graph} k={k}"
        print(f"\n{label}")
        if missing:
            print(f"  Pendiente seeds: {missing}")
        if hvs:
            hv_m, hv_s = stats(hvs)
            d_m,  d_s  = stats(deltas)
            print(f"  Seeds OK : {[s for s in SEEDS if s not in missing]}")
            print(f"  HV       : {hv_m:.4f} +/- {hv_s:.4f}")
            print(f"  Delta    : {d_m:.3f} +/- {d_s:.3f}")
            print(f"  Pts      : {statistics.mean(pts):.1f}")
            results[tag] = {
                "hv_mean": hv_m, "hv_std": hv_s,
                "delta_mean": d_m, "delta_std": d_s,
                "pts_mean": statistics.mean(pts),
                "n_seeds": len(hvs),
            }

    out = BASE / "outputs" / "seed_stats.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    if all_complete:
        print("\n*** TODAS LAS SEEDS COMPLETAS ***")
        print("\nFilas para la tabla del paper:")
        for tag, graph, k in CONFIGS:
            if tag in results:
                r = results[tag]
                print(f"MOPPO {graph} k={k}: "
                      f"HV={r['hv_mean']:.4f}+/-{r['hv_std']:.4f}  "
                      f"Delta={r['delta_mean']:.3f}+/-{r['delta_std']:.3f}  "
                      f"(n={r['n_seeds']})")
    else:
        print("\nSeeds pendientes aun.")

    return all_complete


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
