# Results Summary: MORL for Graph Anonymization

**Project:** Multi-Objective RL for Graph Anonymization with Fairness Audit
**Status:** Awaiting experimental run
**Last updated:** 2026-05-24

---

## Setup

| Parameter | Value |
|-----------|-------|
| Graph | Karate Club (34 nodes, 78 edges) |
| k (anonymity level) | 2 |
| Episode length T | 50 edge flips |
| Training iterations | 100 |
| Steps per iteration | 512 |
| Architecture | MLP actor-critic (256-dim, 2 hidden layers) |
| Optimizer | Adam (lr=3e-4) |
| Weight schedule | alpha ~ Uniform(0,1) per episode |

---

## Metrics to Report (fill after running)

### Pareto Frontier
- Hypervolume indicator (ref = (-1, -1)): **TBD**
- Pareto front size: **TBD**
- Best privacy-only point (alpha=1): r_priv=?, r_util=?
- Best utility-only point (alpha=0): r_priv=?, r_util=?

### Baseline Comparison (planned)
| Method | Hypervolume | Notes |
|--------|-------------|-------|
| MOPPO (ours) | TBD | |
| NSGA-II | TBD | pymoo |
| MOEA/D | TBD | pymoo |
| Pareto Q-learning | TBD | tabular |

### Fairness Audit (alpha=0.5 balanced policy)
| Quartile | Avg degree G0 | Avg edge loss | Normalized loss |
|----------|--------------|---------------|-----------------|
| Q1 peripheral | TBD | TBD | TBD |
| Q2Q3 middle | TBD | TBD | TBD |
| Q4 hub | TBD | TBD | TBD |
| **Disparity index (Q1/Q4)** | | | **TBD** |

> Disparity > 1 means peripheral nodes bear disproportionate anonymization cost.
> Threshold for paper discussion: disparity > 1.5.

---

## How to run

```bash
cd scripts/python

# Train
python train_main.py --graph karate --n_iter 100 --out_dir outputs/moppo

# Evaluate
python evaluate.py --checkpoint outputs/moppo/moppo_checkpoint.pt --out_dir outputs/eval
```

Outputs:
- `outputs/moppo/train_metrics.json` -- loss + mean reward per iteration
- `outputs/eval/pareto_frontier.png` -- scatter plot colored by alpha
- `outputs/eval/eval_results.json` -- all metrics in JSON

---

## Notes / Observations

*(fill during / after run)*
