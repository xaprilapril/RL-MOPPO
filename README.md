# MOPPO: Multi-Objective PPO for Graph Anonymization

Alanis González & Abril Montaño

Codigo del paper:
> **MOPPO: Aprendizaje por Refuerzo Multi-Objetivo con Curriculum Learning para Anonimizacion de Grafos**



---

## Que hace este proyecto

MOPPO entrena un agente PPO condicionado por pesos para anonimizar grafos sociales
balanceando dos objetivos: privacidad (k-anonimato por grado) y utilidad (preservar
el coeficiente de clustering). Un solo modelo cubre todo el frente de Pareto en lugar
de entrenar uno por cada preferencia.

Contribuciones tecnicas:
- Action masking por deficit de k-anonimato (reduce espacio efectivo 25-45x)
- Curriculum learning en 3 fases que estabiliza el entrenamiento
- Analisis de equidad TREX con indice de disparidad por cuartil de grado

---

## Estructura del repositorio

```
moppo-graph-anonymization/
|
|-- scripts/python/
|   |-- train_main.py              <- entrenar MOPPO o baselines PPO
|   |-- evaluate.py                <- evaluar checkpoint (Pareto, TREX, fairness)
|   |-- evaluate_greedy.py         <- evaluar baseline greedy de Liu & Terzi
|   |-- make_comparison_figs.py    <- generar figuras comparativas del paper
|   |-- figures.py                 <- figuras individuales del paper
|   |
|   `-- graph_anon_morl/
|       |-- env.py                 <- GraphAnonEnv (gymnasium)
|       |-- models.py              <- MOPPOActorCritic
|       |-- evaluation.py          <- Pareto front, hypervolume
|       |-- audit.py               <- fairness disparity index
|       |-- trex.py                <- TREX trajectory clustering
|       |-- utils.py               <- reward functions
|       |-- datasets.py            <- carga Facebook ego graph
|       `-- baselines.py           <- PENDIENTE (ver TASK_greedy_baseline.md)
|
|-- outputs/
|   |-- eval_curriculum/           <- resultados MOPPO karate k=2
|   |-- eval_k4/                   <- resultados MOPPO karate k=4
|   |-- eval_fb/                   <- resultados MOPPO facebook n=100
|   |-- eval_baseline_*/           <- resultados baselines PPO alpha fijo
|   `-- figures/                   <- figuras del paper (.png)
|
|-- paper/
|   |-- moppo_paper.tex            <- paper completo en LaTeX
|   `-- moppo_paper.pdf            <- PDF compilado
|
|-- data/
|   `-- facebook_combined.txt.gz   <- Facebook ego network (SNAP)
|
|-- notebooks/
|   |-- graph_anon_eda.ipynb
|   `-- baseline_comparison.ipynb
|
|-- TASK_greedy_baseline.md        <- instrucciones para colaborador
|-- requirements.txt
`-- .gitignore
```

---

## Instalacion

```bash
git clone https://github.com/tu-usuario/moppo-graph-anonymization
cd moppo-graph-anonymization
pip install -r requirements.txt
```

Python 3.10+ recomendado. Para GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Reproducir los experimentos del paper

### MOPPO con curriculum learning

```bash
python scripts/python/train_main.py --graph karate  --k 2 --n_iter 500  --curriculum --out_dir outputs/moppo_curriculum
python scripts/python/train_main.py --graph karate  --k 4 --n_iter 500  --curriculum --out_dir outputs/moppo_k4
python scripts/python/train_main.py --graph facebook --n_nodes 100 --k 2 --n_iter 800 --curriculum --hidden_dim 256 --out_dir outputs/moppo_fb
```

### Baselines PPO alpha fijo

```bash
for ALPHA in 0.0 0.5 1.0; do
  python scripts/python/train_main.py --graph karate --k 2 --fixed_alpha $ALPHA --n_iter 500 --out_dir outputs/baseline_k2_a${ALPHA}
  python scripts/python/train_main.py --graph karate --k 4 --fixed_alpha $ALPHA --n_iter 500 --out_dir outputs/baseline_k4_a${ALPHA}
  python scripts/python/train_main.py --graph facebook --n_nodes 100 --k 2 --fixed_alpha $ALPHA --n_iter 800 --hidden_dim 256 --out_dir outputs/baseline_fb_a${ALPHA}
done
```

### Evaluacion

```bash
python scripts/python/evaluate.py --checkpoint outputs/moppo_curriculum/moppo_checkpoint.pt --out_dir outputs/eval_curriculum
python scripts/python/evaluate.py --checkpoint outputs/moppo_k4/moppo_checkpoint.pt          --out_dir outputs/eval_k4
python scripts/python/evaluate.py --checkpoint outputs/moppo_fb/moppo_checkpoint.pt           --out_dir outputs/eval_fb

python scripts/python/evaluate.py --checkpoint outputs/baseline_k2_a0/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_k2_a0
python scripts/python/evaluate.py --checkpoint outputs/baseline_k2_a05/moppo_checkpoint.pt  --out_dir outputs/eval_baseline_k2_a05
python scripts/python/evaluate.py --checkpoint outputs/baseline_k2_a1/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_k2_a1
python scripts/python/evaluate.py --checkpoint outputs/baseline_k4_a0/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_k4_a0
python scripts/python/evaluate.py --checkpoint outputs/baseline_k4_a05/moppo_checkpoint.pt  --out_dir outputs/eval_baseline_k4_a05
python scripts/python/evaluate.py --checkpoint outputs/baseline_k4_a1/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_k4_a1
python scripts/python/evaluate.py --checkpoint outputs/baseline_fb_a0/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_fb_a0
python scripts/python/evaluate.py --checkpoint outputs/baseline_fb_a05/moppo_checkpoint.pt  --out_dir outputs/eval_baseline_fb_a05
python scripts/python/evaluate.py --checkpoint outputs/baseline_fb_a1/moppo_checkpoint.pt   --out_dir outputs/eval_baseline_fb_a1
```

### Figuras del paper

```bash
python scripts/python/make_comparison_figs.py
```

Los PNG se guardan en `outputs/figures/`.

### Compilar el paper

```bash
cd paper
pdflatex moppo_paper.tex
pdflatex moppo_paper.tex
```

---

## Resultados pre-computados

Los archivos `outputs/eval_*/eval_results.json` estan incluidos en el repo.
Permiten regenerar las figuras sin necesidad de reentrenar (~18 horas de computo).
Los checkpoints `.pt` NO estan en git por tamano (96 MB total).

---

## Tareas abiertas para colaboradores

Ver `TASK_greedy_baseline.md` — instrucciones para implementar el baseline
greedy de Liu & Terzi (2008) en `scripts/python/graph_anon_morl/baselines.py`.

---

## Cita

```bibtex
@article{minerva2026moppo,
  title  = {MOPPO: Aprendizaje por Refuerzo Multi-Objetivo con Curriculum Learning
            para Anonimizacion de Grafos},
  author = {Minerva, Abril},
  year   = {2026}
}
```
# RL-MOPPO
