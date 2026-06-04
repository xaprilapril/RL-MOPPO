# Tarea: Implementar baseline greedy de k-anonimato

## Contexto

Este repo implementa MOPPO, un agente de aprendizaje por refuerzo para anonimizar
grafos sociales. Ya tenemos baselines de PPO con alpha fijo (ver `outputs/baseline_*/`).
Lo que falta es un baseline **deterministico clasico** basado en el algoritmo de
Liu & Terzi (2008) para comparar RL contra un metodo no-aprendido.

Referencia:
> Liu, K., & Terzi, E. (2008). Towards identity anonymization on graphs.
> ACM SIGMOD, pp. 93-106.

---

## Que tienes que crear

Un solo archivo nuevo: `scripts/python/graph_anon_morl/baselines.py`

No toques ningun otro archivo existente. Toda la integracion ya esta preparada en
`scripts/python/evaluate_greedy.py` (ver seccion de ejecucion abajo).

---

## La funcion que debes implementar

```python
import networkx as nx

def greedy_k_anonymize(G0: nx.Graph, k: int) -> nx.Graph:
    """
    Anonimiza G0 por grado usando el algoritmo greedy de Liu & Terzi (2008).

    Parametros
    ----------
    G0 : nx.Graph  -- grafo original (no modificar, trabajar con copia)
    k  : int       -- requisito de k-anonimato por grado

    Retorna
    -------
    G_anon : nx.Graph  -- grafo anonimizado (mismos nodos, aristas modificadas)

    Requisitos
    ----------
    - G_anon debe ser k-anonimo por grado al retornar
    - Minimizar el numero total de aristas modificadas (flips = add + remove)
    - No agregar nodos nuevos al grafo
    - Sin self-loops, sin aristas paralelas
    """
```

---

## Algoritmo a implementar

El algoritmo trabaja sobre la secuencia de grados del grafo:

**Paso 1 - Construir secuencia de grados**
Obtener los grados de todos los nodos y ordenarlos de mayor a menor.
Guardar la correspondencia nodo -> grado (necesitas esto para aplicar los cambios).

**Paso 2 - Agrupar en bloques de k-anonimato**
Recorrer la secuencia ordenada y agrupar nodos en bloques de tamano >= k donde todos
los nodos del bloque tendran el mismo grado objetivo:
- Tomar los primeros k nodos de la secuencia restante
- El grado objetivo del bloque es el grado del PRIMER nodo (el mayor del grupo)
- Todos los nodos del bloque se igualan a ese grado objetivo
- Si al final quedan menos de k nodos sueltos, unirlos al bloque anterior

**Paso 3 - Calcular deltas de grado**
Para cada nodo: `delta = grado_objetivo - grado_actual`
- `delta > 0` hay que agregar `delta` aristas a ese nodo
- `delta < 0` hay que remover `abs(delta)` aristas de ese nodo
- `delta = 0` no hay que hacer nada

**Paso 4 - Aplicar los cambios**
Construir una lista de nodos con delta positivo (necesitan ganar aristas) y una de
nodos con delta negativo (necesitan perder aristas). Luego:

Para REMOVER aristas: iterar sobre nodos con delta negativo y remover aristas
existentes. Priorizar remover aristas donde el otro extremo tambien tiene delta
negativo (eliminar dos pajaros de un tiro).

Para AGREGAR aristas: iterar sobre nodos con delta positivo y agregar aristas a
nodos que no sean vecinos aun. Priorizar conectar con nodos que tambien tengan
delta positivo.

Si despues de la primera pasada quedan nodos con deficit (r_priv != 0), hacer
una segunda pasada sobre esos nodos especificamente.

---

## Archivos de referencia - lee estos antes de empezar

```
scripts/python/graph_anon_morl/utils.py       <- compute_k_anonymity_reward, compute_utility_reward
scripts/python/graph_anon_morl/env.py         <- lineas 5-16, definicion de k-anonimato
scripts/python/graph_anon_morl/audit.py       <- degree_quartile_audit, fairness_disparity_index
```

K-anonimato en este proyecto se define por distribucion de GRADOS (no vecindarios).
Un grafo es k-anonimo si cada valor de grado aparece en al menos k nodos:

```python
degree_counts = {}
for _, d in G.degree():
    degree_counts[d] = degree_counts.get(d, 0) + 1

# grafo es k-anonimo si:
all(count >= k for count in degree_counts.values())
```

---

## Como verificar que funciona

Corre esto desde la raiz del repo:

```python
import networkx as nx
import sys
sys.path.insert(0, 'scripts/python')

from graph_anon_morl.baselines import greedy_k_anonymize
from graph_anon_morl.utils import compute_k_anonymity_reward, compute_utility_reward

G0 = nx.karate_club_graph()
k = 2

G_anon = greedy_k_anonymize(G0, k)

r_priv = compute_k_anonymity_reward(G_anon, k)
r_util = compute_utility_reward(G_anon, G0)

flips = sum(1 for u, v in G0.edges() if not G_anon.has_edge(u, v))
flips += sum(1 for u, v in G_anon.edges() if not G0.has_edge(u, v))

print(f"r_priv = {r_priv:.4f}  # debe ser 0.0 exactamente")
print(f"r_util = {r_util:.4f}  # entre -0.05 y -0.20")
print(f"flips  = {flips}       # entre 5 y 30")
```

Salida esperada:
```
r_priv = 0.0000   <- si no es exactamente 0.0, el algoritmo esta mal
r_util = entre -0.05 y -0.20
flips  = entre 5 y 30
```

Prueba tambien con k=4 y con Facebook:

```python
from graph_anon_morl.datasets import load_facebook_ego

G_fb = load_facebook_ego(n_nodes=100, seed=42)
G_anon_fb = greedy_k_anonymize(G_fb, k=2)

r_priv_fb = compute_k_anonymity_reward(G_anon_fb, 2)
assert r_priv_fb == 0.0, f"No es k-anonimo: r_priv={r_priv_fb}"
print("Facebook k=2: OK")

G_anon_k4 = greedy_k_anonymize(nx.karate_club_graph(), k=4)
assert compute_k_anonymity_reward(G_anon_k4, 4) == 0.0, "karate k=4 fallo"
print("Karate k=4: OK")
```

---

## Como ejecutarlo para el paper

Una vez que `baselines.py` este listo, corre los tres experimentos:

```bash
python scripts/python/evaluate_greedy.py --graph karate  --k 2 --out_dir outputs/eval_greedy_k2
python scripts/python/evaluate_greedy.py --graph karate  --k 4 --out_dir outputs/eval_greedy_k4
python scripts/python/evaluate_greedy.py --graph facebook --n_nodes 100 --k 2 --out_dir outputs/eval_greedy_fb
```

Los resultados van a `outputs/eval_greedy_*/eval_results.json` en este formato:

```json
{
  "hypervolume": 0.xxxx,
  "pareto_front_size": N,
  "pareto_points": [[-0.xx, -0.xx], ...],
  "disparity_index": 2.xx,
  "all_results": [[0.0, -0.xx, -0.xx], ...]
}
```

El script `make_comparison_figs.py` detecta automaticamente esos archivos y los
incluye en las figuras del paper. No hay que tocar nada mas.

---

## Lo que NO debes hacer

- No modificar `env.py`, `models.py`, `train_main.py`, `evaluate.py`
- No agregar dependencias nuevas (solo networkx y numpy que ya estan instaladas)
- No crear clases, solo la funcion `greedy_k_anonymize` con esa firma exacta
- No usar pseudonodos ni agregar nodos nuevos al grafo
- No hacer commit de archivos en `outputs/` ni de checkpoints `.pt`

---

## Pregunta de validacion final

Antes de hacer PR, responde estas tres preguntas con los numeros reales:

1. `greedy_k_anonymize(nx.karate_club_graph(), k=2)` -> r_priv = ?  (debe ser 0.0)
2. Numero de flips en karate k=2 = ?
3. `greedy_k_anonymize(nx.karate_club_graph(), k=4)` -> r_priv = ?  (debe ser 0.0)

Si los tres pasan, el baseline esta listo para integrarse al paper.
