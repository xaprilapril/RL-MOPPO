# Guía Completa del Proyecto: Anonimización de Grafos Sociales con Aprendizaje por Refuerzo Multi-Objetivo

> **Para quién es esta guía:** Para alguien que nunca ha visto este proyecto, que puede no saber qué es un grafo, qué es el aprendizaje por refuerzo, ni qué es la privacidad k-anónima. Todo se explica desde cero con diagramas y ejemplos concretos.

---

## Tabla de Contenidos

1. [El Problema: ¿Por qué importa la privacidad en grafos?](#1-el-problema)
2. [Conceptos Fundamentales](#2-conceptos-fundamentales)
3. [La Solución Propuesta: MOPPO](#3-la-solución-propuesta-moppo)
4. [Formulación Matemática Formal](#4-formulación-matemática-formal)
5. [El Dataset: ego-Facebook](#5-el-dataset-ego-facebook)
6. [Arquitectura del Proyecto](#6-arquitectura-del-proyecto)
7. [Código: Módulo por Módulo](#7-código-módulo-por-módulo)
8. [Los Baselines](#8-los-baselines)
9. [Métricas de Evaluación](#9-métricas-de-evaluación)
10. [Marco Ético](#10-marco-ético)
11. [Los Notebooks](#11-los-notebooks)
12. [Cómo Ejecutar el Proyecto](#12-cómo-ejecutar-el-proyecto)
13. [Resultados Esperados e Interpretación](#13-resultados-esperados)
14. [Glosario](#14-glosario)

---

## 1. El Problema

### ¿Por qué importa la privacidad en redes sociales?

Imagina que Facebook decide publicar un dataset de relaciones de amistad para que investigadores estudien cómo se propagan las enfermedades. Para proteger a los usuarios, **eliminan los nombres** y reemplazan cada persona por un número: usuario_1, usuario_2, etc.

¿Es eso suficiente para proteger la privacidad?

**No.** En 2007, Backstrom et al. demostraron que un atacante que conoce la estructura de amistades de *solo 3 personas* puede re-identificar a esas personas en el dataset anonimizado con alta probabilidad. En 2009, Narayanan & Shmatikoff re-identificaron usuarios de Netflix usando solo información pública de IMDb.

El problema es que **la estructura del grafo en sí misma es información de identidad**. El número de amigos que tienes, con quién están conectados esos amigos, los triángulos que formas — todo eso crea una "huella digital" única.

```
ANTES de anonimizar               DESPUÉS (nombres eliminados)
─────────────────────             ─────────────────────────────

   [María]──[Juan]                    [3]──[7]
      │         │                      │      │
   [Pedro]──[Ana]                    [1]──[5]

  ↑ Se pueden identificar           ↑ Un atacante que sabe que
  por sus nombres                     María tiene 2 amigos que
                                       se conocen entre sí
                                       puede encontrar al nodo [3].
```

### El objetivo de este proyecto

Modificar el grafo (agregar o eliminar conexiones) de manera que sea **difícil re-identificar individuos**, pero sin destruir la utilidad del grafo para investigación.

Este es un problema de **optimización con dos objetivos en conflicto**:
- **Maximizar privacidad** → requiere muchos cambios al grafo
- **Maximizar utilidad** → requiere hacer pocos cambios al grafo

---

## 2. Conceptos Fundamentales

### 2.1 ¿Qué es un grafo social?

Un **grafo** es una estructura matemática compuesta por:
- **Nodos** (también llamados vértices): representan entidades (personas, páginas, cuentas)
- **Aristas** (también llamadas edges): representan relaciones entre entidades (amistades, seguimientos)

```
Ejemplo de grafo social con 5 personas:

    [A]──────[B]
     │  \     │
     │   \    │
    [C]   [D]─[E]

Nodos: A, B, C, D, E  (5 personas)
Aristas: A-B, A-C, A-D, B-E, D-E  (5 amistades)

Grado de un nodo = número de aristas que tiene:
  grado(A) = 3  (conectado a B, C, D)
  grado(B) = 2  (conectado a A, E)
  grado(C) = 1  (conectado solo a A)
  grado(D) = 2  (conectado a A, E)
  grado(E) = 2  (conectado a B, D)
```

En este proyecto usamos una **matriz de adyacencia** para representar el grafo como números que una red neuronal puede leer:

```
Matriz de adyacencia del ejemplo anterior (1=hay arista, 0=no hay):

     A  B  C  D  E
A  [ 0  1  1  1  0 ]
B  [ 1  0  0  0  1 ]
C  [ 1  0  0  0  0 ]
D  [ 1  0  0  0  1 ]
E  [ 0  1  0  1  0 ]

Se "aplana" en un vector de n² = 25 elementos para dárselo a la red neuronal:
[0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,0, 1,0,0,0,1, 0,1,0,1,0]
```

### 2.2 El riesgo de re-identificación

La idea central de los ataques de re-identificación en grafos es usar el **grado de un nodo** (cuántas conexiones tiene) como huella digital. Si eres la única persona en toda la red con exactamente 47 amigos, entonces cualquiera que sepa que tú tienes 47 amigos puede encontrarte en el dataset.

```
Distribución de grados en una red:

  Frecuencia
  (nodos con ese grado)
  │
5 │    ██
4 │    ██  ██
3 │ ██ ██  ██
2 │ ██ ██  ██  ██
1 │ ██ ██  ██  ██  ██
  └──────────────────── Grado (número de amigos)
     1   2   3   4   5

En redes reales (Facebook), muchos nodos tienen grados únicos.
Esos nodos son re-identificables con solo saber su número de amigos.
```

### 2.3 K-anonimato en grafos

El **k-anonimato por secuencia de grados** (Liu & Terzi, 2008) dice:

> Un grafo es **k-anónimo** si cada nodo tiene al menos otros k-1 nodos con el mismo grado.

Dicho de otra forma: nadie puede ser identificado porque hay al menos k personas con la misma "huella de grado".

```
Ejemplo con k=2 (cada grado debe aparecer ≥ 2 veces):

GRAFO NO k-ANÓNIMO:          GRAFO k-ANÓNIMO (k=2):
────────────────────         ────────────────────────

Nodo │ Grado                 Nodo │ Grado
─────┼──────                 ─────┼──────
  A  │   3    ← único!         A  │   3   ┐ par (mismo grado)
  B  │   2    ┐                B  │   3   ┘
  C  │   2    ┘ par            C  │   2   ┐
  D  │   1    ← único!         D  │   2   ┘ par
  E  │   2    ┘                E  │   2   ┘

k-anonimato: FALSO             k-anonimato: VERDADERO
(A y D son re-identificables)  (nadie tiene grado único)
```

**¿Cómo se logra?** Agregando o eliminando aristas para que los grados "se fusionen" en grupos de ≥k nodos. Agregar una arista a A aumenta su grado de 3 a 4; si otro nodo también tiene grado 4, A ya no es único.

### 2.4 El trade-off privacidad vs. utilidad

Hay una tensión fundamental: cuantas más aristas modificamos, más privacidad logramos, pero más destruimos la estructura real del grafo.

```
UTILIDAD: Coeficiente de clustering

El coeficiente de clustering mide qué tan "agrupada" está la red.
Un coeficiente alto significa que los amigos de tus amigos también
son tus amigos (triángulos cerrados).

Ejemplo:
  [A]──[B]    ← A, B, C forman un triángulo → clustering ALTO
    \  /
     [C]

Si eliminamos la arista A-C para cambiar el grado de A:
  [A]──[B]    ← El triángulo se rompe → clustering BAJA
        |
       [C]

Fórmula del coeficiente de clustering para un nodo v:
  c(v) = (aristas existentes entre vecinos de v)
         ─────────────────────────────────────────
         (máximo de aristas posibles entre vecinos de v)

Ejemplo:
  v tiene 3 vecinos: A, B, C
  Aristas posibles entre ellos: A-B, A-C, B-C → 3 pares
  Si solo existe A-B → c(v) = 1/3

C̄(G) = promedio de c(v) sobre todos los nodos

r_util = -|C̄(G_anonimizado) - C̄(G_original)|

Mientras más cerca de 0, mejor conservamos la estructura.
```

---

## 3. La Solución Propuesta: MOPPO

En lugar de usar un algoritmo fijo (que siempre haría la misma transformación), este proyecto entrena un **agente de aprendizaje por refuerzo** que aprende a decidir qué aristas modificar, respetando el balance entre privacidad y utilidad.

### 3.1 ¿Qué es el Aprendizaje por Refuerzo?

El **Aprendizaje por Refuerzo** (RL, Reinforcement Learning) es un paradigma donde un **agente** aprende a tomar decisiones interactuando con un **entorno**:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   AGENTE (red neuronal)          ENTORNO (grafo social)         │
│   ───────────────────            ───────────────────────        │
│                                                                 │
│   ┌────────────────────────────────────────────────────┐        │
│   │                                                    │        │
│   │   1. Observa el estado ──────────────────────────► │        │
│   │      (grafo actual + peso α)                       │        │
│   │                                                    │        │
│   │   2. Decide una acción ◄───────────────────────── │        │
│   │      (¿qué arista flipear?)                        │        │
│   │                                                    │        │
│   │   3. Recibe recompensa y nuevo estado ──────────► │        │
│   │      r_escalar = α·r_priv + (1-α)·r_util          │        │
│   │                                                    │        │
│   │   4. Aprende de esta experiencia                   │        │
│   │      (actualizar pesos de la red)                  │        │
│   │                                                    │        │
│   └────────────────────────────────────────────────────┘        │
│                                                                 │
│   Un episodio = T=50 pasos. Después: reiniciar con G₀.         │
│   Objetivo: maximizar la recompensa acumulada.                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

El agente empieza sin saber nada y aprende por **ensayo y error**:
- Si una acción lleva a mejor k-anonimato → recompensa → aprende a repetirla
- Si una acción destruye la utilidad → penalización → aprende a evitarla

### 3.2 ¿Qué es Multi-Objetivo?

En RL estándar hay **una sola recompensa** que maximizar. Aquí hay **dos recompensas en conflicto**:

```
RL estándar (un objetivo):
  max r = r_privacidad

RL multi-objetivo (dos objetivos, este proyecto):
  max r⃗ = [r_priv, r_util]
           ↑            ↑
       privacidad    utilidad
  
  El problema: mejorar privacidad SUELE empeorar utilidad.
  No existe una solución "perfecta" → existen MUCHAS soluciones
  que representan diferentes compromisos entre los dos objetivos.
```

Cada punto en el plano (r_priv, r_util) representa una solución diferente:

```
r_util (mayor = más útil)
  │
0 ┤ · · · · · · · · · · · ★  ← privacidad perfecta, algo de utilidad perdida
  │                    ★ ·
  │                ★ · ·       ← Frente de Pareto: soluciones óptimas
  │            ★ · · ·          No puedes mejorar uno sin empeorar el otro
  │        ★ · · · · ·
  │    ★ · · · · · ·
-1 ┤ ★ · · · · · · · ·
  └──────────────────────────── r_priv (mayor = más privado)
  -1                          0

★ = Frente de Pareto (soluciones Pareto-óptimas)
· = Soluciones sub-óptimas (existen ★ mejores en ambos objetivos)
```

### 3.3 ¿Qué es PPO?

**PPO** (Proximal Policy Optimization, Schulman et al. 2017) es uno de los algoritmos de RL más populares y estables. La idea central es:

> "Aprende lo más posible de cada experiencia, pero no cambies la política tan drásticamente que se vuelva inestable."

```
Ciclo de entrenamiento PPO (3 etapas por iteración):

┌──────────────────────────────────────────────────────────────────┐
│  ETAPA 1: RECOLECCIÓN DE EXPERIENCIA (collect_rollout)           │
│  ─────────────────────────────────────────────────────           │
│  Para N_STEPS=256 pasos:                                         │
│    observar estado s_t → ejecutar acción a_t → recibir r_t       │
│    guardar: (s_t, a_t, r_t, log π(a_t|s_t), V(s_t))             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  ETAPA 2: ESTIMACIÓN DE VENTAJA (compute_gae)                    │
│  ──────────────────────────────────────────────                  │
│  Ventaja A_t = "¿fue esta acción mejor o peor de lo esperado?"   │
│                                                                  │
│  A_t = Σₗ (γλ)ˡ · δₜ₊ₗ   donde δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)   │
│                                                                  │
│  γ=0.99 (descuento temporal), λ=0.95 (trade-off sesgo/varianza) │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  ETAPA 3: ACTUALIZACIÓN DE PESOS (ppo_update, 4 épocas)          │
│  ──────────────────────────────────────────────────────          │
│                                                                  │
│  ratio = π_nueva(a|s) / π_vieja(a|s)                            │
│                                                                  │
│  L_policy = -min( ratio·A_t ,  clip(ratio, 1-ε, 1+ε)·A_t )     │
│                                     ↑                            │
│                              ε=0.2 → ratio nunca va              │
│                              más allá de [0.8, 1.2]              │
│                                                                  │
│  L_value   = (V(s) - returns)²   (MSE del crítico)              │
│  L_entropy = -H[π(·|s)]          (bonus de exploración)         │
│                                                                  │
│  L_total = L_policy + 0.5·L_value - 0.01·L_entropy              │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 ¿Qué es MOPPO?

**MOPPO** (Multi-Objective PPO) es el nombre que damos en este proyecto a la combinación de:
1. **PPO estándar** (el algoritmo de optimización)
2. **Weight conditioning** (condicionamiento por vector de pesos)

La innovación clave es el **condicionamiento por peso**: en lugar de entrenar un agente separado para cada posible balance privacidad/utilidad, entrenamos **un solo agente** que recibe como parte de su observación el vector de pesos `[α, 1-α]`:

```
Sin weight conditioning (enfoque naive):
  Agente₁: entrenado siempre con α=0.0  → solo cuida privacidad
  Agente₂: entrenado siempre con α=0.5  → balance igual
  Agente₃: entrenado siempre con α=1.0  → solo cuida utilidad
  ──────────────────────────────────────────────────────────
  → Para K puntos en el frente Pareto necesitas K entrenamientos

Con weight conditioning (MOPPO — este proyecto):
  Un solo agente recibe [α, 1-α] concatenado a la observación:

  Observación = [adj_00, adj_01, ..., adj_nn,  α,    1-α  ]
                 ──────────────────────────────  ──────────
                 estado del grafo (10,000 dims)  preferencia (2 dims)
                                                 ↑
                                              Esto le dice al agente
                                              "cuánto le importa cada objetivo"

  Durante entrenamiento: α ~ U(0,1) aleatoriamente cada episodio
  Durante inferencia:   puedes pasar cualquier α sin reentrenar

  Con α=0.0 → el agente priorizará la privacidad (r_priv domina)
  Con α=0.5 → el agente balanceará igual ambos objetivos
  Con α=1.0 → el agente priorizará la utilidad (r_util domina)

  ──────────────────────────────────────────────────────────
  → Un solo entrenamiento cubre TODO el frente Pareto
```

---

## 4. Formulación Matemática Formal

### 4.1 El MOMDP del problema

El problema se formula como un **MOMDP** (Multi-Objective Markov Decision Process). Un MDP es el modelo matemático estándar para RL.

```
Componentes del MOMDP:

┌──────────────────────────────────────────────────────────────┐
│  S = Espacio de estados                                      │
│      sₜ = (Gₜ, w)                                           │
│            ↑       ↑                                         │
│           grafo   vector de pesos [α, 1-α]                   │
│                                                              │
│      Representación numérica:                                │
│      flatten(A_G) ∈ ℝ^(n×n)  +  w ∈ ℝ²                     │
│      = 100×100 + 2 = 10,002 dimensiones                      │
├──────────────────────────────────────────────────────────────┤
│  A = Espacio de acciones                                     │
│      aₜ ∈ {(u,v) : u < v, u,v ∈ {0,...,99}}                 │
│      = todos los pares posibles de nodos (sin dirección)     │
│      Tamaño: C(100,2) = 100×99/2 = 4,950 acciones           │
│                                                              │
│      Semántica de cada acción:                               │
│      Si (u,v) ∈ E(G): eliminar la arista                    │
│      Si (u,v) ∉ E(G): agregar la arista                     │
├──────────────────────────────────────────────────────────────┤
│  T = Horizonte temporal = 50 pasos por episodio              │
│      (el agente hace exactamente 50 flips por episodio)      │
├──────────────────────────────────────────────────────────────┤
│  R⃗ = Función de recompensa vectorial                         │
│      R⃗(sₜ, aₜ) = [r_priv(Gₜ₊₁), r_util(Gₜ₊₁, G₀)]        │
│                                                              │
│      Escalarización para PPO:                                │
│      r_escalar = w · R⃗ = α·r_priv + (1-α)·r_util            │
├──────────────────────────────────────────────────────────────┤
│  γ = Factor de descuento = 0.99                              │
│      (recompensas futuras valen casi lo mismo que las        │
│       inmediatas; apropiado para episodios cortos T=50)      │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Funciones de recompensa

#### Recompensa de privacidad r_priv

```
r_priv(G) = -(1/n) · Σᵥ max(0, k - |C_v|)

Donde:
  n    = número de nodos (100)
  k    = parámetro de anonimato (k=2 en los experimentos)
  C_v  = clase de equivalencia de v = {u ∈ V : grado(u) = grado(v)}
         (todos los nodos con el mismo grado que v)
  |C_v| = tamaño de esa clase (cuántos nodos comparten grado con v)

Intuición paso a paso:
  Para cada nodo v:
    Si |C_v| >= k → no hay penalización (v es anónimo)
    Si |C_v| < k  → penalización = k - |C_v| (v es re-identificable)

Rango garantizado: r_priv ∈ [-1, 0]
  r_priv = 0   → todos los nodos tienen ≥k nodos con su mismo grado (perfecto)
  r_priv = -1  → todos los nodos tienen grado único (máxima vulnerabilidad)

Ejemplo con n=5, k=2:
  Distribución de grados: [3, 2, 2, 1, 2]
  Conteo por grado: {1:1, 2:3, 3:1}

  Nodo A (grado=3): |C_A| = 1 → penalización = max(0, 2-1) = 1
  Nodo B (grado=2): |C_B| = 3 → penalización = max(0, 2-3) = 0
  Nodo C (grado=2): |C_C| = 3 → penalización = max(0, 2-3) = 0
  Nodo D (grado=1): |C_D| = 1 → penalización = max(0, 2-1) = 1
  Nodo E (grado=2): |C_E| = 3 → penalización = max(0, 2-3) = 0

  Total: 1+0+0+1+0 = 2
  r_priv = -(2/5) = -0.4
```

#### Recompensa de utilidad r_util

```
r_util(G, G₀) = -|C̄(G) - C̄(G₀)|

Donde:
  C̄(G)  = coeficiente de clustering promedio del grafo G actual
  C̄(G₀) = coeficiente de clustering del grafo ORIGINAL (sin modificar)
  | · |  = valor absoluto

Coeficiente de clustering de un nodo v:
  c(v) = (aristas entre vecinos de v) / (pares posibles entre vecinos de v)
       = Δ_v / [grado(v) · (grado(v)-1) / 2]

  Donde Δ_v = número de triángulos que pasan por v

  Ejemplo:
    v tiene vecinos {A, B, C}. Aristas entre vecinos: solo A-B existe.
    c(v) = 1 / [3·(3-1)/2] = 1/3 ≈ 0.33

C̄(G) = promedio de c(v) para todos los nodos del grafo.

Rango: r_util ∈ (-∞, 0]
  r_util = 0   → estructura perfectamente preservada (C̄(G) = C̄(G₀))
  r_util = -0.1 → clustering difiere en 0.1 (poca degradación)
  r_util = -0.5 → clustering difiere mucho (estructura muy dañada)
```

#### Escalarización por pesos

```
Conversión de recompensa vectorial a escalar para PPO:

  r_escalar = w · R⃗ = α · r_priv + (1-α) · r_util

Interpretación para distintos valores de α:

  α = 0.0 → r_escalar = r_util      (solo importa preservar el grafo)
  α = 0.3 → r_escalar = 0.3·r_priv + 0.7·r_util  (prioriza utilidad)
  α = 0.5 → r_escalar = 0.5·r_priv + 0.5·r_util  (balance)
  α = 0.7 → r_escalar = 0.7·r_priv + 0.3·r_util  (prioriza privacidad)
  α = 1.0 → r_escalar = r_priv      (solo importa la privacidad)

Durante el entrenamiento de MOPPO:
  α ~ U(0,1) en cada episodio → el agente aprende a ser bueno
  para CUALQUIER preferencia del usuario.
```

### 4.3 El Frente de Pareto

```
Definición de dominancia:
  El punto A = (r_priv_A, r_util_A) DOMINA al punto B si:
    • r_priv_A ≥ r_priv_B  (A es igual o más privado que B)
    • r_util_A ≥ r_util_B  (A es igual o más útil que B)
    • Al menos una desigualdad es estricta
  
  Intuición: A domina a B si A es "mejor en todo" o "mejor en algo sin
  ser peor en nada". No tiene sentido elegir B si A existe.

El FRENTE DE PARETO es el conjunto de puntos no dominados:

  r_util
    │
  0 ┤         ●────●  ← estos son Pareto-óptimos
    │       ●
    │     ●
    │   ●
    │ ●          ○  ← este NO es Pareto-óptimo: el ● más cercano
    │               arriba-derecha lo domina en ambos objetivos
    └──────────────── r_priv
                   0

  ● = Frente de Pareto (soluciones Pareto-óptimas)
  ○ = Punto dominado (sub-óptimo; existe ● que es mejor en ambas métricas)
```

---

## 5. El Dataset: ego-Facebook

### ¿Qué es?

El **ego-Facebook dataset** (Leskovec & Mcauley, 2012) es una red social anónima obtenida de Facebook, disponible públicamente en Stanford Network Analysis Project (SNAP).

```
Dataset completo ego-Facebook:
┌─────────────────────────────────────┐
│  Nodos:        4,039  personas      │
│  Aristas:     88,234  amistades     │
│  Densidad:      ~1.1%               │
│  Clustering:    ~0.61               │
│  Diámetro:      8 saltos            │
│  Tipo:          no dirigido         │
└─────────────────────────────────────┘

Para hacerlo manejable usamos un subgrafo BFS de n=100 nodos:
┌─────────────────────────────────────┐
│  Nodos:          100                │
│  Aristas:       ~390                │
│  Acciones pos.: 4,950               │
│  Dimensión obs: 10,002              │
└─────────────────────────────────────┘
```

### ¿Por qué BFS desde el nodo de mayor grado?

```
Proceso de extracción:

Grafo completo (4039 nodos, 88234 aristas)
       │
       ▼
Paso 1: encontrar el nodo con más conexiones
        seed_node = argmax grado → "el hub principal"

       │
       ▼
Paso 2: BFS (Breadth-First Search) desde ese hub

  Nivel 0 (cola):  [hub]
  ──────────────────────────────────────────
  Nivel 1 (cola):  [vecino₁, vecino₂, ..., vecinoₖ]
                    ↑ todos los amigos directos del hub
  ──────────────────────────────────────────
  Nivel 2 (cola):  [amigos de vecino₁, amigos de vecino₂, ...]
  ──────────────────────────────────────────
  ... (continúa hasta explorar todo el grafo)

       │
       ▼
Paso 3: tomar los primeros 100 nodos en el orden BFS

       │
       ▼
Paso 4: subgrafo inducido
  Solo las aristas entre esos 100 nodos (no importa si vecinos
  de esos nodos están fuera del subconjunto)

       │
       ▼
Paso 5: re-etiquetar nodos a {0, 1, ..., 99}
  (para facilitar indexación como arrays)

Resultado: subgrafo denso, bien conectado y representativo
           de la estructura real de Facebook.
```

### ¿Por qué este dataset para privacidad?

1. **Referencia de facto**: Es el dataset estándar en papers de anonimización de grafos sociales (Liu & Terzi lo usaron, Hay et al. también).
2. **Distribución de grados power-law**: Muchos nodos con pocos amigos (periferia), pocos nodos con muchos amigos (hubs). Esta distribución hace que muchos nodos tengan grados únicos → el problema de k-anonimato es genuinamente difícil.
3. **Alta heterogeneidad**: Los grados varían mucho → muchos nodos con grado único → muchos candidatos k-vulnerables.

---

## 6. Arquitectura del Proyecto

```
clo-author-main/
├── scripts/
│   └── python/
│       ├── graph_anon_morl/          ← Paquete Python principal
│       │   ├── __init__.py           ← Hace que sea un paquete importable
│       │   ├── datasets.py           ← Descarga y extrae ego-Facebook
│       │   ├── env.py                ← Entorno RL (define el MOMDP)
│       │   ├── models.py             ← Red neuronal MOPPOActorCritic
│       │   ├── utils.py              ← Calcula r_priv y r_util
│       │   ├── evaluation.py         ← Barrido del frente de Pareto
│       │   └── audit.py              ← Auditoría de equidad (Rawls/Δ)
│       ├── train_main.py             ← Entrenamiento completo PPO
│       └── evaluate.py               ← Evaluación de checkpoint guardado
│
├── notebooks/
│   ├── graph_anon_eda.ipynb          ← Exploración + prueba de módulos
│   └── baseline_comparison.ipynb     ← MOPPO vs todos los baselines
│
├── data/
│   └── facebook_combined.txt.gz      ← Dataset (auto-descargado)
│
├── outputs/
│   ├── moppo/
│   │   ├── moppo_checkpoint.pt       ← Modelo entrenado (pesos de la red)
│   │   └── train_metrics.json        ← Loss y reward por iteración
│   ├── eval/
│   │   ├── pareto_frontier.png       ← Figura del frente de Pareto
│   │   └── eval_results.json         ← Métricas numéricas completas
│   ├── pareto_comparison.png         ← Comparación todos los métodos
│   └── fairness_comparison.png       ← Auditoría equidad comparativa
│
└── PROJECT_GUIDE.md                  ← Esta guía
```

### Flujo de datos de punta a punta

```
╔══════════════════════════════════════════════════════════════════╗
║                   FLUJO COMPLETO DEL PROYECTO                   ║
╚══════════════════════════════════════════════════════════════════╝

  1. PREPARACIÓN DEL GRAFO
  ────────────────────────
  datasets.py:load_facebook_ego()
       │ descarga + BFS de 100 nodos
       ▼
  G₀ (nx.Graph, 100 nodos, ~390 aristas)

  2. CREACIÓN DEL ENTORNO
  ───────────────────────
  env.py:GraphAnonEnv(G₀, k=2, T=50)
       │ obs_dim = 100²+2 = 10,002
       │ n_actions = C(100,2) = 4,950
       ▼
  entorno RL (interfaz Gymnasium)

  3. CONSTRUCCIÓN DEL MODELO
  ──────────────────────────
  models.py:MOPPOActorCritic(obs_dim=10002, n_actions=4950, hidden_dim=256)
       │ ~2.7M parámetros entrenables
       ▼
  red neuronal (no entrenada)

  4. ENTRENAMIENTO (train_main.py)
  ────────────────────────────────
  Para cada iteración (N_ITER=500):
    collect_rollout() → 512 transiciones (s,a,r,logp,V,done)
         │
    compute_gae() → ventajas A_t y returns
         │
    ppo_update() → 4 épocas × mini-batches de 64
         │
  [Guardar] moppo_checkpoint.pt + train_metrics.json

  5. EVALUACIÓN (evaluate.py)
  ───────────────────────────
  Cargar checkpoint
  Para α ∈ {0.0, 0.05, ..., 1.0} (21 valores):
    Correr 10 episodios deterministas
    Medir r_priv y r_util del grafo final
  compute_pareto_front() → frente de Pareto
  hypervolume_indicator() → HV
  degree_quartile_audit() → índice de disparidad Δ

  [Guardar] pareto_frontier.png + eval_results.json
```

---

## 7. Código: Módulo por Módulo

### 7.1 `datasets.py`

**¿Qué hace?** Descarga el dataset ego-Facebook de internet (solo la primera vez) y extrae un subgrafo representativo de n nodos.

```python
# Código completo explicado línea por línea

FACEBOOK_URL = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
# URL del dataset en Stanford. El archivo pesa ~900 KB comprimido.
# Formato del archivo: una arista por línea, ej: "0 1\n0 2\n1 3\n..."

def load_facebook_ego(n_nodes=100, seed=42, data_dir="data"):

    # ─── Paso 1: Gestión de caché ────────────────────────────────────
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)   # Crear directorio si no existe
    gz_path = data_path / "facebook_combined.txt.gz"

    if not gz_path.exists():    # Solo descarga si no está en caché
        urllib.request.urlretrieve(FACEBOOK_URL, gz_path)
        # urllib.request = módulo estándar de Python para descargar URLs

    # ─── Paso 2: Leer el grafo completo ─────────────────────────────
    with gzip.open(gz_path, "rt") as f:
        G_full = nx.read_edgelist(f, nodetype=int)
    # gzip.open: descomprime el archivo al vuelo sin guardar en disco
    # nx.read_edgelist: lee el formato "nodo1 nodo2" y construye el grafo

    # ─── Paso 3: Encontrar el hub central ────────────────────────────
    seed_node = max(G_full.degree, key=lambda x: x[1])[0]
    # G_full.degree genera pares (nodo, grado)
    # max(..., key=lambda x: x[1]) → el par con mayor grado
    # [0] → tomar solo el nodo (no el grado)

    # ─── Paso 4: BFS desde el hub ────────────────────────────────────
    bfs_order = list(nx.bfs_tree(G_full, seed_node).nodes())
    # nx.bfs_tree: construye el árbol BFS desde seed_node
    # .nodes(): da los nodos en orden de visita BFS

    # ─── Paso 5: Subgrafo inducido ────────────────────────────────────
    sub = G_full.subgraph(bfs_order[:n_nodes]).copy()
    # Tomar los primeros n_nodes nodos
    # .copy() para poder modificar el grafo después

    # ─── Paso 6: Re-etiquetar nodos a {0, ..., n-1} ──────────────────
    return nx.convert_node_labels_to_integers(sub)
```

### 7.2 `env.py`

**¿Qué hace?** Implementa el entorno RL siguiendo la interfaz estándar de Gymnasium. Encapsula el grafo social como un "mundo" en el que el agente puede actuar.

```python
class GraphAnonEnv(gym.Env):
    """
    MOMDP para anonimización de grafos.
    Hereda de gym.Env → interfaz estándar para RL en Python.
    """

    def __init__(self, G0, k=2, T=50):
        # G0: grafo original (nunca se modifica, es la referencia)
        self.G0 = nx.convert_node_labels_to_integers(G0.copy())
        self.n  = self.G0.number_of_nodes()   # 100
        self.k  = k                           # parámetro k-anonimato
        self.T  = T                           # pasos por episodio (50)

        # Generar todas las posibles aristas del grafo completo K_n
        # combinations([0,1,...,99], 2) = todos los pares (u,v) con u<v
        self.all_edges = list(combinations(range(self.n), 2))
        self.n_actions = len(self.all_edges)  # C(100,2) = 4,950

        # Espacio de acciones: entero de 0 a 4949
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # Espacio de observaciones: vector continuo de 10,002 dimensiones
        obs_dim = self.n * self.n + 2   # 10,000 (adj aplanada) + 2 (pesos)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )

    def reset(self, weight=None):
        """Iniciar nuevo episodio: restaurar grafo original y elegir α."""
        self.G = self.G0.copy()   # G es el grafo que se va modificando
        self.t = 0                # Reiniciar contador de pasos

        if weight is not None:
            # Si se pasa [α, 1-α] explícitamente (para evaluación con α fijo)
            alpha = float(np.clip(weight[0], 0.0, 1.0))
        else:
            # Muestrear α ~ U(0,1) (para entrenamiento MOPPO)
            alpha = self.np_random.random()

        self.weight = np.array([alpha, 1.0 - alpha], dtype=np.float32)
        return self._obs(), {}  # Retornar observación inicial + info vacío

    def step(self, action):
        """Ejecutar una acción (flipear una arista) y retornar el resultado."""

        # 1. Decodificar acción: índice entero → par de nodos (u, v)
        u, v = self.all_edges[action]
        #    action=0   → (0,1)
        #    action=1   → (0,2)
        #    action=4949 → (98,99)

        # 2. Flipear la arista
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)   # Si existe → eliminar
        else:
            self.G.add_edge(u, v)      # Si no existe → agregar

        # 3. Calcular ambas recompensas con el grafo modificado
        r_priv = compute_k_anonymity_reward(self.G, self.k)
        r_util = compute_utility_reward(self.G, self.G0)
        reward_vec = np.array([r_priv, r_util], dtype=np.float32)

        # 4. Avanzar contador y verificar si el episodio terminó
        self.t += 1
        terminated = (self.t >= self.T)   # Termina exactamente en T=50 pasos

        # 5. Escalarizar la recompensa para PPO
        scalar_reward = float(self.weight @ reward_vec)
        #                     [α, 1-α] · [r_priv, r_util]
        #                     = α·r_priv + (1-α)·r_util

        return self._obs(), scalar_reward, terminated, False, {"reward_vec": reward_vec}
        #       ↑              ↑               ↑        ↑      ↑
        #    obs nueva     recomp. escalar  ¿terminó? truncated info extra

    def _obs(self):
        """Construir el vector de observación: adj aplanada + pesos."""
        return np.concatenate([
            graph_to_flat_adj(self.G, self.n),  # n×n=10,000 flotantes
            self.weight                          # [α, 1-α]
        ])
        # Resultado: vector de 10,002 flotantes
```

### 7.3 `models.py`

**¿Qué hace?** Define la arquitectura de la red neuronal que actúa como el "cerebro" del agente MOPPO. Usa el patrón Actor-Critic: una cabeza para elegir acciones (actor) y otra para estimar qué tan bueno es el estado (critic).

```
Arquitectura visual completa:

INPUT (10,002 dimensiones por muestra):
  [adj_0,0  adj_0,1  ...  adj_99,99  α  1-α]
                  │
         ┌────────┴────────┐
         │  RED COMPARTIDA │
         │  (backbone)     │
         │                 │
         │ Linear(10002→256)│
         │ Tanh            │
         │ Linear(256→256) │
         │ Tanh            │
         └────────┬────────┘
                  │
         ┌────────┴──────────────────────┐
         │                               │
    ┌────┴─────┐                   ┌─────┴─────┐
    │  ACTOR   │                   │  CRITIC   │
    │  HEAD    │                   │  HEAD     │
    │          │                   │           │
    │Linear    │                   │Linear     │
    │(256→4950)│                   │(256→1)    │
    └────┬─────┘                   └─────┬─────┘
         │                               │
         ▼                               ▼
  logits (4950 valores)           V(s) = 1 escalar
  (uno por cada posible acción)   "¿qué tan bueno
         │                         es este estado?"
         ▼
  Distribución Categorical
  π(a|s) = softmax(logits)
         │
         ▼
  Muestrear acción a ~ π(·|s)
  o tomar argmax para evaluación
```

```python
class MOPPOActorCritic(nn.Module):

    def __init__(self, obs_dim, n_actions, hidden_dim=256):
        super().__init__()

        # Red compartida: transforma la observación en representación interna
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),   # 10,002 → 256
            nn.Tanh(),                        # activación acotada [-1, 1]
            nn.Linear(hidden_dim, hidden_dim), # 256 → 256
            nn.Tanh(),
        )

        # Cabeza actor: predice cuál acción tomar
        self.actor_head = nn.Linear(hidden_dim, n_actions)  # 256 → 4,950

        # Cabeza critic: predice el valor del estado
        self.critic_head = nn.Linear(hidden_dim, 1)         # 256 → 1

        # Inicialización ortogonal (estándar en PPO):
        #   - shared: gain=√2 → gradientes estables
        #   - actor: gain=0.01 → logits pequeños al inicio → exploración uniforme
        #   - critic: gain=1.0 → estimación de valor sin sesgo

    def forward(self, obs):
        h = self.shared(obs)           # Representación interna
        return self.actor_head(h), self.critic_head(h).squeeze(-1)
        #       ↑ logits[4950]          ↑ valor escalar

    def get_action(self, obs, deterministic=False):
        logits, value = self.forward(obs)

        # Distribución de probabilidades sobre las 4,950 acciones
        dist = torch.distributions.Categorical(logits=logits)
        # Categorical: P(a) = softmax(logits)[a]

        if deterministic:
            action = logits.argmax(dim=-1)   # Evaluación: mejor acción conocida
        else:
            action = dist.sample()           # Entrenamiento: explorar

        return action, dist.log_prob(action), dist.entropy(), value
        #      ↑       ↑                       ↑               ↑
        #   acción  log π(a|s)              H[π(·|s)]        V(s)
        #           (para ratio PPO)    (para bonus entropa)  (para GAE)
```

**¿Por qué Tanh y no ReLU?**
- ReLU puede "morir" (neuronas que siempre dan 0) → inestabilidad en RL
- Tanh acota las activaciones en [-1, 1] → gradientes más estables
- Es el estándar de facto en implementaciones de PPO (Stable Baselines, CleanRL, etc.)

**Número total de parámetros:**
```
  shared[0]: 10002 × 256 + 256 =  2,560,768
  shared[2]:   256 × 256 + 256 =     65,792
  actor_head:  256 × 4950 + 4950 = 1,272,150
  critic_head: 256 × 1 + 1 =           257
  ─────────────────────────────────────────
  TOTAL:                          ~3,899,000 parámetros ≈ 3.9M
```

### 7.4 `utils.py`

**¿Qué hace?** Implementa las tres funciones matemáticas fundamentales: las dos funciones de recompensa y la conversión de grafo a array.

```python
def compute_k_anonymity_reward(G, k=2):
    """r_priv = -(1/n) · Σᵥ max(0, k - |C_v|)"""

    # Paso 1: Contar nodos por cada grado
    degree_counts = {}
    for _, d in G.degree():        # G.degree() → iterador de (nodo, grado)
        degree_counts[d] = degree_counts.get(d, 0) + 1
    # Resultado: {grado: frecuencia}
    # Ej: {1: 5, 2: 12, 3: 8, 4: 3, 5: 1, ...}

    # Paso 2: Calcular la suma de penalizaciones
    penalty = sum(
        max(0, k - degree_counts[d])   # 0 si d tiene ≥k nodos, sino k-count
        for _, d in G.degree()         # para cada nodo del grafo
    )

    # Paso 3: Normalizar por el número de nodos
    return -penalty / max(G.number_of_nodes(), 1)
    # max(..., 1) evita división por 0 en grafos vacíos


def compute_utility_reward(G, G0):
    """r_util = -|C̄(G) - C̄(G₀)|"""
    c_t = nx.average_clustering(G)    # C̄ del grafo actual
    c_0 = nx.average_clustering(G0)   # C̄ del grafo original (constante)
    return -abs(c_t - c_0)
    # nx.average_clustering usa el algoritmo de Watts & Strogatz (1998)


def graph_to_flat_adj(G, n):
    """Convierte el grafo a vector numpy de n² flotantes."""
    # nx.to_numpy_array: crea la matriz de adyacencia n×n
    # nodelist=range(n): fuerza el orden exacto 0,1,...,n-1
    adj = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)
    return adj.flatten()    # (n,n) → (n²,)
    # Ejemplo n=3: [[0,1,0],[1,0,1],[0,1,0]] → [0,1,0,1,0,1,0,1,0]
```

### 7.5 `evaluation.py`

**¿Qué hace?** Evalúa un modelo entrenado barriendo el espacio de pesos α para trazar el frente de Pareto. Es la función que se usa para generar las figuras del paper.

**Nota crítica: ¿Por qué medimos el grafo FINAL y no la recompensa acumulada?**

```
INCORRECTO (acumular pasos — bug ya corregido):
  total_r += info["reward_vec"]  ← suma T=50 recompensas de paso

  Problema: r_priv_acumulado = Σₜ r_priv(Gₜ) ≈ -4.5
  Esto es la suma de penalizaciones de TODOS los pasos intermedios,
  no la calidad del grafo final. No es comparable con baselines que
  evalúan directamente el grafo final.

CORRECTO (grafo final — implementación actual):
  Después del episodio: compute_k_anonymity_reward(env.G, env.k)

  Resultado: r_priv_final ∈ [-1, 0]
  Esto es la penalización k-anonimato del grafo RESULTANTE,
  directamente comparable con NSGA-II y Greedy.
```

```python
def evaluate_policy(env, model, n_weights=11, n_episodes=5, device="cpu"):
    results = []
    model.eval()   # Modo evaluación: deshabilita dropout, etc.

    with torch.no_grad():   # No calcular gradientes (no estamos entrenando)
        for alpha in np.linspace(0.0, 1.0, n_weights):
            # α ∈ {0.0, 0.1, 0.2, ..., 1.0} (11 valores)

            ep_finals = []
            for _ in range(n_episodes):   # Promediar sobre 5 episodios
                obs, _ = env.reset(weight=[alpha, 1.0 - alpha])
                done = False

                while not done:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    # unsqueeze(0): añade dimensión de batch → shape (1, 10002)

                    action, _, _, _ = model.get_action(obs_t, deterministic=True)
                    # deterministic=True: tomar siempre la mejor acción conocida
                    # (no explorar → resultados reproducibles)

                    obs, _, terminated, truncated, _ = env.step(action.item())
                    done = terminated or truncated

                # Medir la calidad del GRAFO FINAL (después de T=50 flips)
                final_rp = compute_k_anonymity_reward(env.G, env.k)
                final_ru = compute_utility_reward(env.G, env.G0)
                ep_finals.append([final_rp, final_ru])

            # Promediar sobre los n_episodes episodios
            mean_r = np.mean(ep_finals, axis=0)
            results.append((float(alpha), float(mean_r[0]), float(mean_r[1])))

    return results
    # Estructura: [(α, r_priv, r_util), (α, r_priv, r_util), ...]
    # Ejemplo: [(0.0, -0.06, -0.01), (0.1, -0.08, -0.008), ...]


def compute_pareto_front(reward_vecs):
    """Retorna índices de los puntos Pareto-óptimos."""
    rewards = np.array(reward_vecs)   # Shape: (N, 2)
    n = len(rewards)
    is_pareto = np.ones(n, dtype=bool)   # Inicialmente todos son Pareto

    for i in range(n):
        if not is_pareto[i]:
            continue
        # Para el punto i, buscar si algo lo domina
        dominated = (
            np.all(rewards >= rewards[i], axis=1) &   # ≥ en ambos objetivos
            np.any(rewards > rewards[i], axis=1)       # > en al menos uno
        )
        is_pareto[dominated] = False   # Marcar como no-Pareto

    return np.where(is_pareto)[0]   # Índices de puntos Pareto-óptimos
```

### 7.6 `audit.py`

**¿Qué hace?** Auditar si la anonimización impone costos equitativos entre grupos de nodos según su grado (periferia vs. hubs).

```python
def degree_quartile_audit(G0, G_final):
    """
    Divide nodos en 3 grupos por grado original:
      Q1_peripheral: 25% de menor grado (los más aislados)
      Q2Q3_middle:   50% intermedios
      Q4_hub:        25% de mayor grado (los más conectados)

    Para cada grupo, mide: cambios de aristas normalizados por grado.
    Cambios = aristas eliminadas + aristas agregadas (diferencia simétrica).
    """

    degrees = dict(G0.degree())          # {nodo: grado_original}
    deg_vals = list(degrees.values())    # Lista de todos los grados
    q1_thresh = np.percentile(deg_vals, 25)   # Valor en el percentil 25
    q3_thresh = np.percentile(deg_vals, 75)   # Valor en el percentil 75

    def quartile(d):
        if d <= q1_thresh: return "Q1_peripheral"
        if d >= q3_thresh: return "Q4_hub"
        return "Q2Q3_middle"

    groups = {n: quartile(d) for n, d in degrees.items()}
    # Asignar grupo a cada nodo según su grado original

    # Contar cambios de aristas POR NODO
    edge_changes = {n: 0 for n in G0.nodes()}

    # Aristas ELIMINADAS (estaban en G0, no están en G_final)
    for u, v in G0.edges():
        if not G_final.has_edge(u, v):
            edge_changes[u] += 1
            edge_changes[v] += 1
            # Ambos extremos de la arista eliminada se cuentan

    # Aristas AGREGADAS (no estaban en G0, están en G_final)
    for u, v in G_final.edges():
        if not G0.has_edge(u, v):
            edge_changes[u] += 1
            edge_changes[v] += 1
            # Ambos extremos de la arista agregada se cuentan

    # Calcular estadísticas por grupo
    stats = {}
    for label in ("Q1_peripheral", "Q2Q3_middle", "Q4_hub"):
        nodes = [n for n, g in groups.items() if g == label]
        if not nodes:
            continue
        avg_deg  = float(np.mean([degrees[n] for n in nodes]))
        avg_loss = float(np.mean([edge_changes[n] for n in nodes]))

        stats[label] = {
            "n_nodes":        len(nodes),
            "avg_degree_G0":  avg_deg,
            "avg_edge_loss":  avg_loss,
            "normalized_loss": avg_loss / max(avg_deg, 1e-6),
            # Normalizar por grado: permite comparar entre grupos con
            # diferentes grados promedio. Un hub con grado 20 que pierde
            # 5 aristas (25%) es "comparable" a un nodo periférico
            # con grado 4 que pierde 1 arista (25%).
        }

    return stats


def fairness_disparity_index(audit_stats):
    """
    Δ = normalized_loss(Q1) / normalized_loss(Q4)

    Δ = 1.0 → perfectamente equitativo (mismo costo relativo en todos los grupos)
    Δ > 1.0 → la periferia paga más que los hubs (desigualdad en contra de Q1)
    Δ < 1.0 → los hubs pagan más que la periferia (inusual pero posible)

    Umbral Rawlsiano: Δ ≤ 1.5 es aceptable.
    """
    q1 = audit_stats.get("Q1_peripheral", {}).get("normalized_loss", 0.0)
    q4 = audit_stats.get("Q4_hub",        {}).get("normalized_loss", 1e-6)
    return q1 / max(q4, 1e-6)
    # max(..., 1e-6) evita división por 0 si los hubs no tienen cambios
```

### 7.7 `train_main.py`

**¿Qué hace?** Script completo de entrenamiento. Implementa el loop PPO desde cero.

```python
def collect_rollout(env, model, n_steps, device):
    """
    Ejecuta n_steps=256 pasos en el entorno con la política ACTUAL.
    Guarda todas las experiencias en buffers para el update PPO.
    """
    obs_buf, act_buf, logp_buf = [], [], []
    val_buf, rew_buf, done_buf = [], [], []

    obs, _ = env.reset()  # α aleatorio para MOPPO (sin weight → α ~ U(0,1))

    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            action, logp, _, value = model.get_action(obs_t)
            # action: entero en [0, 4949]
            # logp:   log π(action|obs)  — necesario para el ratio de PPO
            # value:  V(obs)             — necesario para GAE

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        # Guardar esta transición
        obs_buf.append(obs)
        act_buf.append(action.item())
        logp_buf.append(logp.item())
        val_buf.append(value.item())
        rew_buf.append(reward)          # Recompensa escalar α·r_priv+(1-α)·r_util
        done_buf.append(float(done))

        # Avanzar al siguiente estado; si episodio terminó, empezar uno nuevo
        obs = next_obs if not done else env.reset()[0]
        # env.reset() sin weight → nuevo α aleatorio → MOPPO cubre todo el frente

    # Convertir a tensores PyTorch y retornar
    return (
        torch.FloatTensor(np.array(obs_buf)).to(device),   # (256, 10002)
        torch.LongTensor(act_buf).to(device),              # (256,)
        torch.FloatTensor(logp_buf).to(device),            # (256,)
        torch.FloatTensor(val_buf).to(device),             # (256,)
        torch.FloatTensor(rew_buf).to(device),             # (256,)
        torch.FloatTensor(done_buf).to(device),            # (256,)
    )


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    GAE = Generalized Advantage Estimation

    La ventaja A_t responde: "¿fue la acción en t mejor o peor
    de lo que esperaba el crítico?"

    Si A_t > 0: la acción fue mejor de lo esperado → aumentar π(a|s)
    Si A_t < 0: la acción fue peor de lo esperado  → disminuir π(a|s)

    GAE combina estimaciones de distintos horizontes temporales:
      A_t^(1) = δ_t              (solo el siguiente paso)
      A_t^(2) = δ_t + γ·δ_{t+1} (los dos siguientes pasos)
      A_t^(∞) = Σₗ γˡ·δₜ₊ₗ    (horizonte infinito)

      A_t^GAE(λ) = (1-λ)[A_t^(1) + λ·A_t^(2) + λ²·A_t^(3) + ...]
                 = Σₗ (γλ)ˡ·δₜ₊ₗ

    Con λ=0.95:
      - Se da mucho peso a pasos futuros
      - Ventaja de baja varianza pero algo de sesgo
      - Empíricamente mejor que λ=0 (solo inmediato) en RL continuo
    """
    n = len(rewards)
    advantages = torch.zeros(n)
    last_gae = 0.0

    for t in reversed(range(n)):   # Calcular de atrás hacia adelante
        # Valor del siguiente estado (0 si el episodio terminó)
        next_val = values[t + 1].item() if t + 1 < n else 0.0

        # Error TD (Temporal Difference error)
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        #       rₜ     +    γ·V(sₜ₊₁)·(1-done)    -   V(sₜ)
        #       ↑              ↑                        ↑
        #   recompensa    valor futuro              valor actual
        #   inmediata     descontado                estimado

        # Acumular ventaja GAE
        last_gae = delta.item() + gamma * lam * (1 - dones[t].item()) * last_gae
        advantages[t] = last_gae

    returns = advantages + values.cpu()   # V_target = A + V_old
    return advantages, returns


def ppo_update(model, optimizer, obs, actions, old_logps, returns, advantages,
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, n_epochs=4, batch_size=64):
    """
    Actualizar los pesos de la red con PPO.
    Se hacen n_epochs=4 pasadas sobre los datos con mini-batches de 64.
    """
    n = len(obs)
    total_loss = 0.0

    for _ in range(n_epochs):
        # Aleatorizar el orden de los datos en cada época
        idx = torch.randperm(n, device=obs.device)

        for start in range(0, n, batch_size):
            mb = idx[start: start + batch_size]   # Mini-batch de 64 muestras

            # Forward pass con el modelo ACTUALIZADO
            logits, values = model(obs[mb])
            dist = torch.distributions.Categorical(logits=logits)
            logps   = dist.log_prob(actions[mb])    # log π_nueva(a|s)
            entropy = dist.entropy().mean()          # H[π(·|s)]

            # Ratio de importancia: π_nueva / π_vieja
            ratio = (logps - old_logps[mb]).exp()
            # Equivalente a: π_nueva(a|s) / π_vieja(a|s)
            # Se usa log-space para estabilidad numérica

            # Normalizar ventajas (estabilidad en el entrenamiento)
            adv = advantages[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # Loss del actor (PPO clipped)
            policy_loss = -torch.min(
                ratio * adv,                                    # Término normal
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv  # Término clipeado
            ).mean()
            # torch.clamp: limita ratio a [0.8, 1.2]
            # torch.min: toma el más conservador de los dos
            # → Si ratio > 1.2 (política cambió demasiado), usar el clipeado

            # Loss del crítico (MSE entre valor predicho y retorno real)
            value_loss = (values - returns[mb]).pow(2).mean()

            # Loss total
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            #                       ↑ 0.5              ↑ 0.01
            #                   peso del critic    bonus de entropía
            #                                      (fomenta exploración)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # Clip de gradientes: evita explosión de gradientes
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (n_epochs * max(1, n // batch_size))
```

### 7.8 `evaluate.py`

**¿Qué hace?** Script independiente que carga un checkpoint entrenado y genera las figuras y métricas de evaluación final.

```
Cómo funciona evaluate.py:

  1. Cargar el checkpoint:
     ck = torch.load("outputs/moppo/moppo_checkpoint.pt")
     
     Contenido del checkpoint:
     {
       "model_state": {pesos de la red neuronal},
       "obs_dim":     10002,
       "n_nodes":     100,
       "n_actions":   4950,
       "args": {
         "graph": "facebook",
         "n_nodes": 100,
         "k": 2,
         "episode_len": 50,
         ...
       }
     }

  2. Reconstruir el grafo original (mismo que durante entrenamiento):
     G0 = load_facebook_ego(n_nodes=100, seed=42)

  3. Crear el entorno con los mismos parámetros:
     env = GraphAnonEnv(G0, k=2, T=50)

  4. Sweep de α ∈ linspace(0,1,n_weights=11):
     results = evaluate_policy(env, model, n_weights=11, n_episodes=5)
     → lista de (α, r_priv_final, r_util_final)

  5. Calcular frente de Pareto y hypervolume:
     pareto_idx = compute_pareto_front(reward_vecs)
     hv = hypervolume_indicator(pareto_pts)

  6. Auditoría de equidad con α=0.5 (balance):
     audit_stats = degree_quartile_audit(env.G0, env.G)
     disparity = fairness_disparity_index(audit_stats)

  7. Guardar resultados:
     - pareto_frontier.png  (figura matplotlib)
     - eval_results.json    (métricas numéricas)
```

---

## 8. Los Baselines

### ¿Por qué necesitamos baselines?

Para demostrar que MOPPO **aporta algo** frente a alternativas más simples. Un paper científico siempre debe demostrar que el método propuesto supera a métodos ya conocidos.

### 8.1 Política aleatoria

```
DESCRIPCIÓN:
  En cada paso, elegir una de las 4,950 aristas COMPLETAMENTE AL AZAR.
  No usa ninguna información del grafo ni de los objetivos.

PROPÓSITO:
  "Cota inferior": cualquier método inteligente DEBE superar esto.
  Si MOPPO no supera la política aleatoria → hay un bug grave.

EXPECTATIVAS:
  - r_priv_final ≈ -0.85 a -0.95 (muy mala k-anonimato)
    (50 flips aleatorios tienden a diversificar grados, creando más grados únicos)
  - r_util_final ≈ -0.1 a -0.2  (algo de degradación de clustering)
  - HV ≈ 0.01 - 0.05 (muy pequeño)

IMPLEMENTACIÓN (notebook):
  while not done:
      obs, _, terminated, truncated, _ = env_r.step(env_r.action_space.sample())
      done = terminated or truncated
  # env_r.action_space.sample() → entero aleatorio uniforme en [0, 4949]
```

### 8.2 Greedy Liu & Terzi (2008)

```
DESCRIPCIÓN:
  Algoritmo determinístico clásico de la literatura de k-anonimato en grafos.
  Solo AGREGA aristas estratégicamente para fusionar clases de grado.

ALGORITMO (paso a paso):
  ┌─────────────────────────────────────────────────────────────┐
  │  Repetir hasta que no haya nodos en riesgo (o max_ops):     │
  │                                                             │
  │  1. Encontrar nodos "en riesgo":                            │
  │     at_risk = [v : |clase_de_grado(v)| < k]                │
  │                                                             │
  │  2. Tomar el primer nodo en riesgo: u                       │
  │     u_deg = grado(u)                                        │
  │                                                             │
  │  3. Estrategia de conexión (prioridad):                     │
  │     a) ¿Existe w con grado(w) = u_deg y w no vecino de u?  │
  │        → Conectar u-w                                       │
  │        (después: grado(u)=u_deg+1, grado(w)=u_deg+1        │
  │         → los dos cambian juntos, pueden seguir siendo par) │
  │                                                             │
  │     b) Si no existe ese w:                                  │
  │        → Conectar u con cualquier nodo no vecino            │
  │        (subóptimo pero avanza hacia k-anonimato)            │
  └─────────────────────────────────────────────────────────────┘

LIMITACIONES:
  ✗ Solo agrega aristas (nunca elimina)
  ✗ Produce UN SOLO punto en el espacio (r_priv, r_util)
    No puede controlar el balance; siempre maximiza privacidad
  ✗ No generaliza: re-ejecutar para cada nuevo grafo o k
  ✗ No garantiza k-anonimato en todos los casos (depende de la estructura)

VENTAJA PARA EL PAPER:
  ✓ Es la referencia principal de la literatura (Liu & Terzi, KDD 2008)
  ✓ Computable en segundos
  ✓ Determinista: reproducible siempre
```

### 8.3 Fixed-α PPO

```
DESCRIPCIÓN:
  Misma arquitectura que MOPPO, pero con α FIJO durante todo el entrenamiento.
  Para cubrir K puntos del frente Pareto: entrenar K agentes independientes.

IMPLEMENTACIÓN:
  Para α_fixed ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:
    Entrenar un agente con:
      env.reset(weight=[α_fixed, 1-α_fixed])  ← siempre el mismo α
    → El agente aprende SOLO para ese α específico

vs. MOPPO:
  MOPPO:                              Fixed-α K=5:
  ──────────────────────────────      ─────────────────────────────────
  1 entrenamiento                     5 entrenamientos (K×N_ITER)
  α ~ U(0,1) cada episodio            α fijo durante todo el entrenamiento
  1 modelo cubre todo [0,1]           1 modelo por cada α
  Cambiar α = gratis en inferencia    Cambiar α = reentrenar todo

PROPÓSITO EN EL PAPER:
  Aislar el beneficio del weight conditioning:
  "¿Cuánto vale tener un solo modelo multi-objetivo
   vs. K modelos mono-objetivo?"

  Si MOPPO ≈ Fixed-α → weight conditioning no ayuda mucho
  Si MOPPO >> Fixed-α → weight conditioning es clave

IMPLEMENTACIÓN (notebook):
  def collect_rollout_fixed(env, model, n_steps, device, alpha):
      obs, _ = env.reset(weight=[alpha, 1.0 - alpha])
      # ↑ DIFERENCIA CLAVE: siempre el mismo alpha, no aleatorio
      for _ in range(n_steps):
          ...
          obs = env.reset(weight=[alpha, 1.0 - alpha])[0] if done else next_obs
          # ↑ También usar el mismo alpha al reiniciar en medio del rollout
```

### 8.4 NSGA-II

```
DESCRIPCIÓN:
  Algoritmo evolutivo multi-objetivo. No aprende una política de decisiones
  secuenciales: busca directamente el mejor grafo FINAL posible.

REPRESENTACIÓN:
  Cada individuo = vector binario de 4,950 bits
  [b_0, b_1, ..., b_4949] donde bᵢ = 1 significa "flipear la arista i"

  Ejemplo:
    [0, 0, 1, 0, 1, 0, ...]
          ↑        ↑
     flipear (0,2)  flipear (0,4)
     (las demás aristas no se tocan)

EVALUACIÓN DE UN INDIVIDUO:
  1. Aplicar los flips a G₀ → obtener G_final
  2. Medir: [-r_priv(G_final), -r_util(G_final)]
     (NSGA-II minimiza, así que negamos las recompensas)

OPERADORES GENÉTICOS:
  Selección: por torneo con ranking de dominancia Pareto + crowding distance
  Cruce:     uniforme (cada bit tiene 50% de venir de padre₁ o padre₂)
  Mutación:  bit flip con probabilidad 10/4950 ≈ 0.2% por bit
             (en promedio, ~10 flips aleatorios por individuo por generación)

PARÁMETROS:
  pop_size = 25   (individuos por generación)
  n_gen    = 75   (número de generaciones)
  Total evaluaciones = 25 × 75 = 1,875 grafos evaluados

DIFERENCIAS FUNDAMENTALES CON MOPPO:

  NSGA-II:                          MOPPO:
  ──────────────────────────────    ────────────────────────────────
  Busca el grafo FINAL óptimo       Aprende una POLÍTICA de acciones
  No hay "pasos" ni "episodios"     50 pasos de decisión por episodio
  Sin límite de T flips             Limitado a T=50 flips
  No generaliza a nuevos grafos     Un modelo → cualquier grafo parecido
  Debe re-ejecutarse para nuevo α   Cambiar α = gratis en inferencia
  O(pop×gen) evaluaciones de grafo  O(N_ITER×N_STEPS) transiciones

CUÁNDO NSGA-II GANA:
  Cuando el espacio de búsqueda es pequeño y tenemos tiempo de re-optimizar
  para cada α. No necesita "aprender" porque busca directamente.

CUÁNDO MOPPO GANA:
  Cuando necesitamos respuesta en tiempo real, queremos adaptar el α
  sin reentrenar, o el grafo cambia con el tiempo (online learning).
```

---

## 9. Métricas de Evaluación

### 9.1 Hypervolume

El **hypervolume indicator** (HV) mide la calidad de un frente de Pareto con un solo número: el "volumen" del espacio que el frente domina por encima de un punto de referencia.

```
Hypervolume en 2D (área en lugar de volumen):

r_util
  │
0 ┤─────────────────────────────────────── ←línea ref r_util=-1
  │                          ████████████● ← punto en (r_priv=0, r_util=-0.02)
  │               ████████████
  │   ████████████
-1 ┤● ─────────────────────────────────── ← punto de referencia (-1, -1)
  └──────────────────────────────────────── r_priv
  -1                                      0

HV = área de la región sombreada

Cálculo (sweep de izquierda a derecha):
  Puntos del frente ordenados por r_priv:
    p₁ = (-0.8, -0.01)
    p₂ = (-0.4, -0.03)
    p₃ = (-0.1, -0.05)
    p₄ = ( 0.0, -0.10)

  Área:
    Rectángulo 1: (−0.8 − (−1)) × (−0.01 − (−1)) = 0.2 × 0.99 = 0.198
    Rectángulo 2: (−0.4 − (−0.8)) × (−0.03 − (−1)) = 0.4 × 0.97 = 0.388
    Rectángulo 3: (−0.1 − (−0.4)) × (−0.05 − (−1)) = 0.3 × 0.95 = 0.285
    Rectángulo 4: ( 0.0 − (−0.1)) × (−0.10 − (−1)) = 0.1 × 0.90 = 0.090

    HV total = 0.198 + 0.388 + 0.285 + 0.090 = 0.961
```

**Punto de referencia (-1, -1):**
- r_priv = -1: el peor estado posible de k-anonimato (todos los nodos con grado único, k=2)
- r_util = -1: degradación catastrófica del clustering (cambio de 1.0 en coeficiente de clustering)

Si un método no logra que ninguno de sus puntos domine (-1,-1) → HV = 0. Esto significa que el método deja el grafo en un estado comparable o peor al peor caso posible. Esto le pasó a MOPPO con solo 50 iteraciones porque usaba acumulación de recompensas (bug ya corregido).

### 9.2 Frente de Pareto — Visualización comparativa

```
Interpretación del plot Pareto overlay:

r_util (mayor = más útil)
  │
0 ┤ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ X  ← Greedy
  │                                              ■─■    ← NSGA-II (frente)
  │ ◆─────────────────────────────────────◆          ← MOPPO (frente ideal)
  │ ◆                                              ◆    con 500+ iters
  │                                                      ↑ cubre todo el
  │                                                        rango de α
  │ · · · · · · · · · · ·                            ← Aleatorio (mala)
-1 ┤
  └────────────────────────────────────────────────── r_priv
  -1                                                0

Cómo leer el plot:
  ↗ Esquina superior derecha = IDEAL (privacidad perfecta, utilidad perfecta)
    Ningún método llega ahí (son objetivos en conflicto)

  ✗ Greedy: un solo punto fijo. Buena privacidad, pero no puede
    controlar el balance con la utilidad.

  ■ NSGA-II: algunos puntos discretos del frente. Buena calidad
    absoluta, pero re-corre para cada α nuevo.

  ◆ MOPPO (bien entrenado): frente continuo, cubre todo [0,1] en α.
    Puede ir de "priorizar privacidad" a "priorizar utilidad"
    con solo cambiar α en inferencia. Sin reentrenar.
```

---

## 10. Marco Ético

La anonimización de grafos no es solo un problema técnico. Involucra decisiones éticas sobre quién tiene derecho a qué información y quién carga con el costo de la privacidad.

### 10.1 Nissenbaum — Integridad Contextual

**Helen Nissenbaum** (2004) propone la teoría de **integridad contextual**:

> "La privacidad se viola cuando la información fluye fuera del contexto normativo en que fue creada."

```
Aplicación a este proyecto:

Contexto original (C₁: red social Facebook):
  ┌──────────────────────────────────────────────┐
  │  Norma implícita: "mis datos de amistad       │
  │  son solo para mis amigos y Facebook"         │
  └──────────────────────────────────────────────┘
                      │ flujo de información
                      ▼
Contexto destino (C₂: investigación académica):
  ┌──────────────────────────────────────────────┐
  │  Investigadores analizan la estructura        │
  │  de la red para estudiar propagación          │
  │  de enfermedades, influencia social, etc.     │
  └──────────────────────────────────────────────┘

Sin anonimización: la información fluye CON la identidad de los usuarios
→ violación de integridad contextual (fluye fuera del contexto C₁)

Con k-anonimato: la información fluye SIN la identidad (grado compartido)
→ parcialmente preserva la integridad contextual:
  el PATRÓN de la red pasa a C₂, pero no la IDENTIDAD
```

**Justificación del k-anonimato en este marco:**
El k-anonimato garantiza que cualquier información que fluye al nuevo contexto sea atribuible a AL MENOS k personas, no a un individuo específico. Esto reduce la violación de integridad contextual.

### 10.2 Rawls — Principio de Diferencia

**John Rawls** (1971) en *A Theory of Justice* propone:

> "Las desigualdades sociales son justas solo si benefician al grupo menos favorecido."

```
Aplicación al disparity index:

En una red social hay jerarquía natural:
  Q4 (hubs, grado alto): mucho poder, muchas conexiones
  Q1 (periferia, grado bajo): poco poder, pocas conexiones

Los nodos Q1 ya son los "menos favorecidos":
  - Grados únicos más frecuentes en la periferia → más vulnerables
  - Menor influencia en la red → más aislados

Pregunta de Rawls aplicada:
  ¿La anonimización impone costos desproporcionados a los ya desfavorecidos (Q1)?

  SI Δ = Q1_cost/Q4_cost > 1.5:
    → La periferia carga con el precio de la privacidad de los hubs
    → Esto viola el principio de diferencia de Rawls
    → La política de anonimización es injusta

  SI Δ ≤ 1.5:
    → La carga está distribuida de forma aceptablemente proporcional
    → La política satisface el criterio Rawlsiano mínimo

El índice Δ OPERACIONALIZA la justicia distributiva de Rawls:
no es solo un número técnico, es una medida ética.
```

### 10.3 El índice de disparidad Δ en la práctica

```
Cálculo concreto paso a paso:

Ejemplo: grafo después de anonimizar con MOPPO (α=0.5)

Paso 1: Grados originales y cuartiles (n=100 nodos)
  Percentil 25 del grado: por ejemplo, grado ≤ 2  → Q1
  Percentil 75 del grado: por ejemplo, grado ≥ 7  → Q4
  Entre 2 y 7: Q2Q3

Paso 2: Contar cambios de aristas por nodo
  Nodo v ∈ Q1 (grado=2):
    Aristas en G₀ pero no en G_final: 1 (eliminada)
    Aristas en G_final pero no en G₀: 0 (ninguna agregada)
    edge_changes(v) = 1
    normalized_loss(v) = 1/2 = 0.50

  Nodo h ∈ Q4 (grado=15):
    Aristas en G₀ pero no en G_final: 2 (eliminadas)
    Aristas en G_final pero no en G₀: 1 (agregada)
    edge_changes(h) = 3
    normalized_loss(h) = 3/15 = 0.20

Paso 3: Promediar por grupo
  avg_Q1 = promedio de normalized_loss de todos los nodos Q1 → ej: 0.35
  avg_Q4 = promedio de normalized_loss de todos los nodos Q4 → ej: 0.25

Paso 4: Calcular Δ
  Δ = 0.35 / 0.25 = 1.40 → ACEPTABLE (≤ 1.5)

Interpretación:
  "La periferia (Q1) soporta un 40% más de cambios relativos que los hubs (Q4).
   Esto está dentro del umbral Rawlsiano de ≤ 1.5."
```

---

## 11. Los Notebooks

### `graph_anon_eda.ipynb` — Exploración y prueba

Este notebook sirve para **entender el grafo** y **verificar que todos los módulos funcionan** antes de lanzar el entrenamiento.

```
Secciones del notebook:
┌──────────────────────────────────────────────────────────────────┐
│ §0: Configuración — rutas, imports, verificación de módulos      │
├──────────────────────────────────────────────────────────────────┤
│ §1: Comparación de grafos preset                                 │
│     Tabla comparativa: karate (34n), barbell (20n),              │
│     random (20n), facebook-50, facebook-100                      │
│     Métricas: nodos, aristas, densidad, clustering, k-anonimato  │
├──────────────────────────────────────────────────────────────────┤
│ §2: EDA completo de ego-Facebook (n=100)                         │
│     - Estadísticas básicas (densidad, diámetro, asortatividad)   │
│     - Histograma de grados                                       │
│     - Plot log-log (¿power-law?)                                 │
│     - CDF del grado                                              │
│     - Análisis k-anonimato: ¿cuántos nodos son vulnerables?      │
│     - Visualización spring layout (coloreado por cuartil Q1..Q4) │
├──────────────────────────────────────────────────────────────────┤
│ §3: Test de GraphAnonEnv                                         │
│     - Verificar shapes de obs y action_space                     │
│     - Rollout de 50 pasos con política aleatoria                 │
│     - Verificar que r_priv ∈ [-1, 0] y r_util ≤ 0               │
├──────────────────────────────────────────────────────────────────┤
│ §4: Test de MOPPOActorCritic                                     │
│     - Contar parámetros (~3.9M)                                  │
│     - Forward pass: shape de logits y value                      │
│     - Entropía inicial ≈ log(4950) ≈ 8.5 nats (exploración máx) │
├──────────────────────────────────────────────────────────────────┤
│ §5: Mini-entrenamiento (10 iteraciones)                          │
│     - Curva de loss vs. iteración                                │
│     - Verificar que el loss desciende (señal de aprendizaje)     │
├──────────────────────────────────────────────────────────────────┤
│ §6: Sweep Pareto post mini-entrenamiento                         │
│     - Plot r_priv vs. r_util coloreado por α                     │
│     - Hypervolume del frente resultante                          │
├──────────────────────────────────────────────────────────────────┤
│ §7: Fairness audit + TREX clustering de trayectorias             │
│     - Índice Δ con el mini-modelo                                │
│     - k-means sobre perfiles (r_priv, r_util) por episodio       │
└──────────────────────────────────────────────────────────────────┘
```

### `baseline_comparison.ipynb` — Comparación de métodos

Este notebook es el **experimento central del paper**: compara todos los métodos con las mismas métricas y produce las figuras para publicar.

```
Secciones del notebook:
┌──────────────────────────────────────────────────────────────────┐
│ Setup compartido:                                                │
│   - Grafo ego-Facebook n=100                                     │
│   - evaluate_at_alpha() → r_priv, r_util del grafo FINAL        │
│   - evaluate_sweep()    → sweep sobre α ∈ [0,1]                  │
│   - compute_hv()        → hypervolume del frente                 │
│   - apply_flips()       → aplicar vector binario de flips        │
│   - get_final_graph()   → ejecutar un episodio, retornar G_final │
├──────────────────────────────────────────────────────────────────┤
│ MOPPO (N_ITER=200, N_STEPS=256, α~U(0,1)):                       │
│   → Entrenar modelo → evaluar sweep → calcular HV                │
├──────────────────────────────────────────────────────────────────┤
│ Política Aleatoria:                                              │
│   → Sin entrenamiento → evaluar sweep (cota inferior)            │
├──────────────────────────────────────────────────────────────────┤
│ Greedy Liu & Terzi:                                              │
│   → Ejecutar algoritmo greedy → un punto (r_priv, r_util)        │
├──────────────────────────────────────────────────────────────────┤
│ Fixed-α PPO (K=5, N_ITER=100 por agente):                        │
│   α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}                              │
│   → Entrenar 5 agentes → evaluar cada uno en su α fijo          │
├──────────────────────────────────────────────────────────────────┤
│ NSGA-II (pop=25, gen=75, n_var=4950):                            │
│   → Ejecutar optimización evolutiva → frente de Pareto binario   │
├──────────────────────────────────────────────────────────────────┤
│ Plot overlay Pareto:                                             │
│   → outputs/pareto_comparison.png                               │
├──────────────────────────────────────────────────────────────────┤
│ Tabla de hypervolume:                                            │
│   Metodo | HV | Pareto pts | Training runs | Adaptable?          │
├──────────────────────────────────────────────────────────────────┤
│ Fairness audit comparativo:                                      │
│   → outputs/fairness_comparison.png                             │
│   → Tabla Q1/Q2Q3/Q4 normalized_loss + Δ por método             │
├──────────────────────────────────────────────────────────────────┤
│ Resumen ejecutivo automático:                                    │
│   → Genera el argumento para el paper basado en los resultados   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 12. Cómo Ejecutar el Proyecto

### Requisitos de sistema

```
Python:     3.9+
PyTorch:    2.0+  (sin GPU es suficiente para n=100)
RAM:        4 GB mínimo (8 GB recomendado para entrenamiento largo)
Disco:      ~200 MB (dataset + checkpoints)
GPU:        Opcional (la obs_dim es grande, GPU ayuda pero no es necesaria)
```

### Instalar dependencias

```powershell
pip install torch networkx gymnasium numpy matplotlib pandas pymoo scikit-learn
```

### Opción A: Entrenamiento completo por línea de comandos

```powershell
# Desde el directorio raíz del proyecto:
cd scripts\python

# Entrenamiento MOPPO en ego-Facebook (n=100)
# Tiempo estimado: ~15-45 min en CPU
python train_main.py `
  --graph       facebook `
  --n_nodes     100 `
  --k           2 `
  --episode_len 50 `
  --n_iter      500 `
  --n_steps     512 `
  --hidden_dim  256 `
  --lr          3e-4 `
  --gamma       0.99 `
  --lam         0.95 `
  --clip_eps    0.2 `
  --seed        42 `
  --out_dir     outputs/moppo

# Archivos generados:
#   outputs/moppo/moppo_checkpoint.pt   ← Pesos del modelo
#   outputs/moppo/train_metrics.json    ← {iter, loss, mean_reward} por iteración

# Evaluación del checkpoint
python evaluate.py `
  --checkpoint outputs/moppo/moppo_checkpoint.pt `
  --n_weights  21 `
  --n_episodes 10 `
  --out_dir    outputs/eval

# Archivos generados:
#   outputs/eval/pareto_frontier.png   ← Figura del frente Pareto
#   outputs/eval/eval_results.json     ← Métricas completas
```

### Opción B: Notebooks Jupyter (recomendada para experimentar)

```
1. Iniciar Jupyter:
   jupyter notebook
   (abre automáticamente el navegador)

2. Exploración y EDA:
   Abrir: notebooks/graph_anon_eda.ipynb
   Ejecutar: Kernel → Restart & Run All

3. Comparación de baselines (experimento completo):
   Abrir: notebooks/baseline_comparison.ipynb
   Ejecutar: Kernel → Restart & Run All
   
   ⚠ IMPORTANTE: Si el notebook tiene outputs viejos de una ejecución anterior,
   siempre hacer Kernel → Restart & Run All para limpiar el estado del kernel.
   Los outputs almacenados en el notebook pueden ser de código diferente.
```

### Parámetros: demo rápido vs. paper

```
Parámetro          Demo (5-15 min)    Paper (calidad)    Descripción
──────────────     ───────────────    ───────────────    ─────────────────────
N_ITER_MOPPO            200               500-1000      Iteraciones entrenamiento
N_ITER_FIXED            100               300           Iters por agente fixed-α
N_STEPS                 256               512           Pasos por rollout
hidden_dim              128               256           Neuronas por capa
n_weights eval           11                21           Puntos α en el sweep
n_episodes eval           5                10           Episodios por α (promedio)
pop_size NSGA-II         25                50           Individuos por generación
n_gen NSGA-II            75               200           Generaciones evolutivas
```

---

## 13. Resultados Esperados e Interpretación

### Predicciones de hypervolume

```
Con entrenamiento suficiente (500+ iters para MOPPO):

HYPERVOLUME (mayor = mejor, escala 0 a ~1):

  Método              HV Esperado      Notas
  ──────────────────  ───────────      ────────────────────────────────────
  Greedy Liu & Terzi  0.90 – 0.95      Bueno en privacidad, punto único
  NSGA-II             0.85 – 0.93      Buen frente, pero re-corre por α
  MOPPO (500+ iters)  0.30 – 0.65      Frente continuo, 1 solo entrenamiento
  Fixed-α K=5         0.20 – 0.45      K puntos, K entrenamientos
  Aleatorio           0.01 – 0.05      Cota inferior

ADVERTENCIA: Con pocas iteraciones (50-100), MOPPO y Fixed-α tendrán
HV ≈ 0 porque la política no ha aprendido a mejorar el k-anonimato aún.
Con 200 iteraciones ya se debería ver aprendizaje.
```

### El argumento del paper

La clave del argumento de MOPPO **no es tener el HV más alto** — es la **relación HV/costo computacional** y la **adaptabilidad en inferencia**:

```
┌───────────────────────────────────────────────────────────────────┐
│                   ARGUMENTO CENTRAL DEL PAPER                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [1] EFICIENCIA: MOPPO cubre todo el frente Pareto con           │
│      UN SOLO entrenamiento. Greedy/NSGA-II deben re-ejecutarse   │
│      para cada nuevo α. Fixed-α necesita K entrenamientos.       │
│                                                                   │
│  [2] ADAPTABILIDAD: En inferencia, cambiar α no requiere         │
│      reentrenar. Solo se modifica la entrada a la red.            │
│      Esto permite adaptación en tiempo real.                     │
│                                                                   │
│  [3] EQUIDAD: MOPPO puede ser auditado y comparado con          │
│      el índice Δ. Si Δ > 1.5, se puede ajustar el α o           │
│      agregar Δ como tercer objetivo en el entrenamiento.          │
│                                                                   │
│  [4] ESCALABILIDAD: A diferencia de Pareto Q-learning             │
│      (inviable para n=100, espacio de estados 2^4950),           │
│      MOPPO es un algoritmo aproximado eficiente basado           │
│      en redes neuronales.                                         │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### Interpretación del fairness audit

```
Método            Comportamiento esperado del Δ
─────────────     ─────────────────────────────────────────────────
MOPPO (α=0.5)     Δ ≈ 1.0 – 1.3  (tiende a la equidad si converge)
                  El agente aprende a distribuir la carga

Fixed-α (0.5)     Δ similar a MOPPO (misma arquitectura)

NSGA-II           Δ varía según el punto del frente evaluado
                  Puede ser < 1.0 (hubs pagan más, inusual)

Greedy            Δ > 1.0 – 2.0  (la periferia paga más)
                  Greedy agrega aristas a nodos de bajo grado (Q1)
                  → edge_changes(Q1) es alto → normalized_loss(Q1) ALTO
                  → Δ = Q1/Q4 puede superar 1.5 → injusto según Rawls
                  
Aleatorio         Δ ≈ 1.0 (los flips aleatorios no discriminan entre grupos)
```

---

## 14. Glosario

| Término | Definición en este proyecto |
|---------|---------------------------|
| **Grafo** | Estructura matemática de nodos (personas) y aristas (amistades) |
| **Nodo** | Una persona/entidad en la red social |
| **Arista** | Una relación (amistad) entre dos nodos |
| **Grado** | Número de aristas de un nodo (cuántas amistades tiene) |
| **Matriz de adyacencia** | Matriz n×n donde 1 = arista existe, 0 = no existe |
| **k-anonimato** | Propiedad: cada nodo comparte su grado con ≥k-1 otros nodos |
| **Clase de equivalencia** | Conjunto de nodos que comparten el mismo grado |
| **Flip de arista** | Agregar una arista si no existe, o eliminarla si existe |
| **BFS** | Búsqueda en anchura (Breadth-First Search): explora el grafo nivel por nivel |
| **Coeficiente de clustering** | Probabilidad de que los vecinos de un nodo estén conectados entre sí |
| **MOMDP** | Proceso de Decisión de Markov Multi-Objetivo |
| **Episodio** | Una secuencia completa de T=50 acciones desde G₀ hasta G_final |
| **Política (π)** | La función que mapea estados → probabilidades de acción |
| **Ventaja (A_t)** | Cuánto mejor fue una acción comparada con lo esperado por el crítico |
| **GAE** | Estimación Generalizada de la Ventaja (Schulman 2016) |
| **PPO** | Proximal Policy Optimization: algoritmo RL con clip para estabilidad |
| **MOPPO** | PPO + weight conditioning: un modelo cubre todo el frente Pareto |
| **Weight conditioning** | Incluir el vector de pesos [α, 1-α] en la observación del agente |
| **Escalarización** | Convertir [r_priv, r_util] → r_escalar = α·r_priv + (1-α)·r_util |
| **Frente de Pareto** | Conjunto de soluciones no dominadas en el espacio de objetivos |
| **Dominancia Pareto** | A domina a B si A es ≥ B en todos los objetivos y > en al menos uno |
| **Hypervolume (HV)** | Área dominada por el frente Pareto sobre un punto de referencia |
| **Disparity index (Δ)** | Razón de costo normalizado Q1/Q4; mide equidad de Rawls |
| **Integridad contextual** | Teoría de Nissenbaum: privacidad = info que no sale de su contexto |
| **Principio de diferencia** | Rawls: las desigualdades son justas solo si benefician al menos favorecido |
| **Actor-Critic** | Arquitectura RL con dos cabezas: π(a\|s) = actor y V(s) = critic |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II: algoritmo evolutivo multi-objetivo |
| **ego-Facebook** | Dataset de Facebook de SNAP: 4039 nodos, 88234 aristas de amistad |
| **obs_dim** | Dimensión del vector de observación: n²+2 = 10,002 para n=100 |
| **n_actions** | Número de acciones posibles: C(n,2) = 4,950 para n=100 |
| **α (alpha)** | Peso de la preferencia por privacidad vs. utilidad; α ∈ [0,1] |

---

## Diagrama de dependencias entre módulos

```
╔═══════════════════════════════════════════════════════════════════════╗
║                    MAPA COMPLETO DE MÓDULOS                          ║
╚═══════════════════════════════════════════════════════════════════════╝

  datasets.py                            utils.py
  ──────────                             ────────
  load_facebook_ego()               ┌── compute_k_anonymity_reward(G, k)
          │                         │   compute_utility_reward(G, G0)
          │ retorna                 │   graph_to_flat_adj(G, n)
          │ nx.Graph                │
          ▼                         │
         G₀ ─────────────────────► env.py
                                    ───────
                                    GraphAnonEnv(G₀, k, T)
                                    • reset([α,1-α]) → obs
                                    • step(action)   → obs, r, done, info
                                    • _obs() = flatten(A) + [α,1-α]
                                            │
                               ┌────────────┘
                               │
           ┌───────────────────┼────────────────────────┐
           │                   │                        │
           ▼                   ▼                        ▼
      models.py          train_main.py            evaluation.py
      ─────────          ─────────────            ─────────────
  MOPPOActorCritic       collect_rollout()        evaluate_policy()
  • shared (2 capas)     compute_gae()            compute_pareto_front()
  • actor_head           ppo_update()             hypervolume_indicator()
  • critic_head          train() ← punto          └── usa models.py + env.py
  • get_action()           de entrada                       │
                                │                          │
                                │ guarda                   │
                                ▼                          │
                         checkpoint.pt                     │
                                │                          │
                                │ carga                    │
                                └──────────────────────────┤
                                                           ▼
                                                     evaluate.py
                                                     (script final)
                                                           │
                                              ┌────────────┘
                                              │ usa
                                              ▼
                                         audit.py
                                         ────────
                                         degree_quartile_audit()
                                         fairness_disparity_index()
                                         trex_trajectory_cluster()
                                              │
                                              ▼
                                      pareto_frontier.png
                                      eval_results.json
                                      fairness_comparison.png
```

---

*Guía generada para el proyecto "Anonymizing Social Graphs with Multi-Objective Reinforcement Learning".*

*Dataset: ego-Facebook (SNAP, Leskovec & Mcauley 2012), n=100 nodos, k=2.*

*Algoritmo: MOPPO = weight-conditioned PPO (Schulman et al. 2017).*

*Referencias éticas: Rawls 1971, Nissenbaum 2004, Barocas & Moritz 2023.*

*Baselines: Liu & Terzi KDD 2008 (Greedy), Deb et al. 2002 (NSGA-II).*
