"""
GraphAnonEnv v2 — con acción no-op y señal de utilidad por paso.

Cambios clave respecto a v1:
  - Acción n_actions = no-op (el agente no modifica el grafo ese paso)
  - Step reward usa proxy O(1) de utilidad: -|Δedges| / |E_0|
    → da señal inmediata para que el agente aprenda a "no actuar" cuando α es bajo
  - r_util "real" (clustering) solo se computa al final del episodio para evaluación

Por qué esto hace que el weight conditioning funcione:
  - α=0 (maximizar utilidad): cualquier flip degrada edges_changed_ratio → penalización
    → política óptima = no-op siempre → r_util=0, r_priv=baseline
  - α=1 (maximizar privacidad): flipear buenos edges mejora r_priv → recompensa
    → política óptima = flipear agresivamente
  - α intermedio: balance natural → frente de Pareto real
"""

from itertools import combinations

import gymnasium as gym
import networkx as nx
import numpy as np

from .utils import compute_k_anonymity_reward, compute_utility_reward


class GraphAnonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, G0: nx.Graph, k: int = 2, T: int = 50):
        super().__init__()
        self.G0 = nx.convert_node_labels_to_integers(G0.copy())
        self.n  = self.G0.number_of_nodes()
        self.k  = k
        self.T  = T

        self.all_edges  = list(combinations(range(self.n), 2))
        self.n_actions  = len(self.all_edges) + 1   # +1 = no-op
        self.NO_OP      = len(self.all_edges)        # index of no-op action
        self._E0        = self.G0.number_of_edges()  # |E_original| (constant)

        self.action_space = gym.spaces.Discrete(self.n_actions)
        obs_dim = self.n * 2 + 2   # degree_seq + anon_deficit + weight
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.G: nx.Graph = None
        self.t: int = 0
        self.weight: np.ndarray = np.array([0.5, 0.5], dtype=np.float32)
        self._flipped: set = set()   # tracks which edges have been net-flipped

    def reset(self, *, seed=None, options=None, weight=None):
        super().reset(seed=seed)
        self.G = self.G0.copy()
        self.t = 0
        self._flipped = set()
        if weight is not None:
            alpha = float(np.clip(weight[0], 0.0, 1.0))
        else:
            lo, hi = 0.0, 1.0
            if options and "alpha_range" in options:
                lo, hi = options["alpha_range"]
            alpha = float(self.np_random.uniform(lo, hi))
        self.weight = np.array([alpha, 1.0 - alpha], dtype=np.float32)
        return self._obs(), {}

    def step(self, action: int):
        if action == self.NO_OP:
            # No-op: graph unchanged, no privacy gain, no utility cost
            r_priv = compute_k_anonymity_reward(self.G, self.k)
            r_util_proxy = 0.0   # no change → no utility penalty
        else:
            u, v = self.all_edges[action]
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                edge_key = (u, v)
            else:
                self.G.add_edge(u, v)
                edge_key = (u, v)

            # Track net flips: odd number of flips = edge is in different state
            if edge_key in self._flipped:
                self._flipped.discard(edge_key)
            else:
                self._flipped.add(edge_key)

            r_priv = compute_k_anonymity_reward(self.G, self.k)

            # Fast O(1) utility proxy: fraction of net-changed edges
            r_util_proxy = -len(self._flipped) / max(self._E0, 1)

        self.t += 1
        terminated = self.t >= self.T

        # True r_util (clustering) only at episode end — used for eval info
        if terminated:
            r_util_true = compute_utility_reward(self.G, self.G0)
        else:
            r_util_true = r_util_proxy   # proxy during episode

        reward_vec   = np.array([r_priv, r_util_true], dtype=np.float32)
        scalar_reward = float(self.weight @ np.array([r_priv, r_util_proxy]))
        return self._obs(), scalar_reward, terminated, False, {"reward_vec": reward_vec}

    def _obs(self) -> np.ndarray:
        degrees = np.array([self.G.degree(v) for v in range(self.n)],
                           dtype=np.float32) / self.n
        deg_counts: dict = {}
        for _, d in self.G.degree():
            deg_counts[d] = deg_counts.get(d, 0) + 1
        deficit = np.array(
            [max(0, self.k - deg_counts[self.G.degree(v)]) / self.k
             for v in range(self.n)],
            dtype=np.float32,
        )
        return np.concatenate([degrees, deficit, self.weight])

    def get_action_mask(self) -> np.ndarray:
        """
        Returns bool mask (n_actions,) where True = valid action.

        Valid actions:
          - NO_OP always valid
          - edges incident to at least one node with k-anonymity deficit > 0

        When the graph is already k-anonymous, only NO_OP is valid → agent stops.
        Reduces effective action space from ~4,950 to typically 50-200.
        """
        deg_counts: dict = {}
        for _, d in self.G.degree():
            deg_counts[d] = deg_counts.get(d, 0) + 1

        at_risk = {
            v for v in range(self.n)
            if deg_counts.get(self.G.degree(v), 0) < self.k
        }

        mask = np.zeros(self.n_actions, dtype=bool)
        mask[self.NO_OP] = True          # no-op always available

        if at_risk:
            for i, (u, v) in enumerate(self.all_edges):
                if u in at_risk or v in at_risk:
                    mask[i] = True

        return mask
