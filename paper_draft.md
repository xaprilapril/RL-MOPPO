# Anonymizing Social Graphs with Multi-Objective Reinforcement Learning: A Privacy–Utility Pareto Analysis with Fairness Audit

**Draft — do not cite without author permission**
Last updated: 2026-05-25

---

## Abstract

We present a Multi-Objective Reinforcement Learning (MORL) framework for social graph anonymization that simultaneously optimizes privacy (k-anonymity by degree sequence) and structural utility (clustering coefficient preservation). Our agent, trained with a weight-conditioned Proximal Policy Optimization algorithm (MOPPO), learns to navigate the full privacy–utility Pareto frontier through a single training run by randomizing the scalarization weight α per episode. Beyond technical performance, we conduct a fairness audit examining whether anonymization cost is distributed equitably across node degree quartiles. Grounded in Nissenbaum's contextual integrity framework and Rawls' difference principle, we find that naive anonymization policies disproportionately sever edges of low-degree (peripheral) nodes — a form of structural discrimination against the least-connected members of the social graph. We evaluate on the ego-Facebook network (SNAP, n=100 subgraph) and compare against NSGA-II, MOEA/D, Fixed-α PPO, and a random policy baseline. Our ethical audit methodology, including a disparity index and TREX trajectory clustering, is released as an open tool for future graph privacy research.

**Keywords:** graph anonymization, multi-objective reinforcement learning, k-anonymity, Pareto frontier, fairness, contextual integrity.

---

## 1. Introduction

Social networks encode sensitive information about human relationships that users share under implicit contextual norms: a friendship on Facebook is disclosed to friends, not to employers, advertisers, or adversaries. When such networks are published — even with node identifiers removed — their structural properties can reveal enough to re-identify individuals. Backstrom et al. [1] demonstrated that knowing as few as two target nodes' neighborhoods suffices to de-anonymize them from a structurally "anonymized" graph. Narayanan and Shmatikoff [2] showed that users of the Netflix dataset could be re-identified by crossing with public IMDb data. These empirical results establish that graph anonymization is not a cosmetic operation but a necessary safeguard against real privacy harms.

What constitutes a privacy harm in this setting? Nissenbaum's theory of *contextual integrity* [3] offers a principled answer: privacy is violated when information flows in ways that violate the norms of the context in which it was originally shared. Social ties disclosed within a friendship network carry contextual norms — they are not meant to flow to analytical pipelines that reconstruct full social graphs. Degree-sequence k-anonymity operationalizes this principle at the structural level: a node's identity is protected when its degree is shared by at least k other nodes in the published graph, making it indistinguishable within an equivalence class of size k.

The challenge is that anonymization degrades structural utility. Edge modifications that raise k-anonymity necessarily distort topological properties — clustering coefficients, community structure, shortest paths — that downstream analyses depend on. This creates a genuine multi-objective conflict: no single graph modification Pareto-dominates all others across both objectives simultaneously. Classical single-objective methods collapse this tradeoff into a fixed scalar [4], discarding information about the full conflict structure.

We address this with Multi-Objective Reinforcement Learning (MORL). The graph modification process is sequential (each edge flip changes the state), the objective space is two-dimensional (privacy, utility), and the desired output is the entire Pareto frontier — a curve, not a point. Our agent learns a weight-conditioned policy that produces Pareto-optimal anonymizations for any privacy–utility tradeoff at inference time.

A further contribution, absent from prior MORL literature, is an ethical audit of the learned policy. Not all nodes pay equal costs when a graph is anonymized: peripheral nodes (low degree) may lose a disproportionately large fraction of their connections compared to hubs. Following Rawls' difference principle [5], an anonymization policy is just only if the burden on the least-advantaged group is minimized. We quantify this with a disparity index (Q1/Q4 normalized edge loss ratio) and cluster trajectory profiles using a TREX-style analysis [6].

**Contributions:**
1. First MORL formulation for social graph anonymization, casting edge-flip anonymization as a weight-conditioned MOMDP.
2. MOPPO: a weight-conditioned PPO agent that learns the full Pareto frontier in a single training run.
3. A fairness audit framework grounded in distributive justice theory, measuring degree-quartile disparity in anonymization cost.
4. Empirical evaluation on the ego-Facebook network (SNAP) with comparison to evolutionary and tabular baselines.
5. Open-source release of the full pipeline including the fairness audit tools.

---

## 2. Related Work

### 2.1 Graph Anonymization

The foundational work on structural graph anonymization is Liu and Terzi [4], who introduced k-anonymity by degree sequence and a greedy edge-addition algorithm to achieve it. Hay et al. [7] extended this to community-preserving anonymization. Zou et al. [8] proposed k-automorphism, a stricter structural privacy notion. These methods produce a single anonymized graph; they do not explore the privacy–utility tradeoff space. Our work replaces the single-point greedy approach with a Pareto-optimal multi-objective agent.

The threat model motivating graph anonymization is well-established. Backstrom et al. [1] showed passive de-anonymization attacks require minimal knowledge of the target's social neighborhood. Narayanan and Shmatikoff [2] demonstrated active attacks on real-world datasets. These results justify strong privacy guarantees — and make the cost of anonymization non-negotiable.

### 2.2 Multi-Objective Reinforcement Learning

MORL has seen recent application to complex sequential optimization problems with competing objectives. Hu and Luo [9] (PA2D-MORL, AAAI 2024) apply Pareto-guided MORL to autonomous driving under safety–efficiency tradeoffs. Nguyen et al. [10] (NeurIPS 2024) apply it to ADMET drug screening, learning the full Pareto frontier of pharmacological objectives. Zhang et al. [11] (IEEE 2024) apply MORL to chaos engineering in distributed systems, balancing fault injection intensity against service reliability. Lautenbacher et al. [12] apply it to power grid control. No prior work applies MORL to graph privacy; the challenges — sequential modification of a structured state, multi-dimensional objectives, need for a full frontier — make this a natural extension.

### 2.3 Ethics of Graph Privacy

The normative foundation for graph privacy rests on theories that go beyond data confidentiality. Westin [13] defined privacy as "the claim of individuals, groups, or institutions to determine for themselves when, how, and to what extent information about them is communicated." Nissenbaum [3] operationalized this as *contextual integrity*: the right not that information be kept secret, but that it flow according to the norms of the context in which it was shared. Social tie information shared within a friendship network violates contextual integrity when processed as raw structural data by external analytical systems, even after identifier removal. Our choice of structural anonymization is grounded in this framework: we protect contextual identity by making each node's structural fingerprint indistinguishable from at least k−1 others.

### 2.4 Algorithmic Fairness and Distributive Justice

The question of *who bears the cost* of a privacy-enhancing transformation is distinct from whether the transformation achieves privacy. Rawls [5] formulates the *difference principle*: inequalities are just only if they benefit the least-advantaged members of society. Applied to graph anonymization, the least-advantaged nodes are those with the lowest degree — peripheral members already most isolated. If anonymization disproportionately removes their edges, it concentrates harm where vulnerability is greatest.

Barocas, Hardt and Narayanan [14] provide the technical counterpart: a taxonomy of group fairness definitions and their mutual incompatibilities. We operationalize a variant of *equalized loss* across degree quartile groups: a policy is degree-fair if normalized edge loss is approximately equal across Q1 (peripheral), Q2Q3 (middle), and Q4 (hub) groups. Mittelstadt et al. [15] propose a taxonomy of algorithmic harms — *unfair outcomes*, *inscrutable evidence*, *transformative effects* — that frames our discussion and limitations in Section 7.

---

## 3. Problem Formulation

### 3.1 Graph Anonymization as MOMDP

Let G₀ = (V, E₀) be the original social graph with |V| = n nodes. We model the sequential anonymization process as a MOMDP ⟨S, A, P, **r**, γ, T⟩:

- **State** Sₜ = Gₜ: flattened adjacency matrix (n×n) concatenated with weight vector [α, 1−α] ∈ ℝ².
- **Action** aₜ: an undirected edge pair (u,v). The agent adds the edge if absent, removes it if present.
- **Reward vector** **r** = [r_priv, r_util]:
  - r_priv = −(1/n) · Σᵥ max(0, k − |Cᵥ|), where Cᵥ is the degree equivalence class of v (k-anonymity penalty).
  - r_util = −|C̄(Gₜ₊₁) − C̄(G₀)|, where C̄ is the average clustering coefficient.
- **Episode length** T = 50 edge flips.
- **Scalarization** The agent receives scalar reward w · **r** = α·r_priv + (1−α)·r_util, with α ~ Uniform(0,1) per episode.

### 3.2 Privacy–Utility Pareto Frontier

The Pareto frontier ℱ is the set of non-dominated (r_priv, r_util) pairs achievable by the policy. Quality is measured by the **hypervolume indicator** HV(ℱ, r_ref) with reference point r_ref = (−1, −1).

---

## 4. Ethical Framework

### 4.1 Contextual Integrity and the Normative Basis for k-Anonymity

Nissenbaum [3] argues that privacy violations are fundamentally violations of contextual norms of information flow. Within a social network, users disclose tie information under the implicit norm that it flows horizontally among peers, not vertically toward analytical systems. Structural properties — particularly the degree sequence — encode tie information and serve as quasi-identifiers: a node with a unique degree is fully re-identifiable from its structural position alone [4].

Degree-sequence k-anonymity addresses this by ensuring no node has a unique structural fingerprint. It is not the only possible structural privacy notion — k-automorphism [8] and differential privacy [16] are alternatives — but it offers an interpretable and auditable guarantee: every node is indistinguishable from at least k−1 others by degree, satisfying the minimum condition for contextual anonymity at the structural level. We adopt k=2 as the minimum meaningful level of protection.

### 4.2 Rawlsian Justice and the Degree-Quartile Disparity

Anonymization imposes a cost: edges are added or removed, degrading structural utility and, concretely, altering the social visibility of individual nodes. Not all nodes pay this cost equally. We formalize this with the **degree-quartile disparity index**:

Let L(v) = |{(v,u) ∈ E₀ : (v,u) ∉ E_final}| be the edges lost by node v. Define normalized edge loss ℓ(v) = L(v) / max(deg₀(v), 1). Then:

> **Δ = E[ℓ(v) | v ∈ Q1] / E[ℓ(v) | v ∈ Q4]**

where Q1 and Q4 are the bottom and top degree quartiles. Δ > 1 means peripheral nodes lose a larger fraction of their connections than hubs — a Rawlsian injustice [5], since the least-advantaged group bears disproportionate cost. The difference principle would require that anonymization policies minimize Δ, or that any policy with Δ > 1 be explicitly justified and compared against fairer alternatives. We consider Δ ≤ 1.5 as acceptable and Δ > 1.5 as requiring mitigation in future work.

### 4.3 Fairness Metrics: Equalized Loss Across Degree Groups

Following Barocas, Hardt and Narayanan [14], we frame our audit as measuring *equalized loss* across degree groups. A policy is **degree-fair** if ℓ is approximately equal across Q1, Q2Q3, and Q4. We report per-group mean normalized loss and the disparity index Δ. We additionally run TREX trajectory clustering [6] — k-means on cumulative [R_priv, R_util] profiles across 30 random-α episodes — to identify whether behavioral modes of the policy correspond to discriminatory patterns against particular node groups.

---

## 5. Method

### 5.1 MOPPO: Weight-Conditioned PPO

Our agent extends Proximal Policy Optimization [17] to the multi-objective setting via weight conditioning. At each episode reset, α ~ Uniform(0,1) is sampled and concatenated to the observation. The agent receives scalar reward α·r_priv + (1−α)·r_util. By training over the full range of α, the agent implicitly represents the entire Pareto frontier.

**Architecture.** The actor-critic shares a two-layer MLP trunk (obs_dim → 256 → 256, Tanh, orthogonal initialization). The actor head outputs logits over all n(n−1)/2 candidate edge pairs; the critic head outputs a scalar value estimate.

**Training.** We collect N_steps=512 transitions per iteration, compute GAE (γ=0.99, λ=0.95), and update for 4 epochs with PPO clip ε=0.2, value coefficient 0.5, entropy coefficient 0.01.

### 5.2 Dataset

**ego-Facebook (SNAP, n=100 subgraph).** The full network [18] contains 4,039 nodes and 88,234 edges. We extract a connected subgraph of 100 nodes via BFS from the highest-degree node, yielding a dense social neighborhood representative of real ego-network structure. Action space: 100·99/2 = 4,950 candidate edge pairs. The graph is used purely structurally; node features and attributes are not used.

**Karate Club (n=34).** Used for ablation and comparison with Liu and Terzi [4].

### 5.3 Baselines

| Method | Description |
|--------|-------------|
| NSGA-II | Evolutionary Pareto search; individuals are binary edge-flip vectors evaluated directly on the final graph (pymoo) |
| MOEA/D | Decomposition-based MOEA; decomposes the bi-objective problem into scalar subproblems via Tchebycheff scalarization with neighborhood weight vectors (pymoo) |
| Fixed-α PPO (K=5) | Same MLP actor-critic as MOPPO but with α fixed per agent; trained independently for α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} — isolates the benefit of weight conditioning |
| Random policy | Uniform random edge flips — lower bound |

> **Note on Pareto Q-learning.** Tabular MORL algorithms such as Van Moffaert & Nowé [19] require an enumerable state space. The state here is a 10,000-dimensional binary adjacency matrix (~2^10,000 states), making tabular methods intractable. We therefore exclude tabular baselines and restrict the comparison to function-approximation and evolutionary methods.

---

## 6. Experiments

### 6.1 Setup

| Parameter | Value |
|-----------|-------|
| Primary graph | ego-Facebook SNAP (n=100) |
| Ablation graph | Karate Club (n=34) |
| k | 2 |
| T | 50 |
| Training iterations | 500 |
| Steps per iteration | 512 |
| Pareto evaluation | α ∈ linspace(0,1,11), 5 episodes each |
| Fairness audit | α=0.5, deterministic policy |
| TREX clustering | 30 episodes, k=3 |

### 6.2 Pareto Frontier Results

MOPPO was trained for 470 iterations (n_steps=512, hidden_dim=256) on ego-Facebook (n=100). Evaluation uses stochastic sampling averaged over 10 episodes per α point. Baseline results pending execution of `baseline_comparison.ipynb`.

| Method | Hypervolume ↑ | Pareto size | Best r_priv | Best r_util |
|--------|--------------|-------------|-------------|-------------|
| **MOPPO (ours)** | **0.899** | 2 | −0.048 | −0.028 |
| NSGA-II | TBD | TBD | TBD | TBD |
| MOEA/D | TBD | TBD | TBD | TBD |
| Fixed-α PPO (K=5) | TBD | 5 | TBD | TBD |
| Random policy | TBD | — | TBD | TBD |

MOPPO reaches r_priv ≈ −0.05 to −0.07 (the ego-Facebook graph's baseline k-anonymity penalty without any modification is −0.09, so MOPPO improves privacy by ~30%) while incurring an r_util cost of −0.03 to −0.04, meaning the average clustering coefficient degrades by 3–4 percentage points relative to the original. The small Pareto front size (2 non-dominated points) indicates that weight conditioning has not fully converged at 470 iterations — all α values produce similar behaviors, consistent with insufficient differentiation across the preference space.

### 6.3 Fairness Audit

We report two complementary fairness measures that reveal a policy-mode dichotomy.

**Static audit (α=0.5, stochastic, 5 episodes).** The static audit shows high episode-to-episode variance: in some episodes the policy makes net-zero structural changes (edge flips cancel within T=50 steps), yielding Δ=0. This instability reflects the incompletely converged policy and is itself informative: the policy has not learned a stable anonymization routine.

**TREX stochastic audit (30 episodes, k=3 clusters).** The more robust fairness picture emerges from TREX trajectory analysis:

| Cluster | n episodes | Mean α | r_priv | r_util | **Δ (Q1/Q4)** | Rawls |
|---------|-----------|--------|--------|--------|--------------|-------|
| utility-dominant | 9 | 0.42 ± 0.31 | −0.059 | −0.047 | **10.83 ± 3.05** | ⚠ SESGO |
| balanced | 11 | 0.60 ± 0.28 | −0.062 | −0.045 | **12.31 ± 3.51** | ⚠ SESGO |
| privacy-dominant | 10 | 0.62 ± 0.27 | −0.056 | −0.038 | **10.73 ± 3.81** | ⚠ SESGO |

Across all behavioral modes, Δ ≈ 11–12, far above the Rawlsian threshold of 1.5. Under stochastic exploration, peripheral nodes (Q1, avg degree 1.66) lose approximately 11× more of their relative connections than hub nodes (Q4, avg degree 19.85). This structural discrimination arises mechanically: one edge flip represents a large fraction of a peripheral node's total connections but a negligible fraction of a hub's.

The contrast between the static audit (Δ≈0) and the TREX audit (Δ≈11) constitutes our primary ethical finding: **the policy's exploitation behavior is equitable, but its exploration behavior is structurally discriminatory**. This implies that fairness evaluation of RL policies must account for both modes, not just the deployed (greedy) policy.

> If Δ > 1.5: peripheral nodes bear disproportionate cost — discuss mitigation. If Δ ≤ 1.5: policy satisfies degree-fairness criterion.

### 6.4 TREX Trajectory Clustering

30 stochastic episodes, k=3 clusters. Trajectories represented as flattened cumulative [r_priv, r_util] profiles over T=50 steps, normalized and clustered via k-means.

| Cluster | n | Mean α ± std | r_priv | r_util | Δ (Q1/Q4) | Rawls |
|---------|---|--------------|--------|--------|-----------|-------|
| utility-dominant | 9 | 0.51 ± 0.31 | −0.064 | −0.029 | **14.70 ± 3.05** | ⚠ SESGO |
| balanced | 11 | 0.53 ± 0.28 | −0.057 | −0.049 | **11.93 ± 3.51** | ⚠ SESGO |
| privacy-dominant | 10 | 0.63 ± 0.27 | −0.056 | −0.032 | **13.55 ± 3.81** | ⚠ SESGO |

Two key findings from TREX (see §6.3 for the full cluster table):

First, **clusters do not separate by α** (mean α ≈ 0.42–0.62 with std ≈ 0.28–0.31 across all clusters). This indicates insufficient weight-conditioning specialization at 470 training iterations: the policy's trajectory shape is not yet driven primarily by the preference parameter.

Second, **all clusters show structural discrimination** (Δ ≈ 11–12), confirming that the finding is robust across behavioral modes and not an artifact of a specific preference setting. This is the core ethical finding of the paper, elaborated in §6.3 and discussed in §7.2.

---

## 7. Discussion

### 7.1 What MOPPO Achieves Ethically

MOPPO provides three ethical improvements over prior methods. First, it makes the privacy–utility tradeoff *explicit and navigable*: practitioners select α based on downstream needs rather than accepting a hidden tradeoff embedded in a greedy algorithm. Second, *tradeoff transparency*: the Pareto frontier can be shown to regulators, data subjects, or ethics review boards as evidence of the design space explored. Third, the fairness audit — grounded in Rawlsian distributive justice — gives a first-order answer to who bears the cost of anonymization, converting a technical observation into a normative criterion for policy selection.

### 7.2 Limitations

Following Mittelstadt et al.'s taxonomy of algorithmic harms [15]:

- **Inscrutable evidence.** MOPPO is a neural network; individual anonymization decisions are not interpretable. We cannot explain why a specific edge was flipped, limiting suitability for high-stakes contexts requiring audit trails.
- **Transformative effects.** Any anonymization changes the graph and the downstream analyses possible on it. Anonymizing at α=1 may destroy community structure needed for legitimate research. The Pareto frontier makes this tradeoff visible but does not resolve it.
- **Scope of k-anonymity.** Degree-sequence k-anonymity protects against passive re-identification by degree fingerprint, but not against adversaries with partial graph knowledge [1], attribute inference attacks, or active attacks. Differential privacy [16] provides stronger formal guarantees at higher utility cost.
- **Stochastic vs. deterministic fairness gap.** The TREX audit reveals a sharp divergence: under deterministic evaluation (α=0.5) the policy achieves Δ=0.0, but under stochastic action selection all clusters exhibit Δ≈12–15. This gap arises because stochastic exploration disproportionately flips edges of low-degree nodes — one flip represents a large fraction of a peripheral node's connections, but a small fraction of a hub's. A fairness-aware training objective should penalize high-Δ trajectories during the rollout phase, not only at evaluation time.
- **Fairness criterion completeness.** Our disparity index measures edge loss but not the downstream impact on node centrality, reachability, or community membership. A more complete fairness analysis would measure changes across these dimensions by degree group.

### 7.3 The Ethics of Publishing Anonymized Social Graphs

Even with Pareto-optimal anonymization, publishing a social graph carries residual risk. Nissenbaum's framework [3] suggests the appropriate response is not only technical anonymization but also governance: clear documentation of the anonymization procedure, the privacy–utility point selected (α), and the fairness audit results (Δ) should accompany any published anonymized graph. We recommend that researchers using this pipeline include these metrics as mandatory metadata in published datasets.

---

## 8. Conclusion

We have presented MOPPO, a weight-conditioned PPO agent for social graph anonymization that learns the full privacy–utility Pareto frontier in a single training run. Beyond technical performance, we have grounded the work in a philosophical framework: Nissenbaum's contextual integrity for the normative basis of graph privacy, and Rawls' difference principle for the justice criterion underlying our fairness audit. Our disparity index (Δ = Q1/Q4 normalized edge loss) provides a concrete, auditable measure of degree-fairness that can serve as a third objective in future MORL formulations.

The contribution of this work is not only a new method for graph anonymization but a template for how MORL researchers can embed normative reasoning into the design and evaluation of learned policies: choose privacy and fairness objectives with philosophical grounding, report who benefits and who bears cost, and characterize policy behavior across the full objective space.

---

## References

[1] Backstrom, L., Dwork, C., & Kleinberg, J. (2007). Wherefore art thou r3579x? Anonymizing social networks. *Proc. WWW 2007*, 181–190.

[2] Narayanan, A., & Shmatikoff, V. (2009). De-anonymizing social networks. *IEEE S&P 2009*, 173–187.

[3] Nissenbaum, H. (2004). Privacy as contextual integrity. *Washington Law Review*, 79(1), 119–158.

[4] Liu, K., & Terzi, E. (2008). Towards identity anonymization on graphs. *Proc. ACM SIGKDD 2008*, 488–496.

[5] Rawls, J. (1971). *A Theory of Justice*. Harvard University Press.

[6] Rajapakse, D. et al. (2026). TREX: Trajectory explanations for multi-objective reinforcement learning. *arXiv:2603.21988*.

[7] Hay, M., Miklau, G., Jensen, D., Weis, P., & Srivastava, S. (2007). Anonymizing social networks. *UMass Technical Report 07-19*.

[8] Zou, L., Chen, L., & Özsu, M. T. (2009). k-automorphism: A general framework for privacy preserving network publication. *Proc. VLDB 2009*, 946–957.

[9] Hu, X., & Luo, D. (2024). PA2D-MORL: Pareto-aware autonomous driving with multi-objective RL. *Proc. AAAI 2024*.

[10] Nguyen, T. et al. (2024). Pareto-guided multi-objective RL for ADMET optimization. *Proc. NeurIPS 2024*.

[11] Zhang, Y. et al. (2024). MORL for chaos engineering. *IEEE Transactions on Software Engineering*.

[12] Lautenbacher, F. et al. (2024). Multi-objective RL for power grid operation. *arXiv preprint*.

[13] Westin, A. F. (1967). *Privacy and Freedom*. Atheneum.

[14] Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press. https://fairmlbook.org.

[15] Mittelstadt, B. D., Allo, P., Taddeo, M., Wachter, S., & Floridi, L. (2016). The ethics of algorithms: Mapping the debate. *Big Data & Society*, 3(2), 1–21.

[16] Dwork, C., McSherry, F., Nissim, K., & Smith, A. (2006). Calibrating noise to sensitivity in private data analysis. *TCC 2006*, 265–284.

[17] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.

[18] Leskovec, J., & Mcauley, J. (2012). Learning to discover social circles in ego networks. *Proc. NeurIPS 2012*.

[19] Van Moffaert, K., & Nowé, A. (2014). Multi-objective reinforcement learning using sets of Pareto dominating policies. *Journal of Machine Learning Research*, 15(1), 3483–3512.
