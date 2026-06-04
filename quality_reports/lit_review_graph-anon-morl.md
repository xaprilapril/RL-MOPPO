# Literature Review: MORL for Graph Anonymization
**Date:** 2026-05-23
**Status:** Complete — librarian-critic approved (round 2)
**Total papers:** 34 | **Proximity 1:** 0 | **Proximity 2:** 10 | **Proximity 3:** 16 | **Proximity 4:** 7 | **Proximity 5:** 1

---

## Key Finding

**The combination of MORL + graph anonymization/privacy is an open research gap.**
No paper found applies multi-policy MORL (Pareto frontier) to graph privacy or anonymization. The project occupies uncontested territory at the intersection of three literatures.

---

## Gap Summary (from frontier_map.md)

**Dimension 1 — Methodological:** No paper combines MORL (multi-policy, multi-objective) with graph anonymization. Closest is Bernini et al. (KDD 2024): DRL + MDP for community hiding, single-objective, no Pareto front.

**Dimension 2 — Privacy-utility Pareto:** Classical methods optimize a single objective (minimum edges for fixed k). No prior work maps the complete Pareto frontier between privacy and structural utility.

**Dimension 3 — Ethical audit:** No graph anonymization paper (classical or ML-based) audits differential impact on low-degree peripheral nodes.

---

## Contribution Statement

> We propose the first multi-objective reinforcement learning framework for social graph anonymization, training a GNN-augmented MOPPO agent to simultaneously optimize k-degree privacy and clustering-coefficient utility by approximating the complete Pareto frontier via rotating weight vectors. Beyond the privacy-utility trade-off, we introduce a degree-quartile disconnection audit using TREX trajectory analysis to determine whether learned Pareto-optimal policies achieve privacy by disproportionately isolating structurally peripheral nodes, providing the first ethically audited graph anonymization framework.

---

## 3 Papers Reviewers Will Cite (with prepared responses)

| Reviewer cite | Objection | Our differentiator |
|---|---|---|
| **Bernini et al. (KDD 2024)** | "This is just Bernini extended with a second objective" | Different privacy objective (k-degree vs. community hiding); MOMDP vs. constrained single-objective MDP; ethical audit absent in Bernini |
| **Lautenbacher et al. (arXiv 2025)** | "You imported MOPPO from power grids — what is novel?" | Novel domain (social graph privacy), novel k-anonymity reward formulation (population-level constraint), ethical audit entirely absent from grid paper |
| **Liu & Terzi (SIGMOD 2008)** | "k-anonymity for graphs was solved in 2008" | Dynamic policy vs. static one-shot algorithm; multi-objective framing (utility tracked explicitly); structural learning via GNN |

---

## Target Venues

| Priority | Venue | Rationale |
|---|---|---|
| **Tier 1** | KDD, AAAI | Closest papers appeared here; covers graph mining + RL + ethics |
| **Tier 2** | WWW, IJCAI, WSDM | Strong fit for social graph privacy and MORL theory |
| **Tier 3** | NeurIPS | Viable if theoretical guarantees are strengthened |
| **Journal** | TKDE, TKDD, PETs | Extended version targets |

---

## Full Detail Files

| File | Content |
|---|---|
| `quality_reports/literature/graph-anon-morl/annotated_bibliography.md` | 34 papers, full annotations |
| `quality_reports/literature/graph-anon-morl/references.bib` | 34 BibTeX entries (4 marked UNVERIFIED) |
| `quality_reports/literature/graph-anon-morl/frontier_map.md` | Field map, gap analysis, reviewer table |
| `quality_reports/literature/graph-anon-morl/positioning.md` | Contribution statement, reviewer responses, venue rankings |
| `quality_reports/literature/graph-anon-morl/critic_review.md` | Librarian-critic score: 70 → ~85/100 after second pass |

---

## Papers to Verify Manually Before Submission

4 BibTeX entries marked `% UNVERIFIED` — confirm before submitting:
- `Zhang2024_morl_chaos_engineering` — IEEE venue/DOI not confirmed
- `Kasiviswanathan2013_node_dp_graphs` — DOI/page range not confirmed
- `CasasRoma2017_community_preserving` — page range not confirmed
- `Esmaeilpour2021_epsilon_k_anonymization_gnn` — article number/DOI inferred
