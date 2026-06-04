"""Run social network analysis and generate all figures for the presentation."""
import sys, os, warnings, json
from pathlib import Path
from itertools import combinations
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from scipy import stats

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR  = str(PROJECT_ROOT / 'scripts' / 'python')
DATA_DIR     = str(PROJECT_ROOT / 'data')
OUT_DIR      = str(PROJECT_ROOT / 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, SCRIPTS_DIR)

from graph_anon_morl.datasets import load_facebook_ego
from graph_anon_morl.utils    import compute_k_anonymity_reward, compute_utility_reward
from graph_anon_morl.audit    import degree_quartile_audit, fairness_disparity_index

# ── 1. LOAD GRAPH ────────────────────────────────────────────────────────────
print("Loading graph...")
G = load_facebook_ego(n_nodes=100, seed=42, data_dir=DATA_DIR)
n, m = G.number_of_nodes(), G.number_of_edges()

density        = nx.density(G)
avg_clustering = nx.average_clustering(G)
transitivity   = nx.transitivity(G)
avg_degree     = 2 * m / n
degree_seq     = sorted([d for _, d in G.degree()], reverse=True)
max_degree, min_degree = degree_seq[0], degree_seq[-1]

giant        = max(nx.connected_components(G), key=len)
G_giant      = G.subgraph(giant).copy()
diameter     = nx.diameter(G_giant)
avg_path_len = nx.average_shortest_path_length(G_giant)
n_components = nx.number_connected_components(G)

print(f"  n={n}, m={m}, density={density:.4f}, avg_deg={avg_degree:.2f}")
print(f"  diameter={diameter}, avg_path={avg_path_len:.3f}, clustering={avg_clustering:.4f}")

# ── 2. DEGREE DISTRIBUTION ───────────────────────────────────────────────────
print("Degree distribution...")
degrees     = [d for _, d in G.degree()]
deg_counter = Counter(degrees)
unique_degs = sorted(deg_counter.keys())
freqs       = [deg_counter[d] for d in unique_degs]
deg_array   = np.array(degrees)

log_k = np.log(unique_degs)
log_p = np.log([f / n for f in freqs])
slope, intercept, r_val, *_ = stats.linregress(log_k, log_p)
gamma = -slope

singleton_degrees = [d for d, cnt in deg_counter.items() if cnt == 1]
singleton_nodes   = [nd for nd, d in G.degree() if d in singleton_degrees]
r_priv_initial    = compute_k_anonymity_reward(G, k=2)

print(f"  gamma={gamma:.3f}, R2={r_val**2:.3f}, singletons={len(singleton_nodes)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(unique_degs, freqs, color='#1E88E5', alpha=0.85, edgecolor='white')
axes[0].set_xlabel('Grado k'); axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribucion de grados')

k_fit = np.linspace(min(unique_degs), max(unique_degs), 100)
axes[1].scatter(unique_degs, [f/n for f in freqs], color='#1E88E5', s=50, zorder=3, label='Datos')
axes[1].plot(k_fit, np.exp(intercept)*k_fit**slope, 'r--', lw=1.5, label=f'Ajuste gamma={gamma:.2f}')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel('log k'); axes[1].set_ylabel('log P(k)')
axes[1].set_title(f'Log-log (R2={r_val**2:.3f})'); axes[1].legend(fontsize=8)

sorted_deg = np.sort(deg_array)
ccdf = 1 - np.arange(1, n+1) / n
axes[2].step(sorted_deg, ccdf, color='#43A047', lw=2)
axes[2].set_xscale('log'); axes[2].set_yscale('log')
axes[2].set_xlabel('Grado k'); axes[2].set_ylabel('P(K > k)')
axes[2].set_title('CCDF (cola de distribucion)')

plt.suptitle('Analisis de distribucion de grados - ego-Facebook (n=100)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_degree_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_degree_distribution.png")

# ── 3. CENTRALITY ────────────────────────────────────────────────────────────
print("Computing centralities...")
cent_degree    = nx.degree_centrality(G)
cent_between   = nx.betweenness_centrality(G, normalized=True)
cent_closeness = nx.closeness_centrality(G)
cent_eigenvec  = nx.eigenvector_centrality(G, max_iter=1000)
cent_pagerank  = nx.pagerank(G, alpha=0.85)

df_cent = pd.DataFrame({
    'Degree': cent_degree, 'Betweenness': cent_between,
    'Closeness': cent_closeness, 'Eigenvector': cent_eigenvec, 'PageRank': cent_pagerank
})
df_cent['Degree_raw'] = [G.degree(nd) for nd in df_cent.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc = axes[0].scatter(df_cent['Degree'], df_cent['Betweenness'],
                     c=df_cent['PageRank'], cmap='plasma', s=60, alpha=0.8)
plt.colorbar(sc, ax=axes[0], label='PageRank')
axes[0].set_xlabel('Degree centrality'); axes[0].set_ylabel('Betweenness centrality')
axes[0].set_title('Degree vs. Betweenness\n(color = PageRank)')

metrics_c = ['Degree','Betweenness','Closeness','Eigenvector']
colors_c  = ['#1E88E5','#E53935','#43A047','#FB8C00']
x = np.arange(5); w = 0.2
for i, (met, col) in enumerate(zip(metrics_c, colors_c)):
    top5 = df_cent.nlargest(5, met)
    vals = top5[met].values
    axes[1].bar(x + i*w, vals/vals.max(), w, label=met, color=col, alpha=0.85)
axes[1].set_xticks(x + 1.5*w)
axes[1].set_xticklabels([f'#{i+1}' for i in range(5)])
axes[1].set_ylabel('Centralidad normalizada'); axes[1].set_title('Top-5 por cada centralidad')
axes[1].legend(fontsize=8)

plt.suptitle('Centralidades - ego-Facebook', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_centrality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_centrality.png")

# ── 4. COMMUNITIES ───────────────────────────────────────────────────────────
print("Detecting communities...")
communities_gen = nx.community.greedy_modularity_communities(G)
partition = {}
for cid, comm in enumerate(communities_gen):
    for node in comm:
        partition[node] = cid

n_communities = len(set(partition.values()))
modularity    = nx.community.modularity(
    G, [{nd for nd, c in partition.items() if c == cid} for cid in set(partition.values())]
)
comm_sizes = Counter(partition.values())
print(f"  Communities={n_communities}, Q={modularity:.4f}, sizes={sorted(comm_sizes.values(),reverse=True)}")

palette     = list(mcolors.TABLEAU_COLORS.values())
node_colors = [palette[partition[nd] % len(palette)] for nd in G.nodes()]
node_sizes  = [30 + 5*G.degree(nd) for nd in G.nodes()]
pos         = nx.spring_layout(G, seed=42, k=0.4)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.2, edge_color='gray', width=0.5)
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color=node_colors, node_size=node_sizes, alpha=0.9)
top5 = df_cent.nlargest(5, 'Degree').index.tolist()
nx.draw_networkx_labels(G, pos, {nd: str(nd) for nd in top5}, ax=axes[0], font_size=7, font_weight='bold')
axes[0].set_title(f'Red ego-Facebook - {n_communities} comunidades\nn={n}, m={m}, Q={modularity:.3f}')
axes[0].axis('off')

comm_ids  = sorted(set(partition.values()))
inter_mat = np.zeros((len(comm_ids), len(comm_ids)))
for u, v in G.edges():
    cu, cv = partition[u], partition[v]
    inter_mat[cu][cv] += 1
    if cu != cv: inter_mat[cv][cu] += 1

im = axes[1].imshow(inter_mat, cmap='Blues', aspect='auto')
axes[1].set_xticks(range(len(comm_ids))); axes[1].set_yticks(range(len(comm_ids)))
axes[1].set_xticklabels([f'C{c}' for c in comm_ids])
axes[1].set_yticklabels([f'C{c}' for c in comm_ids])
axes[1].set_title('Matriz de conexiones inter-comunidades')
plt.colorbar(im, ax=axes[1], label='N aristas')
for i in range(len(comm_ids)):
    for j in range(len(comm_ids)):
        axes[1].text(j, i, int(inter_mat[i,j]), ha='center', va='center', fontsize=8,
                     color='white' if inter_mat[i,j] > inter_mat.max()*0.6 else 'black')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_communities.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_communities.png")

# ── 5. CLUSTERING ────────────────────────────────────────────────────────────
print("Clustering analysis...")
clust_local  = nx.clustering(G)
deg_to_clust = {}
for node in G.nodes():
    d = G.degree(node)
    deg_to_clust.setdefault(d, []).append(clust_local[node])
ck_x = sorted(deg_to_clust.keys())
ck_y = [np.mean(deg_to_clust[d]) for d in ck_x]

valid = [(x, y) for x, y in zip(ck_x, ck_y) if x > 0 and y > 0]
if len(valid) > 1:
    lx, ly = zip(*valid)
    b_slope, b_int, *_ = stats.linregress(np.log(lx), np.log(ly))
    beta = -b_slope
else:
    beta, b_slope, b_int = float('nan'), 0, 0

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(list(clust_local.keys()), list(clust_local.values()),
                c=[G.degree(nd) for nd in clust_local], cmap='viridis', s=50, alpha=0.7)
axes[0].set_xlabel('Nodo'); axes[0].set_ylabel('Clustering local C_v')
axes[0].set_title('Clustering local (color=grado)')

axes[1].scatter(ck_x, ck_y, color='#1E88E5', s=60, zorder=3, label='C(k) observado')
if not np.isnan(beta) and len(valid) > 1:
    k_fit = np.linspace(min(ck_x)+0.1, max(ck_x), 100)
    axes[1].plot(k_fit, np.exp(b_int)*k_fit**b_slope, 'r--', label=f'C(k)~k^(-{beta:.2f})')
axes[1].set_xscale('log'); axes[1].set_yscale('log')
axes[1].set_xlabel('Grado k (log)'); axes[1].set_ylabel('C(k) promedio (log)')
axes[1].set_title('Clustering vs. grado C(k)'); axes[1].legend(fontsize=8)

plt.suptitle('Analisis de clustering - ego-Facebook', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_clustering.png")

# ── 6. SMALL WORLD ───────────────────────────────────────────────────────────
print("Small-world analysis...")
p_rand = 2*m / (n*(n-1))
G_rand = nx.gnp_random_graph(n, p_rand, seed=42)
gc_r   = max(nx.connected_components(G_rand), key=len)
G_rand = nx.relabel_nodes(G_rand.subgraph(gc_r).copy(),
                           {old: i for i, old in enumerate(G_rand.subgraph(gc_r).nodes())})

L_real = avg_path_len
C_real = avg_clustering
L_rand = nx.average_shortest_path_length(G_rand)
C_rand = nx.average_clustering(G_rand)
sigma  = (C_real/C_rand) / (L_real/L_rand)
print(f"  sigma={sigma:.3f}")

all_dists = []
for source in G_giant.nodes():
    sp = nx.single_source_shortest_path_length(G_giant, source)
    all_dists.extend(v for v in sp.values() if v > 0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(all_dists, bins=range(1, diameter+2), color='#1E88E5', alpha=0.8, edgecolor='white', rwidth=0.85)
axes[0].axvline(L_real, color='red', linestyle='--', label=f'L={L_real:.2f}')
axes[0].set_xlabel('Distancia geodesica'); axes[0].set_ylabel('Frecuencia')
axes[0].set_title('Distribucion de distancias\nentre todos los pares'); axes[0].legend()

cats = ['Clustering C', 'Camino L (norm.)']
vr   = [C_real, L_real/max(L_real, L_rand)]
vrnd = [C_rand, L_rand/max(L_real, L_rand)]
xb = np.arange(len(cats)); wb = 0.35
axes[1].bar(xb-wb/2, vr,  wb, label='ego-Facebook', color='#1E88E5')
axes[1].bar(xb+wb/2, vrnd, wb, label='Erdos-Renyi',  color='#90A4AE')
axes[1].set_xticks(xb); axes[1].set_xticklabels(cats)
axes[1].set_title(f'Small-world: sigma={sigma:.2f}\nAlto C + bajo L vs. aleatorio'); axes[1].legend()

plt.suptitle('Analisis de caminos - ego-Facebook', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_paths.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_paths.png")

# ── 7. NETWORK VIZ ───────────────────────────────────────────────────────────
print("Network visualization...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
deg_vals    = np.array([G.degree(nd) for nd in G.nodes()])
node_sz2    = 30 + 8*deg_vals
node_col2   = [palette[partition[nd] % len(palette)] for nd in G.nodes()]
edge_bet    = nx.edge_betweenness_centrality(G, normalized=True)
edge_widths = [0.3 + 6*edge_bet.get(e, edge_bet.get((e[1],e[0]),0)) for e in G.edges()]

pos_spring = nx.spring_layout(G, seed=42, k=0.5, iterations=100)
nx.draw_networkx_edges(G, pos_spring, ax=axes[0], alpha=0.25, edge_color='gray', width=edge_widths)
nx.draw_networkx_nodes(G, pos_spring, ax=axes[0], node_color=node_col2, node_size=node_sz2, alpha=0.92)
top8 = df_cent.nlargest(8, 'Degree').index.tolist()
nx.draw_networkx_labels(G, pos_spring, {nd: str(nd) for nd in top8}, ax=axes[0], font_size=7, font_weight='bold')
axes[0].set_title(f'Spring layout - tamano=grado, color=comunidad\nn={n}, m={m}, Q={modularity:.3f}, sigma={sigma:.2f}')
axes[0].axis('off')

pos_kk   = nx.kamada_kawai_layout(G)
bet_vals = np.array([cent_between[nd] for nd in G.nodes()])
nx.draw_networkx_edges(G, pos_kk, ax=axes[1], alpha=0.2, edge_color='gray', width=0.6)
nx.draw_networkx_nodes(G, pos_kk, ax=axes[1], node_color=bet_vals, cmap='hot_r', node_size=node_sz2, alpha=0.9)
nx.draw_networkx_labels(G, pos_kk, {nd: str(nd) for nd in top8}, ax=axes[1], font_size=7)
axes[1].set_title('Kamada-Kawai - color=betweenness'); axes[1].axis('off')
sm = plt.cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(bet_vals.min(), bet_vals.max()))
plt.colorbar(sm, ax=axes[1], label='Betweenness', shrink=0.7)

plt.suptitle('Red ego-Facebook - propiedades estructurales', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_network_viz.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_network_viz.png")

# ── 8. ANONYMIZATION IMPACT ──────────────────────────────────────────────────
print("Anonymization impact analysis...")

def greedy_k_anonymize(G0, k=2, max_ops=3000):
    Ga = G0.copy()
    for _ in range(max_ops):
        dc = {}
        for _, d in Ga.degree(): dc[d] = dc.get(d,0) + 1
        at_risk = sorted([nd for nd, d in Ga.degree() if dc[d] < k], key=lambda nd: Ga.degree(nd))
        if not at_risk: break
        made = False
        for u in at_risk:
            ud = Ga.degree(u)
            same = [v for v, d in Ga.degree() if d == ud and v != u and not Ga.has_edge(u, v)]
            if same: Ga.add_edge(u, same[0]); made = True; break
            other = sorted([v for v in Ga.nodes() if v != u and not Ga.has_edge(u, v)],
                           key=lambda v: -dc.get(Ga.degree(v), 0))
            if other: Ga.add_edge(u, other[0]); made = True; break
        if not made: break
    return Ga

def random_modify(G0, T=50, seed=0):
    rng = np.random.default_rng(seed)
    Gr  = G0.copy()
    all_e = list(combinations(list(G0.nodes()), 2))
    for _ in range(T):
        u, v = all_e[rng.integers(len(all_e))]
        Gr.remove_edge(u,v) if Gr.has_edge(u,v) else Gr.add_edge(u,v)
    return Gr

G_greedy = greedy_k_anonymize(G, k=2)
G_rnd    = random_modify(G, T=50)

dc_g    = Counter(d for _, d in G_greedy.degree())
min_cls = min(dc_g.values())
added   = G_greedy.number_of_edges() - m
print(f"  Greedy: +{added} edges, min_class={min_cls}, k-anon={min_cls>=2}")

def net_metrics(Gx, label):
    gc = Gx.subgraph(max(nx.connected_components(Gx), key=len)).copy()
    return {
        'label': label,
        'n_edges': Gx.number_of_edges(),
        'avg_clustering': round(nx.average_clustering(Gx), 5),
        'avg_path': round(nx.average_shortest_path_length(gc), 4),
        'modularity': round(nx.community.modularity(
            Gx, [{nd for nd, c in partition.items() if c==cid}
                 for cid in set(partition.values())]), 4),
        'r_priv': round(compute_k_anonymity_reward(Gx, k=2), 5),
        'r_util': round(compute_utility_reward(Gx, G), 5),
    }

ri = [net_metrics(G,'Original'), net_metrics(G_greedy,'Greedy (k=2)'), net_metrics(G_rnd,'Aleatorio')]
df_impact = pd.DataFrame(ri).set_index('label')

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes_flat = axes.flatten()
mtp = [('avg_clustering','Clustering C (mayor mejor)',True),
       ('avg_path','Camino promedio L (menor mejor)',False),
       ('modularity','Modularidad Q (mayor mejor)',True),
       ('r_priv','r_priv - k-anonimato (mayor mejor)',True),
       ('r_util','r_util - utility loss (mayor mejor)',True),
       ('n_edges','Numero de aristas',False)]
barcols = ['#1E88E5','#43A047','#E53935']
labels_b = df_impact.index.tolist()
for ax, (metric, title, _) in zip(axes_flat, mtp):
    vals = df_impact[metric].values.astype(float)
    bars = ax.bar(labels_b, vals, color=barcols, alpha=0.85, edgecolor='white')
    ax.set_title(title); ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+abs(vals).max()*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    ax.axhline(vals[0], color='navy', linestyle=':', lw=1, alpha=0.5)
plt.suptitle('Impacto de la anonimizacion sobre metricas de red\nego-Facebook (n=100)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_anon_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_anon_impact.png")

# Degree dist comparison + radar
fig = plt.figure(figsize=(14, 5))
ax_l = fig.add_subplot(121)
for Gx, label, color in [(G,'Original','#1E88E5'),(G_greedy,'Greedy (k=2)','#43A047'),(G_rnd,'Aleatorio','#E53935')]:
    degs = [d for _, d in Gx.degree()]
    cnt  = Counter(degs)
    xs, ys = zip(*sorted(cnt.items()))
    ax_l.step(xs, ys, label=label, color=color, lw=2, alpha=0.85)
ax_l.set_xlabel('Grado k'); ax_l.set_ylabel('Frecuencia')
ax_l.set_title('Distribucion de grados antes y despues de anonimizar'); ax_l.legend()

radar_m = ['avg_clustering','modularity','r_priv','r_util']
radar_l = ['Clustering','Modularidad','Privacidad','Utilidad']
angles  = np.linspace(0, 2*np.pi, len(radar_m), endpoint=False).tolist()
angles += angles[:1]
ax_r = fig.add_subplot(122, polar=True)
for Gx, label, color in [(G,'Original','#1E88E5'),(G_greedy,'Greedy (k=2)','#43A047'),(G_rnd,'Aleatorio','#E53935')]:
    row    = df_impact.loc[label, radar_m].values.astype(float)
    colmax = df_impact[radar_m].max().values
    colmin = df_impact[radar_m].min().values
    denom  = np.where(colmax-colmin==0, 1, colmax-colmin)
    norm   = (row-colmin)/denom
    vals   = norm.tolist() + norm[:1].tolist()
    ax_r.plot(angles, vals, 'o-', lw=2, color=color, label=label)
    ax_r.fill(angles, vals, alpha=0.08, color=color)
ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(radar_l)
ax_r.set_title('Radar: metricas normalizadas', pad=15)
ax_r.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=8)
plt.suptitle('Comparacion estructural: original vs. anonimizado', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_degree_compare.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_degree_compare.png")

# ── 9. FAIRNESS AUDIT ────────────────────────────────────────────────────────
print("Fairness audit...")
audit_greedy = degree_quartile_audit(G, G_greedy)
audit_rnd    = degree_quartile_audit(G, G_rnd)
disp_greedy  = fairness_disparity_index(audit_greedy)
disp_rnd     = fairness_disparity_index(audit_rnd)
print(f"  Greedy: Delta={disp_greedy:.4f}  Aleatorio: Delta={disp_rnd:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
methods_f   = ['Greedy (k=2)', 'Aleatorio']
disparities = [disp_greedy, disp_rnd]
bar_colors2 = ['#43A047' if d <= 1.5 else '#E53935' for d in disparities]
bars = axes[0].bar(methods_f, disparities, color=bar_colors2, edgecolor='white')
axes[0].axhline(1.0, color='green', linestyle='--', lw=1.5, label='Delta=1 (ideal)')
axes[0].axhline(1.5, color='red',   linestyle='--', lw=1.5, label='Delta=1.5 (umbral Rawls)')
for bar, val in zip(bars, disparities):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)
axes[0].set_ylabel('Disparity Index Delta = Q1/Q4')
axes[0].set_title('Equidad - quien paga el costo\nde la anonimizacion?')
axes[0].legend(fontsize=8)

x2 = np.arange(2); w2 = 0.25
q1v  = [audit_greedy.get('Q1_peripheral',{}).get('normalized_loss',0),
        audit_rnd.get('Q1_peripheral',{}).get('normalized_loss',0)]
midv = [audit_greedy.get('Q2Q3_middle',{}).get('normalized_loss',0),
        audit_rnd.get('Q2Q3_middle',{}).get('normalized_loss',0)]
q4v  = [audit_greedy.get('Q4_hub',{}).get('normalized_loss',0),
        audit_rnd.get('Q4_hub',{}).get('normalized_loss',0)]
axes[1].bar(x2-w2, q1v,  w2, label='Q1 perifericos', color='#2196F3')
axes[1].bar(x2,    midv, w2, label='Q2Q3 medios',    color='#4CAF50')
axes[1].bar(x2+w2, q4v,  w2, label='Q4 hubs',        color='#F44336')
axes[1].set_xticks(x2); axes[1].set_xticklabels(methods_f)
axes[1].set_ylabel('Perdida de aristas normalizada')
axes[1].set_title('Costo por cuartil de grado\n(ideal: barras iguales)')
axes[1].legend(fontsize=8)

plt.suptitle('Auditoria de equidad - Criterio de Rawls', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/sna_fairness.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: sna_fairness.png")

# ── SAVE STATS JSON ──────────────────────────────────────────────────────────
stats_out = {
    'n': n, 'm': m, 'density': density, 'avg_degree': avg_degree,
    'max_degree': max_degree, 'min_degree': min_degree,
    'avg_clustering': avg_clustering, 'transitivity': transitivity,
    'diameter': diameter, 'avg_path_len': avg_path_len,
    'sigma': sigma, 'gamma': gamma, 'r2_powerlaw': r_val**2,
    'n_communities': n_communities, 'modularity': modularity,
    'singleton_nodes': len(singleton_nodes),
    'r_priv_initial': r_priv_initial,
    'greedy_edges_added': added,
    'greedy_k_anon': bool(min_cls >= 2),
    'disparity_greedy': disp_greedy,
    'disparity_random': disp_rnd,
    'C_rand': C_rand, 'L_rand': L_rand,
}
with open(f'{OUT_DIR}/sna_stats.json', 'w') as f:
    json.dump(stats_out, f, indent=2)

print("\n" + "="*60)
print("ALL FIGURES GENERATED")
print("="*60)
for fname in ['sna_degree_distribution','sna_centrality','sna_communities',
              'sna_clustering','sna_paths','sna_network_viz',
              'sna_anon_impact','sna_degree_compare','sna_fairness']:
    ok = os.path.exists(f'{OUT_DIR}/{fname}.png')
    print(f"  {'OK' if ok else 'MISSING'} {fname}.png")
