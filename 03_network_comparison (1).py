"""
Network Comparison
- Erdos-Renyi, Barabasi-Albert, Shared-Language vs my network
"""

import csv, os, collections
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CSV_PATH = "linguistic_transfer.csv"
os.makedirs("results/figures", exist_ok=True)

# Load colonial network
G_col = nx.DiGraph()
edges_by_source = collections.defaultdict(list)

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        G_col.add_edge(row["source"], row["target"], edgetype=row["type"])
        edges_by_source[row["source"]].append(row["target"])

N = G_col.number_of_nodes()
E = G_col.number_of_edges()
p = E / (N * (N - 1))
m = max(1, round(E / N))

# Shared language network
G_lang = nx.Graph()
for src, targets in edges_by_source.items():
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            G_lang.add_edge(targets[i], targets[j], shared_colonizer=src)

# Comparison networks
np.random.seed(42)
G_er = nx.erdos_renyi_graph(N, p, directed=True, seed=42)
G_ba = nx.barabasi_albert_graph(N, m, seed=42)

def network_metrics(G, name, directed=True):
    G_und = G.to_undirected() if directed else G
    wcc   = max(nx.connected_components(G_und), key=len)
    G_sub = G_und.subgraph(wcc)
    avg_clust = nx.average_clustering(G_und)
    try:
        avg_path  = nx.average_shortest_path_length(G_sub)
        diameter  = nx.diameter(G_sub)
    except:
        avg_path, diameter = float("nan"), float("nan")
    degs   = [d for _, d in G_und.degree()]
    avg_deg = np.mean(degs)
    max_deg = max(degs)
    density = nx.density(G)
    assort  = nx.degree_assortativity_coefficient(G)

    print(f"\n{'='*45}\n{name}")
    print(f"  Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print(f"  Density: {density:.5f}  Avg degree: {avg_deg:.3f}  Max degree: {max_deg}")
    print(f"  Avg clustering: {avg_clust:.5f}")
    print(f"  Avg path length: {avg_path:.4f}" if not np.isnan(avg_path) else "  Avg path length: N/A")
    print(f"  Diameter: {diameter}" if not np.isnan(diameter) else "  Diameter: N/A")
    print(f"  Assortativity: {assort:.4f}")

    return dict(name=name, nodes=G.number_of_nodes(), edges=G.number_of_edges(),
                density=density, avg_degree=avg_deg, max_degree=max_deg,
                avg_clustering=avg_clust, avg_path_length=avg_path,
                diameter=diameter, assortativity=assort)

metrics = [
    network_metrics(G_col,  "Colonial Network (Yours)", directed=True),
    network_metrics(G_er,   "Erdos-Renyi Random",       directed=True),
    network_metrics(G_ba,   "Barabasi-Albert Scale-Free",directed=False),
    network_metrics(G_lang, "Shared-Language Network",  directed=False),
]

# Fit power law for trendline
def fit_pl(deg_seq):
    counts = collections.Counter(deg_seq)
    ks = np.array(sorted(counts.keys()))
    pk = np.array([counts[k]/len(deg_seq) for k in ks])
    mask = (ks >= 1) & (pk > 0)
    ks_f, pk_f = ks[mask], pk[mask]
    if len(ks_f) < 2:
        return None, None, None, ks_f, pk_f
    slope, intercept, r, _, _ = stats.linregress(np.log(ks_f), np.log(pk_f))
    return -slope, np.exp(intercept), r**2, ks_f, pk_f

# Figure: 2x2 log log degree distributions with trendlines
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Degree Distribution Comparison (Log-Log + Power Law Fit)",
             fontsize=14, fontweight="bold")

nets = [
    (G_col,  "Colonial Network",       "#1C7293"),
    (G_er,   "Erdos-Renyi Random",     "#64748B"),
    (G_ba,   "Barabasi-Albert",        "#B85042"),
    (G_lang, "Shared-Language",        "#2C5F2D"),
]

for ax, (G, label, color) in zip(axes.flat, nets):
    G_und  = G.to_undirected()
    degs   = [d for _, d in G_und.degree()]
    alpha, C, r2, ks_f, pk_f = fit_pl(degs)

    # Observed dots
    ax.scatter(ks_f, pk_f, color=color, s=40, alpha=0.8,
               edgecolors="white", linewidths=0.4, zorder=3, label="Observed")

    # Trendline
    if alpha is not None:
        k_range = np.linspace(ks_f.min(), ks_f.max(), 300)
        ax.plot(k_range, C * k_range**(-alpha), color="black", lw=1.8,
                linestyle="--", label=f"α={alpha:.2f},  R²={r2:.3f}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(label, fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel("Degree k (log)"); ax.set_ylabel("P(k) (log)")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/03_network_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 03_network_comparison.png")

# Figure: Clustering + Path Length bars
labels  = [m["name"] for m in metrics]
clusts  = [m["avg_clustering"] for m in metrics]
paths   = [m["avg_path_length"] for m in metrics]
colors  = ["#1C7293","#64748B","#B85042","#2C5F2D"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Clustering
bars = axes[0].bar(labels, clusts, color=colors, edgecolor="white")
axes[0].set_title("Avg Clustering Coefficient\n(high in Shared-Language = territories share colonizer → triangles form)",
                  fontweight="bold", fontsize=10)
axes[0].set_ylabel("Clustering Coefficient")
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, clusts):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

valid = [(l, p, c) for l, p, c in zip(labels, paths, colors) if not np.isnan(p)]
bars2 = axes[1].bar([v[0] for v in valid], [v[1] for v in valid],
                    color=[v[2] for v in valid], edgecolor="white")
axes[1].set_title("Average Shortest Path Length", fontweight="bold")
axes[1].set_ylabel("Avg Path Length")
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars2, [v[1] for v in valid]):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+0.03, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=9)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/03b_small_world_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 03b_small_world_comparison.png")
