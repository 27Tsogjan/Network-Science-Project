"""
Community Detection & Clustering

Runs three community detection methods:
  1. Louvain (via greedy modularity — networkx built-in)
  2. Girvan-Newman (edge betweenness)
  3. Label Propagation

Tests statistical significance vs a null (randomized) model.
Prints community membership and saves plots.
"""

import csv, os, random, collections
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms import community as nx_comm

CSV_PATH = "linguistic_transfer.csv"
os.makedirs("results/figures", exist_ok=True)

# Load graph (undirected for community detection)
G_dir = nx.DiGraph()
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        G_dir.add_edge(row["source"], row["target"], edgetype=row["type"])

G = G_dir.to_undirected()
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Method 1: Greedy Modularity (approximation of Louvain) 
print("\n--- Method 1: Greedy Modularity (Louvain-style) ---")
communities_gm = list(nx_comm.greedy_modularity_communities(G))
communities_gm = sorted(communities_gm, key=len, reverse=True)
Q_gm = nx_comm.modularity(G, communities_gm)
print(f"Number of communities: {len(communities_gm)}")
print(f"Modularity (Q): {Q_gm:.4f}")
for i, comm in enumerate(communities_gm[:8]):
    print(f"  Community {i+1} ({len(comm)} nodes): {sorted(comm)[:8]}{'...' if len(comm)>8 else ''}")

# Method 2: Label Propagation
print("\n--- Method 2: Label Propagation ---")
communities_lp = list(nx_comm.label_propagation_communities(G))
communities_lp = sorted(communities_lp, key=len, reverse=True)
Q_lp = nx_comm.modularity(G, communities_lp)
print(f"Number of communities: {len(communities_lp)}")
print(f"Modularity (Q): {Q_lp:.4f}")
for i, comm in enumerate(communities_lp[:8]):
    print(f"  Community {i+1} ({len(comm)} nodes): {sorted(comm)[:8]}{'...' if len(comm)>8 else ''}")

#Method 3: Girvan-Newman 
print("\n--- Method 3: Girvan-Newman (top-level split) ---")
gn_gen = nx_comm.girvan_newman(G)
communities_gn = list(next(gn_gen))  # just top-level split into 2
Q_gn = nx_comm.modularity(G, communities_gn)
print(f"Number of communities (top split): {len(communities_gn)}")
print(f"Modularity (Q): {Q_gn:.4f}")
for i, comm in enumerate(communities_gn):
    print(f"  Community {i+1} ({len(comm)} nodes): {sorted(comm)[:8]}{'...' if len(comm)>8 else ''}")

#Statistical significance vs null model 
print("\n--- Statistical Significance (vs randomized null) ---")
print("Randomizing graph 100 times and computing modularity distribution...")

null_Q_values = []
for trial in range(100):
    G_rand = nx.configuration_model([d for _, d in G.degree()])
    G_rand = nx.Graph(G_rand)
    G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
    try:
        comm_rand = list(nx_comm.greedy_modularity_communities(G_rand))
        null_Q_values.append(nx_comm.modularity(G_rand, comm_rand))
    except:
        pass

null_mean = np.mean(null_Q_values)
null_std  = np.std(null_Q_values)
z_score   = (Q_gm - null_mean) / null_std if null_std > 0 else float("inf")

print(f"Null model Q: mean={null_mean:.4f}, std={null_std:.4f}")
print(f"Your network Q (Greedy Modularity): {Q_gm:.4f}")
print(f"Z-score: {z_score:.2f}  → {'SIGNIFICANT' if z_score > 2 else 'not significant'}")

#Figure: Community size distributions 
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Community Detection Results", fontsize=14, fontweight="bold")

for ax, comms, label, color, Q in [
    (axes[0], communities_gm, f"Greedy Modularity\nQ={Q_gm:.3f}", "#1C7293", Q_gm),
    (axes[1], communities_lp, f"Label Propagation\nQ={Q_lp:.3f}", "#B85042", Q_lp),
    (axes[2], communities_gn, f"Girvan-Newman\nQ={Q_gn:.3f}", "#2C5F2D", Q_gn),
]:
    sizes = sorted([len(c) for c in comms], reverse=True)
    ax.bar(range(1, len(sizes)+1), sizes, color=color, alpha=0.85, edgecolor="white")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Community rank")
    ax.set_ylabel("Size (nodes)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/04_communities.png", dpi=150, bbox_inches="tight")
plt.close()

#Figure: Null model comparison
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(null_Q_values, bins=20, color="#64748B", alpha=0.7, edgecolor="white", label="Null model Q")
ax.axvline(Q_gm, color="#B85042", linewidth=2.5, linestyle="--",
           label=f"Your network Q={Q_gm:.3f}\n(z={z_score:.1f})")
ax.set_xlabel("Modularity Q", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
ax.set_title("Modularity vs Null Model (100 randomizations)", fontsize=13, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/04b_null_model.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved: 04_communities.png, 04b_null_model.png")
print("\nNote: For Gephi visualization, use Modularity plugin → run community detection → color nodes by 'Modularity Class'")
