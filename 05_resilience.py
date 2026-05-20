"""
Network Resilience

Tests network resilience under:
  1. Random failure (nodes removed randomly)
  2. Targeted degree attack (highest degree first)
  3. Targeted betweenness attack (highest betweenness first)

Tracks largest connected component size as nodes are removed.
Also estimates the percolation threshold.

Output: saves resilience plots to results/figures/
"""

import csv, os, random, copy
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "linguistic_transfer.csv"
os.makedirs("results/figures", exist_ok=True)

# Load graph
G_dir = nx.DiGraph()
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        G_dir.add_edge(row["source"], row["target"])

G = G_dir.to_undirected()
N = G.number_of_nodes()
print(f"Graph: {N} nodes, {G.number_of_edges()} edges")

def largest_cc_fraction(G_rem):
    """Return size of largest CC as fraction of original N."""
    if G_rem.number_of_nodes() == 0:
        return 0.0
    comps = nx.connected_components(G_rem)
    return max(len(c) for c in comps) / N

def simulate_removal(G, order):
    """Remove nodes in given order, track LCC fraction at each step."""
    G_copy = G.copy()
    fractions = [largest_cc_fraction(G_copy)]
    for node in order:
        if node in G_copy:
            G_copy.remove_node(node)
        fractions.append(largest_cc_fraction(G_copy))
    return fractions

def find_threshold(fracs, steps):
    """Find fraction of removed nodes where LCC drops below 0.5."""
    for i, f in enumerate(fracs):
        if f < 0.5:
            return steps[i]
    return 1.0

nodes = list(G.nodes())
steps = np.linspace(0, 1, N + 1)

# Random failure (average of 10 trials)
print("Running random failure (10 trials)...")
random_fracs_all = []
for trial in range(10):
    random.seed(trial)
    order_rand = random.sample(nodes, len(nodes))
    random_fracs_all.append(simulate_removal(G, order_rand))
random_fracs = np.mean(random_fracs_all, axis=0)
random_threshold = find_threshold(random_fracs, steps)
print(f"  Random failure threshold (LCC < 0.5): {random_threshold:.3f} of nodes removed")

# Targeted degree attack
print("Running targeted degree attack...")
G_temp = G.copy()
order_degree = []
while G_temp.number_of_nodes() > 0:
    highest = max(G_temp.degree(), key=lambda x: x[1])[0]
    order_degree.append(highest)
    G_temp.remove_node(highest)
degree_fracs = simulate_removal(G, order_degree)
degree_threshold = find_threshold(degree_fracs, steps)
print(f"  Degree attack threshold (LCC < 0.5): {degree_threshold:.3f} of nodes removed")

# Targeted betweenness attack
print("Running targeted betweenness attack (recalculated at each step)...")
G_temp = G.copy()
order_bc = []
# Recalculate every 5 steps
step = 0
while G_temp.number_of_nodes() > 0:
    if step % 5 == 0:
        bc = nx.betweenness_centrality(G_temp)
    highest = max(bc.keys(), key=lambda x: bc.get(x, 0))
    if highest not in G_temp:
        bc.pop(highest, None)
        continue
    order_bc.append(highest)
    bc.pop(highest, None)
    G_temp.remove_node(highest)
    step += 1

bc_fracs = simulate_removal(G, order_bc)
bc_threshold = find_threshold(bc_fracs, steps)
print(f"  Betweenness attack threshold (LCC < 0.5): {bc_threshold:.3f} of nodes removed")

# Theoretical percolation threshold for random network
degs = np.array([d for _, d in G.degree()])
k1 = np.mean(degs)
k2 = np.mean(degs**2)
# Molloy-Reed criterion: network percolates if k2/k1 - k1 > 1
# Critical threshold: fc = 1 - 1/(k2/k1 - 1)
molloy_reed = k2 / k1 - 1
if molloy_reed > 1:
    fc_predicted = 1 - 1 / (k2 / k1 - 1)
else:
    fc_predicted = 0.0
print(f"\nMolloy-Reed criterion (κ = k2/k1 - 1): {molloy_reed:.3f}")
print(f"Predicted random failure threshold (fc): {fc_predicted:.3f}")
print(f"Empirical random failure threshold:      {random_threshold:.3f}")

# Figure: Resilience curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Network Resilience Analysis", fontsize=14, fontweight="bold")

x = steps[:len(random_fracs)]

# Left: all three attack strategies
axes[0].plot(x, random_fracs[:len(x)], color="#2C5F2D", linewidth=2.5, label="Random Failure")
axes[0].plot(x, degree_fracs[:len(x)], color="#B85042", linewidth=2.5, linestyle="--", label="Degree Attack")
axes[0].plot(x, bc_fracs[:len(x)],     color="#1C7293", linewidth=2.5, linestyle=":",  label="Betweenness Attack")
axes[0].axhline(0.5, color="gray", linewidth=1, linestyle="-.", label="LCC = 0.5 threshold")
axes[0].axvline(fc_predicted, color="orange", linewidth=1.5, linestyle="--",
                label=f"Predicted fc={fc_predicted:.2f}")
axes[0].set_xlabel("Fraction of nodes removed", fontsize=12)
axes[0].set_ylabel("Largest CC / N", fontsize=12)
axes[0].set_title("Failure vs Attack Strategies", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Right: zoom in on the 0 to 0.4 removal range (most revealing)
x_zoom = steps[:int(N * 0.4) + 1]
axes[1].plot(x_zoom, random_fracs[:len(x_zoom)], color="#2C5F2D", linewidth=2.5, label="Random Failure")
axes[1].plot(x_zoom, degree_fracs[:len(x_zoom)], color="#B85042", linewidth=2.5, linestyle="--", label="Degree Attack")
axes[1].plot(x_zoom, bc_fracs[:len(x_zoom)],     color="#1C7293", linewidth=2.5, linestyle=":",  label="Betweenness Attack")
axes[1].axhline(0.5, color="gray", linewidth=1, linestyle="-.")
axes[1].set_xlabel("Fraction of nodes removed", fontsize=12)
axes[1].set_ylabel("Largest CC / N", fontsize=12)
axes[1].set_title("Zoomed: First 40% of Removals", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/05_resilience.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: 05_resilience.png")

print(f"""
Summary:
  Random failure collapses at:     {random_threshold:.1%} removed
  Degree attack collapses at:      {degree_threshold:.1%} removed
  Betweenness attack collapses at: {bc_threshold:.1%} removed
  Predicted threshold (fc):        {fc_predicted:.1%} removed

If degree/betweenness attacks collapse the network much faster than
random failure, your network is vulnerable like most scale-free networks —
remove the big colonial power hubs and the language diffusion network falls apart.
""")
