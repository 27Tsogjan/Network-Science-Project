"""
Spreading Models: SI and SIR
- R0 = (beta/mu) * mean_k  (corrected formula)
- SI model: language spread as contagion
- SIR model: spread with recovery rate mu
- Compares colonial network vs Erdos-Renyi random network
"""

import csv, os, random, collections
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "linguistic_transfer.csv"
os.makedirs("results/figures", exist_ok=True)
N_TRIALS = 50
T_STEPS  = 30

G_dir = nx.DiGraph()
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        G_dir.add_edge(row["source"], row["target"])

G = G_dir.to_undirected()
N = G.number_of_nodes()
nodes = list(G.nodes())
p_er  = G.number_of_edges() / (N * (N - 1) / 2)
G_er  = nx.erdos_renyi_graph(N, p_er, seed=42)

top_hub = max(G.degree(), key=lambda x: x[1])[0]
print(f"Top hub (seed): {top_hub}")

def simulate_SI(G, seed_node, beta, T):
    infected = {seed_node}
    curve = [len(infected) / G.number_of_nodes()]
    for _ in range(T):
        new_infected = set()
        for node in infected:
            for nb in G.neighbors(node):
                if nb not in infected and random.random() < beta:
                    new_infected.add(nb)
        infected |= new_infected
        curve.append(len(infected) / G.number_of_nodes())
    return curve

def simulate_SIR(G, seed_node, beta, mu, T):
    n = G.number_of_nodes()
    susceptible = set(G.nodes()) - {seed_node}
    infected    = {seed_node}
    recovered   = set()
    S, I, R = [len(susceptible)/n], [len(infected)/n], [0.0]
    for _ in range(T):
        new_inf, new_rec = set(), set()
        for node in infected:
            for nb in G.neighbors(node):
                if nb in susceptible and random.random() < beta:
                    new_inf.add(nb)
            if random.random() < mu:
                new_rec.add(node)
        susceptible -= new_inf
        infected    |= new_inf
        infected    -= new_rec
        recovered   |= new_rec
        S.append(len(susceptible)/n)
        I.append(len(infected)/n)
        R.append(len(recovered)/n)
    return S, I, R

def avg_curves(G, seed_node, model, T, trials, **kwargs):
    all_curves = []
    for _ in range(trials):
        if model == "SI":
            all_curves.append(simulate_SI(G, seed_node, T=T, **kwargs))
        elif model == "SIR":
            _, _, R = simulate_SIR(G, seed_node, T=T, **kwargs)
            all_curves.append(R)
    return np.mean(all_curves, axis=0)

# Parameters: beta=transmission, mu=recovery
BETA = 0.15
MU   = 0.05
random_seed = random.choice(nodes)

print(f"Running SI  (β={BETA}, {N_TRIALS} trials)...")
si_col_hub  = avg_curves(G,    top_hub,     "SI", T_STEPS, N_TRIALS, beta=BETA)
si_col_rand = avg_curves(G,    random_seed, "SI", T_STEPS, N_TRIALS, beta=BETA)
si_er_hub   = avg_curves(G_er, 0,           "SI", T_STEPS, N_TRIALS, beta=BETA)
si_er_rand  = avg_curves(G_er, random.randint(0,N-1), "SI", T_STEPS, N_TRIALS, beta=BETA)

print(f"Running SIR (β={BETA}, μ={MU}, {N_TRIALS} trials)...")
sir_col_hub  = avg_curves(G,    top_hub,     "SIR", T_STEPS, N_TRIALS, beta=BETA, mu=MU)
sir_col_rand = avg_curves(G,    random_seed, "SIR", T_STEPS, N_TRIALS, beta=BETA, mu=MU)
sir_er_hub   = avg_curves(G_er, 0,           "SIR", T_STEPS, N_TRIALS, beta=BETA, mu=MU)
sir_er_rand  = avg_curves(G_er, random.randint(0,N-1), "SIR", T_STEPS, N_TRIALS, beta=BETA, mu=MU)

t = np.arange(T_STEPS + 1)
labels_bar = ["Colonial\nHub seed","Colonial\nRandom seed","Random Net\nHub seed","Random Net\nRandom seed"]
colors_bar = ["#1C7293","#5BA8C4","#64748B","#9AAAB8"]

# SI plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"SI Model: Language Spread  (β={BETA})", fontsize=14, fontweight="bold")

axes[0].plot(t, si_col_hub,  color="#1C7293", lw=2.5, label=f"Colonial — Hub ({top_hub})")
axes[0].plot(t, si_col_rand, color="#1C7293", lw=2.5, ls="--", label="Colonial — Random seed")
axes[0].plot(t, si_er_hub,   color="#64748B", lw=2.5, label="Random Net — Hub seed")
axes[0].plot(t, si_er_rand,  color="#64748B", lw=2.5, ls="--", label="Random Net — Random seed")
axes[0].set_xlabel("Time steps", fontsize=12)
axes[0].set_ylabel("Fraction infected", fontsize=12)
axes[0].set_title("Infected fraction over time", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

final_si = [si_col_hub[-1], si_col_rand[-1], si_er_hub[-1], si_er_rand[-1]]
axes[1].bar(labels_bar, final_si, color=colors_bar, edgecolor="white")
axes[1].set_ylabel("Final fraction infected", fontsize=12)
axes[1].set_title("Final spread size", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1.05)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/06a_SI_spreading.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06a_SI_spreading.png")

# SIR plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"SIR Model: Language Spread  (β={BETA}, μ={MU})", fontsize=14, fontweight="bold")

axes[0].plot(t, sir_col_hub,  color="#1C7293", lw=2.5, label=f"Colonial — Hub ({top_hub})")
axes[0].plot(t, sir_col_rand, color="#1C7293", lw=2.5, ls="--", label="Colonial — Random seed")
axes[0].plot(t, sir_er_hub,   color="#64748B", lw=2.5, label="Random Net — Hub seed")
axes[0].plot(t, sir_er_rand,  color="#64748B", lw=2.5, ls="--", label="Random Net — Random seed")
axes[0].set_xlabel("Time steps", fontsize=12)
axes[0].set_ylabel("Fraction recovered (reached)", fontsize=12)
axes[0].set_title("Cumulative spread (recovered) over time", fontsize=12, fontweight="bold")
axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

final_sir = [sir_col_hub[-1], sir_col_rand[-1], sir_er_hub[-1], sir_er_rand[-1]]
axes[1].bar(labels_bar, final_sir, color=colors_bar, edgecolor="white")
axes[1].set_ylabel("Final fraction reached", fontsize=12)
axes[1].set_title("Final epidemic size", fontsize=12, fontweight="bold")
axes[1].set_ylim(0, 1.05)
axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/06b_SIR_spreading.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 06b_SIR_spreading.png")

# R0 formula: R0 = (beta/mu) * mean_k
degs  = np.array([d for _, d in G.degree()])
mean_k = np.mean(degs)
R0 = (BETA / MU) * mean_k
beta_c = MU / mean_k   # epidemic threshold

print(f"\nR0 = (β/μ) × ⟨k⟩ = ({BETA}/{MU}) × {mean_k:.3f} = {R0:.3f}")
print(f"{'Epidemic spreads network-wide (R0 > 1)' if R0 > 1 else 'Epidemic dies out (R0 < 1)'}")
print(f"Epidemic threshold: β_c = μ/⟨k⟩ = {MU}/{mean_k:.3f} = {beta_c:.4f}")
