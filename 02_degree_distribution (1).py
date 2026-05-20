
# Degree Distribution & Scale Free Analysis

import csv, collections, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

CSV_PATH = "linguistic_transfer.csv"
os.makedirs("results/figures", exist_ok=True)

G = nx.DiGraph()
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        G.add_edge(row["source"], row["target"], edgetype=row["type"])

in_deg  = [d for _, d in G.in_degree()]
out_deg = [d for _, d in G.out_degree()]
tot_deg = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]

def degree_distribution(deg_seq):
    counts = collections.Counter(deg_seq)
    total  = len(deg_seq)
    ks = sorted(counts.keys())
    pk = [counts[k] / total for k in ks]
    return np.array(ks), np.array(pk)

def fit_power_law(deg_seq):
    counts = collections.Counter(deg_seq)
    ks = np.array(sorted(counts.keys()))
    pk = np.array([counts[k] / len(deg_seq) for k in ks])
    mask = (ks >= 1) & (pk > 0)
    ks_f, pk_f = ks[mask], pk[mask]
    slope, intercept, r, _, _ = stats.linregress(np.log(ks_f), np.log(pk_f))
    return -slope, np.exp(intercept), r**2, ks_f, pk_f

# Figure 1: Dot plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Degree Distribution: Colonial Language Network", fontsize=14, fontweight="bold")

for ax, deg, label, color in zip(
    axes,
    [in_deg, out_deg, tot_deg],
    ["In-Degree", "Out-Degree", "Total Degree"],
    ["#1C7293", "#B85042", "#2C5F2D"]
):
    ks, pk = degree_distribution(deg)

    # dot size proportional to frequency
    ax.scatter(ks, pk, color=color, s=pk*3000, alpha=0.75,
               edgecolors="white", linewidths=0.5, zorder=3)

    # thin stem lines for readability
    for k, p in zip(ks, pk):
        ax.plot([k, k], [0, p], color=color, alpha=0.3, linewidth=1.2, zorder=2)

    # zoom x-axis to meaningful range (P(k) >= 0.005)
    meaningful = ks[pk >= 0.005]
    if len(meaningful) > 1 and ks.max() > meaningful[-1]:
        ax.set_xlim(-0.3, meaningful[-1] + 1.5)
        ax.annotate(f"GB outlier\nk={ks.max()}",
                    xy=(meaningful[-1], pk[ks == meaningful[-1]][0]),
                    xytext=(max(0.5, meaningful[-1]*0.4), pk.max()*0.75),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1),
                    fontsize=8, color="gray")
    ax.set_ylim(0, pk.max() * 1.18)

    ax.set_xlabel("Degree k", fontsize=11)
    ax.set_ylabel("P(k)", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("results/figures/02a_degree_distribution_linear.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02a_degree_distribution_linear.png")

# Figure 2: Log log + trendline
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Log Log Degree Distribution: Power Law Test", fontsize=14, fontweight="bold")

for ax, deg, label, color in zip(
    axes,
    [in_deg, out_deg, tot_deg],
    ["In-Degree", "Out-Degree", "Total Degree"],
    ["#1C7293", "#B85042", "#2C5F2D"]
):
    alpha, C, r2, ks_fit, pk_fit = fit_power_law(deg)

    ax.scatter(ks_fit, pk_fit, color=color, s=55, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=3, label="Observed P(k)")

    k_range = np.linspace(ks_fit.min(), ks_fit.max(), 300)
    ax.plot(k_range, C * k_range**(-alpha), color="black", linewidth=2,
            linestyle="--", label=f"Fit: P(k) ~ k$^{{-{alpha:.2f}}}$\n$R^2$={r2:.3f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree k (log)", fontsize=11)
    ax.set_ylabel("P(k) (log)", fontsize=11)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sf = "scale free" if 2 <= alpha <= 3 else "NOT scale free"
    print(f"  {label}: α={alpha:.3f}, R²={r2:.3f} → {sf}")

plt.tight_layout()
plt.savefig("results/figures/02b_degree_distribution_loglog.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02b_degree_distribution_loglog.png")

# Figure 3: CCDF
fig, ax = plt.subplots(figsize=(7, 5))
for deg, label, color in [
    (in_deg,  "In-Degree",  "#1C7293"),
    (out_deg, "Out-Degree", "#B85042"),
    (tot_deg, "Total Degree","#2C5F2D"),
]:
    deg_arr = np.array(sorted(set(deg)))
    ccdf = np.array([sum(d >= k for d in deg) / len(deg) for k in deg_arr])
    ax.plot(deg_arr, ccdf, color=color, linewidth=2.5, label=label,
            marker="o", markersize=5)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Degree k (log)", fontsize=12)
ax.set_ylabel("P(K ≥ k): CCDF (log)", fontsize=12)
ax.set_title("Complementary CDF: Heavy Tail Test", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.2, linestyle="--")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/02c_ccdf.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 02c_ccdf.png")
