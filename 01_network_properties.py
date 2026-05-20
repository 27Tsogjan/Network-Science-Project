"""
Basic Network Properties

Calculates: node/edge counts, density, degree stats, average degree,
betweenness centrality, hubs, diameter, avg path length,
clustering coefficient, assortativity, reciprocity.

Output: prints a summary table + saves properties.json
"""

import csv, json, collections
import networkx as nx

CSV_PATH = "linguistic_transfer.csv"   #adjust path if needed
OUT_JSON = "results/01_properties.json"

#Load graph
G = nx.DiGraph()
edge_types = {}

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        src, tgt, typ = row["source"], row["target"], row["type"]
        G.add_edge(src, tgt, edgetype=typ)
        edge_types[(src, tgt)] = typ

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

#Edge type breakdown
type_counts = collections.Counter(edge_types.values())
print(f"\nEdge types:")
for t, c in type_counts.items():
    print(f"  {t}: {c}")

#Density
density = nx.density(G)
print(f"\nDensity: {density:.6f}")

#Degree stats
in_deg  = dict(G.in_degree())
out_deg = dict(G.out_degree())
tot_deg = {n: in_deg[n] + out_deg[n] for n in G.nodes()}

avg_in  = sum(in_deg.values())  / G.number_of_nodes()
avg_out = sum(out_deg.values()) / G.number_of_nodes()
avg_tot = sum(tot_deg.values()) / G.number_of_nodes()

print(f"\nAverage in-degree:  {avg_in:.3f}")
print(f"Average out-degree: {avg_out:.3f}")
print(f"Average total degree: {avg_tot:.3f}")

#Top hubs by out-degree (colonial powers)
print("\nTop 10 nodes by OUT-degree (colonial powers):")
top_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10]
for n, d in top_out:
    print(f"  {n:6s}  out={d}  in={in_deg[n]}")

print("\nTop 10 nodes by IN-degree (most colonized):")
top_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:10]
for n, d in top_in:
    print(f"  {n:6s}  in={d}  out={out_deg[n]}")

#Betweenness centrality
print("\nCalculating betweenness centrality (this may take a moment)...")
bc = nx.betweenness_centrality(G, normalized=True)
top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 nodes by betweenness centrality:")
for n, v in top_bc:
    print(f"  {n:6s}  {v:.6f}")

#Clustering coefficient
# Use undirected version for standard clustering coefficient
G_und = G.to_undirected()
avg_clust = nx.average_clustering(G_und)
print(f"\nAverage clustering coefficient (undirected): {avg_clust:.6f}")

# Diameter & avg path length (on largest weakly connected component)
largest_wcc = max(nx.weakly_connected_components(G), key=len)
G_sub = G.subgraph(largest_wcc).copy()
G_sub_und = G_sub.to_undirected()

diameter = nx.diameter(G_sub_und)
avg_path = nx.average_shortest_path_length(G_sub_und)
print(f"\nLargest WCC size: {len(largest_wcc)} nodes")
print(f"Diameter: {diameter}")
print(f"Average shortest path length: {avg_path:.4f}")

# Assortativity
assort_degree = nx.degree_assortativity_coefficient(G)
print(f"\nDegree assortativity coefficient: {assort_degree:.4f}")
print("  (negative = hubs connect to low-degree nodes; expected for colonial networks)")

# Reciprocity
recip = nx.reciprocity(G)
print(f"\nReciprocity: {recip:.4f}")
print("  (fraction of edges that have a reciprocal edge)")

# Save results
import os; os.makedirs("results", exist_ok=True)

results = {
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "edge_type_counts": dict(type_counts),
    "density": density,
    "avg_in_degree": avg_in,
    "avg_out_degree": avg_out,
    "avg_total_degree": avg_tot,
    "top_out_degree": top_out,
    "top_in_degree": top_in,
    "top_betweenness": top_bc,
    "avg_clustering": avg_clust,
    "diameter": diameter,
    "avg_path_length": avg_path,
    "assortativity": assort_degree,
    "reciprocity": recip,
    "largest_wcc_size": len(largest_wcc),
}

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUT_JSON}")
