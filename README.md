
# Network Science Project — Analysis Scripts

This repository contains analysis scripts for a colonial language transfer network. The scripts
compute network properties, visualize degree distributions, compare with null models, detect
communities, evaluate resilience under node removal, and simulate simple spreading models.

## Requirements
- Python 3.8+ and the following Python packages: `networkx`, `numpy`, `scipy`, `matplotlib`.

## Data
- The primary input is `linguistic_transfer.csv` with columns `source`, `target`, and `type`.
	Adjust the `CSV_PATH` variable at the top of each script if your data file is named differently.

## Quick usage
Run a script with Python, for example:

```bash
python 01_network_properties.py
python "02_degree_distribution (1).py"
python "03_network_comparison (1).py"
python 04_community_detection.py
python 05_resilience.py
python "06_spreading_models (1).py"
```

## Outputs
- Scripts write outputs into the `results/` directory. Expect JSON summaries (e.g. `results/01_properties.json`)
	and PNG figures under `results/figures/` (e.g. `02a_degree_distribution_linear.png`).

## File descriptions

- `01_network_properties.py`
	- Purpose: Compute core metrics for the directed linguistic-transfer network.
	- What it does: builds a `networkx.DiGraph` from `linguistic_transfer.csv`, reports node/edge
		counts and edge-type breakdown, computes density, in/out/total degree statistics, betweenness
		centrality, top hubs, average clustering (on undirected projection), diameter and average
		shortest-path length (on the largest weakly connected component), assortativity, and reciprocity.
	- Output: prints a summary and saves `results/01_properties.json` with the computed metrics.

- `02_degree_distribution (1).py`
	- Purpose: Analyze and visualize the network's degree distributions for in-, out-, and total-degree.
	- What it does: computes degree sequences, produces linear dot/stem plots, fits a power-law trend
		(linear regression on log-log data) to estimate the exponent α and R², and plots CCDFs.
	- Output: PNGs saved to `results/figures/` (files prefixed `02*`) and printed fit statistics.

- `03_network_comparison (1).py`
	- Purpose: Compare empirical network structure with standard null/synthetic models.
	- What it does: constructs a shared-language undirected network, generates an Erdos–Rényi random
		graph and a Barabási–Albert graph using empirical N/p/m parameters, computes metrics (density,
		average degree, clustering, avg path length, diameter, assortativity), and plots log–log degree
		distributions with power-law fits for each network.
	- Output: comparison PNGs in `results/figures/` (e.g. `03_network_comparison.png`,
		`03b_small_world_comparison.png`) and console metric summaries.

- `04_community_detection.py`
	- Purpose: Detect communities and assess whether observed modularity is significant.
	- What it does: converts the directed graph to undirected, runs greedy modularity (Louvain-style),
		label propagation, and Girvan–Newman (top-level split), computes modularity Q, and compares
		Q to a null distribution via randomized configuration-model graphs (100 trials by default).
	- Output: community-size and null-model comparison PNGs in `results/figures/` and console output
		with modularity and z-score significance.

- `05_resilience.py`
	- Purpose: Evaluate network robustness under node removals (random failure and targeted attacks).
	- What it does: simulates repeated random node removals (averaged over trials), computes targeted
		removal orders (degree-based and betweenness-based), tracks the largest connected component
		fraction as nodes are removed, and estimates percolation thresholds (empirical and Molloy–Reed
		theoretical estimate).
	- Output: `results/figures/05_resilience.png` and a printed summary of collapse thresholds.

- `06_spreading_models (1).py`
	- Purpose: Simulate SI and SIR spreading processes to model language diffusion.
	- What it does: runs SI and SIR models seeded at the empirically-identified top hub and
		at random nodes, averages multiple trials, and compares epidemic dynamics on the empirical
		network vs an Erdos–Rényi random graph. Prints R0 and epidemic threshold estimates.
	- Output: PNGs (e.g. `results/figures/06a_SI_spreading.png`, `06b_SIR_spreading.png`) and console
		output with R0/threshold values.

## Other files and notes
- `data.py`: optional utilities for data handling (inspect the file head to see exported helpers).
- Several CSV and GML files exist (e.g. `Edges.csv`, `Nodes.csv`, `linguistic_network.gml`) which
	may serve as alternative inputs or visualization resources for Gephi.


