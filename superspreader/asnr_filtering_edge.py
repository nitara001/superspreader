import os
import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import spearmanr

# === Paths ===
results_dir = "/Users/nitarawijayatilake/Documents/PhD/superspreader/superspreader/Results_gillespie_grid1"
k_file = os.path.join(results_dir, "super_plots", "k_dispersion_summary.csv")

# === Step 1: Load network-level stats from each node stats file ===
node_stat_files = glob(os.path.join(results_dir, "*_node_and_network_stats.csv"))
network_stats = []

for f in node_stat_files:
    try:
        df = pd.read_csv(f)
        if not df.empty:
            row = df.iloc[0]
            network_stats.append({
                "network": row["network"],
                "density": row.get("density", None),
                "modularity": row.get("modularity", None),
                "node_count": row.get("node_count", None),
                "edge_count": row.get("edge_count", None)
            })
    except Exception as e:
        print(f"Skipping {f}: {e}")

stats_df = pd.DataFrame(network_stats)

# === Step 2: Merge with k-dispersion summary ===
k_df = pd.read_csv(k_file)
k_df['k_dispersion'] = pd.to_numeric(k_df['k_dispersion'], errors='coerce')
merged = pd.merge(k_df, stats_df, on="network")

# === Step 3: Clean and filter ===
merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["k_dispersion", "density", "modularity"])

# === Step 4: Correlation tests ===
r_dens, p_dens = spearmanr(merged["k_dispersion"], merged["density"])
r_mod, p_mod = spearmanr(merged["k_dispersion"], merged["modularity"])
r_nodes, p_nodes = spearmanr(merged["k_dispersion"], merged["node_count"])
r_edges, p_edges = spearmanr(merged["k_dispersion"], merged["edge_count"])

print(f"Spearman (k ~ density):     r = {r_dens:.3f}, p = {p_dens:.3g}")
print(f"Spearman (k ~ modularity):  r = {r_mod:.3f}, p = {p_mod:.3g}")
print(f"Spearman (k ~ node_count):  r = {r_nodes:.3f}, p = {p_nodes:.3g}")
print(f"Spearman (k ~ edge_count):  r = {r_edges:.3f}, p = {p_edges:.3g}")
