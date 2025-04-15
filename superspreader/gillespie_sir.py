import networkx as nx
import EoN
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from itertools import product
from networkx.algorithms.community import greedy_modularity_communities

# === Paths ===
graph_dir = r"C:\Users\s2607536\Documents\asnr_2025\Networks"
output_dir = r"C:\Users\s2607536\OneDrive - University of Edinburgh\Code\superspreader\Results_gillespie_grid"
os.makedirs(output_dir, exist_ok=True)

# === Load ASNR graphs ===
graphs = {}
for file in os.listdir(graph_dir):
    if file.endswith(".graphml"):
        G = nx.read_graphml(os.path.join(graph_dir, file))
        if G.number_of_nodes() > 10 and G.number_of_edges() > 5:
            graphs[file] = G

# === Parameters ===
tau_vals = [0.1, 0.5, 0.9]
gamma_vals = [0.1, 0.5, 0.9]
n_reps = 5  # number of repetitions per parameter combo

# === Main loop ===
for name, G in list(graphs.items()):
    base = os.path.splitext(name)[0]

    try:
        # Network stats
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        density = nx.density(G)
        communities = list(greedy_modularity_communities(G))
        modularity = nx.algorithms.community.quality.modularity(G, communities)

        degree = dict(G.degree())
        eig = nx.eigenvector_centrality_numpy(G)
        btw = nx.betweenness_centrality(G)
        core = nx.core_number(G)
        node_attributes = list(next(iter(G.nodes(data=True)))[1].keys()) if len(G.nodes) > 0 else []

        records = []
        for node in G.nodes():
            record = {
                "network": base,
                "node": node,
                "degree": degree[node],
                "eigenvector": eig[node],
                "betweenness": btw[node],
                "kshell": core[node],
                "node_count": node_count,
                "edge_count": edge_count,
                "density": density,
                "modularity": modularity
            }
            for attr in node_attributes:
                record[attr] = G.nodes[node].get(attr, None)
            records.append(record)

        full_df = pd.DataFrame.from_records(records)
        full_df.to_csv(os.path.join(output_dir, f"{base}_node_and_network_stats.csv"), index=False)

    except Exception as e:
        print(f"❌ didnt work for {name}: {e}")
        continue

    # Epidemic simulations with repeats
    for tau, gamma in product(tau_vals, gamma_vals):
        prefix = f"{base}_tau{tau}_gamma{gamma}"
        all_transmats = []
        all_series = []

        for run in range(1, n_reps + 1):
            sim = EoN.Gillespie_SIR(
                G,
                tau=tau,
                gamma=gamma,
                initial_infecteds=random.sample(list(G.nodes()), 1),
                return_full_data=True
            )

            df_trans = pd.DataFrame(sim.transmissions(), columns=["time", "infector", "infectee"])
            df_trans["run_id"] = run
            all_transmats.append(df_trans)

            df_series = pd.DataFrame({
                "t": sim.t(),
                "S": sim.S(),
                "I": sim.I(),
                "R": sim.R()
            })
            df_series["run_id"] = run
            all_series.append(df_series)

        pd.concat(all_transmats).to_csv(
            os.path.join(output_dir, f"{prefix}_transmat.csv"), index=False
        )
        pd.concat(all_series).to_csv(
            os.path.join(output_dir, f"{prefix}_series.csv"), index=False
        )
