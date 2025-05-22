import os
import networkx as nx
import numpy as np
import pandas as pd
import EoN
import random
from joblib import Parallel, delayed
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
random.seed(61)

# function to filter networks to get strong edges only
def extract_strong_edges(G):
    edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True) if 'weight' in d] 
    if not edges:
        return None
    median_wt = np.median([w for _, _, w in edges]) 
    G_strong = G.copy() 
    G_strong.remove_edges_from([(u, v) for u, v, w in edges if w <= median_wt]) 
    return G_strong if G_strong.number_of_edges() > 0 else None

# simulation function
def run_simulation(args):
    G, tau, gamma, run_id = args
    sim = EoN.Gillespie_SIR(
        G, tau=tau, gamma=gamma,
        initial_infecteds=random.sample(list(G.nodes()), 1),
        return_full_data=True
    )
    df_trans = pd.DataFrame(sim.transmissions(), columns=["time", "infector", "infectee"])
    df_trans["run_id"] = run_id

    df_series = pd.DataFrame({
        "t": sim.t(),
        "S": sim.S(),
        "I": sim.I(),
        "R": sim.R(),
        "run_id": run_id
    })

    return df_trans, df_series

if __name__ == "__main__":
    graph_dir = r"C:\Users\s2607536\Documents\asnr_2025\Networks"
    output_dir = r"C:\Users\s2607536\Documents\superspreader\superspreader\Results2025"
    os.makedirs(output_dir, exist_ok=True)

    graphs = {} 
    strong_graphs = {} 
    for file in os.listdir(graph_dir):
        if file.endswith(".graphml"):
            G = nx.read_graphml(os.path.join(graph_dir, file))
            if G.number_of_nodes() > 10 and G.number_of_edges() > 10:
                edge_weights = nx.get_edge_attributes(G, 'weight')
                if edge_weights:
                    try:
                        for (u, v), wt in edge_weights.items():
                            G[u][v]['weight'] = float(wt)
                        graphs[file] = G
                        strong_G = extract_strong_edges(G)
                        if strong_G:
                            strong_graphs[file] = strong_G
                    except:
                        print(f"{file} has non-numeric weights, skipping")

    tau_vals = [0.01, 0.05, 0.1, 0.3, 0.6]
    gamma_vals = [0.01, 0.08, 0.3]
    n_reps = 1000

    for name in list(graphs.keys()): 
        base = os.path.splitext(name)[0]
        for version, graph in {"full": graphs[name], "strong": strong_graphs.get(name)}.items():
            if graph is None or graph.number_of_edges() == 0:
                continue

            G_version = graph.subgraph(max(nx.connected_components(graph), key=len)).copy() 
            if G_version.number_of_nodes() < 5:
                continue

            suffix = f"{base}_{version}"
            node_count = G_version.number_of_nodes()
            edge_count = G_version.number_of_edges()
            density = nx.density(G_version)
            transitivity = nx.transitivity(G_version)
            try:
                diameter = nx.diameter(G_version)
            except nx.NetworkXError:
                diameter = np.nan  # disconnected
            degrees = np.array(list(dict(G_version.degree()).values()))
            degree_cv = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else np.nan
            communities = list(greedy_modularity_communities(G_version))
            modularity = nx.algorithms.community.quality.modularity(G_version, communities)

            degree = dict(G_version.degree())
            eig = nx.eigenvector_centrality_numpy(G_version)
            btw = nx.betweenness_centrality(G_version)
            core = nx.core_number(G_version)
            node_attrs = list(next(iter(G_version.nodes(data=True)))[1].keys()) if G_version.nodes else []

            records = []
            for node in G_version.nodes():
                rec = {"network": suffix,
                    "node": node,
                    "degree": degree[node],
                    "eigenvector": eig[node],
                    "betweenness": btw[node],
                    "kshell": core[node],
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "density": density,
                    "modularity": modularity, 
                    "transitivity": transitivity,
                    "network_size": node_count,
                    "diameter": diameter,
                    "degree_cv": degree_cv}
                for attr in node_attrs:
                    rec[attr] = G_version.nodes[node].get(attr, None)
                records.append(rec)

            pd.DataFrame.from_records(records).to_csv(
                os.path.join(output_dir, f"{suffix}_node_and_network_stats.csv"), index=False)

            for tau in tau_vals:
                for gamma in gamma_vals:
                    prefix = f"{suffix}_tau{tau}_gamma{gamma}"
                    args_list = [(G_version, tau, gamma, run_id) for run_id in range(1, n_reps + 1)]

                    results_parallel = Parallel(n_jobs=-1, backend="loky")(
                        delayed(run_simulation)(args) for args in args_list
                    )

                    all_transmats, all_series = zip(*results_parallel)

                    pd.concat(all_transmats).to_csv(
                        os.path.join(output_dir, f"{prefix}_transmat.csv"), index=False)
                    pd.concat(all_series).to_csv(
                        os.path.join(output_dir, f"{prefix}_series.csv"), index=False)
