import EoN
import networkx as nx
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# === Setup paths ===
graph_dir = r"C:\Users\s2607536\Documents\asnr_2025\Networks"
output_dir = r"C:\Users\s2607536\OneDrive - University of Edinburgh\Code\superspreader\Results"
os.makedirs(output_dir, exist_ok=True)

# === Load ASNR graphs ===
graphs = {}
for file in os.listdir(graph_dir):
    if file.endswith(".graphml"):
        G = nx.read_graphml(os.path.join(graph_dir, file))
        if G.number_of_nodes() > 10 and nx.is_connected(G):
            graphs[file] = G

# === Parameters ===
simple_beta = 0.2 #infection transmission rate 
max_infection_rate = 0.8 #the max rate of information transmission even if all neighbours are informed
min_infection_rate = 0.001 #min rate even if no neighbours are informed- some personal learning
sigmoid_steepness = 20
complex_params = (max_infection_rate, min_infection_rate, sigmoid_steepness)

# === Run each model ===
for name, G in graphs.items():
    for mode in ["simple", "complex"]:
        IC = defaultdict(lambda: 'S') #set up all nodes as S
        initial_infected = random.sample(list(G.nodes()), 1) #seed 1 random node
        seed_set = set(initial_infected)
        infection_log = {}
        for node in initial_infected:
            IC[node] = 'I'
            infection_log[node] = ['SEED']

        last_infector = {}  # to track infector in simple contagion

        # === Define rate functions ===
        if mode == "simple":
            def rate_function(G, node, status, parameters):
                beta = parameters[0]
                if status[node] == 'S':
                    infectors = [nbr for nbr in G.neighbors(node) if status[nbr] == 'I'] #'this is a list of all currently infected neighbours of the S node (possible sources of infection)
                    if not infectors:
                        return 0 #can't get infected if it has no infected neighbours
                    chosen = random.choice(infectors)
                    last_infector[node] = chosen
                    return 1 - (1 - beta)**len(infectors)
                return 0
            parameters = (simple_beta,)
        else:
            def rate_function(G, node, status, parameters):
                max_prob, min_prob, steepness = parameters
                if status[node] == 'S':
                    neighbors = list(G.neighbors(node))
                    if not neighbors:
                        return 0
                    infected_neighbors = sum(1 for nbr in neighbors if status[nbr] == 'I')
                    degree = len(neighbors)
                    frac_infected = infected_neighbors / degree
                    x = steepness * (frac_infected - 0.5)
                    prob = max_prob * (1 - (1 - 2 * min_prob) / (1 + np.exp(x))) + min_prob
                    return min(max(prob, min_prob), max_prob)
                return 0
            parameters = complex_params

        # === Track who caused infection ===
        def transition_choice(G, node, status, parameters):
            if node not in seed_set and status[node] == 'S':
                if mode == "simple":
                    infection_log[node] = [last_infector.get(node, "UNKNOWN")]
                else:
                    infectors = [nbr for nbr in G.neighbors(node) if status[nbr] == 'I']
                    if infectors:
                        infection_log[node] = infectors
            return 'I'

        def get_influence_set(G, node, status, parameters):
            return G.neighbors(node)

        try:
            t, S, I = EoN.Gillespie_complex_contagion(
                G, rate_function, transition_choice, get_influence_set,
                IC, return_statuses=('S', 'I'), parameters=parameters, tmax=1000
            )
        except:
            continue

        # === Save results ===
        base = os.path.splitext(name)[0]
        prefix = f"{base}_{mode}"

        plt.figure(figsize=(6, 4))
        plt.plot(t, S, label='Susceptible')
        plt.plot(t, I, label='Infected')
        plt.title(prefix)
        plt.xlabel("Time")
        plt.ylabel("Node count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_curve.png"))
        plt.close()

        pd.DataFrame([(k, v) for k, v in infection_log.items()],
                     columns=["infectee", "infectors"]).to_csv(
            os.path.join(output_dir, f"{prefix}_log.csv"), index=False)

        pd.DataFrame({"t": t, "S": S, "I": I}).to_csv(
            os.path.join(output_dir, f"{prefix}_series.csv"), index=False)

# === Save sigmoid curve figure ===
degree = 10
infected_counts = np.arange(0, degree + 1)
frac_infected = infected_counts / degree

sigmoid_probs = []
for frac in frac_infected:
    x = sigmoid_steepness * (frac - 0.5)
    prob = max_infection_rate * (1 - (1 - 2 * min_infection_rate) / (1 + np.exp(x))) + min_infection_rate
    sigmoid_probs.append(prob)

plt.figure(figsize=(7, 4))
plt.plot(infected_counts, sigmoid_probs, marker='o')
plt.xlabel("Number of infected neighbors (out of 10)")
plt.ylabel("Infection probability")
plt.title("Sigmoid-shaped infection curve (complex contagion)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "sigmoid_shape.png"))
plt.close()


## ===  sigmoid shape  ===
plt.figure(figsize=(8, 5))
degree = 10
infected_counts = np.arange(0, degree + 1)
frac_infected = infected_counts / degree

sigmoid_probs = []
for frac in frac_infected:
    x = psi * (frac - 0.5)
    prob = max_rate * (1 - (1 - 2 * eta) / (1 + np.exp(x))) + eta
    sigmoid_probs.append(prob)  # <-- this was missing!

plt.plot(infected_counts, sigmoid_probs, marker='o')
plt.xlabel("Number of infected neighbors (out of 10)")
plt.ylabel("Infection probability")
plt.title(" sigmoid-shaped infection probability")
plt.grid(True)
plt.tight_layout()
plt.show()
