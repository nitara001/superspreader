import networkx as nx
import EoN
import numpy as np
import random

def run_secondary_infection(G, tau, gamma):
    sim = EoN.Gillespie_SIR(
        G, tau=tau, gamma=gamma,
        initial_infecteds=random.sample(list(G.nodes()), 1),
        return_full_data=True
    )
    return len(sim.transmissions())

def omega_test(G, tau, gamma, omega=0.1, Q=5, x_start=10, x_max=1000):
    secondary_counts = []
    Z_values = []
    ΔZ_series = []

    for i in range(1, x_max + 1):
        secondary_counts.append(run_secondary_infection(G, tau, gamma))
        if i >= x_start:
            μ = np.mean(secondary_counts)
            σ2 = np.var(secondary_counts, ddof=1)
            Z = (σ2 / μ) * 100 if μ != 0 else float("inf")
            Z_values.append(Z)

            if len(Z_values) > 1:
                ΔZ = abs(Z_values[-2] - Z_values[-1])
                ΔZ_series.append(ΔZ)

                if len(ΔZ_series) >= Q and all(dz < omega for dz in ΔZ_series[-Q:]):
                    print(f"✅ Converged at {i} replicates (Ω={omega}, Q={Q})")
                    return i, secondary_counts

    print(f"⚠️ Did not converge within {x_max} replicates.")
    return x_max, secondary_counts
