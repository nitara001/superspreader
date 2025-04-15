import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def estimate_dispersion_k(secondary_infections):
    if len(secondary_infections) < 2:
        return np.nan  # Too few to estimate
    mean_si = np.mean(secondary_infections)
    var_si = np.var(secondary_infections, ddof=1)
    if var_si <= mean_si:
        return np.inf  # No overdispersion
    return mean_si**2 / (var_si - mean_si)

transmat_dir = r"C:\Users\s2607536\OneDrive - University of Edinburgh\Code\superspreader\Results_gillespie_grid"
output_dir = os.path.join(transmat_dir, "super_plots")
os.makedirs(output_dir, exist_ok=True)

dispersion_log = []

for filepath in glob(os.path.join(transmat_dir, "*_transmat.csv")):
    df = pd.read_csv(filepath)
    if df.empty or 'infector' not in df.columns:
        continue

    base = os.path.splitext(os.path.basename(filepath))[0]
    title = base.replace("_transmat", "")

    all_nodes = pd.unique(df[['infector', 'infectee']].values.ravel('K'))
    infect_counts = df['infector'].value_counts().reindex(all_nodes, fill_value=0).sort_values(ascending=False)

    if infect_counts.empty:
        continue

    # Dispersion parameter
    k = estimate_dispersion_k(infect_counts.values)
    dispersion_log.append({"network": title, "k_dispersion": k})

    # Ranked node plot
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(infect_counts)+1), infect_counts.values, marker='o')
    plt.xlabel("Ranked node")
    plt.ylabel("Secondary infections")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_spread_ranked.png"))
    plt.close()

    # Histogram plot
    plt.figure(figsize=(6, 4))
    plt.hist(infect_counts.values, bins=range(0, max(infect_counts)+2), align='left', edgecolor='black')
    plt.xlabel("Number of secondary infections")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base}_spread_hist.png"))
    plt.close()

pd.DataFrame(dispersion_log).to_csv(os.path.join(output_dir, "dispersion_summary.csv"), index=False)
