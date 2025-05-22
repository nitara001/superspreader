import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import nbinom
from scipy.optimize import minimize
from tqdm import tqdm

# ─── USER PARAMETERS ──────────────────────────────────────────────────────────
transmat_dir = "/Users/nitarawijayatilake/Documents/PhD/superspreader/superspreader/Results_gillespie_grid1"
output_dir   = os.path.join(transmat_dir, "super_plots")
os.makedirs(output_dir, exist_ok=True)

BATCH = 100   # number of runs to combine into one plot

# ─── UTILS ────────────────────────────────────────────────────────────────────
def fit_negative_binomial(data):
    mean_val, var_val = data.mean(), data.var(ddof=1)
    if var_val <= mean_val: return None
    def neg_ll(params):
        r, p = params
        if r <= 0 or not (0 < p < 1): return np.inf
        return -np.sum(nbinom.logpmf(data, r, p))
    r0 = mean_val**2 / (var_val - mean_val)
    p0 = mean_val / var_val
    res = minimize(
        neg_ll,
        x0=[r0, p0],
        bounds=[(1e-5, None), (1e-5, 1-1e-5)]
    )
    return res.x if res.success else None

def estimate_dispersion_k(data):
    if len(data) < 2: return np.nan
    m, v = data.mean(), data.var(ddof=1)
    return np.inf if v <= m else m**2 / (v - m)

# ─── GATHER ALL FILES ─────────────────────────────────────────────────────────
pattern   = os.path.join(transmat_dir, "*_transmat.csv")
filepaths = sorted(glob(pattern))

# ─── PROCESS IN BATCHES ───────────────────────────────────────────────────────
batch_counts = []   # list of 1D arrays of counts
batch_titles = []   # list of network titles in this batch
summary      = []   # per-run metadata

for idx, fp in enumerate(tqdm(filepaths, desc="All runs"), start=1):
    title = os.path.basename(fp).replace("_transmat.csv", "")
    try:
        df = pd.read_csv(fp, usecols=["infector","infectee"])
    except:
        continue
    if df.empty:
        continue

    all_nodes = pd.unique(df[['infector','infectee']].values.ravel('K'))
    infect_counts = (
        df['infector']
        .value_counts()
        .reindex(all_nodes, fill_value=0)
        .sort_values(ascending=False)
    )
    values = infect_counts.values
    if values.sum() == 0 or len(values) < 2:
        continue

    # record summary
    nb = fit_negative_binomial(values)
    k_disp = estimate_dispersion_k(values)
    r, p = (nb if nb is not None else (np.nan, np.nan))
    summary.append({"network": title, "nb_r": r, "nb_p": p, "k_dispersion": k_disp})

    # accumulate for this batch
    batch_counts.append(values)
    batch_titles.append(title)

    # when batch is full, plot & clear
    if idx % BATCH == 0:
        start, end = idx - BATCH + 1, idx
        first_title, last_title = batch_titles[0], batch_titles[-1]

        # overlaid ranked‐node plot
        plt.figure(figsize=(6,4))
        for arr in batch_counts:
            plt.plot(np.sort(arr)[::-1], alpha=0.3, color='black')
        plt.xlabel("Ranked node")
        plt.ylabel("Secondary infections")
        plt.title(f"Runs {start}–{end}")
        plt.tight_layout()
        fname = f"batch_ranked_{start}_{end}_{first_title}__to__{last_title}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

        # aggregated histogram
        all_vals = np.concatenate(batch_counts)
        plt.figure(figsize=(6,4))
        plt.hist(all_vals,
                 bins=range(0, all_vals.max()+2),
                 align='left',
                 edgecolor='black')
        plt.xlabel("Number of secondary infections")
        plt.ylabel(f"Frequency ({BATCH} runs)")
        plt.title(f"Histogram Runs {start}–{end}")
        plt.tight_layout()
        fname = f"batch_hist_{start}_{end}_{first_title}__to__{last_title}.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

        batch_counts.clear()
        batch_titles.clear()

# ─── SAVE RUN SUMMARY ──────────────────────────────────────────────────────────
pd.DataFrame(summary).to_csv(
    os.path.join(output_dir, "nb_and_dispersion_summary.csv"),
    index=False
)
