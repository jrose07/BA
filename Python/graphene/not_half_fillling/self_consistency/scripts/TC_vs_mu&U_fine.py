import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from graphene_mu import *
from concurrent.futures import ProcessPoolExecutor


import time

"""Es soll TC in abhängigkeit von mu und U ohne contour plotten."""


def get_T_C(U, mu, E_D, T_array, start):
    U = np.asarray(U)
    mu = np.asarray(mu)
    T_array = np.asarray(T_array)
    U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    def T_C_scalar(u, m):
        deltas = []
        for T in T_array:
            try:
                delta = get_delta(start=start, T = T, U = u, E_D = E_D, mu = m, iterations=4, num_points=1009)
            except (RuntimeError, OverflowError, ValueError):
                delta = np.nan
            deltas.append(delta)
        deltas = np.asarray(deltas)
        abs_deltas = np.abs(deltas)
        # handle all-NaN or empty cases robustly
        if abs_deltas.size == 0 or np.all(np.isnan(abs_deltas)):
            return np.nan
        try:
            min_abs_delta = np.nanmin(abs_deltas)
        except (ValueError, FloatingPointError):
            return np.nan
        mask = np.isclose(abs_deltas, min_abs_delta, rtol=1e-12)
        T_candidates = T_array[mask]
        if T_candidates.size == 0:
            return np.nan
        T_C = np.min(T_candidates)
        # print(T_C_theo, T_C)
        return T_C
    return np.vectorize(T_C_scalar, otypes=[float])(U_b, mu_b)


"""Ab hier richtige Rechnung"""

#Params
E_D = mev2t(200)
U = np.linspace(0,mev2t(2e3),100)
mu = np.array([mev2t(2.6e3), mev2t(2.7e3), mev2t(2.8e3)]) # t
# mu = np.linspace(mev2t(0.4e3), mev2t(2.8e3), 5)
T = np.linspace(0,100,1000)

version = "3"

#Rechnung und plots
t1= time.perf_counter()

"""Do Multiprocessing with AI hallucinations"""

def worker_func(mu_chunk):
    # keep full U for each worker, split only along mu to ensure consistent second axis
    return get_T_C(U, mu_chunk, E_D, T, 1)

def main():
    # choose number of processes not exceeding number of mu chunks
    nproc = 8
    mu_chunks = np.array_split(mu, nproc)
    nproc = min(nproc, len(mu_chunks))
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        results = pool.map(worker_func, mu_chunks)

    # concatenate along the mu axis (axis=0) to reconstruct full grid
    T_C = np.concatenate(list(results), axis=0)

    print(np.any(np.isnan(T_C)))

    T_C_df = pd.DataFrame(T_C, index=mu, columns=t2mev(U)*1e-3)
    T_C_df.index.name = r"$\mu / [t]$"
    T_C_df.columns.name = r"$U / [eV]$"
    T_C_df.to_csv(f"./data/TC_vs_mu&U_fine_{version}.csv")



    t2 = time.perf_counter()
    dt = t2-t1
    print(f"{np.floor((t2-t1)/60)} mins {((dt/60 - np.floor(dt/60))*60):.2f} sec")

    fig, ax = plt.subplots()

    print(np.nanmax(T_C), np.max(T_C))

    U_eV = t2mev(U)*1e-3
    for i, mu_val in enumerate(mu):
        mu_eV = t2mev(mu_val)*1e-3
        valid = np.isfinite(T_C[i])
        if np.any(valid):
            ax.plot(U_eV[valid], T_C[i][valid], label=rf"$\mu = {mu_eV:.2f}$ eV")

    ax.set(
        xlabel=r"$U \, / \, [eV]$",
        ylabel=r"$T_C \, / \, [K]$"
    )
    # ax.set_facecolor(color='#5C51A3')
    ax.legend()
    ax.grid()
    fig.savefig(f"../plots/TC_vs_mu&U_fine_{version}.pdf")

if __name__ == "__main__":
    main()


"""
Notes wegen runtime:
T array muss gar nicht so groß sein -> Macht einfach nur ungenauer ob man das T_C trifft oder nicht, aber für grobes Bild ist das egal.
Naja okay nvmd, das sagt ja auch aus wie detailed das bild sein kann.. 
Also so mindestens 50 sollte es schon sein 
für mu x U = 50 x 50 schon gutes Bild ->
num_points x 10 approx x1.5 runtime, aber das kann man gut sparen, ist nur für so nachkommastellen relevant bei T_C
"""
