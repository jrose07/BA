import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from allgemein_mu import *
from concurrent.futures import ProcessPoolExecutor

"""In diesem Skript ist die Idee herauszufinden, wie die nicht-analytische (denke ich) Abhägigkeit der DOS von mu sein wird.
Hierbei ist diesmal E_D eine Konstante und zwar die Größtmöglkiche (es geht darum was maximal in Graphen möglich ist -> E_D = 0.07t)"""
import time


def get_Delta_0(U, mu, E_D, start):
    U = np.asarray(U)
    mu = np.asarray(mu)
    U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    def Delta_0_scalar(u, m):
        try:
            delta0 = get_delta(start=start, T = 0, U = u, E_D = E_D, mu = m, iterations=5, num_points=10009)
        except (RuntimeError, OverflowError, ValueError):
            delta0 = np.nan
        return delta0
    return np.vectorize(Delta_0_scalar, otypes=[float])(U_b, mu_b)

"""Ab hier richtige Rechnung"""

#Params
E_D = mev2t(200)
U = np.linspace(mev2t(0e3),mev2t(250e3),100) # t
mu = np.linspace(-3,3,100) # t

version = "0"

#Rechnung und plots
t1= time.perf_counter()

"""Do Multiprocessing with AI hallucinations"""

def worker_func(mu_chunk):
    # keep full U for each worker, split only along mu to ensure consistent second axis
    return get_Delta_0(U, mu_chunk, E_D, 1)

def main():
    
    # choose number of processes not exceeding number of mu chunks
    nproc = 8
    mu_chunks = np.array_split(mu, nproc)
    nproc = min(nproc, len(mu_chunks))
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        results = pool.map(worker_func, mu_chunks)

    # concatenate along the mu axis (axis=0) to reconstruct full grid
    Delta0 = np.concatenate(list(results), axis=0)

    print(np.any(np.isnan(Delta0)))
    
    Delta0_df = pd.DataFrame(Delta0/6, index=mu/6, columns=U/6) # meV
    Delta0_df.index.name = r"$\mu / [W]$"
    Delta0_df.columns.name = r"$U / [W]$"
    Delta0_df.to_csv(f"./data/DELTA0_vs_mu&U_{version}.csv")



    t2 = time.perf_counter()
    dt = t2-t1
    print(f"{np.floor((t2-t1)/60)} mins {((dt/60 - np.floor(dt/60))*60):.2f} sec")
    print(np.nanmin(Delta0/6), np.nanmax(Delta0/6))
    levels = np.linspace(np.nanmin(Delta0/6), np.nanmax(Delta0/6), 100)
    fig, ax = plt.subplots()

    """Plot The underlying mesh"""
    # U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    # ax.plot(t2mev(U_b)*1e-3, mu_b, "b.")

    colorbar = ax.contourf(U/6, mu/6, Delta0/6, levels=levels, cmap='Spectral_r')
    fig.colorbar(colorbar, ax=ax, label=r"$\Delta_0 \, / \, [W]$")
    ax.set(
        xlabel=r"$U \, / \, [W]$",
        ylabel=r"$\mu \, / \, [W]$"
    )
    # ax.set_facecolor(color='black')
    ax.set_facecolor(color='#5C51A3')
    fig.savefig(f"../plots/DELTA0_vs_mu&U_{version}.pdf")

if __name__ == "__main__":
    main()


"""
Notes wegen runtime:
T array muss gar nicht so groß sein -> Macht einfach nur ungenauer ob man das Delta0 trifft oder nicht, aber für grobes Bild ist das egal.
Naja okay nvmd, das sagt ja auch aus wie detailed das bild sein kann.. 
Also so mindestens 50 sollte es schon sein 
für mu x U = 50 x 50 schon gutes Bild ->
num_points x 10 approx x1.5 runtime, aber das kann man gut sparen, ist nur für so nachkommastellen relevant bei Delta0
"""
