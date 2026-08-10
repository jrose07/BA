import matplotlib.pyplot as plt
from matplotlib import ticker
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
E_D = mev2t(200) # t
U = np.linspace(mev2t(0e3),mev2t(250e3),100) # t
mu = np.linspace(-3,3,100) # t

version = "1"

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
    vmin = np.nanmin(Delta0/6)
    vmax = np.nanmax(Delta0/6)
    levels = ticker.MaxNLocator(nbins=100, steps=[1, 2, 2.5, 4, 5, 10]).tick_values(vmin, vmax)
    fig, ax = plt.subplots()

    """Plot The underlying mesh"""
    # U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    # ax.plot(t2mev(U_b)*1e-3, mu_b, "b.")

    colorbar = ax.contourf(U/6, mu/6, Delta0/6, levels=levels, cmap='Spectral_r')
    cbar = fig.colorbar(colorbar, ax=ax, label=r"$\Delta_0 \, / \, [W]$")
    cbar.locator = ticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 4, 5, 10])
    cbar.update_ticks()
    # ax.set(
    #     xlabel=r"$U \, / \, [W]$",
    #     ylabel=r"$\mu \, / \, [W]$"
    # )

    #Critical U: 
    # E_D = E_D / 6 # W
    A = 0.184080       # 1/t^2
    A = A * 6**2        #1/W^2
    U_C = 1 / (A * (E_D/6))

    ax.axvline(
        U_C,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=5
    )

    ax.plot(
        U_C,
        0,
        marker="x",
        markersize=8,
        markeredgewidth=2,
        color="red",
        zorder=6
    )

    ax.annotate(
        r"$U_C$",
        xy=(U_C, 0),
        xytext=(7, 10),
        textcoords="offset points",
        color="red",
        fontsize=11,
        ha="left",
        va="bottom"
    )
    
    
    
    
    ax.set_xlabel(
    r"$U\,/\,[W]$",
    fontsize=12
    )

    ax.set_ylabel(
        r"$\mu\,/\,[W]$",
        fontsize=12
    )

    ax.set_xlim(
        np.min(U/6),
        np.max(U/6)
    )

    ax.set_ylim(
        np.min(mu/6),
        np.max(mu/6)
    )

    ax.tick_params(
        which="major",
        direction="in",
        length=5,
        width=0.9,
        labelsize=10,
        top=True,
        right=True
    )

    ax.tick_params(
        which="minor",
        direction="in",
        length=3,
        width=0.7,
        top=True,
        right=True
    )

    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

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
