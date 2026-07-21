import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants as const
from graphene_mu import *
from concurrent.futures import ProcessPoolExecutor

"""In diesem Skript ist die Idee herauszufinden, wie die nicht-analytische (denke ich) Abhägigkeit der DOS von mu sein wird.
Hierbei ist diesmal E_D eine Konstante und zwar die Größtmöglkiche (es geht darum was maximal in Graphen möglich ist -> E_D = 0.07t)"""
import time

def get_T_C(U, mu, E_D):
    U = np.asarray(U)
    mu = np.asarray(mu)
    U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    def T_C_scalar(u, m):
        rho_0 = func_DOS(m)
        gamma = 0.57721566490153286060651209008240243104215933593992 # Euler constant (Wikipedia)
        denom = rho_0 * u
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            T_C = 2*np.exp(gamma)/np.pi * (t2mev(E_D)*1e-3 * const.e) / const.k * np.exp(-1/denom)
            print(T_C) # E_D from t -> J because [k] = J/K
        # return T_C
        return T_C if np.isfinite(T_C) else np.nan
    return np.vectorize(T_C_scalar, otypes=[float])(U_b, mu_b)


"""Ab hier richtige Rechnung"""

#Params
E_D = mev2t(200)
U = np.linspace(0,mev2t(10e3),1000)
mu = np.linspace(0,3,1000)
# T = np.linspace(0,35e3,100)

version = 7

#Rechnung und plots
t1= time.perf_counter()

"""Do Multiprocessing with AI hallucinations"""

def worker_func(mu_chunk):
    # keep full U for each worker, split only along mu to ensure consistent second axis
    return get_T_C(U, mu_chunk, E_D)

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
T_C_df.index.name = r"$\mu / t$"
T_C_df.columns.name = r"$U / eV$"
T_C_df.to_csv(f"./TC_vs_mu&U_{version}.csv")



t2 = time.perf_counter()
dt = t2-t1
print(f"{np.floor((t2-t1)/60)} mins {((dt/60 - np.floor(dt/60))*60):.2f} sec")

levels = np.linspace(np.nanmin(T_C), np.nanmax(T_C), 100)
T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots()

print(np.nanmax(T_C), np.max(T_C))

"""Plot The underlying mesh"""
# U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
# ax.plot(t2mev(U_b)*1e-3, mu_b, "b.")

colorbar = ax.contourf(t2mev(U)*1e-3, mu, T_C, levels=levels, cmap='Spectral')
fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$\mu \, / \, t$"
)
ax.set_facecolor(color='black')
# ax.legend()
fig.savefig(f"../plots/TC_vs_mu&U_theo_{version}.pdf")



"""
Notes wegen runtime:
T array muss gar nicht so groß sein -> Macht einfach nur ungenauer ob man das T_C trifft oder nicht, aber für grobes Bild ist das egal.
Naja okay nvmd, das sagt ja auch aus wie detailed das bild sein kann.. 
Also so mindestens 50 sollte es schon sein 
für mu x U = 50 x 50 schon gutes Bild ->
num_points x 10 approx x1.5 runtime, aber das kann man gut sparen, ist nur für so nachkommastellen relevant bei T_C
"""
