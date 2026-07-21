import matplotlib.pyplot as plt
import numpy as np
from graphene_mu import *
import scipy.constants as const


T = np.linspace(0, 400, 1000)  #K
U = mev2t(200e3)
E_D = mev2t(200)
# mu = np.linspace(0,0.2,5)
mu = np.linspace(0,0.2,10)
# mu = 1
fig, ax = plt.subplots()
for m in mu:
    deltas = np.array([])
    for elem in T:
            deltas = np.append(deltas, get_delta(U=U, T=elem, E_D=E_D, mu=m, iterations=5, start=1, num_points=10009))
    
    #Determine T_C
    abs_deltas = np.abs(deltas)
    min_abs_delta = np.nanmin(abs_deltas)
    mask = np.isclose(abs_deltas, min_abs_delta, rtol=1e-10)
    if not np.any(mask):
        T_C = np.nan
    T_C = np.min(T[mask])
    color = np.random.rand(3,)
    line, = ax.plot(T, t2mev(deltas)*1e-3, label=rf"$\mu = $ {m:.2f} t $\,$ T_C = {T_C:.2g} K", color=color)
    mark, = ax.plot(T_C, 0, "x", color=color)
    
    # store plotted artists and their T_C for sorted legend
    if 'legend_items' not in locals():
        legend_items = []
    legend_items.append((T_C, line))

ax.set(
    ylabel=r"$\Delta / eV$",
    xlabel=r"$T [K]$", 
    title=rf"$U = {t2mev(U)*1e-3:.0f}eV \; E_D = {t2mev(E_D):.0f} meV$"
)
ax.grid()
# sort legend entries by T_C (highest first)
if 'legend_items' in locals() and len(legend_items) > 0:
    # filter out nan T_C
    legend_items = [it for it in legend_items if not np.isnan(it[0])]
    legend_items.sort(key=lambda x: x[0], reverse=True)
    handles = [it[1] for it in legend_items]
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
else:
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"../plots/U_{t2mev(U)*1e-3:.0f}ev_Debye_{E_D*100:.0f}mev.pdf")