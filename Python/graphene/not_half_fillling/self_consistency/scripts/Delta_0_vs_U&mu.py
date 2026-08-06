import matplotlib.pyplot as plt
import numpy as np
from graphene_mu import *
import scipy.constants as const

T = 0
# print(t2mev(mu))
E_D = mev2t(200) # t
version = "2"

U = np.linspace(mev2t(0e3), mev2t(2e3), 1000)
# , mev2t(180), mev2t(170),
mu = np.array([mev2t(2.6e3), mev2t(2.7e3), mev2t(2.8e3)]) # t

delta_max = np.array([])
fig, ax = plt.subplots(layout='tight')
for m in mu:
    deltas = np.array([])
    for elem in U:
            deltas = np.append(deltas, get_delta(U=elem, T=T, E_D=E_D, mu=m, iterations=5, start=1, num_points=10009))
    
    abs_deltas = abs(deltas)
    # min_abs_delta = np.nanmin(abs_deltas)
    mask = np.isclose(abs_deltas, 0, rtol=1e-6)
    mask = False
    step = (U.max() - U.min())/(len(U))
    color = np.random.rand(3,)
    if np.any(mask):
        # U_C = np.nan
        U_C = np.max(U[mask]) 
        lbl = rf"$\mu = {t2mev(m)*1e-3:.2f} eV, U_C={t2mev(U_C)*1e-3:.2f}eV$"
        line, = ax.plot(t2mev(deltas),t2mev(U)*1e-3, label=lbl, color=color)
        if 'legend_items' not in locals():
            legend_items = []
        legend_items.append((U_C, line))
    else:
        U_C = np.nan
        lbl = rf"$\mu = {t2mev(m)*1e-3:.2f} eV$"
        line, = ax.plot(t2mev(deltas),t2mev(U)*1e-3, label=lbl, color=color)
    delta_max = np.append(delta_max, np.max(deltas))
    print(U_C)


    mark, = ax.plot(0,t2mev(U_C)*1e-3, "x", color=color)
    
    # store plotted artists and their T_C for sorted legend
print(delta_max, np.max(delta_max))
delta_max = np.max(delta_max)
ax.set(
    xlabel=r"$\Delta_0 \, / \, meV$",
    ylabel=r"$U \, / \, eV$", 
    # xlim=[0, t2mev(delta_max)],
    ylim=[np.min(t2mev(U)*1e-3),np.max(t2mev(U)*1e-3)],
    # xscale='log',
    # yscale='log',
    # ylim = [0,1e-6],
    title=rf"$E_D = {t2mev(E_D):.0f} meV$"
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
fig.savefig(f"../plots/Delta0_vs_U_{version}.pdf")
# plt.show()