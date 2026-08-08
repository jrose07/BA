import matplotlib.pyplot as plt
import numpy as np
from graphene_mu import *
import scipy.constants as const


T = 3.5 #K
E_D = mev2t(200) # t
version = "0"

U = np.linspace(mev2t(0e3), mev2t(22e3), 100)
# , mev2t(180), mev2t(170),
# mu = np.array([mev2t(2.6e3), mev2t(2.7e3), mev2t(2.8e3)]) # t
mu = np.array([mev2t(0.6e3), mev2t(0.8e3), mev2t(1e3)]) # t
mu = np.linspace(mev2t(0.3e3), mev2t(3e3),10)

delta_max = np.array([])
fig, ax = plt.subplots(layout='tight')
for m in mu:
    deltas = np.array([])
    for elem in U:
            deltas = np.append(deltas, get_delta(U=elem, T=T, E_D=E_D, mu=m, iterations=5, start=1, num_points=10009))
    
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(delta_max) % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
    delta_max = np.append(delta_max, np.max(deltas))

    """Comparison to the Li decorated Graphene Measurement Delta = 0.9meV -> Corresponding U"""
    
    U_fine = np.linspace(U.min(), U.max(), 1000000)
    delta_spl = np.interp(U_fine, U, deltas)

    U_9mev = np.mean(U_fine[np.isclose(t2mev(delta_spl), 0.9, rtol=1e-3)])
    ax.plot(0.9,t2mev(U_9mev)*1e-3, marker="x", color=color)
    if np.isnan(U_9mev):
        lbl = rf"$\mu = {t2mev(m)*1e-3:.2f} eV$"
    else:
        lbl = rf"$\mu = {t2mev(m)*1e-3:.2f} eV, U(0.9 meV) = {t2mev(U_9mev)*1e-3:.2f} eV$"
    line, = ax.plot(t2mev(deltas),t2mev(U)*1e-3, label=lbl, color=color)
    
print(delta_max, np.max(delta_max))

ax.vlines(0.9, ymin=0,ymax=np.max(t2mev(U)*1e-3))


delta_max = np.max(delta_max)
ax.set(
    xlabel=r"$\Delta \, / \, meV$",
    ylabel=r"$U \, / \, eV$", 
    # xlim=[0, t2mev(delta_max)],
    ylim=[np.min(t2mev(U)*1e-3),np.max(t2mev(U)*1e-3)],
    xlim=[-0.1,10],
    # xscale='log',
    # yscale='log',
    # ylim = [0,1e-6],
    title=rf"$E_D = {t2mev(E_D):.0f} meV, T = {T:.1f}K$"
)
ax.grid()
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"../plots/Delta_vs_U&mu_{version}.pdf")
# plt.show()