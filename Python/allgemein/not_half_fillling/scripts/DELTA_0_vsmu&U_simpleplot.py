import matplotlib.pyplot as plt
import numpy as np
from allgemein_mu import *


# T = 3.5 #K
T = 0 #K
E_D = mev2t(200) # t
version = "0"

U = np.linspace(mev2t(0e3), mev2t(300e3), 100)
# , mev2t(180), mev2t(170),
# mu = np.array([mev2t(2.6e3), mev2t(2.7e3), mev2t(2.8e3)]) # t
mu = np.array([mev2t(0e3), mev2t(0.1e3), mev2t(0.2e3), mev2t(0.3e3)]) # t


fig, ax = plt.subplots(layout='tight')
for m in mu:
    deltas = np.array([])
    for elem in U:
        deltas = np.append(deltas, get_delta(U=elem, T=T, E_D=E_D, mu=m, iterations=5, start=1, num_points=10009))

    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][len(ax.lines) % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
    lbl = rf"$\mu = {m/6*1e3:.0f} mW$"
    ax.plot(U/6, deltas/6, label=lbl, color=color)

        # Store the mu = 0 curve
    if m == 0:
        delta_mu0 = deltas.copy()
    

    if delta_mu0 is not None:

        tol = 1e-10

        # First point where Delta becomes non-zero
        positive_indices = np.where(delta_mu0 > tol)[0]

        if len(positive_indices) > 0:
            idx_c = positive_indices[0]
            U_c = U[idx_c]

            # ax.axvline(
            #     U_c / 6,
            #     linestyle='--',
            #     color='black',
            #     label=rf"$U_C = {U_c/6:.3g}\,\mathrm{{W}}$"
            # )

            ax.annotate(
            rf"$U_C = {U_c/6:.3g}\,\mathrm{{W}}$",
            xy=(U_c / 6, 0),
            xytext=(20, 5),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->'),
            )
            
            ax.scatter(U_c / 6, 0, color='black', zorder=5)

            print(f"Critical coupling U_C = {U_c}")
            print(f"Critical coupling U_C / 6 = {U_c/6} W")




# delta_max = np.max(delta_max)
ax.set(
    ylabel=r"$\Delta_0 \, / \, [W]$",
    xlabel=r"$U \, / \, [W]$", 
    # xlim=[0, t2mev(delta_max)],
    # ylim=[np.min(t2mev(U)*1e-3),np.max(t2mev(U)*1e-3)],
    # xlim=[-0.1,10],
    # xscale='log',
    # yscale='log',
    # ylim = [0,1e-6],
    title=rf"$E_D = {E_D/6*1e3:.0f} mW$"
)
ax.grid()
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"../plots/Delta_vs_U&mu_{version}.pdf", bbox_inches='tight')
# plt.show()