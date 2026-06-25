from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt

E = np.linspace(-3,3,10001)

U = np.linspace(0,20,20)
E_D = np.linspace(0.1,20,20)

deltas = np.zeros((len(E_D), len(U)))
for i, u in enumerate(U):
    for j, e_d in enumerate(E_D):
        deltas[j, i] = get_delta(u, T=1, E=E, E_debye=e_d, n_fixpunkt=20, start=1)
        print(f"{(i*len(E_D) + j)/(len(E_D)*len(U))*100} % finished")

U, E_D = np.meshgrid(U, E_D)
levels= np.linspace(deltas.min(), deltas.max(), 100)


fig, ax = plt.subplots()
# contourf_ = ax.contourf(U, E_D, deltas, levels=levels)
# cbar =  fig.colorbar(contourf_, label=r"$\D'elta / t$")
mesh_ = ax.imshow(deltas, extent=[U.min(), U.max(), E_D.min(), E_D.max()], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(mesh_, label=r"$\Delta / t$")
ax.set(
    xlabel=r"$U / t$",
    ylabel=r"$E_D / t$"
)

fig.savefig("plots/U_Debye_vs_Delta.pdf")