from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt

E = np.linspace(-3,3,10001)

# #U_vs_Delta

U = np.linspace(0,5,100)
deltas = np.array([])
for elem in U:
    deltas = np.append(deltas, get_delta(elem, T=1, E=E, E_debye=3, n_fixpunkt=200, start=1 ))

fig, ax = plt.subplots()
ax.plot(deltas, U)
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$U / t$"
)
ax.grid()
fig.savefig("plots/U_vs_delta.pdf")


# #T_vs_Delta
T = np.linspace(0.1, 100000, 100)  #K
deltas = np.array([])
for elem in T:
        deltas = np.append(deltas, get_delta(U=3, T=elem, E=E, E_debye=3, n_fixpunkt=200, start=1 ))

fig, ax = plt.subplots()
ax.plot(deltas, T)
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$T [K]$"
)
ax.grid()
fig.savefig("plots/T_vs_delta.pdf")


# #E_Debye vs Delta
E_debye = np.linspace(0.1, 3, 100)  #K
U = 6
T= 1
deltas = np.array([])
for elem in E_debye:
        deltas = np.append(deltas, get_delta(U=U, T=T, E=E, E_debye=elem, n_fixpunkt=200, start=1 ))

fig, ax = plt.subplots()
ax.plot(deltas, E_debye)
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$E_\text{debye} / t$",
    title=rf"$U = {U}t, T={T}K$"
)
ax.grid()
fig.savefig("plots/Debye_vs_delta.pdf")