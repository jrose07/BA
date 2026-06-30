from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B


# E = np.linspace(-3,3,10001)

# #U_vs_Delta

U = np.linspace(0,10,100)
T = 0
E_D = 1
deltas = np.array([])
for elem in U:
    deltas = np.append(deltas, get_delta(U=elem, T=T, E_debye=E_D, num_max=200, start=1, num_points=108))

fig, ax = plt.subplots()
ax.plot(U,deltas)
ax.set(
    ylabel=r"$\Delta / t$",
    xlabel=r"$U / t$",
    title=rf"$T={T}K, E_D = {E_D}t$"
)
ax.grid()
fig.savefig("plots/U_vs_delta.pdf")
