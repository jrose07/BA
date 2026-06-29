from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B




# #E_Debye vs Delta
E_debye = np.linspace(0.1, 3, 1000)  #K
U = 3
T= 0
deltas = np.array([])
for elem in E_debye:
        deltas = np.append(deltas, get_delta(U=U, T=T, E_debye=elem, num_max=200, start=1, num_points=1011 ))

fig, ax = plt.subplots()
ax.plot(E_debye, deltas)
ax.set(
    ylabel=r"$\Delta / t$",
    xlabel=r"$E_\text{debye} / t$",
    title=rf"$U = {U}t, T={T}K$"
)
ax.grid()
fig.savefig("plots/Debye_vs_delta.pdf")