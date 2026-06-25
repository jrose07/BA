from graphenemodeling.graphene import _constants as _c
from graphene import get_delta, integral
import numpy as np
import matplotlib.pyplot as plt

E = np.linspace(-3,3,10001)
U = 3
T = 1
E_D = 2

delta_arr = np.linspace(0,10,100)
f = np.array([])
for delta in delta_arr:
    f = np.append(f,integral(delta, E, U, T, E_D))
    
fig, ax = plt.subplots()
ax.plot(delta_arr, f, label=r"$\int \rho(E) \cdot f(E, \Delta) \mathrm{d} E$")
ax.plot(delta_arr,delta_arr, label=r"$\Delta$")
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$g(\Delta) / t$"
)
ax.grid()
ax.legend()
fig.savefig("plots/schnittpunkte.pdf")