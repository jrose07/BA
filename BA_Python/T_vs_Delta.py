from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B

t = _c.g0

# #T_vs_Delta 
"""Delta eher 2-5meV, U -> Delta~10meV"""
"""T_C im Weak Coupling Limit"""

T = np.linspace(0, 1000, 100)  #K
U = 30
E_D = 0.1
deltas = np.array([])
for elem in T:
        deltas = np.append(deltas, get_delta(U=U, T=elem, E_debye=E_D, num_max=200, start=1, num_points=1009))


fig, ax = plt.subplots()
ax.plot(T, deltas, "b-",label="Verlauf")


# Theoretische Vorhersage von T_C.
delta_0 = deltas[0]
T_C_theo = 2*(delta_0 *t)/(k_B * 3.52)
print(f"T_C_Theo = {T_C_theo:.2f} K" )

T_C = np.min(T[np.isclose(deltas,0, atol=delta_0/100)])
print(f"T_C anhand von Kurve: T_C = {T_C:.2f} K")
ax.plot(T_C, 0, "rx", label=rf"$T_C = {T_C:.2f} K$")



ax.set(
    ylabel=r"$\Delta / t$",
    xlabel=r"$T [K]$", 
    title=rf"$U = {U}t, E_D = {E_D} t$"
)
ax.grid()
ax.legend()
fig.savefig("plots/T_vs_delta.pdf")