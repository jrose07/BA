# from graphenemodeling.graphene import _constants as _c
from graphene import get_delta
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B
import scipy.constants as const


t = 2.7*const.e

# #T_vs_Delta 
"""Delta eher 2-5meV, U -> Delta~10meV"""
"""T_C im Weak Coupling Limit"""

T = np.linspace(0, 100, 1000)  #K
U = 90
E_D = 0.07
deltas = np.array([])
for elem in T:
        deltas = np.append(deltas, get_delta(U=U, T=elem, E_debye=E_D, num_max=20, start=1, num_points=10009))


fig, ax = plt.subplots()
ax.plot(T, deltas, "b-",label="Verlauf")


# Theoretische Vorhersage von T_C.
delta_0 = deltas[0]
T_C_theo = 2*(delta_0 *t)/(k_B * 3.52)
print(f"T_C_Theo = {T_C_theo:.2f} K" )

# only determine and plot T_C from the curve if a zero (within tolerance) is found
mask = np.isclose(deltas, 0, atol=delta_0/200, rtol=1e-8)
if mask.any():
    T_C = np.min(T[mask])
    print(f"T_C anhand von Kurve: T_C = {T_C:.2f} K")
    ax.plot(T_C, 0, "rx", label=rf"$T_C = {T_C:.2f} K$")
else:
    print(rf"No T_C found from curve, $\Delta_min$ = {np.min(deltas):.2f} t")

ax.set(
    ylabel=r"$\Delta / t$",
    xlabel=r"$T [K]$", 
    title=rf"$U = {U}t, E_D = {E_D} t$"
)
ax.grid()
ax.legend()
fig.savefig(f"../plots/T_vs_Delta/U_{U:.0f}t_Debye_{E_D*100:.0f}mt.pdf")