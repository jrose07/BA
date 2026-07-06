from graphenemodeling.graphene import _constants as _c
from graphene import get_delta, integral
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# E = np.linspace(-3,3,10001)
U = 79
rtol = 1e-4
T = 2 #K
E_D = 0.07#t

delta_arr = np.linspace(0,0.005,1000)
f = np.array([])
for delta in delta_arr:
    f = np.append(f,integral(delta, U, T, E_D))

#Schnittpunkt 
SP_mask = np.isclose(f,delta_arr, rtol=rtol)
# print(SP_mask)


fig, ax = plt.subplots()
ax.plot(delta_arr, f,label=r"$\int \rho(E) \cdot f(E, \Delta) \mathrm{d} E$")
ax.plot(delta_arr,delta_arr, label=r"$\Delta$")
ax.plot(delta_arr[SP_mask], f[SP_mask], "rx", label="Schnittpunkte")
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$g(\Delta) / t$"
)
ax.grid()
ax.legend()
fig.savefig(f"plots/schnittpunkte/schnittpunkte_U{U:.0f}t.pdf")

ax.cla()
conv = _c.g0 /const.e * 1e3 #from [t] -> [meV]
ax.plot(delta_arr*conv, f*conv,label=r"$\int \rho(E) \cdot f(E, \Delta) \mathrm{d} E$")
ax.plot(delta_arr*conv,delta_arr*conv, label=r"$\Delta$")
ax.plot(delta_arr[SP_mask]*conv, f[SP_mask]*conv, "rx", label="Schnittpunkte")
ax.set(
    xlabel=r"$\Delta / [meV]$",
    ylabel=r"$g(\Delta) / [meV]$",
    title=f"U = {U:.2f}t = {U*conv/1e3:.2f}eV"
)
ax.grid()
ax.legend()
fig.savefig(f"plots/schnittpunkte/schnittpunkte_U{U*conv/1e3:.0f}ev.pdf")

#Schnittpunkte_delta
delta_SP = delta_arr[SP_mask]
notnullmask = delta_SP != 0
delta_SP = np.mean(delta_SP[notnullmask])

print(f"Schnittpunkte sind bei Delta = {delta_SP*conv:.2f} meV")