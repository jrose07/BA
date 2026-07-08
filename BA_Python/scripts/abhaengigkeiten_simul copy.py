import matplotlib.pyplot as plt
import numpy as np
from graphene import newton_func
# import pandas as pd
from scipy.optimize import newton
import scipy.integrate as integrate
import scipy.constants as const

# Schreibe eine Funktion die für ein T_array einfach die kritische Temperatur T_C (In Abhängigkeit von U,E_D, A) zurückgibt  

t = 2.7 * const.e #J (t = 2.7eV)

def get_T_C(U, E_D, T_array, start):
    U = np.asarray(U)
    E_D = np.asarray(E_D)
    T_array = np.asarray(T_array)
    U_b, E_D_b = np.meshgrid(U, E_D, indexing='xy')
    def T_C_scalar(u, e):
        deltas = []
        for T in T_array:
            try:
                delta = newton(newton_func, args=(u, T, e, 1009), x0=start, maxiter=200, tol=1e-8)
            except (RuntimeError, OverflowError, ValueError):
                delta = np.nan
            deltas.append(delta)
        deltas = np.asarray(deltas)
        abs_deltas = np.abs(deltas)
        min_abs_delta = np.nanmin(abs_deltas)
        mask = np.isclose(abs_deltas, min_abs_delta, rtol=1e-10)
        if not np.any(mask):
            T_C = np.nan
        T_C = np.min(T_array[mask])
        return T_C
    return np.vectorize(T_C_scalar, otypes=[float])(U_b, E_D_b)

from graphene import mev2t
U = np.linspace(75,110,10)
E_D = np.linspace(mev2t(150),mev2t(200),10)
T = np.linspace(1,700,100)

T_C = get_T_C(U,E_D, T, 0.5)
A = 0.184080
U_C = 1/(E_D*A)

conv = t / const.e * 1e3

levels = np.linspace(np.nanmin(T_C), np.nanmax(T_C), 100)
T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots()
colorbar = ax.contourf(U*conv*1e-3, E_D*conv, T_C_masked, levels=levels, cmap='inferno')
ax.plot(U_C*conv*1e-3, E_D*conv, label=r"$U_C$")
fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$E_D \, / \, meV$"
)
ax.set_facecolor(color='black')
ax.legend()
fig.savefig("../plots/T_C_with_delta_2.pdf")