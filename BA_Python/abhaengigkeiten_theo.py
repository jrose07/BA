import matplotlib.pyplot as plt
import numpy as np
# from graphene import get_delta
from graphenemodeling.graphene import _constants as _c
# import pandas as pd
import scipy.constants as const
from scipy.optimize import newton

# def E_D_theo(U, T_C, A):
#     return 2*const.k*T_C*np.arccosh(np.exp(1/(2*U*A*const.k*T_C)))
t = _c.g0

def y_func(y, U, A, E_D):
    y = float(y)
    # if abs(y) < 1e-12:
    #     return -U * A * E_D
    logcosh = np.logaddexp(y, -y) - np.log(2) #Das Gleiche wie ln(cosh(y)), logaddexp(y,-y) = log(exp(y)+exp(-y))
    return logcosh / y - U * A * E_D


def solve_with_y(U, A, E_D, start):
    U_arr = np.asarray(U)
    E_D_arr = np.asarray(E_D)
    U_b, E_D_b = np.meshgrid(U_arr, E_D_arr, indexing='xy')
    C = U_b * A * E_D_b
    y = np.full(U_b.shape, np.nan, dtype=float)

    # If C is outside this scope, then there is no root to be found.
    valid = np.logical_and(C > -1, C < 1)

    def solve_scalar(u, e):
        try:
            return newton(y_func, args=(u, A, e), x0=start, maxiter=200, tol=1e-8)
        except (RuntimeError, OverflowError, ZeroDivisionError):
            return np.nan

    if np.any(valid):
        y[valid] = np.vectorize(solve_scalar, otypes=[float])(U_b[valid], E_D_b[valid])

    T_C = np.where(np.isfinite(y), E_D_b * t / (2 * const.k * y), np.nan)
    return T_C

A = 0.184058 #(1/t^2)
U = np.linspace(70,110,1000)
E_D = np.linspace(0.05,0.07,100)

T_C = solve_with_y(U, A, E_D, start=0.07*t/(2*const.k*10))

conv = _c.g0 / const.e * 1e3

levels = np.linspace(np.nanmin(T_C), np.nanmax(T_C), 100)
T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots()
colorbar = ax.contourf(U*conv*1e-3, E_D*conv, T_C_masked, levels=levels)
fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$E_D \, / \, meV$"
)
ax.set_facecolor(color='black')
fig.savefig("plots/T_C_theo.pdf")