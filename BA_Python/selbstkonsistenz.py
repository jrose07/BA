from graphenemodeling.graphene import monolayer as mlg
from graphenemodeling.graphene import _constants as _c
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_b #In J/K
import scipy.integrate as integrate
# from functools import partial


t = _c.g0 # in J
E = np.linspace(-3, 3, 10001) *t
DOS = mlg.DensityOfStates(E, model="FullTightBinding")

def func(E):
    return mlg.DensityOfStates(E, model="FullTightBinding")

"""To perform the integral over the DOS lets use Integrating using samples by scipy.integrate.simpson
It needs an uneven amount of points where to integrate between.
"""


def fixpunkt_alogorithmus_0K(start, U, num_max):
    deltas = np.array([start])
    # x = E
    for i in range(num_max):
        # print(deltas)
        delta_next = U*integrate.simpson(func(E)*deltas[i]/np.sqrt(deltas[i]**2 + E**2), x=E)
        # delta_next = U*integrate.quad(func, -3, 3)[0]
        deltas = np.append(deltas, delta_next)
        # print(i, deltas[i], delta_next)
    return deltas

def fixpunkt_algorithmus(start, T, U ,num_max, E_debye):
    mask = np.logical_and(E< np.abs(E_debye), E> - np.abs(E_debye))
    x = E[mask]
    deltas = np.array([start])
    for i in range(num_max):
        delta_next = U*integrate.simpson(func(x)*deltas[i]/np.sqrt(deltas[i]**2 + x**2) * np.tanh(np.sqrt(deltas[i]**2 + x**2)/(2*k_b * T * t)), x=x)
        deltas = np.append(deltas, delta_next)
        # print(i, deltas[i], delta_next)
    return deltas

""" Also E = {-3,3}*t = {-3,3} * 4.49e-19 J, so not in units of t but J.
    Delta as well.
    -> kb * T muss nicht noch * t sein, weil ansonsten in unit J**2
"""

deltas = fixpunkt_algorithmus(1, 1, 1e-19*t, 200, 0.5*t)
index = np.arange(0,len(deltas))
plt.plot(index[1:], deltas[1:], "r.")
plt.savefig("Deltas.pdf")

print(deltas[-1])

# Plot Delta(delta) vs. integral(delta)

delta_linspace=np.linspace(0, 50, 1000)*t
U = 1e-18*t
E_debye = 1*t
mask = np.logical_and(E< np.abs(E_debye), E> - np.abs(E_debye))
x = E[mask]
T = 1e# K
DOS = mlg.DensityOfStates(x, model='FullTightBinding')
y = np.array([])
for delta in delta_linspace:
    print(np.sqrt(delta**2 + E[0]**2)/(2*k_b * T ), np.sqrt(E[0]**2 + delta**2))
    y = np.append(y, U*integrate.simpson(DOS*delta/np.sqrt(delta**2 + x**2)* np.tanh(np.sqrt(delta**2 + x**2)/(2*k_b * T)), x=x))

fig, ax = plt.subplots()
ax.plot(delta_linspace/t, delta_linspace/t, label=r"$\Delta$") # in units of t
ax.plot(delta_linspace/t, y/t, label=r"$\int \rho(E)\cdot f(\Delta)$") # in units of t? 
ax.set_title(f"U = {U/t} t")
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$f(\Delta) / t$"
)
ax.grid()
ax.legend()

fig.savefig("Delta_vs_func.pdf")