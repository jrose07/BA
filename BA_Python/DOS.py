from graphenemodeling.graphene import _constants as _c
from graphene import get_delta, func_DOS
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B
import scipy.constants as const
from scipy.stats import linregress
from uncertainties import ufloat, std_dev as stds, nominal_value as noms

conv = _c.g0 /const.e * 1e3 
t = _c.g0 / const.e
E_D = 200*1e-3/ t # t
E = np.linspace(-E_D, E_D, 10001) # in t

DOS = func_DOS(E)
# Lineare Regression des DOS in diesem Limit:
result = linregress(E[E>0], DOS[E>0])
m = ufloat(result.slope, result.stderr)
b = ufloat(result.intercept, result.intercept_stderr)

# print(m, b)

fig, ax = plt.subplots()
ax.plot(E, DOS, label=f"E_D = {E_D:.2f}t")
ax.plot(E[E>0], noms(m)*E[E>0]+noms(b), label=f"Lineare Regression")
ax.plot(E[E<0], -noms(m)*E[E<0] + noms(b), label=f"Lineare Regression negativ")
ax.legend()
ax.grid()
ax.set(
    xlabel=r"$E / t$",
    ylabel=r"$\rho(E) \, [1/t]$"
)

fig.savefig("plots/DOS.pdf")


print(f"Wie man sieht ist das DOS in dem Debye-Frequenz-Bereich komplett linear.\n")
print(f"Damit gilt DOS(epsilon) = abs( {m:.6f} 1/t^2 * epsilon + {b:.2f} 1/t")


#Dasselbe nochmal mit eV:

m = m/(conv**2)
b = b/(conv)

fig, ax = plt.subplots()
ax.plot(E*conv, DOS/conv, label=f"E_D = {E_D*conv:.2f}meV")
ax.plot(E[E>0]*conv, noms(m)*E[E>0]*conv+noms(b), label=f"Lineare Regression")
ax.plot(E[E<0]*conv, -noms(m)*E[E<0]*conv + noms(b), label=f"Lineare Regression negativ")
ax.legend()
ax.grid()
ax.set(
    xlabel=r"$E / meV$",
    ylabel=r"$\rho(E) \, [1/meV]$"
)

fig.savefig("plots/DOS_meV.pdf")


print(f"Wie man sieht ist das DOS in dem Debye-Frequenz-Bereich komplett linear.\n")
print(f"Damit gilt DOS(epsilon) = abs( {m} 1/(meV**2) * epsilon + {b} 1/meV")