# from graphenemodeling.graphene import _constants as _c
from graphene import t2mev, mev2t, func_DOS
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B
import scipy.constants as const
from scipy.stats import linregress
from uncertainties import ufloat, std_dev as stds, nominal_value as noms

t = 2.7 # eV
print(t2mev(1)*1e-3)
conv = t /const.e * 1e3 
E_D = 6*20*1e-3 # t
E = np.linspace(-E_D, E_D, 10001) # in t

DOS = func_DOS(E)

#In W
E = E/6 #W
DOS = DOS*6 #1/W

# Lineare Regression des DOS in diesem Limit:
result = linregress(E[E>0], DOS[E>0])
m = ufloat(result.slope, result.stderr)
b = ufloat(result.intercept, result.intercept_stderr)

# print(m, b)

fig, ax = plt.subplots()
ax.plot(E, DOS, label=rf"DOS")
ax.plot(E[E>0], noms(m)*E[E>0]+noms(b), "r-", label=f"Lineare Regression")
ax.plot(E[E<0], -noms(m)*E[E<0] + noms(b), "r-")
ax.legend()
ax.grid()
ax.set(
    xlabel=r"$E \, / \, [W]$",
    ylabel=r"DOS $ \, / \, [1/W]$",
    title=rf"$E_D = {E_D/6*1e3:.2f} mW$"
)

fig.savefig("../plots/DOS.pdf")


print(f"Wie man sieht ist das DOS in dem Debye-Frequenz-Bereich komplett linear.\n")
print(f"Damit gilt DOS(epsilon) = abs( {m:.6f} 1/t^2 * epsilon + {b:.2f} 1/t")


#Dasselbe nochmal mit eV:
E_D = mev2t(200) # t
E = np.linspace(-E_D, E_D, 10001) # in t

DOS = func_DOS(E) # in 1/t

#Umwandeln in eV Größen
E = t2mev(E)  #eV
DOS = DOS / (t2mev(1)) #1/eV

result = linregress(E[E>0], DOS[E>0])
m = ufloat(result.slope, result.stderr)
b = ufloat(result.intercept, result.intercept_stderr)

fig, ax = plt.subplots()
ax.plot(E, DOS, label=rf"$E_D = {t2mev(E_D):.2f}meV$")
ax.plot(E[E>0], noms(m)*E[E>0]+noms(b), label=f"Lineare Regression")
ax.plot(E[E<0], -noms(m)*E[E<0] + noms(b), label=f"Lineare Regression negativ")
ax.legend()
ax.grid()
ax.set(
    xlabel=r"$E \, / \, [meV]$",
    ylabel=r"DOS $\, / \, [1/meV]$"
)

fig.savefig("../plots/DOS_meV.pdf")


print(f"Wie man sieht ist das DOS in dem Debye-Frequenz-Bereich komplett linear.\n")
print(f"Damit gilt DOS(epsilon) = abs( {m} 1/(meV**2) * epsilon + {b} 1/meV bzw. A = m = {m/(1e-3)**2} 1/eV**2")