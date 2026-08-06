# from graphenemodeling.graphene import _constants as _c
from graphene_mu import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as k_B
import scipy.constants as const
from scipy.stats import linregress
from uncertainties import ufloat, std_dev as stds, nominal_value as noms

t = 2.7 # eV

conv = t /const.e * 1e3 
E_D = mev2t(200) # t
mu = mev2t(0.5e3) #t

def dos_regress_single(mu, E_D):
    E = np.linspace(mu-E_D, mu+E_D, 10001) # in t
    DOS = func_DOS(E)
    # Lineare Regression des DOS in diesem Limit:
    result = linregress(E[E>0], DOS[E>0])
    m = ufloat(result.slope, result.stderr)
    b = ufloat(result.intercept, result.intercept_stderr)

    # perr = np.sqrt(result.stderr) * 100
    
    y_predict = noms(m)*E+noms(b)
    mape = np.mean(np.abs((DOS - y_predict) / DOS)) * 100
    relative_error = np.mean(np.abs(y_predict - DOS)) / (DOS.max() - DOS.min()) * 100
# print(f"MAPE = {mape:.2f}%")
    # print(m, b)
    # fig, ax = plt.subplots()
    # ax.plot(E, DOS, label=rf"$\mu = {mu:.2f}t$")
    # ax.plot(E[E>0], noms(m)*E[E>0]+noms(b), label=f"Lineare Regression {mape:.2f} %")
    # # ax.plot(E[E<0], -noms(m)*E[E<0] + noms(b), label=f"Lineare Regression negativ")
    # ax.legend()
    # ax.grid()
    # ax.set(
    #     xlabel=r"$E / t$",
    #     ylabel=r"$\rho(E) \, [1/t]$"
    # )
    # plt.show()
    return mape, relative_error

mu = np.linspace(E_D,3,1000)
mapes = []
relerrors = []
for m in mu:
    mape, relerror = dos_regress_single(m, E_D)
    mapes.append(mape)
    relerrors.append(relerror)

fig, ax = plt.subplots(layout='tight')
ax.plot(mu, mapes, label="MAPE")
ax.plot(mu, relerrors, label="Relative Errors")
ax.set(
    xlabel=r"$\mu \, / \, t$",
    ylabel=r"Error $\, / \,$ %"
)
ax.grid()
ax.legend()
plt.show()
