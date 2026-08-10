from pathlib import Path
from allgemein_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

A = 0.184080 # 1/t^2
mu = t2mev(0.05)*1e-3 #eV
print(mu)
half_filling = True
# mu = t2mev(0.1)*1e-3 
# mu = t2mev(0.1)*1e-3
mu = 0
version = "_theo3"
safe = True
levels = 100

if half_filling == True:
    csv_path = Path(rf"/home/jrose/Dokumente/Uni/SoSe26/BA_Git/Python/graphene/half_filling/scripts/data/TC_vs_E_D&U{version}.csv")
else:
    csv_path = Path(rf"/home/jrose/Dokumente/Uni/SoSe26/BA_Git/Python/graphene/not_half_fillling/self_consistency/scripts/data/TC_vs_E_D&U{version}.csv")
    
data = pd.read_csv(csv_path, header=None)
U = pd.to_numeric(data.iloc[0, 1:], errors="coerce").to_numpy() #eV
E_D = pd.to_numeric(data.iloc[1:, 0], errors="coerce").to_numpy() #meV
T_C = data.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy() # K

"""Get everything into W-units"""
U = mev2t(U*1e3) / 6 #W
E_D = mev2t(E_D) / 6 #W
T_C = T_C/6 # K*W/eV
A = A*6**2 #1/W^2
mu = mev2t(mu*1e3) / 6 #W

if np.isnan(U).all():
    # Ensure U is a range if all values are NaN
    U = np.arange(T_C.shape[1])

T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots(layout='tight')

"""Plot The underlying mesh"""
# U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
# ax.plot(t2mev(U_b)*1e-3, mu_b, "b.")
"""
good cmaps: ['Spectral', 'jet', 'inferno', 'viridis', 'Spectral_r', ]
"""
zero_mask = np.ma.masked_where(T_C_masked != 0, T_C_masked)
positive_mask = np.ma.masked_where(T_C_masked <= 0, T_C_masked)
positive_values = positive_mask.compressed()

"""When T_C = 0 shouldnt be plotted"""
# levels = np.linspace(np.nanmin(positive_values), np.nanmax(positive_values), 100)
# colorbar = ax.contourf(U, t2mev(mu)*1e-3, positive_mask, levels=levels, cmap='Spectral_r')

"""WHen T_C = 0 should be plotted"""
# levels = np.linspace(np.nanmin(T_C_masked), np.nanmax(T_C_masked), levels)
vmin = np.nanmin(T_C)
vmax = np.nanmax(T_C)
levels = ticker.MaxNLocator(nbins=100, steps=[1, 2, 2.5, 4, 5, 10]).tick_values(vmin, vmax)
colorbar = ax.contourf(U, E_D, T_C_masked, levels=levels, cmap='Spectral_r')
cbar = fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, [K\cdot\frac{W}{eV}]$")
cbar.locator = ticker.MaxNLocator(nbins=10, steps=[1, 2, 2.5, 4, 5, 10])
cbar.update_ticks()


"""Plot U_C"""
# Compute U_C for each E_D safely and vectorized
E_D_arr = np.asarray(E_D)
U_C = np.full(E_D_arr.shape, 0, dtype=float)

# # valid where mu is nonzero and more than E_D (avoid divide-by-zero / log issues)
valid = (mu > E_D_arr)
if np.any(valid):
    e = E_D_arr[valid]
    # use m in the formula (was mistakenly using the full mu array inside the loop)
    with np.errstate(divide='ignore', invalid='ignore'):
        # arg = m / ((E_D - m) * (E_D + m) ** 2)
        arg = mu**2 /(mu**2 - e**2)
        # print(arg)
        Uvals = 2 / (A *mu* np.log(arg))
    # keep only finite results
    Uvals[~np.isfinite(Uvals)] = np.nan
    U_C[valid] = Uvals
    # U_C[~valid] = 0

valid_zero = mu == 0
if np.any(valid_zero):
    U_C[valid_zero] = 1/(A*E_D_arr[valid_zero])

# # plot only finite pairs
# mask = np.isfinite(U_C) & np.isfinite(E_D_arr)
# if np.any(mask):
    # ax.plot(U_C[mask], E_D_arr[mask])
# print(U_C)
ax.plot(U_C, E_D, "r-", label=r"$U_C$")


ax.set(
    xlabel=r"$U \, / \, [W]$",
    ylabel=r"$E_D \, / \, [W]$",
    xlim=[np.min(U),np.max(U)],
    title=rf"$\mu = {mu:.2f}W$"
)
# ax.set_facecolor(color='black')


ax.tick_params(
    which="major",
    direction="in",
    length=5,
    width=0.9,
    labelsize=10,
    top=True,
    right=True
)

ax.tick_params(
    which="minor",
    direction="in",
    length=3,
    width=0.7,
    top=True,
    right=True
)

ax.minorticks_on()

for spine in ax.spines.values():
    spine.set_linewidth(0.9)



ax.set_facecolor(color='#5C51A3')
ax.legend()
if safe == True:
    fig.savefig(f"../plots/TC_vs_E_D&U{version}.pdf")
plt.show()
