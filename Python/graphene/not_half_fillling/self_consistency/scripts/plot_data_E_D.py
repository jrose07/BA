from pathlib import Path
from graphene_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = 0.02525098 #1/eV**2
mu = t2mev(0.05)*1e-3 #eV
print(mu)

csv_path = Path(r"./data/TC_vs_E_D&U_3.csv")
data = pd.read_csv(csv_path, header=None)
U = pd.to_numeric(data.iloc[0, 1:], errors="coerce").to_numpy() #eV
E_D = pd.to_numeric(data.iloc[1:, 0], errors="coerce").to_numpy() #meV
T_C = data.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()

if np.isnan(U).all():
    # Ensure U is a range if all values are NaN
    U = np.arange(T_C.shape[1])

T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots()

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
levels = np.linspace(np.nanmin(T_C_masked), np.nanmax(T_C_masked), 100)
colorbar = ax.contourf(U, E_D, T_C_masked, levels=levels, cmap='inferno')


"""Plot U_C"""
# Compute U_C for each E_D safely and vectorized
E_D_arr = np.asarray(E_D) * 1e-3 #eV
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

# # plot only finite pairs
# mask = np.isfinite(U_C) & np.isfinite(E_D_arr)
# if np.any(mask):
    # ax.plot(U_C[mask], E_D_arr[mask])
# print(U_C)
ax.plot(U_C, E_D)


fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$E_D \, / \, meV$",
    xlim=[np.min(U),np.max(U)],
    title=rf"$\mu = {mu:.2f}$"
)
ax.set_facecolor(color='black')
# fig.savefig(f"../plots/TC_vs_mu&U_6.pdf")
plt.show()