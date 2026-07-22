from pathlib import Path
from graphene_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = 0.18
E_D = mev2t(200)

csv_path = Path(r"TC_vs_mu&U_19.csv")
data = pd.read_csv(csv_path, header=None)
U = pd.to_numeric(data.iloc[0, 1:], errors="coerce").to_numpy() #eV
mu = pd.to_numeric(data.iloc[1:, 0], errors="coerce").to_numpy()
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
colorbar = ax.contourf(U, t2mev(mu)*1e-3, T_C_masked, levels=levels, cmap='inferno')


"""Plot U_C"""
# Compute U_C for each mu safely and vectorized
mu_arr = np.asarray(mu)
U_C = np.full(mu_arr.shape, 0, dtype=float)

# valid where mu is nonzero and more than E_D (avoid divide-by-zero / log issues)
valid = (mu_arr != 0) & (mu_arr > E_D)
if np.any(valid):
    m = mu_arr[valid]
    # use m in the formula (was mistakenly using the full mu array inside the loop)
    with np.errstate(divide='ignore', invalid='ignore'):
        # arg = m / ((E_D - m) * (E_D + m) ** 2)
        arg = m**2 /(m**2 - E_D**2)
        Uvals = 2 / (A *m * np.log(arg))
    # keep only finite results
    Uvals[~np.isfinite(Uvals)] = np.nan
    U_C[valid] = Uvals

# plot only finite pairs
mask = np.isfinite(U_C) & np.isfinite(mu_arr)
if np.any(mask):
    ax.plot(U_C[mask], mu_arr[mask])


fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$\mu \, / \, eV$",
    xlim=[np.min(U),np.max(U)]
)
ax.set_facecolor(color='black')
# fig.savefig(f"../plots/TC_vs_mu&U_6.pdf")
plt.show()