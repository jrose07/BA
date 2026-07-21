from pathlib import Path
from graphene_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


csv_path = Path(r"TC_vs_mu&U_12.csv")
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
colorbar = ax.contourf(U, t2mev(mu)*1e-3, T_C_masked, levels=levels, cmap='Spectral_r')

fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$\mu \, / \, eV$"
)
ax.set_facecolor(color='black')
# fig.savefig(f"../plots/TC_vs_mu&U_6.pdf")
plt.show()