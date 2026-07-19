from pathlib import Path
from graphene_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


csv_path = Path(r"TC_vs_mu&U_7.csv")
data = pd.read_csv(csv_path, header=None)
U = pd.to_numeric(data.iloc[0, 1:], errors="coerce").to_numpy() #eV
mu = pd.to_numeric(data.iloc[1:, 0], errors="coerce").to_numpy()
T_C = data.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy()

if np.isnan(U).all():
    U = np.arange(T_C.shape[1])

print(np.nanmax(T_C), np.nanmax(T_C))

levels = np.linspace(np.nanmin(T_C), np.nanmax(T_C), 100)
T_C_masked = np.ma.masked_invalid(T_C)
fig, ax = plt.subplots()

print(np.nanmax(T_C), np.max(T_C))

"""Plot The underlying mesh"""
# U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
# ax.plot(t2mev(U_b)*1e-3, mu_b, "b.")

colorbar = ax.contourf(U, mu, T_C_masked, levels=levels, cmap='jet')
# ax.plot(U_C*conv*1e-3, E_D*conv, label=r"$U_C$")
fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$\mu \, / \, t$"
)
ax.set_facecolor(color='black')
# ax.legend()
# fig.savefig(f"../plots/TC_vs_mu&U_6.pdf")
plt.show()