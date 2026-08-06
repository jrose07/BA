from pathlib import Path
from allgemein_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = 0.184080 # 1/t^2
E_D = mev2t(200) #t

version = "6"
safe = True

csv_path = Path(rf"/home/jrose/Dokumente/Uni/SoSe26/BA_Git/Python/graphene/not_half_fillling/self_consistency/scripts/data/TC_vs_mu&U_{version}.csv")
data = pd.read_csv(csv_path, header=None)
U = pd.to_numeric(data.iloc[0, 1:], errors="coerce").to_numpy() #eV
mu = pd.to_numeric(data.iloc[1:, 0], errors="coerce").to_numpy() # in t
# mu = t2mev(mu)*1e-3 # t -> eV
U = mev2t(U*1e3) #t
T_C = data.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy() # K 
T_C = T_C / 2.7 #K -> K*t/eV

"""Now we go from 1/t -> 1/W mit Bandwith W = 6t"""
U = U/6 #W
mu = mu/6 #W
T_C = T_C/6 #K*W/ev
A = A*(6**2) #1/W**2
E_D = E_D / 6 #W

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
colorbar = ax.contourf(U, mu, T_C_masked, levels=levels, cmap='Spectral_r')


"""Plot U_C"""
# # Compute U_C for each mu safely and vectorized
# mu_arr = np.linspace(mu.min(), mu.max(), 1000)
# U_C = np.full(mu_arr.shape, 0, dtype=float)

# # valid where mu is nonzero and more than E_D (avoid divide-by-zero / log issues)
# valid = (mu_arr != 0) & (mu_arr > E_D) 
# valid = np.logical_or(valid, mu_arr < -E_D)
# if np.any(valid):
#     m = mu_arr[valid]
#     # use m in the formula (was mistakenly using the full mu array inside the loop)
#     with np.errstate(divide='ignore', invalid='ignore'):
#         arg = m**2 /(m**2 - E_D**2)
#         Uvals = 2 / (A *np.abs(m) * np.log(arg))
#     # keep only finite results
#     Uvals[~np.isfinite(Uvals)] = np.nan
#     U_C[valid] = Uvals
    
# mask = np.logical_and(np.isfinite(U_C), 0==0)
# if np.any(mask):
#     ax.plot(U_C[mask], mu_arr[mask], "r-", label=r"$U_C$")

# mask_zero = mu_arr == 0
# if np.any(mask_zero):
#     U_C[mask_zero] = 1/(A*E_D)

# # print(U_C, mu_arr)
# # plot only finite pairs
# if np.any(mask_zero):
#     ax.hlines(0,xmin=0,xmax=U_C[mask_zero], color="red")
    # ax.plot(U_C[mask_zero], 0, "r.")
ax.plot(1/(A*E_D), 0, "r.", label=r"$U_C$")
# ax.vlines(1/(A*E_D), ymin=0.01*mu.min(), ymax=0.01*mu.max(),color="red", label=r"$U_C$")
ax.legend()

fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, [K\cdot \frac{W}{eV}]$")
ax.set(
    xlabel=r"$U \, / \, W$",
    ylabel=r"$\mu \, / \, W$",
    xlim=[np.min(U),np.max(U)]
)
# ax.set_facecolor(color='black')
ax.set_facecolor(color='#5C51A3')

# ax.legend()
if safe == True:
    fig.savefig(f"../plots/TC_vs_mu&U_{version}.pdf")
plt.show()




"""Mache noch einen Plot für ein bestimmtes mu."""

fig, ax = plt.subplots()
num = 10
mu_targets = [np.percentile(mu, i * 100 / num) for i in range(num)] # W
mu_targets = np.array([0,0.04,0.05, 0.06, 0.1])/6 #t -> W
mu_targets = np.array([-0.4, -0.2, -1/6, 0, 1/6, 0.2, 0.4])
# mu_targets = np.array([])

for m in mu_targets:
    mask = np.abs(mu - m) == np.min(np.abs(mu - m))
    T_C_mu = T_C[mask][0]
    T_C_mu[np.isnan(T_C_mu)] = 0
    U_grenz = np.max(U[np.isclose(T_C_mu, 0)])
    line, = ax.plot(U, T_C_mu, label=rf"$\mu = {m*1e3:.1f}$ mW, $U_G= ${U_grenz:.2f} W")
    ax.plot(U_grenz, 0, marker="x", color=line.get_color())

ax.set(
    xlabel=r"$U \, / \, W$",
    ylabel=r"$T_C \; / \; [K\cdot \frac{W}{eV}]$",
    title=r"Selected $T_C(U)$ for different $\mu$",
)
ax.grid()
ax.legend()
plt.show()