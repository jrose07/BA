import matplotlib.pyplot as plt
import numpy as np
from graphene_mu import *

#Params
U = mev2t(2e3)
E_D = mev2t(200)

amnt = 2
mu = np.linspace(1,1.5,amnt)
T = 1000#K
rtol = 1e-3

fig, ax = plt.subplots()
delta_arr = np.linspace(0,mev2t(20),1000)
for m in mu:
    f = np.array([])
    for delta in delta_arr:
        f = np.append(f,integral(delta, U=U, T=T, E_D=E_D, mu=m, num_points=10009))
    
    SP_mask = np.logical_and(delta_arr != 0, np.isclose(f, delta_arr, rtol=rtol))
    if np.any(SP_mask):
        delta_SP = np.mean(delta_arr[SP_mask])
        f_SP = np.mean(f[SP_mask])
    else:
        delta_SP = np.nan
        f_SP = np.nan
    sp_text = "SP" if np.any(SP_mask) else "no SP"
    
    color = np.random.rand(3,)
    
    ax.plot(delta_arr, f, label=rf"$\mu = {m:.2f}t$, {sp_text}", zorder=2, color=color)
    ax.plot(delta_SP, f_SP, "x", color=color, zorder=10, markersize=10)

ax.plot(delta_arr,delta_arr, label=r"$\Delta$", zorder=2)
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$g(\Delta) / t$",
    title=rf"$E_D$ = {E_D:.2f}t, $U$ = {U:.2f}t, $T = {T:.0f} K$"
)
ax.grid()
ax.legend()
fig.savefig(f"../plots/schnittpunkte/mu_schnittpunkte_{T:.0f}K_{t2mev(U)*1e-3:.2f}eV_{t2mev(E_D):.0f}mev_{amnt}.pdf")