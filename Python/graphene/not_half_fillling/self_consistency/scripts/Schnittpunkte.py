import matplotlib.pyplot as plt
import numpy as np
from graphene_mu import *

#Params
mu = 1 #t
E_D = mev2t(200)
U = np.linspace(mev2t(120e3), mev2t(200e3),10)
T = 18000 #K
rtol = 1e-3

fig, ax = plt.subplots()
delta_arr = np.linspace(0,2,1000)
for u in U:
    f = np.array([])
    for delta in delta_arr:
        f = np.append(f,integral(delta, u, T, E_D, mu, 10009))
    
    SP_mask = np.logical_and(delta_arr != 0, np.isclose(f, delta_arr, rtol=rtol))
    sp_text = "SP" if np.any(SP_mask) else "no SP"
    
    ax.plot(delta_arr, f, label=rf"$U = {t2mev(u)*1e-3:.2f}eV$, {sp_text}")
    ax.plot(delta_arr[SP_mask], f[SP_mask], "rx")

ax.plot(delta_arr,delta_arr, label=r"$\Delta$")
ax.set(
    xlabel=r"$\Delta / t$",
    ylabel=r"$g(\Delta) / t$",
    title=rf"$E_D$ = {E_D:.2f}t, $\mu$ = {mu:.2f}t, $T = {T:.0f} K$"
)
ax.grid()
ax.legend()
fig.savefig(f"../plots/schnittpunkte.pdf")