import matplotlib.pyplot as plt
import numpy as np
from allgemein_mu import *
import scipy.constants as const


T = np.linspace(0, 240*2.7*6, 100)/2.7 #K -> K * t/eV
U = 20 # t
E_D = mev2t(200) # t
mu = np.linspace(0,0.7,5) # t
# mu = np.array([0.4,0.5])
safe = True
version="1"



fig, ax = plt.subplots()
for idx, m in enumerate(mu):
    deltas = np.array([])
    for elem in T:
            deltas = np.append(deltas, get_delta(U=U, T=elem, E_D=E_D, mu=m, iterations=5, start=1, num_points=10009)) # t
    
    #Determine T_C
    deltas = deltas/6 # W
    T_scaled = T/6 #W
    m_scaled = m/6 # W
    abs_deltas = np.abs(deltas)
    min_abs_delta = np.nanmin(abs_deltas)
    mask = np.isclose(abs_deltas, min_abs_delta, rtol=1e-10)
    if not np.any(mask):
        T_C = np.nan
    T_C = np.min(T_scaled[mask])
    color = f"C{idx % 10}"
    ax.plot(T_scaled, deltas, label=rf"$\mu = $ {m_scaled*1e3:.0f} mW $\,$ $T_C = {T_C:.0f}$" + r"$ \, K \cdot \frac{W}{eV}$", color=color)
    ax.plot(T_C, 0, "x", color=color)
    print(f"mu = {m_scaled}")
    print(deltas)
    print("next")

ax.set(
    ylabel=r"$\Delta \, / \, [W]$",
    xlabel=r"$T \, / \, [K \cdot \frac{W}{eV}]$", 
    title=rf"$U = {U/6:.1f} W \; E_D = {E_D/6*1e3:.0f} mW$"
)
ax.grid()
# sort legend entries by T_C (highest first)
# if 'legend_items' in locals() and len(legend_items) > 0:
#     # filter out nan T_C
#     legend_items = [it for it in legend_items if not np.isnan(it[0])]
#     legend_items.sort(key=lambda x: x[0], reverse=True)
#     handles = [it[1] for it in legend_items]
#     labels = [h.get_label() for h in handles]
#     ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
# else:
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
if safe == True:
    fig.savefig(f"../plots/Delta_vs_T&mu_{version}.pdf")
plt.show()