import numpy as np
import matplotlib.pyplot as plt

A = 0.184080 #1/t^2
E_D = 0.07 # t
U_C = 1/(A*E_D) #t
print(U_C)
U = np.linspace(1,3*U_C,1000)
delta_0 =np.clip((E_D/2 * (U/U_C - U_C/U)),a_min=-0.2, a_max=np.inf)

fig, ax = plt.subplots()
ax.plot(U, delta_0, label=r"$\Delta_0 (U)$")
ax.vlines(U_C, ymin=-0.04, ymax=0.04, color="green",linestyles="dotted", label=rf"$U_C = {U_C:.2f}t$")
ax.grid()
ax.set(
    xlabel=r"$U \, / \, t$",
    ylabel=r"$\Delta_0 \, / \, t$",
    title=rf"$E_D = {E_D:.2f}t$"
)
ax.legend()
fig.savefig("../plots/delta01D.pdf")


#Jetzt wie bei T_C nur mit 2D Plot (variates E_D und U):
# Hier natürlich T = 0 (delta_0)

def get_delta_0(U, E_D):
    U_C = 1/(A*E_D)
    return E_D/2 * (U/U_C - U_C/U)

from graphene import mev2t, t2mev

U = np.linspace(75,110,100) #t
E_D = np.linspace(mev2t(150), mev2t(200),100)
U_b, E_D_b = np.meshgrid(U, E_D, indexing='xy')
U_C = 1/(A*E_D)
delta_0 = get_delta_0(U_b, E_D_b)
delta_0 = t2mev(delta_0)
# delta_0 = np.where(delta_0<0, 0, delta_0)
print(delta_0)
levels = np.append([np.min(delta_0)], np.linspace(0,np.max(delta_0), 100))
# levels = np.linspace(np.min(delta_0), np.max(delta_0), 100)
fig, ax = plt.subplots()
ax.plot(t2mev(U_C)*1e-3, t2mev(E_D), label=r"$U_C$")
colorbar = ax.contourf(t2mev(U)*1e-3, t2mev(E_D), delta_0, levels=levels, cmap='inferno')
fig.colorbar(colorbar, ax=ax, label=r"$\Delta_0 \, / \, meV$")
ax.set(
    xlabel=r"$U \, / \, eV$",
    ylabel=r"$E_D \, / \, meV$",
    xlim=[np.min(t2mev(U)*1e-3), np.max(t2mev(U)*1e-3)]
)
ax.legend()
ax.set_facecolor(color='black')
fig.savefig("../plots/delta02D.pdf")