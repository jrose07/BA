import numpy as np
import matplotlib.pyplot as plt

A = 0.18 #1/t^2
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
fig.savefig("plots/delta0.pdf")