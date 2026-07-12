import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
# from graphenemodeling.graphene import _constants as _c
import scipy.constants as const

t = 2.7*const.e #J
E_D = 0.07*t

def U_min(T):
    A = 0.18  #1/t^2
    x = 2*const.k * T * A * np.log(np.cosh(E_D/(2*const.k*T)))
    # return 1/x #t^2/J 
    return t/x #t

T = np.linspace(0,2.5,1000)
fig, ax = plt.subplots()
ax.plot(T, U_min(T))
ax.grid()
ax.set(
    xlabel=r"$T\, / \,K$",
    ylabel=r"$U_{\text{\min}} \, / \,t$"
)
fig.savefig("plots/U_min.pdf")


print(f"Hieran kann man erkennen, dass für erst bei einer kritischen Temperatur von ca. 1.5 K ein endliches U für eine Lösung von Delta verantwortlich ist, die anfängt ungleich null zu sein. Allerdings ist hierfür ein U von 80t also ca. 160eV nötig, was nicht realistisch ist.")