import matplotlib.pyplot as plt
import numpy as np


plt.style.use('_mpl-gallery')
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
})

t= 1 
a = 1 # m

def get_eps(k_x, k_y):
    return t * np.sqrt(3 + 2*np.cos(np.sqrt(3)*k_y*a) + 4*np.cos(np.sqrt(3)*k_y * a/2) * np.cos(3*k_x * a/2)  )

k_max = 2*np.pi/(3*a) * 1.5

k_x = np.linspace(-k_max, k_max, 100)
k_y = np.linspace(-k_max,k_max,100)
k_x, k_y = np.meshgrid(k_x, k_y)

epsp = get_eps(k_x, k_y)
epsm = -epsp

# Make data
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface
cmap = "viridis"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surface = ax.plot_surface(
    k_x,
    k_y,
    epsp,
    vmin=epsm.min(),
    vmax=epsp.max(),
    cmap=cmap,
    linewidth=0,
    antialiased=True,
    alpha=0.96,
)
ax.plot_surface(
    k_x,
    k_y,
    epsm,
    vmin=epsm.min(),
    vmax=epsp.max(),
    cmap=cmap,
    linewidth=0,
    antialiased=True,
    alpha=0.96,
)
fig.colorbar(surface, ax=ax)
ax.set(
    xlabel=r"$k_x \, / \, a.u.$",
    ylabel=r"$k_y \, / \, a.u.$",
    zlabel=r"$\epsilon_\pm(\vec{k}) \, / \, a.u.$",
    # title="Band structure",
)
ax.view_init(elev=28, azim=-55)
ax.grid(False)

fig.tight_layout()

# fig.savefig("Baenderplot.pdf", bbox_inches="tight", dpi=300)
plt.show()