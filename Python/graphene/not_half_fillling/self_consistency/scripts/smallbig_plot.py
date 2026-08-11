import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from graphene_mu import t2mev

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
    "axes.titlesize": 20,
})

cbar_labelsize = 15
# ============================================================
# Files
# ============================================================

file_large = "data/TC_vs_mu&U_6.csv"
file_zoom = "data/TC_vs_mu&U_7.csv"
version = "4"
critical = True

# ============================================================
# Load CSV files
# ============================================================

data_large = pd.read_csv(file_large)
data_zoom = pd.read_csv(file_zoom)


# ============================================================
# Read data
# ============================================================

mu_large = data_large.iloc[:, 0].to_numpy(float)
U_large = data_large.columns[1:].astype(float).to_numpy()
TC_large = data_large.iloc[:, 1:].to_numpy(float)

mu_zoom = data_zoom.iloc[:, 0].to_numpy(float)
U_zoom = data_zoom.columns[1:].astype(float).to_numpy()
TC_zoom = data_zoom.iloc[:, 1:].to_numpy(float)


# ============================================================
# Convert mu to eV
#
# t2mev() converts to meV
# 1e-3 converts meV -> eV
#
# U is intentionally left unchanged here.
# ============================================================

mu_large = t2mev(mu_large) * 1e-3
mu_zoom = t2mev(mu_zoom) * 1e-3


# ============================================================
# Make sure the data orientation is correct
#
# TC has shape:
#     (number of mu values, number of U values)
#
# imshow expects:
#     rows    -> y / mu
#     columns -> x / U
# ============================================================

if TC_large.shape != (len(mu_large), len(U_large)):
    raise ValueError(
        "Shape of TC_large does not match mu_large and U_large."
    )

if TC_zoom.shape != (len(mu_zoom), len(U_zoom)):
    raise ValueError(
        "Shape of TC_zoom does not match mu_zoom and U_zoom."
    )


# ============================================================
# Determine zoom region
# ============================================================

mu_min = mu_zoom.min()
mu_max = mu_zoom.max()

U_min = U_zoom.min()
U_max = U_zoom.max()


print("Zoom region:")
print(f"U  = {U_min:.6g} ... {U_max:.6g}")
print(f"mu = {mu_min:.6g} ... {mu_max:.6g}")


# ============================================================
# Independent color scales
# ============================================================

vmin_large = np.nanmin(TC_large)
vmax_large = np.nanmax(TC_large)

vmin_zoom = np.nanmin(TC_zoom)
vmax_zoom = np.nanmax(TC_zoom)


print("\nColor scale:")
print(f"Main: {vmin_large:.8g} ... {vmax_large:.8g}")
print(f"Zoom: {vmin_zoom:.8g} ... {vmax_zoom:.8g}")


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


# ============================================================
# MAIN PLOT
# ============================================================

im = ax.imshow(
    TC_large,
    origin="lower",
    aspect="auto",
    extent=[
        U_large.min(),
        U_large.max(),
        mu_large.min(),
        mu_large.max()
    ],
    cmap="Spectral_r",
    vmin=vmin_large,
    vmax=vmax_large,
    interpolation="bicubic",
    rasterized=True
)

if critical == True:
    # Critical
    E_D = 0.2 #eV
    A = 0.02525098 
    U_C = 1 / (A * E_D)

    ax.axvline(
        U_C,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=5
    )

    ax.plot(
        U_C,
        0,
        marker="x",
        markersize=8,
        markeredgewidth=2,
        color="red",
        zorder=6
    )

    ax.annotate(
        r"$U_C$",
        xy=(U_C, 0),
        xytext=(7, 10),
        textcoords="offset points",
        color="red",
        fontsize=11,
        ha="left",
        va="bottom"
    )




# ============================================================
# Main axis labels
# ============================================================

ax.set_xlabel(
    r"$U \,/\, [\mathrm{eV}]$"
)

ax.set_ylabel(
    r"$\mu \,/\, [\mathrm{eV}]$"
)


# ============================================================
# Main colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    r"$T_C \,/\, [\mathrm{K}]$",
    fontsize=cbar_labelsize
)


# ============================================================
# Rectangle around zoom region
# ============================================================

rect = Rectangle(
    (U_min, mu_min),
    U_max - U_min,
    mu_max - mu_min,
    fill=False,
    edgecolor="black",
    linewidth=1.5,
    zorder=10
)

ax.add_patch(rect)


# ============================================================
# ZOOM AXES
#
# [left, bottom, width, height]
# ============================================================

axins = fig.add_axes(
    [0.48, 0.58, 0.27, 0.25],
    zorder=20
)


# ============================================================
# ZOOM PLOT
# ============================================================

im_zoom = axins.imshow(
    TC_zoom,
    origin="lower",
    aspect="auto",
    extent=[
        U_zoom.min(),
        U_zoom.max(),
        mu_zoom.min(),
        mu_zoom.max()
    ],
    cmap="plasma",
    vmin=vmin_zoom,
    vmax=vmax_zoom,
    interpolation="bicubic",
    rasterized=True
)
if critical == True:
    U_C = 1 / (A * E_D)

    axins.axvline(
        U_C,
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=5
    )

    axins.plot(
        U_C,
        0,
        marker="x",
        markersize=8,
        markeredgewidth=2,
        color="red",
        zorder=6
    )

    axins.annotate(
        r"$U_C$",
        xy=(U_C, 0),
        xytext=(7, 10),
        textcoords="offset points",
        color="red",
        fontsize=11,
        ha="left",
        va="bottom"
    )


# ============================================================
# Zoom limits
# ============================================================

axins.set_xlim(
    U_min,
    U_max
)

axins.set_ylim(
    mu_min,
    mu_max
)


# ============================================================
# Zoom labels
# ============================================================

axins.set_xlabel(
    r"$U \,/\, [\mathrm{eV}]$",
    fontsize=9
)

axins.set_ylabel(
    r"$\mu \,/\, [\mathrm{eV}]$",
    fontsize=9
)


axins.tick_params(
    axis="both",
    which="major",
    labelsize=8
)


# ============================================================
# Zoom colorbar
# ============================================================

cbar_zoom = fig.colorbar(
    im_zoom,
    ax=axins,
    orientation="horizontal",
    pad=0.22,
    fraction=0.08,
    aspect=25
)

cbar_zoom.set_label(
    r"$T_C \,/\, [\mathrm{K}]$",
    fontsize=cbar_labelsize - 5
)

cbar_zoom.ax.tick_params(
    labelsize=7
)


# ============================================================
# Connection lines
# ============================================================

mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none",
    ec="black",
    linewidth=1.2
)


# ============================================================
# Save as PDF
# ============================================================

fig.savefig(
    f"../plots/nice_plots/Plot_zusammen{version}.pdf",
    bbox_inches="tight",
    dpi=300
)


# ============================================================
# Show
# ============================================================

# plt.show()

plt.close(fig)