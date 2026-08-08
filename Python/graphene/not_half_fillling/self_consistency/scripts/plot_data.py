from pathlib import Path
from graphene_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Parameters
# ============================================================

A = 0.02525098       # 1/eV**2
E_D = 200e-3         # eV

print(f"U_C = {1/(A*E_D)} eV")

version = "2"
safe = True


# ============================================================
# Load data
# ============================================================

csv_path = Path(
    rf"./data/TC_vs_mu&U_{version}.csv"
)

data = pd.read_csv(
    csv_path,
    header=None
)

U = pd.to_numeric(
    data.iloc[0, 1:],
    errors="coerce"
).to_numpy()  # eV

mu = pd.to_numeric(
    data.iloc[1:, 0],
    errors="coerce"
).to_numpy()  # t

mu = t2mev(mu) * 1e-3  # t -> eV

T_C = data.iloc[1:, 1:].apply(
    pd.to_numeric,
    errors="coerce"
).to_numpy()  # K


# ============================================================
# Safety fallback
# ============================================================

if np.isnan(U).all():
    U = np.arange(T_C.shape[1])


# ============================================================
# Mask invalid values
# ============================================================

T_C_masked = np.ma.masked_invalid(T_C)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.0, 5.2),
    layout="tight"
)


# ============================================================
# Colormap / background
# ============================================================

cmap = "Spectral_r"

# Color used for masked / invalid regions
ax.set_facecolor("#5C51A3")


# ============================================================
# Main imshow plot
# ============================================================

image = ax.imshow(
    T_C_masked,
    extent=[
        np.min(U),
        np.max(U),
        np.min(mu),
        np.max(mu)
    ],
    origin="lower",
    aspect="auto",
    cmap=cmap,
    interpolation="bicubic"
)


# ============================================================
# Colorbar
# ============================================================

cbar = fig.colorbar(
    image,
    ax=ax,
    pad=0.025,
    fraction=0.046
)

cbar.set_label(
    r"$T_C\, / \, [K]$",
    fontsize=12
)

cbar.ax.tick_params(
    labelsize=10,
    direction="in"
)


# ============================================================
# Axes
# ============================================================

ax.set_xlabel(
    r"$U\, / \,[eV]$",
    fontsize=12
)

ax.set_ylabel(
    r"$\mu\, / \,[eV]$",
    fontsize=12
)

ax.set_xlim(
    np.min(U),
    np.max(U)
)

ax.set_ylim(
    np.min(mu),
    np.max(mu)
)


# ============================================================
# Tick appearance
# ============================================================

ax.tick_params(
    which="major",
    direction="in",
    length=5,
    width=0.9,
    labelsize=10,
    top=True,
    right=True
)

ax.tick_params(
    which="minor",
    direction="in",
    length=3,
    width=0.7,
    top=True,
    right=True
)

ax.minorticks_on()


# ============================================================
# Spines
# ============================================================

for spine in ax.spines.values():
    spine.set_linewidth(0.9)


# ============================================================
# Save / show
# ============================================================

if safe:
    fig.savefig(
        f"../plots/nice_plots/TC_vs_mu&U_{version}.pdf",
        bbox_inches="tight"
    )

plt.show()