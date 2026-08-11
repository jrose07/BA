from pathlib import Path
from allgemein_mu import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.labelsize": 15,
    "axes.titlesize": 20,
})

# ============================================================
# Parameters
# ============================================================

A = 0.184080       # 1/t^2
E_D = mev2t(200)   # t

version = "6"
safe = True
critical = True
inferno = False


# ============================================================
# Load data
# ============================================================

csv_path = Path(
    rf"/home/jrose/Dokumente/Uni/SoSe26/BA_Git/Python/graphene/"
    rf"not_half_fillling/self_consistency/scripts/data/"
    rf"TC_vs_mu&U_{version}.csv"
)

data = pd.read_csv(csv_path, header=None)

U = pd.to_numeric(
    data.iloc[0, 1:], errors="coerce"
).to_numpy()  # eV

mu = pd.to_numeric(
    data.iloc[1:, 0], errors="coerce"
).to_numpy()  # t

T_C = data.iloc[1:, 1:].apply(
    pd.to_numeric, errors="coerce"
).to_numpy()  # K


# ============================================================
# Unit conversions
# ============================================================

# U: eV -> t
U = mev2t(U * 1e3)

# T_C: K -> K*t/eV
T_C = T_C / 2.7


# Now go from 1/t -> 1/W with bandwidth W = 6t

U = U / 6       # W
mu = mu / 6     # W
T_C = T_C / 6   # K*W/eV

A = A * 6**2    # 1/W^2
E_D = E_D / 6   # W


# ============================================================
# Safety fallback
# ============================================================

if np.isnan(U).all():
    U = np.arange(T_C.shape[1])


# ============================================================
# Prepare data
# ============================================================

T_C_masked = np.ma.masked_invalid(T_C)

fig, ax = plt.subplots(
    # figsize=(7.0, 5.2),
    layout='tight'
    # constrained_layout=True
)

# Colormap
if inferno:
    cmap = "inferno"
else:
    cmap = "Spectral_r"

# Background for invalid values
ax.set_facecolor("0.85")


# ============================================================
# Main plot
# ============================================================

im = ax.imshow(
    T_C_masked,
    extent=[
        U.min(),
        U.max(),
        mu.min(),
        mu.max()
    ],
    origin="lower",
    aspect="auto",
    cmap=cmap,
    interpolation="bicubic"
)


# ============================================================
# Critical interaction U_C
# ============================================================

if critical:

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
# Colorbar
# ============================================================

cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.025,
    fraction=0.046
)

cbar.set_label(
    r"$T_C\,/\,[K\cdot W/eV]$",
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
    r"$U\,/\,[W]$",
    fontsize=12
)

ax.set_ylabel(
    r"$\mu\,/\,[W]$",
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

ax.set_title(
    rf"$E_D = {E_D:.3f}W$"
)
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



for spine in ax.spines.values():
    spine.set_linewidth(0.9)


# ============================================================
# Save / show
# ============================================================

if safe:
    output_path = Path(
        f"../plots/TC_vs_mu&U_{version}.pdf"
    )

    fig.savefig(
        output_path,
        # bbox_inches="tight"
    )

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