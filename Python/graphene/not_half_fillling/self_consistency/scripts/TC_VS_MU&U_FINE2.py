import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphene_mu import *
from concurrent.futures import ProcessPoolExecutor

import time


# ============================================================
# Parameter
# ============================================================

E_D = mev2t(200)

U = np.linspace(
    0,
    mev2t(2e3),
    100
)

mu = np.array([
    mev2t(0.5e3),
    mev2t(0.6e3),
    mev2t(0.7e3),
    mev2t(0.8e3),
    mev2t(0.9e3),
    mev2t(1e3),
    # mev2t(2.5e3),
    # mev2t(2.6e3),
    # mev2t(2.7e3),
    # mev2t(2.8e3),
    # mev2t(2.9e3)
])

# Maximale Temperatur für den ERSTEN groben Scan
T_MAX = 0.1

# Anzahl grober Temperaturpunkte
N_COARSE = 50

# Anzahl Punkte für Verfeinerung des Tc-Intervalls
N_REFINE = 20

# Numerische Schwelle, ab der Delta als "0" betrachtet wird.
#
# Diese Zahl ggf. an die Größenordnung von Delta anpassen!
DELTA_TOL = 1e-7

# --------------------------------------------------------
# NEU: Adaptive Erweiterung des Temperaturbereichs
#
# Falls innerhalb von [0, T_max] kein Übergang gefunden wird,
# obwohl Delta am oberen Rand noch nicht verschwunden ist
# (T_C liegt also höher als der gescannte Bereich), wird
# T_max verdoppelt und neu gescannt. Das verhindert, dass
# Kurven bei hohem U einfach "abbrechen", nur weil T_C
# außerhalb von [0, T_MAX] liegt.
# --------------------------------------------------------

# Harte Obergrenze, damit die Erweiterung nicht unbegrenzt läuft
T_MAX_HARD_LIMIT = 200.0

# Wie oft T_max maximal verdoppelt werden darf
MAX_EXTENSIONS = 6

# Solver-Parameter
ITERATIONS = 4
NUM_POINTS = 1009
START = 1

version = "5"


# ============================================================
# Delta berechnen
# ============================================================

def calculate_delta(T, u, m):
    """
    Berechnet |Delta(T)| für einen einzelnen (U, mu)-Punkt.
    """

    try:
        delta = get_delta(
            start=START,
            T=T,
            U=u,
            E_D=E_D,
            mu=m,
            iterations=ITERATIONS,
            num_points=NUM_POINTS
        )

        delta = abs(delta)

        if not np.isfinite(delta):
            return np.nan

        return delta

    except (
        RuntimeError,
        OverflowError,
        ValueError,
        FloatingPointError
    ):
        return np.nan


# ============================================================
# Tc für einen einzelnen (U, mu)-Punkt
# ============================================================

def get_T_C_scalar(
    u,
    m,
    T_max=T_MAX,
    n_coarse=N_COARSE,
    n_refine=N_REFINE,
    delta_tol=DELTA_TOL,
    T_max_hard_limit=T_MAX_HARD_LIMIT,
    max_extensions=MAX_EXTENSIONS
):
    """
    Bestimmt Tc für einen einzelnen (U, mu)-Punkt.

    Vorgehen:
        1. Grobes Scanning über T (mit adaptiver Erweiterung
           von T_max, falls T_C außerhalb des Bereichs liegt)
        2. Übergangsintervall bestimmen
        3. Dieses Intervall verfeinern
        4. Tc interpolieren
    """

    current_T_max = T_max
    extension = 0

    # --------------------------------------------------------
    # 1. Coarse Graining (mit adaptiver Erweiterung)
    # --------------------------------------------------------

    while True:

        T_coarse = np.linspace(
            0,
            current_T_max,
            n_coarse
        )

        delta_coarse = np.array([
            calculate_delta(T, u, m)
            for T in T_coarse
        ])

        # Nur finite Werte betrachten
        valid = np.isfinite(delta_coarse)

        if not np.any(valid):
            return np.nan

        # ----------------------------------------------------
        # Falls Delta bereits bei T=0 verschwunden ist
        # ----------------------------------------------------

        if np.isfinite(delta_coarse[0]):

            if delta_coarse[0] < delta_tol:
                return np.nan

        # ----------------------------------------------------
        # 2. Übergangsintervall finden
        #
        # Wir suchen:
        #
        # Delta(T_i) >= delta_tol
        # Delta(T_i+1) < delta_tol
        # ----------------------------------------------------

        crossing = np.where(
            (delta_coarse[:-1] >= delta_tol) &
            (delta_coarse[1:] < delta_tol)
        )[0]

        if crossing.size > 0:
            # Übergang gefunden -> weiter mit Verfeinerung
            break

        # ----------------------------------------------------
        # Kein Übergang gefunden. Prüfen, ob Delta am oberen
        # Rand des gescannten Bereichs noch nicht verschwunden
        # ist -> T_C liegt vermutlich oberhalb von current_T_max.
        # ----------------------------------------------------

        last_valid_idx = np.where(valid)[0][-1]

        still_finite_at_edge = (
            delta_coarse[last_valid_idx] >= delta_tol
        )

        can_extend = (
            extension < max_extensions
            and current_T_max < T_max_hard_limit
        )

        if still_finite_at_edge and can_extend:

            extension += 1
            current_T_max = min(
                current_T_max * 2,
                T_max_hard_limit
            )
            continue

        # Kein Übergang innerhalb des (ggf. erweiterten)
        # Temperaturbereichs gefunden -> tatsächlich kein Tc
        return np.nan

    # --------------------------------------------------------
    # Erstes Auftreten des Übergangs nehmen
    # --------------------------------------------------------

    i = crossing[0]

    T_left = T_coarse[i]
    T_right = T_coarse[i + 1]

    # --------------------------------------------------------
    # 3. Intervall um Tc verfeinern
    # --------------------------------------------------------

    T_refine = np.linspace(
        T_left,
        T_right,
        n_refine
    )

    delta_refine = np.array([
        calculate_delta(T, u, m)
        for T in T_refine
    ])

    valid = np.isfinite(delta_refine)

    if not np.any(valid):
        return np.nan

    T_refine = T_refine[valid]
    delta_refine = delta_refine[valid]

    # --------------------------------------------------------
    # 4. Ersten Punkt unterhalb der Schwelle finden
    # --------------------------------------------------------

    below = delta_refine < delta_tol

    if not np.any(below):
        return np.nan

    j = np.argmax(below)

    # Falls schon der erste Punkt unterhalb der Schwelle liegt
    if j == 0:
        return T_refine[0]

    # --------------------------------------------------------
    # 5. Lineare Interpolation
    # --------------------------------------------------------

    T1 = T_refine[j - 1]
    T2 = T_refine[j]

    d1 = delta_refine[j - 1]
    d2 = delta_refine[j]

    if d2 == d1:
        return T2

    Tc = (
        T1
        + (delta_tol - d1)
        * (T2 - T1)
        / (d2 - d1)
    )

    return Tc


# ============================================================
# Worker für multiprocessing
# ============================================================

def worker_func(args):
    """
    Berechnet Tc für einen einzelnen (mu, U)-Punkt.
    """

    i, j, u, m = args

    Tc = get_T_C_scalar(
        u=u,
        m=m
    )

    return i, j, Tc


# ============================================================
# Hauptrechnung
# ============================================================

def main():

    t1 = time.perf_counter()

    # --------------------------------------------------------
    # Alle (mu, U)-Punkte als unabhängige Tasks
    # --------------------------------------------------------

    tasks = [
        (i, j, u, m)
        for i, m in enumerate(mu)
        for j, u in enumerate(U)
    ]

    T_C = np.full(
        (len(mu), len(U)),
        np.nan
    )

    # --------------------------------------------------------
    # Multiprocessing
    # --------------------------------------------------------

    nproc = min(
        8,
        len(tasks)
    )

    with ProcessPoolExecutor(
        max_workers=nproc
    ) as pool:

        for i, j, Tc in pool.map(
            worker_func,
            tasks
        ):

            T_C[i, j] = Tc

    # --------------------------------------------------------
    # Kontrolle
    # --------------------------------------------------------

    print(
        "NaNs:",
        np.any(np.isnan(T_C))
    )

    print(
        "max(T_C):",
        np.nanmax(T_C)
    )

    print(
        "min(T_C):",
        np.nanmin(T_C)
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    U_eV = t2mev(U) * 1e-3
    mu_eV = t2mev(mu) * 1e-3

    T_C_df = pd.DataFrame(
        T_C,
        index=mu_eV,
        columns=U_eV
    )

    T_C_df.index.name = r"$\mu / [eV]$"
    T_C_df.columns.name = r"$U / [eV]$"

    T_C_df.to_csv(
        f"./data/TC_vs_mu&U_fine_{version}.csv"
    )

    # --------------------------------------------------------
    # Laufzeit
    # --------------------------------------------------------

    t2 = time.perf_counter()

    dt = t2 - t1

    print(
        f"{np.floor(dt / 60)} mins "
        f"{((dt / 60 - np.floor(dt / 60)) * 60):.2f} sec"
    )

    # ========================================================
    # Plot
    # ========================================================

    fig, ax = plt.subplots()

    for i, mu_val in enumerate(mu):

        mu_eV_val = t2mev(mu_val) * 1e-3

        valid = np.isfinite(T_C[i])

        if np.any(valid):

            ax.plot(
                U_eV[valid],
                T_C[i][valid],
                label=rf"$\mu = {mu_eV_val:.2f}$ eV"
            )

    ax.set(
        xlabel=r"$U \, / \, [eV]$",
        ylabel=r"$T_C \, / \, [K]$",
        # ylim=[-0.1, 15.0]
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid()

    fig.tight_layout()

    fig.savefig(
        f"../plots/TC_vs_mu&U_fine_{version}.pdf"
    )

    plt.show()


# ============================================================
# Multiprocessing entry point
# ============================================================

if __name__ == "__main__":
    main()