import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np


# ============================================================
# CSV-Dateien
# ============================================================

csv_main = "./data/TC_vs_mu&U_fine_8.csv"
csv_zoom = "./data/TC_vs_mu&U_fine_7.csv"


# ============================================================
# Daten einlesen
# ============================================================

df_main = pd.read_csv(csv_main)
df_zoom = pd.read_csv(csv_zoom)


# ============================================================
# Welche Spalten sollen geplottet werden?
# ============================================================

# ERSTE Spalte = x
x_main = pd.to_numeric(df_main.iloc[:, 0], errors="coerce")
x_zoom = pd.to_numeric(df_zoom.iloc[:, 0], errors="coerce")


# Nur die 5 Y-Spalten verwenden
# -> Spalten 2 bis 6 der CSV
y_columns_main = df_main.columns[1:6]
y_columns_zoom = df_zoom.columns[1:6]

print("Geplottete Linien:")
print(y_columns_main)


# ============================================================
# Figure + Hauptplot
# ============================================================

fig, ax = plt.subplots(figsize=(9, 6))


# ============================================================
# Hauptplot
# ============================================================

for col in y_columns_main:

    y = pd.to_numeric(df_main[col], errors="coerce")

    # Nur gültige Daten verwenden
    mask = np.isfinite(x_main) & np.isfinite(y)

    ax.plot(
        x_main[mask],
        y[mask],
        label=col
    )


ax.set_xlabel(df_main.columns[0])
ax.set_ylabel("y")
ax.grid(True)

ax.legend()


# ============================================================
# Zoom-Grenzen automatisch aus CSV 2 bestimmen
# ============================================================

# Gültige x-Werte
x_zoom_valid = x_zoom[np.isfinite(x_zoom)]

if len(x_zoom_valid) == 0:
    raise ValueError("Keine gültigen x-Werte in der Zoom-CSV gefunden.")


x_min = x_zoom_valid.min()
x_max = x_zoom_valid.max()


# Alle Y-Werte der 5 Zoom-Kurven sammeln
zoom_y_values = []

for col in y_columns_zoom:

    y = pd.to_numeric(df_zoom[col], errors="coerce")

    y = y[np.isfinite(y)]

    zoom_y_values.extend(y.values)


zoom_y_values = np.array(zoom_y_values)


if len(zoom_y_values) == 0:
    raise ValueError("Keine gültigen y-Werte in der Zoom-CSV gefunden.")


y_min = zoom_y_values.min()
y_max = zoom_y_values.max()


# ============================================================
# Rand um Zoom-Bereich
# ============================================================

x_range = x_max - x_min
y_range = y_max - y_min

x_margin = 0.05 * x_range
y_margin = 0.05 * y_range


# Falls die Daten nur einen einzelnen Wert enthalten
if x_margin == 0:
    x_margin = 1

if y_margin == 0:
    y_margin = 1


x_zoom_min = x_min - x_margin
x_zoom_max = x_max + x_margin

y_zoom_min = y_min - y_margin
y_zoom_max = y_max + y_margin


# ============================================================
# Inset
# ============================================================

axins = inset_axes(
    ax,
    width="40%",
    height="40%",
    loc="upper right"
)


# ============================================================
# Nur die 5 Zoom-Kurven plotten
# ============================================================

for col in y_columns_zoom:

    y = pd.to_numeric(df_zoom[col], errors="coerce")

    mask = np.isfinite(x_zoom) & np.isfinite(y)

    axins.plot(
        x_zoom[mask],
        y[mask]
    )


# Zoom-Grenzen
axins.set_xlim(
    x_zoom_min,
    x_zoom_max
)

axins.set_ylim(
    y_zoom_min,
    y_zoom_max
)


axins.grid(True)


# ============================================================
# Zoom-Bereich im Hauptplot markieren
# ============================================================

mark_inset(
    ax,
    axins,
    loc1=2,
    loc2=4,
    fc="none"
)


plt.tight_layout()
plt.show()