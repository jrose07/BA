from graphenemodeling.graphene import monolayer as mlg
from graphenemodeling.graphene import _constants as _c
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

"""_c.g0 ist der tight-binding parameter t. Der muss an E heranmultipliziert werden, damit im FullTightBinding Modell die Energieabschätzungen zur Berechnung der elliptischen Intregrale gut funktionieren. Beim Plotten kann dann wieder durch geteilt werden -> Nur E im Verhältnis zu t plotten."""

t = 2.7*const.e
"""Die DOS geht von -3 bis 3t, siehe "The electronic properties of graphene"."""
E = np.linspace(-3, 3, 10001) *t

DOS = mlg.DensityOfStates(E, model="FullTightBinding")

fig, ax = plt.subplots()
ax.plot(E/t, DOS/np.max(DOS))
ax.grid()

plt.show()