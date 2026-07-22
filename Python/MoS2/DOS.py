import matplotlib.pyplot as plt
import numpy as np

E, DOS = np.genfromtxt("./DOS_3.csv", delimiter=",",skip_header=1, unpack=True)
E = E + 1
E_D = 200e-3 #eV

E_before=E
DOS_before=DOS
cutoff_before = np.logical_and(E<E_D, -E_D < E)
E_unique = np.unique(E)
DOS_mean = np.array([DOS[E == e].mean() for e in E_unique])

E = E_unique
DOS = DOS_mean

cutoff = np.logical_and(E < E_D, -E_D < E)
E = E[cutoff]
DOS = DOS[cutoff]
E_old = E
DOS_old = DOS

# Interpolate DOS on a finer energy grid
E_fine = np.linspace(E.min(), E.max(), 100000)
DOS_interp = np.interp(E_fine, E, DOS)

# use interpolated arrays from here on
E = E_fine
DOS = DOS_interp
print(len(DOS))

plt.plot(E_old, DOS_old, "g.")
plt.plot(E_before[cutoff_before], DOS_before[cutoff_before], "b.")
plt.plot(E, DOS, "r-")
plt.show()