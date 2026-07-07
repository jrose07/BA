import matplotlib.pyplot as plt
import numpy as np
from graphene import get_delta
from graphenemodeling.graphene import _constants as _c
import pandas as pd

#Schreibe eine Funktion die für ein T_array einfach die kritische Temperatur T_C (In Abhängigkeit von U,E_D, A) zurückgibt

def get_T_C(U, E_D, T_array):
    deltas = np.array([])
    for elem in T_array:
        deltas = np.append(deltas, get_delta(U=U, T=elem, E_debye=E_D, num_max=20, start=1, num_points=10009))
    
    # Suche T_C:
    rtol = 1e-4
    atol = 1e-8
    mask = np.isclose(deltas, 0, rtol=rtol, atol=atol)
    print(np.min(deltas))
    if not np.any(mask):
        T_C = np.nan
        # atol *= 1.5
        # mask = np.isclose(deltas, 0, rtol=rtol, atol=atol)
    else:
        T_C = np.min(T_array[mask])
    return T_C


U = np.linspace(10,100,3)
# E_D = np.linspace(0,1,2)
E_D = np.array([1])
T_array = np.linspace(0,1000,1000)

T_C_array = np.array([])
print("START")
for i in range(len(U)):
    for j in range(len(E_D)):
        T_C_array = np.append(T_C_array, get_T_C(U=U[i], E_D=E_D[j], T_array=T_array))
        # index_j = np.isclose(E_D, ed)
        # index_i = np.isclose(U, u)
        # print(index_i, index_j)
        print(f"{(i * len(E_D) + j +1)/(len(E_D)*len(U))*100:.2f} % finished")
print("STOP")
T_C_matrix = T_C_array.reshape(len(U), len(E_D))

df = pd.DataFrame(
    T_C_matrix, 
    index=U,
    columns=E_D
)
df.index.name="U in t"
df.columns.name="E_D in t"
print(df)
df.to_csv("T_C.csv")