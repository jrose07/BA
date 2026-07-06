import matplotlib.pyplot as plt
import numpy as np
from graphene import get_delta
from graphenemodeling.graphene import _constants as _c
import pandas as pd
import scipy.constants as const

def E_D_theo(U, T_C, A):
    return 2*const.k*T_C*np.arccosh(np.exp(1/(2*U*A*const.k*T_C)))

A = 0.18 #(1/t^2)