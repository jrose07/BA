import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import scipy.integrate as integrate
from scipy.optimize import newton
from graphenemodeling.graphene import _constants as _c
from scipy.constants import k as k_b #In J/K

t = _c.g0 #Braucht man nur für die voncersion von k_b*T in die unit t (dazu mache k_b*T /t. )

def rho_triangle(x):
    """x: arraylike with unit 1/t -> x=3 --> x = 3*t
    return DOS of triangle Lattice in units of 1/t -> to get real 1/J you have to divide DOS with value of t.
    """
    E = x
    z_0 = np.zeros_like(E)
    z_1 = np.zeros_like(E)
    mask_upper = np.logical_and(2<=E, 3>=E)
    mask_lower = np.logical_and(E>=-6, E<=2)
    z_0[mask_upper] = 3 + 2*np.sqrt(3-E[mask_upper]) - (E[mask_upper]/2)**2
    z_0[mask_lower] = 4*np.sqrt(3-E[mask_lower])
    z_1[mask_upper] = 4*np.sqrt(3-E[mask_upper])
    z_1[mask_lower] = 3 + 2*np.sqrt(3-E[mask_lower]) - (E[mask_lower]/2)**2
    return 1/(np.sqrt(z_0)*np.pi**2) * ellipk(z_1/z_0)

def func_DOS(E):
    """E:arraylike with unit E = epsilon/t with [epsilon] = J (real energy). 
    return DOS of honeycomb-lattice in units of 1/t.
    """
    return np.abs(E)*rho_triangle(3-E**2)
    
def newton_func(delta, E, U, T, E_debye):
    mask = np.logical_and(E< np.abs(E_debye), E> - np.abs(E_debye))
    x = E[mask]
    DOS = func_DOS(x)
    E_k = np.sqrt(delta**2 + x**2)
    return U*integrate.simpson(DOS*delta/E_k * np.tanh(E_k/(2*k_b*T/t)), x=x) - delta

def integral(delta, E, U, T, E_debye):
    mask = np.logical_and(E< np.abs(E_debye), E> - np.abs(E_debye))
    x = E[mask]
    DOS = func_DOS(x)
    E_k = np.sqrt(delta**2 + x**2)
    return U*integrate.simpson(DOS*delta/E_k * np.tanh(E_k/(2*k_b*T/t)), x=x)

def fixpunkt_algo(start, E, T, U, num_max, E_debye):
    """start: Starting Value of Delta in units of t
    E: Energy linspace, in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    num_max: maximum number of iterations
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    mask = np.logical_and(E< np.abs(E_debye), E> - np.abs(E_debye))
    x = E[mask]
    DOS = func_DOS(x) #in units of 1/t
    deltas = np.array([start])
    for i in range(num_max):
        E_k = np.sqrt(deltas[i]**2 + x**2)
        delta_next = U*integrate.simpson(DOS*deltas[i]/E_k * np.tanh(E_k/(2*k_b*T/t)), x=x) #Sollte in units of t sein, weil [DOS] = 1/t, [delta] = t, [E_k] = t und durch integration noch mal t
        deltas = np.append(deltas, delta_next)
    return deltas

def get_delta(U,T,E,E_debye, n_fixpunkt, start):
    deltas = fixpunkt_algo(start, E, T, U, n_fixpunkt, E_debye)
    delta = deltas[-1]
    #Newton Verfahren
    delta = newton(newton_func, args=(E,U,T,E_debye),x0=delta)
    return delta