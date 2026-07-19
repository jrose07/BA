import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import scipy.integrate as integrate
from scipy.optimize import newton
# from graphenemodeling.graphene import _constants as _c
from scipy.constants import k as k_b #In J/K
import scipy.constants as const

t = 2.7 * const.e #Braucht man nur für die voncersion von k_b*T in die unit t (dazu mache k_b*T /t. )
# t = 2.7 eV (nach Quellen anscheinend bester Wert)

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
    
def newton_func(delta, U, T, E_debye, num_points):
    x = np.linspace(-E_debye, E_debye, num_points)
    DOS = func_DOS(x)
    E_k = np.sqrt(delta**2 + x**2)
    U_arr = np.asarray(U)
    T_arr = np.asarray(T)
    integrand_base = DOS * delta / E_k
    if np.all(T_arr == 0):
        integral = integrate.simpson(integrand_base, x=x)
    else:
        T_b = T_arr[..., np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = E_k / (2 * k_b * T_b / t)
            tanh_val = np.tanh(arg)
        if np.any(T_b == 0):
            tanh_val = np.where(T_b == 0, 1.0, tanh_val)
        integral = integrate.simpson(integrand_base * tanh_val, x=x, axis=-1)

    return U_arr / 2 * integral - delta

def integral(delta, U, T, E_debye):
    x = np.linspace(-E_debye, E_debye, 10009)
    DOS = func_DOS(x)
    mask = np.logical_and(DOS < np.inf, x != 0)
    x = x[mask]
    DOS = DOS[mask]
    E_k = np.sqrt(delta**2 + x**2)
    integrand_base = DOS * delta / E_k
    if T == 0:
        integral = integrate.simpson(integrand_base, x=x)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = E_k / (2 * k_b * T / t)
            tanh_val = np.tanh(arg)
        integral = integrate.simpson(integrand_base * tanh_val, x=x, axis=-1)
    return U / 2 * integral 

def fixpunkt_algo(start, T, U, E_debye, num_max, num_points):
    """start: Starting Value of Delta in units of t
    E: Energy linspace, in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    num_max: maximum number of iterations
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    x = np.linspace(-E_debye, E_debye, num_points)
    DOS = func_DOS(x) #in units of 1/t
    #Bei x = 1 ist DOS unendlich -> Filtere Punkte heraus
    mask = np.logical_and(DOS < np.inf, x != 0)
    x = x[mask]
    DOS = DOS[mask]
    deltas = np.array([start])
    for i in range(num_max):
        E_k = np.sqrt(deltas[i]**2 + x**2)
        if T == 0:
            delta_next = U/2*integrate.simpson(DOS*deltas[i]/E_k, x=x)
        else:
            delta_next = U/2*integrate.simpson(DOS*deltas[i]/E_k * np.tanh(E_k/(2*k_b*T/t)), x=x) #Sollte in units of t sein, weil [DOS] = 1/t, [delta] = t, [E_k] = t und durch integration noch mal t
        deltas = np.append(deltas, delta_next)
    return deltas

def get_delta(start, T,U, E_debye, num_max, num_points ):
    deltas = fixpunkt_algo(start, T, U, E_debye, num_max, num_points)
    delta = deltas[-1]
    # return delta
    #Newton Verfahren
    try:
        delta = newton(newton_func, args=(U,T,E_debye, num_points), x0=delta, maxiter=200)
        return delta
    except Exception:
        return deltas[-1]

def t2mev(x):
    conv = t /const.e * 1e3 #from [t] -> [meV]
    return x * conv

def mev2t(x):
    return x * const.e*1e-3/t

# E=np.linspace(-3,3,100001)
# T=1
# U=3
# E_debye=2

# deltas = fixpunkt_algo(1,T=T, U=U, num_max=200, E_debye=E_debye, num_points=10019)
# delta_FP = deltas[-1]
# delta_Newton = newton(newton_func, args=(E,U,T,E_debye), x0=deltas[-1])
# m = delta_Newton-delta_FP
# fig,ax=plt.subplots()
# x = np.linspace(-E_debye, E_debye, 10001)
# DOS = func_DOS(x)
# ax.plot(x, DOS)
# ax.plot(np.arange(len(deltas)), deltas, "r.")
# ax.plot(len(deltas)+1, delta_Newton, "b.")
# x = np.linspace(len(deltas),len(deltas)+1,100)
# ax.plot(x, m*x+delta_FP,"r--")
# plt.show()


if __name__ == "__main__":
    conv = t /const.e * 1e3 #from [t] -> [meV]
    U = 9.4 #eV 
    U = 9.4*const.e/t

    print(f"t = {t/const.e:.2f}eV von Graphenemodeling")
    print(f"Debye Energien für Graphen sind ca. {1800*k_b/const.e*1e3:.2f} - {2300*k_b/const.e*1e3:.2f} meV, also 0.05- 0.07t")
    print(f"Delta Werte sind im Bereich ca. 1-10 meV")
    print(f"1t entspricht {conv:.2f}meV")
    print(f"U ist im Bereich von {U:.2f}t = {U*conv*1e-3:.2f}eV")