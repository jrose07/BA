import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import scipy.integrate as integrate
from scipy.optimize import newton
from scipy.constants import k as k_b #In J/K
import scipy.constants as const

t = 2.7 * const.e #Braucht man nur für die conversion von k_b*T in die unit t (dazu mache k_b*T /t. )
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
    mask_outside = np.logical_or(E<-6, E>3)
    z_0[mask_upper] = 3 + 2*np.sqrt(3-E[mask_upper]) - (E[mask_upper]/2)**2
    z_0[mask_lower] = 4*np.sqrt(3-E[mask_lower])
    z_1[mask_upper] = 4*np.sqrt(3-E[mask_upper])
    z_1[mask_lower] = 3 + 2*np.sqrt(3-E[mask_lower]) - (E[mask_lower]/2)**2
    DOS_tr = np.where(mask_outside,0,1/(np.sqrt(z_0)*np.pi**2) * ellipk(z_1/z_0))
    return DOS_tr

def func_DOS(E):
    """E:arraylike with unit E = epsilon/t with [epsilon] = J (real energy). 
    return DOS of honeycomb-lattice in units of 1/t for mu=0.
    """
    return np.abs(E)*rho_triangle(3-E**2)
    
def newton_func(delta, U, T, E_D, mu, num_points):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    mu: chemical Potential in Units of t
    iterations: maximum number of iterations
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    x = np.linspace(-E_D, E_D, num_points)
    DOS = func_DOS(x + mu)
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
    return U / 2 * integral - delta

def integral(delta, U, T, E_D, mu, num_points):
    """Get the integral of the DOS-Formula for a delta and mu!=0
    E: arraylike
    delta: delta
    U, T, E_D: scalar
    returns integral of DOS and Self_consistency
    """
    x = np.linspace(-E_D, E_D, num_points)
    DOS = func_DOS(x + mu)
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

def get_next_delta(delta, T, U, mu, E):
    """Get the next delta in a fixpoint algorithm using the DOS-Formula with mu!=0
    E: arraylike
    delta: delta_n
    U, T: scalar
    returns delta_n+1 of fipoint algorithm
    """
    E_k = np.sqrt(E**2 + delta**2)
    DOS = func_DOS(E + mu)
    if T==0:
        integral = integrate.simpson(DOS*delta/E_k, x=E) 
    else:
        integral = integrate.simpson(DOS*delta/E_k * np.tanh(E_k/(2*k_b*T/t)), x=E)
    return U/2 * integral #Sollte in units of t sein, weil [DOS] = 1/t, [delta] = t, [E_k] = t und durch integration noch mal t -> [integral] = 1, [U] = t


def fixpunkt_algo(start, T, U, E_D, mu, iterations, num_points):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    mu: chemical Potential in Units of t
    iterations: maximum number of iterations
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    x = np.linspace(-E_D, E_D, num_points)
    DOS = func_DOS(x + mu) #in units of 1/t
    #Bei x = 1 ist DOS unendlich -> Filtere Punkte heraus
    mask = np.logical_and(DOS < np.inf, x != 0)
    x = x[mask]
    deltas = np.array([start])
    for i in range(iterations):
        delta_next = get_next_delta(deltas[i], T=T, U=U, mu=mu, E=x)
        deltas = np.append(deltas, delta_next)
    return deltas

def get_delta(start, T, U, E_D, iterations, num_points, mu=0):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    mu: chemical Potential in Units of t
    iterations: maximum number of iterations
    num_pints: number of points in E_linspace, integration
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    deltas = fixpunkt_algo(start, T, U, E_D, mu, iterations, num_points)
    delta = deltas[-1]
    # return delta
    #Newton Verfahren
    try:
        delta = newton(newton_func, args=(U,T,E_D, mu, num_points), x0=delta, maxiter=200)
        return delta
    except Exception:
        return deltas[-1]

"""Conversions"""
def t2mev(x):
    conv = t /const.e * 1e3 #from [t] -> [meV]
    return x * conv

def mev2t(x):
    return x * const.e*1e-3/t


if __name__ == "__main__":
    conv = t /const.e * 1e3 #from [t] -> [meV]
    U = 9.4 #eV 
    U = 9.4*const.e/t

    print(f"t = {t/const.e:.2f}eV von Paper")
    print(f"Debye Energien für Graphen sind ca. {1800*k_b/const.e*1e3:.2f} - {2300*k_b/const.e*1e3:.2f} meV, also 0.05- 0.07t")
    print(f"Delta Werte sind im Bereich ca. 1-10 meV")
    print(f"1t entspricht {conv:.2f}meV")
    print(f"U ist im Bereich von {U:.2f}t = {U*conv*1e-3:.2f}eV")
    
    
    """Test functions"""
    
    # E = np.linspace(-2,2,1000)
    # DOS = func_DOS(E + 0.5)
    # plt.plot(E, DOS)
    # plt.show()
    import time
    t1= time.perf_counter()
    T = np.linspace(0, 20000, 10)  #K
    U = mev2t(200*1e3)
    E_D = 0.07
    mu = [0,0.5,1,1.5,2] # t
    fig, ax = plt.subplots()
    for m in mu:
        deltas = np.array([])
        for elem in T:
                deltas = np.append(deltas, get_delta(U=U, T=elem, E_D=E_D, mu=m, iterations=2, start=1, num_points=10009))
        ax.plot(T, deltas, "b-",label="Verlauf")
    t2 = time.perf_counter()
    print(t2-t1)
    plt.show()
    """Tipp: Wennmöglich wenig Fixpunktiterationen machen, newton funktioniert genauso gut, nur nicht für irgendwelche anfangsvalues"""