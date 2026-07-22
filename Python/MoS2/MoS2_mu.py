import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import newton
from scipy.constants import k as k_b #In J/K
import scipy.constants as const

"""Because DOS is given in eV here everything will be calulcaled in eV"""

def get_E_DOS(E_D, num_points):
    E, DOS = np.genfromtxt("./DOS_3.csv", delimiter=",",skip_header=1, unpack=True)
    E = E + 1 # Reparametrisiere E dass Fermikante bei E = 0 liegt.
    #Make unique with means
    E_unique = np.unique(E)
    DOS = np.array([DOS[E==e].mean() for e in E_unique])
    E = E_unique
    #Cutoff
    cutoff = np.logical_and(E<E_D, -E_D<E)
    E = E[cutoff]
    DOS = DOS[cutoff]
    # print(len(E))
    #Interpolate
    E_fine = np.linspace(E.min(), E.max(), num_points)
    DOS = np.interp(E_fine, E, DOS)
    E = E_fine
    # mask = DOS < np.inf
    return E, DOS
    
def newton_func(delta, U, T, E, DOS, mu, num_points):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of eV
    mu: chemical Potential in Units of eV
    iterations: maximum number of iterations
    E_debye: Debye-Energy in units of eV
    returns list of delta for each iteration in units of t
    """
    # E, DOS = get_E_DOS(E_D, num_points) #eV
    E_k = np.sqrt(delta**2 + (E-mu)**2)
    integrand_base = DOS * delta / E_k
    if T == 0:
        integral = integrate.simpson(integrand_base, x=E)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = E_k * const.e / (2 * k_b * T ) # E_k in eV-> J
            tanh_val = np.tanh(arg)
        integral = integrate.simpson(integrand_base * tanh_val, x=E, axis=-1)
    return U / 2 * integral - delta # in eV

def integral(delta, U, T, E_D, mu, num_points):
    """Get the integral of the DOS-Formula for a delta and mu!=0
    E: arraylike
    delta: delta
    U, T, E_D: scalar
    returns integral of DOS and Self_consistency
    """
    E, DOS = get_E_DOS(E_D, num_points) #eV
    E_k = np.sqrt(delta**2 + (E-mu)**2) #eV
    integrand_base = DOS * delta / E_k
    if T == 0:
        integral = integrate.simpson(integrand_base, x=E)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = E_k * const.e / (2 * k_b * T ) #E_k in eV -> J
            tanh_val = np.tanh(arg)
        integral = integrate.simpson(integrand_base * tanh_val, x=E, axis=-1)
    return U / 2 * integral #in eV

def get_next_delta(delta, T, U, mu, E, DOS):
    """Get the next delta in a fixpoint algorithm using the DOS-Formula with mu!=0
    E: arraylike
    delta: delta_n
    U, T: scalar
    returns delta_n+1 of fixpoint algorithm
    """
    E_k = np.sqrt((E-mu)**2 + delta**2)
    # print(E_k * const.e / (2*k_b*T))
    if T==0:
        integral = integrate.simpson(DOS*delta/E_k, x=E) 
    else:
        integral = integrate.simpson(DOS*delta/E_k * np.tanh(E_k * const.e/(2*k_b*T)), x=E)
    return U/2 * integral #Ist in units of eV


def fixpunkt_algo(start, T, U, E, DOS, mu, iterations, num_points):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    mu: chemical Potential in Units of t
    iterations: maximum number of iterations
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    # E, DOS = get_E_DOS(E_D, num_points) #in eV
    deltas = np.array([start])
    for i in range(iterations):
        delta_next = get_next_delta(deltas[i], T=T, U=U, mu=mu, E=E, DOS=DOS)
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
    E, DOS = get_E_DOS(E_D, num_points)
    deltas = fixpunkt_algo(start, T, U, E, DOS, mu, iterations, num_points)
    delta = deltas[-1]
    # return delta
    #Newton Verfahren
    try:
        delta = newton(newton_func, args=(U,T,E,DOS, mu, num_points), x0=delta, maxiter=200)
        return delta
    except Exception:
        return deltas[-1]

def get_delta_2(start, T, U, E, DOS, iterations, num_points, mu=0):
    """start: Starting Value of Delta in units of t
    T: Temperature in Kelvin
    U: Coupling Constant in Units of t
    mu: chemical Potential in Units of t
    iterations: maximum number of iterations
    num_pints: number of points in E_linspace, integration
    E_debye: Debye-Energy in units of t
    returns list of delta for each iteration in units of t
    """
    # E, DOS = get_E_DOS(E_D, num_points)
    deltas = fixpunkt_algo(start, T, U, E, DOS, mu, iterations, num_points)
    delta = deltas[-1]
    # return delta
    #Newton Verfahren
    try:
        delta = newton(newton_func, args=(U,T,E,DOS, mu, num_points), x0=delta, maxiter=200)
        return delta
    except Exception:
        return deltas[-1]


if __name__ == "__main__":
    # conv = t /const.e * 1e3 #from [t] -> [meV]
    # U = 9.4 #eV 
    # U = 9.4*const.e/t

    # print(f"t = {t/const.e:.2f}eV von Paper")
    # print(f"Debye Energien für Graphen sind ca. {1800*k_b/const.e*1e3:.2f} - {2300*k_b/const.e*1e3:.2f} meV, also 0.05- 0.07t")
    # print(f"Delta Werte sind im Bereich ca. 1-10 meV")
    # print(f"1t entspricht {conv:.2f}meV")
    # print(f"U ist im Bereich von {U:.2f}t = {U*conv*1e-3:.2f}eV")
    
    
    """Test functions"""
    Theta_D = 262.3 #K Debye-Temperatur. 
    E_D = Theta_D * const.k / const.e #in eV
    
    E, DOS = get_E_DOS(E_D, 1000)
    plt.plot(E, DOS)
    plt.show()
    import time
    t1= time.perf_counter()
    T = np.linspace(0, 2000, 100)  #K
    U = 10 #eV
    # mu = [0,0.5,1,1.5,2] #eV
    mu = [0,1,2]
    fig, ax = plt.subplots()
    for m in mu:
        deltas = np.array([])
        for elem in T:
                deltas = np.append(deltas, get_delta(U=U, T=elem, E_D=E_D, mu=m, iterations=2, start=1, num_points=10009))
        ax.plot(T, deltas,label=f"Verlauf bei mu={m}eV")
        ax.set(
            xlabel=r"$T \, K$",
            ylabel=r"$\Delta \, eV$"
        )
    t2 = time.perf_counter()
    print(t2-t1)
    plt.show()
    """Tipp: Wennmöglich wenig Fixpunktiterationen machen, newton funktioniert genauso gut, nur nicht für irgendwelche anfangsvalues"""