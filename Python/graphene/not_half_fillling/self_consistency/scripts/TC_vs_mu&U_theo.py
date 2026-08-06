import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from graphene_mu import *
from concurrent.futures import ProcessPoolExecutor
import scipy.constants as const
from scipy.optimize import brentq

import time

A = 0.184080 # /t^2
# gamma = 0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495

def logcosh(z):
    """Numerisch stabileres Verfahren auch für großes z"""
    """ln(cosh(x)) = ln(1/2 * (e^x + e^-x)) = ln(e^x ( 1 + e^(-2x))) - ln(2) = abs(x) - ln(2) + ln(1+e^(-2abs(x)))"""
    z = abs(z)
    return z - np.log(2) + np.log(1 + np.exp(-2*z))

def F(z, beta_c):
    return 2/beta_c * logcosh(beta_c/2 * z)

def G(z, beta_c):
    def integrand(u, beta):
        return np.tanh(beta/2 * u) / u
    total, err = integrate.quad(integrand, 0, z, args=(beta_c,), limit=500)
    # print(err)
    return total


def bracket(f, blo, bhi, args, blo_min=1e-30, bhi_max=1e30):
    flo, fhi = f(blo, *args), f(bhi, *args)
    while (not np.isfinite(flo) or flo > 0) and blo > blo_min:
        blo /= 10
        flo = f(blo, *args)
    while (not np.isfinite(fhi) or fhi < 0) and bhi < bhi_max:
        bhi *= 10
        fhi = f(bhi, *args)
    return blo, bhi


def I_exact(mu, E_D, beta_c):
    #Nur für mu < 1 gut (Ist eine lineare Regression xD)
    mu = abs(mu)
    if mu < E_D:
        F_parts = 2*F(E_D, beta_c) - 2*F(mu, beta_c)
        with np.errstate(divide='ignore', invalid='ignore'):
            if mu == 0:
                G_parts = 0.0
            else:
                G_parts = 2*mu*G(mu, beta_c)
        return F_parts + G_parts
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            if mu == 0:
                G_parts = 0.0
            else:
                G_parts = 2*mu*G(E_D, beta_c)
        return G_parts
    
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     rho_0 = func_DOS(mu)
    #     if mu == 0:
    #         G_parts = 0.0
    #     else:
    #         G_parts = 2*G(E_D, beta_c)
    # return rho_0*G_parts

print(1/mev2t(const.k * 1000 / const.e * 1e3))

def solve_Tc_exact(mu, ED, U, A, blo=1e-4, bhi=1.0):
    UA = U*A # Only for mu<1
    # UA = U #With rho_0
    def gap_eq(beta_c):
        return UA/2 * I_exact(mu, ED, beta_c) - 1.0
    
    blo, bhi = bracket(lambda b: gap_eq(b), blo, bhi, args=())
    flo, fhi = gap_eq(blo), gap_eq(bhi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan, np.nan
    try:
        beta_c = brentq(gap_eq, blo, bhi, xtol=1e-12, rtol=1e-12)
    except ValueError:
        return np.nan, np.nan
    T_C = 1/(const.k * beta_c / (t2mev(1) * 1e-3 * const.e)) # beta_c ist in units 1/t-> 1/J
    return T_C , beta_c


def get_T_C(U, mu, E_D):
    U = np.asarray(U)
    mu = np.asarray(mu)
    U_b, mu_b = np.meshgrid(U, mu, indexing='xy')
    def T_C_scalar(U_val, mu_val):
        T_C, _ = solve_Tc_exact(mu_val, E_D, U_val, A)
        return T_C
    return np.vectorize(T_C_scalar, otypes=[float])(U_b, mu_b)


"""Ab hier richtige Rechnung"""

#Params
E_D = mev2t(200) # t
U = np.linspace(0,mev2t(20e3),200) # t
mu = np.linspace(0,0.7,200) # t

version = 2

#Rechnung und plots
t1= time.perf_counter()

"""Do Multiprocessing with AI hallucinations"""

def worker_func(mu_chunk):
    # keep full U for each worker, split only along mu to ensure consistent second axis
    return get_T_C(U, mu_chunk, E_D)

def main():
    # choose number of processes not exceeding number of mu chunks
    nproc = 8
    mu_chunks = np.array_split(mu, nproc)
    nproc = min(nproc, len(mu_chunks))
    with ProcessPoolExecutor(max_workers=nproc) as pool:
        results = pool.map(worker_func, mu_chunks)

    # concatenate along the mu axis (axis=0) to reconstruct full grid
    T_C = np.concatenate(list(results), axis=0)

    print(np.any(np.isnan(T_C)))

    T_C_df = pd.DataFrame(T_C, index=mu, columns=t2mev(U)*1e-3)
    T_C_df.index.name = r"$\mu / t$"
    T_C_df.columns.name = r"$U / eV$"
    T_C_df.to_csv(f"./data/TC_vs_mu&U_theo_{version}.csv")



    t2 = time.perf_counter()
    dt = t2-t1
    print(f"{np.floor((t2-t1)/60)} mins {((dt/60 - np.floor(dt/60))*60):.2f} sec")

    finite = np.isfinite(T_C)
    if not np.any(finite):
        print("No finite T_C values found")
        return
    levels = np.linspace(np.nanmin(T_C), np.nanmax(T_C), 100)
    T_C_masked = np.ma.masked_invalid(T_C)
    fig, ax = plt.subplots()

    print(np.nanmax(T_C), np.max(T_C))

    colorbar = ax.contourf(t2mev(U)*1e-3, mu, T_C_masked, levels=levels, cmap='Spectral_r')
    fig.colorbar(colorbar, ax=ax, label=r"$T_C \, / \, K$")
    ax.set(
        xlabel=r"$U \, / \, eV$",
        ylabel=r"$\mu \, / \, t$"
    )
    # ax.set_facecolor(color='black')
    ax.set_facecolor(color='#5C51A3')
    fig.savefig(f"../plots/TC_vs_mu&U_theo_{version}.pdf")

if __name__ == "__main__":
    main()


"""
Notes wegen runtime:
T array muss gar nicht so groß sein -> Macht einfach nur ungenauer ob man das T_C trifft oder nicht, aber für grobes Bild ist das egal.
Naja okay nvmd, das sagt ja auch aus wie detailed das bild sein kann.. 
Also so mindestens 50 sollte es schon sein 
für mu x U = 50 x 50 schon gutes Bild ->
num_points x 10 approx x1.5 runtime, aber das kann man gut sparen, ist nur für so nachkommastellen relevant bei T_C
"""
