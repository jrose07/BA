"""
11-band Slater-Koster tight-binding model of monolayer MoS2 and its
electronic density of states (DOS), computed via the Green's-function
(spectral) method exactly as in Eq. (21)-(23) of

    S. Jalilvand and H. Mousavi,
    "Multi-band Tight-Binding Model of MoS2 Monolayer",
    J. Electron. Mater. 49, 3599 (2020), https://doi.org/10.1007/s11664-020-08069-y

The 11x11 Bloch Hamiltonian (5 Mo 4d orbitals + 3+3 S 3p orbitals of the
top/bottom sulfur layer) and the Slater-Koster parameter set are taken
from the reference that the paper itself uses (Ref. 41 there):

    E. Cappelluti, R. Roldan, J. A. Silva-Guillen, P. Ordejon, F. Guinea,
    "Tight-binding model and direct-gap/indirect-gap transition in
    single-layer and multi-layer MoS2", Phys. Rev. B 88, 075409 (2013).
    (crystal field Delta_1 for d_xz,d_yz completed in
     R. Roldan et al., 2D Mater. 1, 034003 (2014), Table I)

METHOD
------
D(E) = -(1/pi) * Im Tr G(E)  with  G(k,E) = [(E + i*eta) I - H(k)]^-1

is evaluated using the spectral representation of the resolvent,

    -(1/pi) Im G_ll(k,E) = sum_n |<l|n,k>|^2 * eta/pi / [(E-E_n(k))^2+eta^2]

i.e. a Lorentzian-broadened histogram of the eigenvalues weighted by the
orbital projections |<l|n,k>|^2. This is mathematically identical to
inverting the matrix at every (k,E) as done "by hand" in the paper
(Eqs. 6-20), but only needs one diagonalization per k-point and is
numerically far more efficient/stable.

Jalilvand & Mousavi note that, because of a sign convention in their
Green's-function equation of motion, the signs of all 12 Slater-Koster
parameters of Ref. 41 have to be reversed there. Here we instead build
the physical Hamiltonian H(k) directly with the ORIGINAL (un-reversed)
Cappelluti signs and diagonalize/resolve it -- the two routes give the
same Hamiltonian and hence the same DOS, band structure, gap, and
Van Hove singularities, without the risk of a sign slip.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Slater-Koster tight-binding parameters for monolayer MoS2 (in eV)
# ----------------------------------------------------------------------
PARAMS = dict(
    D0=-1.512,    # Delta_0   on-site energy of d_{3z^2-r^2}
    D1=0.419,     # Delta_1   on-site energy of d_xz, d_yz
    D2=-3.025,    # Delta_2   on-site energy of d_{x^2-y^2}, d_xy
    Dp=-1.276,    # Delta_p   on-site energy of p_x, p_y (S)
    Dz=-8.236,    # Delta_z   on-site energy of p_z (S)
    Vpds=-2.619,  # V_pd_sigma  (Mo-S)
    Vpdp=-1.396,  # V_pd_pi     (Mo-S)
    Vdds=-0.933,  # V_dd_sigma  (Mo-Mo)
    Vddp=-0.478,  # V_dd_pi     (Mo-Mo)
    Vddd=-0.442,  # V_dd_delta  (Mo-Mo)
    Vpps=0.696,   # V_pp_sigma  (S-S)
    Vppp=0.278,   # V_pp_pi     (S-S)
)

A_LAT = 3.16  # in-plane lattice constant (Angstrom); only sets the k-scale

# Orbital bookkeeping -----------------------------------------------------
# basis order (Cappelluti convention):
# 0:px_t 1:py_t 2:pz_t | 3:dz2 4:dx2-y2 5:dxy 6:dxz 7:dyz | 8:px_b 9:py_b 10:pz_b
ORB_LABELS = ["px(top)", "py(top)", "pz(top)",
              "d3z2-r2", "dx2-y2", "dxy", "dxz", "dyz",
              "px(bot)", "py(bot)", "pz(bot)"]
MO_IDX = [3, 4, 5, 6, 7]
S_TOP_IDX = [0, 1, 2]
S_BOT_IDX = [8, 9, 10]


def sk_prefactors(p):
    """Angular Slater-Koster combinations E1..E16, see Cappelluti et al.
    PRB 88, 075409 (2013), Appendix A (Eqs. A8-A23)."""
    phi = np.arccos(np.sqrt(4.0 / 7.0))   # ideal trigonal-prism angle
    c, s = np.cos(phi), np.sin(phi)
    Vpds, Vpdp = p["Vpds"], p["Vpdp"]
    Vdds, Vddp, Vddd = p["Vdds"], p["Vddp"], p["Vddd"]
    Vpps, Vppp = p["Vpps"], p["Vppp"]
    r3 = np.sqrt(3.0)

    E1 = 0.5 * (-Vpds * (s**2 - 0.5 * c**2) + r3 * Vpdp * s**2) * c
    E2 = (-Vpds * (s**2 - 0.5 * c**2) - r3 * Vpdp * c**2) * s
    E3 = 0.25 * ((r3 / 2) * Vpds * c**3 + Vpdp * c * s**2)
    E4 = 0.5 * ((r3 / 2) * Vpds * s * c**2 - Vpdp * s * c**2)
    E5 = -0.75 * Vpdp * c
    E6 = -0.75 * Vpdp * s
    E7 = 0.25 * (-r3 * Vpds * c**2 - Vpdp * (1 - 2 * c**2)) * s
    E8 = 0.5 * (-r3 * Vpds * s**2 - Vpdp * (1 - 2 * s**2)) * c
    E9 = 0.25 * Vdds + 0.75 * Vddd
    E10 = -(r3 / 4) * (Vdds - Vddd)
    E11 = 0.75 * Vdds + 0.25 * Vddd
    E12 = Vddp
    E13 = Vddp
    E14 = Vddd
    E15 = Vpps
    E16 = Vppp
    return dict(E1=E1, E2=E2, E3=E3, E4=E4, E5=E5, E6=E6, E7=E7, E8=E8,
                E9=E9, E10=E10, E11=E11, E12=E12, E13=E13, E14=E14,
                E15=E15, E16=E16)


def hamiltonian(kx, ky, p, E, a=A_LAT):
    """Build the 11x11 Bloch Hamiltonian H(kx,ky) (in eV)."""
    xi = kx * a / 2.0
    eta = np.sqrt(3.0) * ky * a / 2.0
    r3 = np.sqrt(3.0)

    C1 = (2 * np.cos(xi) * np.cos(eta / 3) + np.cos(2 * eta / 3)
          + 1j * (2 * np.cos(xi) * np.sin(eta / 3) - np.sin(2 * eta / 3)))
    C2 = (np.cos(xi) * np.cos(eta / 3) - np.cos(2 * eta / 3)
          + 1j * (np.cos(xi) * np.sin(eta / 3) + np.sin(2 * eta / 3)))
    C3 = (np.cos(xi) * np.cos(eta / 3) + 2 * np.cos(2 * eta / 3)
          + 1j * (np.cos(xi) * np.sin(eta / 3) - 2 * np.sin(2 * eta / 3)))
    d1 = np.sin(eta / 3) - 1j * np.cos(eta / 3)
    l1 = np.cos(2 * xi) + 2 * np.cos(xi) * np.cos(eta)
    l2 = np.cos(2 * xi) - np.cos(xi) * np.cos(eta)
    l3 = 2 * np.cos(2 * xi) + np.cos(xi) * np.cos(eta)
    cc = np.cos(xi) * np.cos(eta)
    ss = np.sin(xi) * np.sin(eta)

    Dp, Dz, D0, D2, D1v = p["Dp"], p["Dz"], p["D0"], p["D2"], p["D1"]
    E1, E2, E3, E4 = E["E1"], E["E2"], E["E3"], E["E4"]
    E5, E6, E7, E8 = E["E5"], E["E6"], E["E7"], E["E8"]
    E9, E10, E11, E12 = E["E9"], E["E10"], E["E11"], E["E12"]
    E13, E14, E15, E16 = E["E13"], E["E14"], E["E15"], E["E16"]

    # ---- in-plane (same layer) matrix elements ----
    Hxx = Dp + E15 * l3 + 3 * E16 * cc
    Hyy = Dp + E16 * l3 + 3 * E15 * cc
    Hzz = Dz + 2 * E16 * l1
    Hz2z2 = D0 + 2 * E9 * l1
    Hx2x2 = D2 + E11 * l3 + 3 * E12 * cc
    Hxyxy = D2 + E12 * l3 + 3 * E11 * cc
    Hxzxz = D1v + E13 * l3 + 3 * E14 * cc
    Hyzyz = D1v + E14 * l3 + 3 * E13 * cc
    Hxy = -r3 * (E15 - E16) * ss
    Hz2x2 = 2 * E10 * l2
    Hz2xy = -2 * r3 * E10 * ss
    Hx2xy = r3 * (E11 - E12) * ss
    Hxzyz = r3 * (E14 - E13) * ss

    # ---- Mo(d) - S(p) matrix elements (same in-plane cell) ----
    Hz2x = -2 * r3 * E1 * np.sin(xi) * d1
    Hz2y = 2 * E1 * C2
    Hz2zc = E2 * C1
    Hx2x = -2 * r3 * (E5 / 3 - E3) * np.sin(xi) * d1
    Hx2y = -2 * E3 * C3 - 2j * E5 * np.cos(xi) * d1
    Hx2z = -2 * E4 * C2
    Hxyx = -(2.0 / 3) * E5 * C3 - 6j * E3 * np.cos(xi) * d1
    Hxyy = Hx2x
    Hxyz = 2 * r3 * E4 * np.sin(xi) * d1
    Hxzx = (2.0 / 3) * E6 * C3 + 6j * E7 * np.cos(xi) * d1
    Hxzy = 2 * r3 * (E6 / 3 - E7) * np.sin(xi) * d1
    Hxzz = -2 * r3 * E8 * np.sin(xi) * d1
    Hyzx = Hxzy
    Hyzy = 2 * E7 * C3 + 2j * E6 * np.cos(xi) * d1
    Hyzz = 2 * E8 * C2

    H = np.zeros((11, 11), dtype=complex)

    # p-p block (top-top and bottom-bottom, identical), Eq. (4)
    Hpp = np.array([[Hxx, Hxy, 0], [np.conj(Hxy), Hyy, 0], [0, 0, Hzz]],
                    dtype=complex)
    H[np.ix_(S_TOP_IDX, S_TOP_IDX)] = Hpp
    H[np.ix_(S_BOT_IDX, S_BOT_IDX)] = Hpp

    # d-d block, Eq. (5)
    Hdd = np.array([
        [Hz2z2, Hz2x2, Hz2xy, 0, 0],
        [np.conj(Hz2x2), Hx2x2, Hx2xy, 0, 0],
        [np.conj(Hz2xy), np.conj(Hx2xy), Hxyxy, 0, 0],
        [0, 0, 0, Hxzxz, Hxzyz],
        [0, 0, 0, np.conj(Hxzyz), Hyzyz],
    ], dtype=complex)
    H[np.ix_(MO_IDX, MO_IDX)] = Hdd

    # top-bottom vertical S-S coupling, Eq. (6) (k-independent)
    Vppp_, Vpps_ = p["Vppp"], p["Vpps"]
    Hptpb = np.array([[Vppp_, 0, 0], [0, Vppp_, 0], [0, 0, Vpps_]],
                      dtype=complex)
    H[np.ix_(S_TOP_IDX, S_BOT_IDX)] = Hptpb
    H[np.ix_(S_BOT_IDX, S_TOP_IDX)] = Hptpb.conj().T

    # Mo(d) - top S(p), Eq. (7)
    Hdpt = np.array([
        [Hz2x, Hz2y, Hz2zc],
        [Hx2x, Hx2y, Hx2z],
        [Hxyx, Hxyy, Hxyz],
        [Hxzx, Hxzy, Hxzz],
        [Hyzx, Hyzy, Hyzz],
    ], dtype=complex)
    H[np.ix_(MO_IDX, S_TOP_IDX)] = Hdpt
    H[np.ix_(S_TOP_IDX, MO_IDX)] = Hdpt.conj().T

    # Mo(d) - bottom S(p), Eq. (8) (z-odd orbitals change sign)
    Hdpb = Hdpt.copy()
    Hdpb[:, 2] *= -1        # pz column flips sign
    Hdpb[3:5, :] *= -1      # dxz,dyz rows flip sign  (indices 3,4 of the 5)
    H[np.ix_(MO_IDX, S_BOT_IDX)] = Hdpb
    H[np.ix_(S_BOT_IDX, MO_IDX)] = Hdpb.conj().T

    return H


# ----------------------------------------------------------------------
# 2. Brillouin-zone sampling (triangular Mo sublattice, constant a)
# ----------------------------------------------------------------------
def bz_kmesh(nk, a=A_LAT):
    """Uniform Monkhorst-Pack-like mesh tiling one primitive cell of the
    reciprocal lattice (equivalent to summing over the full BZ)."""
    s = (np.arange(nk) + 0.5) / nk
    S, T = np.meshgrid(s, s, indexing="ij")
    S, T = S.ravel(), T.ravel()
    kx = S * 2 * np.pi / a
    ky = (2 * T - S) * 2 * np.pi / (a * np.sqrt(3.0))
    return kx, ky


# ----------------------------------------------------------------------
# 3. Band structure and DOS
# ----------------------------------------------------------------------
def band_structure(params, npts=150, a=A_LAT):
    Ecoef = sk_prefactors(params)
    G = np.array([0.0, 0.0])
    K = np.array([4 * np.pi / (3 * a), 0.0])
    M = np.array([0.0, 2 * np.pi / (np.sqrt(3) * a)])

    def seg(p0, p1, n):
        return np.linspace(p0, p1, n, endpoint=False)

    path = np.vstack([seg(G, M, npts), seg(M, K, npts), seg(K, G, npts + 1)])
    dists = [0.0]
    for i in range(1, len(path)):
        dists.append(dists[-1] + np.linalg.norm(path[i] - path[i - 1]))
    ticks = [0.0, np.linalg.norm(M - G),
              np.linalg.norm(M - G) + np.linalg.norm(K - M),
              np.linalg.norm(M - G) + np.linalg.norm(K - M) + np.linalg.norm(G - K)]

    bands = np.zeros((len(path), 11))
    for i, (kx, ky) in enumerate(path):
        H = hamiltonian(kx, ky, params, Ecoef, a=a)
        bands[i] = np.linalg.eigvalsh(H)
    return np.array(dists), bands, ticks


def valence_band_max(params, nk=80, a=A_LAT):
    """Reference energy = top of the 7 valence bands (bands 0..6)."""
    Ecoef = sk_prefactors(params)
    kx, ky = bz_kmesh(nk, a=a)
    vbm = -np.inf
    for x, y in zip(kx, ky):
        H = hamiltonian(x, y, params, Ecoef, a=a)
        w = np.linalg.eigvalsh(H)
        vbm = max(vbm, w[6])
    return vbm


def compute_dos(params, E_grid, eta=0.04, nk=48, a=A_LAT):
    """Total + orbital-resolved DOS, Eqs. (21)-(23) of Jalilvand & Mousavi,
    evaluated via the spectral (eigenvector-weighted Lorentzian) method."""
    Ecoef = sk_prefactors(params)
    kx, ky = bz_kmesh(nk, a=a)
    nkpts = len(kx)
    nE = len(E_grid)
    dos_orb = np.zeros((11, nE))

    for x, y in zip(kx, ky):
        H = hamiltonian(x, y, params, Ecoef, a=a)
        w, v = np.linalg.eigh(H)                  # w: (11,) v: (11,11)
        weight = np.abs(v) ** 2                    # weight[orb, band]
        # Lorentzian(E - w_n) for every band n, shape (11 bands, nE)
        dE = E_grid[None, :] - w[:, None]
        lorentz = (eta / np.pi) / (dE ** 2 + eta ** 2)   # (band, nE)
        dos_orb += weight @ lorentz                 # (orb,band)@(band,nE)

    dos_orb /= nkpts
    return dos_orb   # shape (11, nE), states/eV per unit cell, per orbital


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Tunable numerical settings
    # ------------------------------------------------------------------
    NK = 60          # k-mesh is NK x NK over the Brillouin zone
    ETA = 0.05        # Lorentzian broadening (eV) -> plays the role of 0+
    E_MIN, E_MAX = -13, 9   # energy window (eV) relative to the raw TB energies

    # ---- reference energy: top of valence band -> E_F = 0 -------------
    E0 = valence_band_max(PARAMS, nk=90)
    print(f"Valence-band maximum (before shift): {E0:.4f} eV -> shifted to 0")

    # ---- band structure --------------------------------------------
    dists, bands, ticks = band_structure(PARAMS, npts=150)
    bands -= E0

    # ---- direct gap check at K ---------------------------------------
    Ecoef = sk_prefactors(PARAMS)
    a = A_LAT
    K = (4 * np.pi / (3 * a), 0.0)
    wK = np.linalg.eigvalsh(hamiltonian(*K, PARAMS, Ecoef)) - E0
    gap = wK[7] - wK[6]
    print(f"Direct gap at K: {gap:.3f} eV  (valence top {wK[6]:.3f} eV, "
          f"conduction bottom {wK[7]:.3f} eV)")

    # ---- DOS -----------------------------------------------------------
    E_grid = np.linspace(E_MIN, E_MAX, 1400)
    dos_orb = compute_dos(PARAMS, E_grid, eta=ETA, nk=NK)
    dos_orb_shifted_E = E_grid - E0

    total_dos_per_orbital = dos_orb.sum(axis=0) / 11.0   # paper's Eq.(21) norm.
    mo_dos = dos_orb[MO_IDX, :].sum(axis=0)
    s_dos_top = dos_orb[S_TOP_IDX, :]
    s_dos_bot = dos_orb[S_BOT_IDX, :]
    s_dos_avg_px = 0.5 * (s_dos_top[0] + s_dos_bot[0])
    s_dos_avg_py = 0.5 * (s_dos_top[1] + s_dos_bot[1])
    s_dos_avg_pz = 0.5 * (s_dos_top[2] + s_dos_bot[2])
    s_dos_total = s_dos_top.sum(axis=0) + s_dos_bot.sum(axis=0)

    d0 = dos_orb[3]                          # d_3z2-r2
    d1o = 0.5 * (dos_orb[4] + dos_orb[5])    # dx2-y2, dxy (symmetric)
    d2o = 0.5 * (dos_orb[6] + dos_orb[7])    # dxz, dyz    (symmetric)

    # =====================================================================
    # PLOTS  (mirrors Fig. 2 of Jalilvand & Mousavi 2020)
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5),
                              gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    for n in range(11):
        ax.plot(dists, bands[:, n], color="tab:blue", lw=1.3)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([r"$\Gamma$", "M", "K", r"$\Gamma$"])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("11-band structure of MoS$_2$ monolayer")
    ax.set_xlim(dists[0], dists[-1])
    ax.set_ylim(E_MIN, E_MAX)
    for t in ticks:
        ax.axvline(t, color="gray", lw=0.5)

    ax = axes[1]
    ax.fill_betweenx(E_grid - E0, 0, total_dos_per_orbital,
                      color="lightgray", label="total (per orbital, Eq.21)")
    ax.plot(mo_dos, E_grid - E0, color="tab:red", lw=1.3, label="Mo atom")
    ax.plot(s_dos_total, E_grid - E0, color="tab:green", lw=1.3, ls="--",
            label="S atoms")
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("DOS (states/eV/cell)")
    ax.set_title("Total & atom-resolved DOS")
    ax.set_ylim(E_MIN, E_MAX)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig("./mos2_bands_and_dos.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(d0, E_grid - E0, color="k", lw=1.4, label=r"$d_{3z^2-r^2}$")
    ax2.plot(d1o, E_grid - E0, color="tab:orange", lw=1.2, ls="--",
             label=r"$d_{x^2-y^2},\,d_{xy}$")
    ax2.plot(d2o, E_grid - E0, color="tab:purple", lw=1.2, ls=":",
             label=r"$d_{xz},\,d_{yz}$")
    ax2.plot(s_dos_avg_px + s_dos_avg_py, E_grid - E0, color="tab:green",
              lw=1.2, ls="-.", label=r"$p_x,\,p_y$ (S)")
    ax2.plot(s_dos_avg_pz, E_grid - E0, color="tab:blue", lw=1.2,
              label=r"$p_z$ (S)")
    ax2.axhline(0, color="k", ls="--", lw=0.6)
    ax2.set_xlabel("Partial DOS (states/eV/cell)")
    ax2.set_ylabel("Energy (eV)")
    ax2.set_title("Orbital-resolved partial DOS")
    ax2.set_ylim(E_MIN, E_MAX)
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig("./mos2_partial_dos.png", dpi=150)

    print("Saved: mos2_bands_and_dos.png, mos2_partial_dos.png")
