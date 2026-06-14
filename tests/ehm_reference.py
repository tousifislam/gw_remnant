"""
Conservative, orbit-averaged binding energy and angular momentum for
*eccentric, aligned-spin* circular-to-eccentric binaries, translated from the
supplementary file ``EOB_Keplerian.dat.m`` of

    A. Gamboa, M. Khalil, A. Buonanno, "Third post-Newtonian dynamics for
    eccentric orbits and aligned spins in the effective-one-body waveform
    model SEOBNRv5EHM".

Implements (dropping the instantaneous Cos[zeta]/Sin[zeta] pieces, i.e. the
orbit-averaged conservative parts):
  - ``Ehat`` : the energy  ``Heob``  (Eq. block 'Energy / Hamiltonian')
  - ``Lhat`` : the angular momentum ``Leob`` (Eq. block 'Angular momentum')

Both are *per reduced mass* mu = M nu (with M = 1), to 3PN, with:
  x      = <M Omega>^(2/3)              (orbit-averaged frequency parameter)
  e      = Keplerian eccentricity
  chiS   = (chi1 + chi2)/2,  chiA = (chi1 - chi2)/2   (aligned, dimensionless)
  delta  = (m1 - m2)/M
  kapS, kapA : spin-quadrupole combinations; ZERO for black holes (kappa_i = 1)

Physical quantities (M = 1, mu = nu):
  E_binding = nu * Ehat        (note: code's _E0_from_PN returns -E_binding)
  L_orbital = nu * Lhat

The PN counting parameter epsilon of the .m file is set to 1 here; it is kept
symbolically in comments to mark PN orders.
"""

from __future__ import annotations

import numpy as np

PI = np.pi


def _params(q, chi1, chi2):
    X1 = q / (1.0 + q)
    X2 = 1.0 / (1.0 + q)
    nu = X1 * X2
    delta = (X1 - X2)  # = (m1 - m2)/M
    chiS = 0.5 * (chi1 + chi2)
    chiA = 0.5 * (chi1 - chi2)
    return nu, delta, chiS, chiA


def Ehat(q, x, e=0.0, chi1=0.0, chi2=0.0, kapS=0.0, kapA=0.0):
    """Conservative orbit-averaged energy per reduced mass, Ehat = E/mu, 3PN.

    Translation of ``Heob`` minus the rest-mass constant 1/(eps^2 nu) and the
    instantaneous Cos/Sin[zeta] term.
    """
    nu, delta, chiS, chiA = _params(q, chi1, chi2)
    om = 1.0 - e**2
    sq = np.sqrt(om)

    # non-spinning: 0PN, 1PN, 2PN, 3PN
    E = -x / 2
    E += (x**2 * (9 - e**2 * (-15 + nu) + nu)) / (24 * om)
    E += (x**3 * (201 - 105 * nu + nu**2 + 24 * om**1.5 * (-5 + 2 * nu)
                  + e**4 * (-15 + 15 * nu + nu**2)
                  - 2 * e**2 * (87 + 15 * nu + nu**2))) / (48 * om**2)
    E += -(x**4 * (-67635 - 27 * (-18319 + 492 * PI**2) * nu - 27378 * nu**2 - 35 * nu**3
                   + 5 * e**6 * (999 + 1215 * nu + 90 * nu**2 + 7 * nu**3)
                   + 3 * e**2 * (55539 - 9 * (-1651 + 123 * PI**2) * nu - 6078 * nu**2 + 35 * nu**3)
                   - 3 * e**4 * (33507 - 261 * nu + 66 * nu**2 + 35 * nu**3)
                   + 18 * om**1.5 * (720 + (-10256 + 123 * PI**2) * nu + 1056 * nu**2
                                     + 48 * e**2 * (-105 + 43 * nu + 8 * nu**2)))) / (10368 * om**3)

    # spin-orbit: 1.5PN
    E += (2 * x**2.5 * (-2 * delta * chiA + (-2 + nu) * chiS)) / (3 * om**1.5)
    # spin-spin: 2PN
    E += (x**3 * (delta * kapA + kapS - 2 * kapS * nu
                  + (1 - 4 * nu) * chiA**2 + 2 * delta * chiA * chiS + chiS**2)) / (2 * om**2)
    # spin-orbit: 2.5PN
    E += (x**3.5 * (delta * (-144 + 55 * nu + 8 * e**2 * (21 + nu)) * chiA
                    + (-144 + 217 * nu - 14 * nu**2 - 4 * e**2 * (-42 + 10 * nu + nu**2)) * chiS
                    + om**1.5 * (-24 * delta * (-3 + nu) * chiA
                                 + 12 * (6 - 8 * nu + nu**2) * chiS))) / (18 * om**2.5)
    # spin-spin: 3PN
    E += (x**4 * (-3 * delta * kapA * (7 * (-9 + 5 * nu) + e**2 * (21 + 10 * nu))
                  + 3 * kapS * (63 - 161 * nu + 22 * nu**2 + e**2 * (-21 + 32 * nu + 8 * nu**2))
                  + (197 - 857 * nu + 132 * nu**2 + 3 * e**2 * (-117 + 461 * nu + 16 * nu**2)) * chiA**2
                  - 2 * delta * (-197 + 253 * nu + 3 * e**2 * (117 + nu)) * chiA * chiS
                  + (197 - 437 * nu + 128 * nu**2 + 3 * e**2 * (-117 + 5 * nu)) * chiS**2
                  + om**1.5 * (6 * (delta * kapA * (-14 + 5 * nu) + kapS * (-14 + 33 * nu - 6 * nu**2))
                               - 12 * (11 - 46 * nu + 6 * nu**2) * chiA**2
                               + 24 * delta * (-11 + 9 * nu) * chiA * chiS
                               - 12 * (11 - 16 * nu + 4 * nu**2) * chiS**2))) / (36 * om**3)
    return E


def Lhat(q, x, e=0.0, chi1=0.0, chi2=0.0, kapS=0.0, kapA=0.0):
    """Conservative orbit-averaged angular momentum per reduced mass,
    Lhat = L/mu, 3PN.

    Translation of ``Leob`` minus the instantaneous Cos/Sin[zeta] term.
    """
    nu, delta, chiS, chiA = _params(q, chi1, chi2)
    om = 1.0 - e**2
    sq = np.sqrt(om)

    # non-spinning: 0PN, 1PN, 2PN, 3PN
    L = sq / np.sqrt(x)
    L += (np.sqrt(x) * (9 - e**2 * (-9 + nu) + nu)) / (6 * sq)
    L += (x**1.5 * (141 - 81 * nu + nu**2 - 2 * e**2 * nu * (19 + nu)
                    + om**1.5 * (-60 + 24 * nu)
                    + e**4 * (-3 + 11 * nu + nu**2))) / (24 * om**1.5)
    L += (x**2.5 * (100440 + 54 * (-12604 + 369 * PI**2) * nu + 32400 * nu**2 + 56 * nu**3
                    + 3 * e**2 * (-7992 + 9 * (-23512 + 861 * PI**2) * nu + 6144 * nu**2 - 56 * nu**3)
                    + 24 * e**4 * (-1431 + 1521 * nu - 204 * nu**2 + 7 * nu**3)
                    - 8 * e**6 * (837 + 621 * nu + 72 * nu**2 + 7 * nu**3)
                    - 18 * om**1.5 * (720 + (-10256 + 123 * PI**2) * nu + 1056 * nu**2
                                      + 48 * e**2 * (-75 + 31 * nu + 8 * nu**2)))) / (10368 * om**2.5)

    # spin-orbit: 1.5PN
    L += -((5 + 3 * e**2) * x * (2 * delta * chiA + (2 - nu) * chiS)) / (3 * om)
    # spin-spin: 2PN
    L += (x**1.5 * ((2 + e**2) * (delta * kapA + kapS - 2 * kapS * nu)
                    - (2 + 3 * e**2) * (-1 + 4 * nu) * chiA**2
                    + 2 * (2 + 3 * e**2) * delta * chiA * chiS
                    + (2 + 3 * e**2) * chiS**2)) / (2 * om**1.5)
    # spin-orbit: 2.5PN
    L += (x**2 * (delta * (-1584 + 626 * nu + 1063 * e**2 * nu + 3 * e**4 * (144 + 49 * nu)) * chiA
                  + (e**2 * (3133 - 410 * nu) * nu + e**4 * (432 + 201 * nu - 114 * nu**2)
                     - 2 * (792 - 1231 * nu + 62 * nu**2)) * chiS
                  + om**1.5 * (-192 * delta * (-3 + nu) * chiA
                               + 96 * (6 - 8 * nu + nu**2) * chiS))) / (144 * om**2)
    # spin-spin: 3PN
    L += -(x**2.5 * (-3 * (delta * kapA * (84 + e**2 * (63 - 46 * nu) + 3 * e**4 * (-5 + nu) - 50 * nu)
                          + kapS * (84 - 218 * nu + 28 * nu**2 - 3 * e**4 * (5 - 11 * nu + 2 * nu**2)
                                    + e**2 * (63 - 172 * nu + 8 * nu**2)))
                     + (27 * e**4 * (9 - 37 * nu + 4 * nu**2) - 3 * e**2 * (-67 + 237 * nu + 40 * nu**2)
                        - 4 * (59 - 260 * nu + 42 * nu**2)) * chiA**2
                     + 2 * delta * (e**4 * (243 - 99 * nu) + 3 * e**2 * (67 + 77 * nu)
                                    + 4 * (-59 + 85 * nu)) * chiA * chiS
                     + (e**2 * (201 + 369 * nu - 132 * nu**2) + 9 * e**4 * (27 - 19 * nu + 4 * nu**2)
                        - 4 * (59 - 146 * nu + 44 * nu**2)) * chiS**2
                     + om**1.5 * (6 * (delta * kapA * (14 - 5 * nu) + kapS * (14 - 33 * nu + 6 * nu**2))
                                  + 12 * (11 - 46 * nu + 6 * nu**2) * chiA**2
                                  + 24 * delta * (11 - 9 * nu) * chiA * chiS
                                  + 12 * (11 - 16 * nu + 4 * nu**2) * chiS**2))) / (36 * om**2.5)
    return L
