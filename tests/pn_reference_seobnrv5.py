"""
Reference PN-expanded binding energy E(v) and orbital angular momentum L(v)
for aligned-spin, circular-orbit binaries.

This is a faithful, line-by-line Python translation of the supplementary
Mathematica file ``PNexp_EOMs.m`` accompanying

    M. Khalil et al., "Theoretical groundwork supporting the precessing-spin
    two-body dynamics of the effective-one-body waveform models SEOBNRv5",
    arXiv:2303.18143.

Specifically it implements:
  - the binding energy, Eqs. (67)  -> ``binding_energy``
  - the orbital angular momentum vector L, Eqs. (65) -> ``orbital_angular_momentum``

reduced to the *aligned-spin* case (both spins parallel to the orbital
angular momentum unit vector lN = z-hat). In that case all cross-product
terms vanish and every dot product collapses to a z-component, so the vector
L is along z and only its magnitude L_z is returned.

This module is intended purely as ground truth for the test-suite; it is not
imported by the package itself.

Notation (matching the .m file), with total mass M = 1:
    nu   = m1 m2 / M^2          symmetric mass ratio
    mu   = M nu                 reduced mass
    X1   = m1/M, X2 = m2/M      (X1 >= X2, i.e. q = m1/m2 >= 1)
    v    = (M Omega)^(1/3)      so x = v^2 = (M Omega)^(2/3)
    S1, S2                      spin *vectors*; for aligned spins the
                                z-component is S_i = chi_i * m_i^2
    lNS1 = lN.S1 = S1 (aligned),  S1sq = S1.S1 = S1^2, etc.
    CES2_i                      spin-induced quadrupole constant
                                (= 1 for black holes -> C-tilde = 0)
"""

from __future__ import annotations

import numpy as np

PI = np.pi
GAMMA = np.euler_gamma
LOG2 = np.log(2.0)


def _masses(q):
    """Return (X1, X2, nu) for mass ratio q = m1/m2 >= 1, with M = 1."""
    X1 = q / (1.0 + q)
    X2 = 1.0 / (1.0 + q)
    nu = X1 * X2
    return X1, X2, nu


def _spin_scalars(chi1z, chi2z, X1, X2):
    """Aligned-spin scalar projections used by the PN expressions.

    Spin angular momenta S_i = chi_i * m_i^2 (z-components), with M = 1.
    For aligned spins every dot product reduces to these scalars.
    """
    S1 = chi1z * X1**2
    S2 = chi2z * X2**2
    lNS1 = S1
    lNS2 = S2
    S1sq = S1**2
    S2sq = S2**2
    S1S2 = S1 * S2
    lNS1sq = lNS1**2
    lNS2sq = lNS2**2
    return S1, S2, lNS1, lNS2, S1sq, S2sq, S1S2, lNS1sq, lNS2sq


def binding_energy(q, v, chi1z=0.0, chi2z=0.0, CES2_1=1.0, CES2_2=1.0):
    """Binding energy E(v), Eqs. (67) of arXiv:2303.18143, aligned spins, M = 1.

    Returns the (negative, for bound systems) binding energy. To compare with
    ``InitialEnergyMomenta._E0_from_PN`` note the code uses the opposite sign
    convention: code value  ==  -binding_energy.
    """
    X1, X2, nu = _masses(q)
    mu = nu  # M = 1
    M = 1.0
    (S1, S2, lNS1, lNS2, S1sq, S2sq, S1S2, lNS1sq, lNS2sq) = _spin_scalars(
        chi1z, chi2z, X1, X2
    )
    C21t = CES2_1 - 1.0
    C22t = CES2_2 - 1.0

    # non-spinning, ES0 (0PN..4PN)
    ES0 = mu * (
        -v**2 / 2
        + v**4 * (3/8 + nu/24)
        + v**6 * (27/16 - 19*nu/16 + nu**2/48)
        + v**8 * (675/128 + (-34445/1152 + 205*PI**2/192)*nu
                  + 155*nu**2/192 + 35*nu**3/10368)
        + v**10 * (3969/256 + (498449/6912 - 3157*PI**2/1152)*nu**2
                   - 301*nu**3/3456 - 77*nu**4/62208
                   + nu*(123671/11520 - 448*GAMMA/15 - 9037*PI**2/3072
                         - 896*LOG2/15 - 448*np.log(v)/15))
    )

    # spin-orbit, ESO (1.5PN, 2.5PN, 3.5PN)
    ESO = (
        lNS2 * (v**5 * (-X1 - nu/3)
                + v**7 * ((-5*nu/2 + nu**2/18) + X1*(-3/2 + 5*nu/3))
                + v**9 * ((-81*nu/8 + 55*nu**2/8 + nu**3/24)
                          + X1*(-27/8 + 39*nu/2 - 5*nu**2/8)))
        + lNS1 * (v**5 * (-X2 - nu/3)
                  + v**7 * ((-5*nu/2 + nu**2/18) + X2*(-3/2 + 5*nu/3))
                  + v**9 * ((-81*nu/8 + 55*nu**2/8 + nu**3/24)
                            + X2*(-27/8 + 39*nu/2 - 5*nu**2/8)))
    ) / M

    # spin1-spin2, ES1S2 (2PN, 3PN, 4PN)
    ES1S2 = (
        v**6 * (3*lNS1*lNS2*nu/2 - S1S2*nu/2)
        + v**8 * (5*S1S2*nu**2/12 + lNS1*lNS2*(5*nu/6 - 5*nu**2/36))
        + v**10 * (lNS1*lNS2*(21*nu/4 - 973*nu**2/144 - 721*nu**3/432)
                   + S1S2*(-7*nu/8 - 343*nu**2/48 - 7*nu**3/144))
    ) / M**3

    # spin-squared, ESsq (2PN, 3PN, 4PN)
    ESsq = (
        v**6 * (lNS2sq*(3*X1/4 - 3*nu/4) + lNS1sq*(3*X2/4 - 3*nu/4)
                + S2sq*(-X1/4 + nu/4) + S1sq*(-X2/4 + nu/4))
        + v**8 * (S2sq*(X1*(5/8 - 25*nu/24) - 5*nu/8 - 5*nu**2/24)
                  + S1sq*(X2*(5/8 - 25*nu/24) - 5*nu/8 - 5*nu**2/24)
                  + lNS2sq*(15*nu/8 + 85*nu**2/72 + X1*(-15/8 + 25*nu/8))
                  + lNS1sq*(15*nu/8 + 85*nu**2/72 + X2*(-15/8 + 25*nu/8)))
        + v**10 * (lNS2sq*(63*nu/32 + 637*nu**2/96 - 847*nu**3/864
                           + X1*(-63/32 + 679*nu/96 - 2695*nu**2/288))
                   + lNS1sq*(63*nu/32 + 637*nu**2/96 - 847*nu**3/864
                             + X2*(-63/32 + 679*nu/96 - 2695*nu**2/288))
                   + S2sq*(-21*nu/32 + 385*nu**2/96 + 7*nu**3/288
                           + X1*(21/32 - 637*nu/96 + 413*nu**2/288))
                   + S1sq*(-21*nu/32 + 385*nu**2/96 + 7*nu**3/288
                           + X2*(21/32 - 637*nu/96 + 413*nu**2/288)))
    ) / (M**2 * mu)

    # spin-quadrupole (C-tilde), ESsqC; vanishes for black holes
    ESsqC = (
        C22t * (
            v**6 * (lNS2sq*(3*X1/4 - 3*nu/4) + S2sq*(-X1/4 + nu/4))
            + v**8 * (S2sq*(X1*(-5/8 - 5*nu/8) + 5*nu/8 - 5*nu**2/24)
                      + lNS2sq*(-15*nu/8 + 5*nu**2/8 + X1*(15/8 + 15*nu/8)))
            + v**10 * (lNS2sq*(-189*nu/32 + 295*nu**2/32 - 7*nu**3/96
                               + X1*(189/32 + 77*nu/32 - 91*nu**2/32))
                       + S2sq*(63*nu/32 - 295*nu**2/96 + 7*nu**3/288
                               + X1*(-63/32 - 77*nu/96 + 91*nu**2/96)))
        )
        + C21t * (
            v**6 * (lNS1sq*(3*X2/4 - 3*nu/4) + S1sq*(-X2/4 + nu/4))
            + v**8 * (S1sq*(X2*(-5/8 - 5*nu/8) + 5*nu/8 - 5*nu**2/24)
                      + lNS1sq*(-15*nu/8 + 5*nu**2/8 + X2*(15/8 + 15*nu/8)))
            + v**10 * (lNS1sq*(-189*nu/32 + 295*nu**2/32 - 7*nu**3/96
                               + X2*(189/32 + 77*nu/32 - 91*nu**2/32))
                       + S1sq*(63*nu/32 - 295*nu**2/96 + 7*nu**3/288
                               + X2*(-63/32 - 77*nu/96 + 91*nu**2/96)))
        )
    ) / (M**2 * mu)

    return ES0 + ESO + ES1S2 + ESsq + ESsqC


def orbital_angular_momentum(q, v, chi1z=0.0, chi2z=0.0, CES2_1=1.0, CES2_2=1.0):
    """Orbital angular momentum magnitude L_z(v), Eqs. (65) of arXiv:2303.18143,
    aligned spins, M = 1.

    For aligned spins L points along z; we return L_z = L . lN. Each block in
    the .m file has the form (A)*lN[i] + (B1)*S1[i] + (B2)*S2[i]; the
    z-projection is A + B1*S1 + B2*S2, with lNS1->S1, lNS2->S2, lNS1^2->S1sq.
    """
    X1, X2, nu = _masses(q)
    mu = nu  # M = 1
    M = 1.0
    (S1, S2, lNS1, lNS2, S1sq, S2sq, S1S2, lNS1sq, lNS2sq) = _spin_scalars(
        chi1z, chi2z, X1, X2
    )
    C21t = CES2_1 - 1.0
    C22t = CES2_2 - 1.0

    # non-spinning, LS0 (Newtonian..4PN); coefficient of lN
    LS0 = M * mu * (
        v**(-1)
        + v * (3/2 + nu/6)
        + v**3 * (27/8 - 19*nu/8 + nu**2/24)
        + v**5 * (135/16 + (-6889/144 + 41*PI**2/24)*nu + 31*nu**2/24 + 7*nu**3/1296)
        + v**7 * (2835/128 + (356035/3456 - 2255*PI**2/576)*nu**2
                  - 215*nu**3/1728 - 55*nu**4/31104
                  + nu*(98869/5760 - 128*GAMMA/3 - 6455*PI**2/1536
                        - 256*LOG2/3 - 128*np.log(v)/3))
    )

    # spin-orbit, LSO (1.5PN, 2.5PN, 3.5PN)
    LSO = (
        v**2 * ((lNS2*(-7*X1/4 - 7*nu/12) + lNS1*(-7*X2/4 - 7*nu/12))
                + (-3*X2/4 - nu/4)*S1 + (-3*X1/4 - nu/4)*S2)
        + v**4 * ((lNS2*(-55*nu/16 + 11*nu**2/144 + X1*(-33/16 + 55*nu/24))
                   + lNS1*(-55*nu/16 + 11*nu**2/144 + X2*(-33/16 + 55*nu/24)))
                  + (X2*(-9/16 + 5*nu/8) - 15*nu/16 + nu**2/48)*S1
                  + (X1*(-9/16 + 5*nu/8) - 15*nu/16 + nu**2/48)*S2)
        + v**6 * ((lNS2*(-405*nu/32 + 275*nu**2/32 + 5*nu**3/96
                         + X1*(-135/32 + 195*nu/8 - 25*nu**2/32))
                   + lNS1*(-405*nu/32 + 275*nu**2/32 + 5*nu**3/96
                           + X2*(-135/32 + 195*nu/8 - 25*nu**2/32)))
                  + (-81*nu/32 + 55*nu**2/32 + nu**3/96
                     + X2*(-27/32 + 39*nu/8 - 5*nu**2/32))*S1
                  + (-81*nu/32 + 55*nu**2/32 + nu**3/96
                     + X1*(-27/32 + 39*nu/8 - 5*nu**2/32))*S2)
    )

    # spin1-spin2, LS1S2 (1.5PN..)
    LS1S2 = (nu * (
        v**3 * ((2*lNS1*lNS2 - S1S2) + (lNS2/2)*S1 + (lNS1/2)*S2)
        + v**5 * ((lNS1*lNS2*(-7/6 + 13*nu/36) + 2*S1S2*nu/3)
                  + lNS2*(5/4 - 7*nu/24)*S1 + lNS1*(5/4 - 7*nu/24)*S2)
        + v**7 * ((lNS1*lNS2*(15/4 + 361*nu/288 - 361*nu**2/432)
                   + S1S2*(-5/4 - 245*nu/24 - 5*nu**2/72))
                  + lNS2*(15/8 - 349*nu/64 - 223*nu**2/288)*S1
                  + lNS1*(15/8 - 349*nu/64 - 223*nu**2/288)*S2)
    )) / (M * mu)

    # spin-squared, LSsq
    LSsq = (
        v**3 * ((lNS2sq*(X1 - nu) + lNS1sq*(X2 - nu)
                 + S2sq*(-X1/2 + nu/2) + S1sq*(-X2/2 + nu/2))
                + lNS1*(X2/2 - nu/2)*S1 + lNS2*(X1/2 - nu/2)*S2)
        + v**5 * ((S2sq*(X1*(1 - 5*nu/3) - nu - nu**2/3)
                   + S1sq*(X2*(1 - 5*nu/3) - nu - nu**2/3)
                   + lNS2sq*(35*nu/8 + 121*nu**2/72 + X1*(-35/8 + 11*nu/2))
                   + lNS1sq*(35*nu/8 + 121*nu**2/72 + X2*(-35/8 + 11*nu/2)))
                  + lNS1*(X2*(11/8 - nu/2) - 11*nu/8 + 5*nu**2/24)*S1
                  + lNS2*(X1*(11/8 - nu/2) - 11*nu/8 + 5*nu**2/24)*S2)
        + v**7 * ((lNS2sq*(111*nu/32 + 347*nu**2/96 - 505*nu**3/864
                           + X1*(-111/32 + 199*nu/16 - 2833*nu**2/288))
                   + lNS1sq*(111*nu/32 + 347*nu**2/96 - 505*nu**3/864
                             + X2*(-111/32 + 199*nu/16 - 2833*nu**2/288))
                   + S2sq*(-15*nu/16 + 275*nu**2/48 + 5*nu**3/144
                           + X1*(15/16 - 455*nu/48 + 295*nu**2/144))
                   + S1sq*(-15*nu/16 + 275*nu**2/48 + 5*nu**3/144
                           + X2*(15/16 - 455*nu/48 + 295*nu**2/144)))
                  + lNS1*(-21*nu/32 + 563*nu**2/96 - 235*nu**3/288
                          + X2*(21/32 - 7*nu/3 - 113*nu**2/32))*S1
                  + lNS2*(-21*nu/32 + 563*nu**2/96 - 235*nu**3/288
                          + X1*(21/32 - 7*nu/3 - 113*nu**2/32))*S2)
    ) / (M * mu)

    # spin-quadrupole (C-tilde), LSsqC; vanishes for black holes (coeff of lN)
    LSsqC = (
        C22t * (
            v**3 * (lNS2sq*(3*X1/2 - 3*nu/2) + S2sq*(-X1/2 + nu/2))
            + v**5 * (S2sq*(X1*(-1 - nu) + nu - nu**2/3)
                      + lNS2sq*(-3*nu + nu**2 + X1*(3 + 3*nu)))
            + v**7 * (lNS2sq*(-135*nu/16 + 1475*nu**2/112 - 5*nu**3/48
                              + X1*(135/16 + 55*nu/16 - 65*nu**2/16))
                      + S2sq*(45*nu/16 - 1475*nu**2/336 + 5*nu**3/144
                              + X1*(-45/16 - 55*nu/48 + 65*nu**2/48)))
        ) / (M * mu)
        + C21t * (
            v**3 * (lNS1sq*(3*X2/2 - 3*nu/2) + S1sq*(-X2/2 + nu/2))
            + v**5 * (S1sq*(X2*(-1 - nu) + nu - nu**2/3)
                      + lNS1sq*(-3*nu + nu**2 + X2*(3 + 3*nu)))
            + v**7 * (lNS1sq*(-135*nu/16 + 1475*nu**2/112 - 5*nu**3/48
                              + X2*(135/16 + 55*nu/16 - 65*nu**2/16))
                      + S1sq*(45*nu/16 - 1475*nu**2/336 + 5*nu**3/144
                              + X2*(-45/16 - 55*nu/48 + 65*nu**2/48)))
        ) / (M * mu)
    )

    return LS0 + LSO + LS1S2 + LSsq + LSsqC
