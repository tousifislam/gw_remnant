"""
Experimental combined PN binding energy and angular momentum.

Recipe under investigation (per user request):
  base   : SEOBNRv5EHM Heob/Leob  -> 3PN, eccentric + aligned spin
           (investigations/ehm_reference.py)
  + 4PN  : exact analytic non-spinning, circular (e = 0) term
  + 5PN  : Le Tiec et al. NR-calibrated pseudo non-spinning, circular term
  (+ optional precessing-spin augmentation from File 1 -- not yet added)

All "augmentation" terms are non-spinning and circular: they are added only at
e = 0 because the 4PN/5PN eccentricity corrections are not available here.

Returns are per reduced mass (hat = quantity / mu, M = 1). Physical values are
nu * hat. Sign bridge to the package:
    code._E0_from_PN  ==  -nu * Ehat_combined   (= -E_binding)
    code._L0_from_PN  ==  +nu * Lhat_combined
"""

from __future__ import annotations

import numpy as np

import ehm_reference as ehm

PI = np.pi
GAMMA = np.euler_gamma
LOG2 = np.log(2.0)
GAMMA5 = -37.72                      # Le Tiec et al., Table II
E5 = 3 * GAMMA5 + 9359293 / 161280   # Eq. (4.24b) of arXiv:1111.5378


def _nu(q):
    return (q / (1 + q)) * (1 / (1 + q))


def energy_4pn_circular(nu, x):
    """Exact 4PN non-spinning circular term of Ehat = E/mu, i.e. the x^5 term
    of -x/2 (1 + ... + e4 x^4). Matches File 1 ES0/mu at v^10."""
    e4 = (-3969/128
          + (-123671/5760 + 9037/1536 * PI**2 + 896/15 * GAMMA
             + 448/15 * np.log(16 * x)) * nu
          + (-498449/3456 + 3157/576 * PI**2) * nu**2
          + 301/1728 * nu**3 + 77/31104 * nu**4)
    return -0.5 * x * (e4 * x**4)


def energy_5pn_circular(nu, x):
    """5PN pseudo (NR-calibrated) non-spinning circular term of Ehat."""
    e5term = (45927/512 + nu * E5 + (4988/35 - 656 * nu / 5) * nu * np.log(x))
    return -0.5 * x * (e5term * x**5)


def L_4pn_circular(nu, x):
    """Exact analytic 4PN non-spinning circular term of Lhat = L/mu, i.e. the
    x^4 term of (1/sqrt(x)) (1 + ... + l4 x^4). From File 1 LS0/mu at v^7."""
    l4 = (2835/128
          + (356035/3456 - 2255/576 * PI**2) * nu**2
          - 215/1728 * nu**3 - 55/31104 * nu**4
          + nu * (98869/5760 - 128/3 * GAMMA - 6455/1536 * PI**2
                  - 256/3 * LOG2 - 128/3 * 0.5 * np.log(x)))
    return (1 / np.sqrt(x)) * (l4 * x**4)


def L_5pn_circular(nu, x):
    """5PN pseudo (NR-calibrated) non-spinning circular term of Lhat."""
    j5 = -(2/3) * E5 - 4988/945 - 656 * nu / 135
    l5 = (15309/256 + nu * j5 + (9976/105 + 1312 * nu / 15) * nu * np.log(x))
    return (1 / np.sqrt(x)) * (l5 * x**5)


def Ehat_combined(q, x, e=0.0, chi1=0.0, chi2=0.0):
    """Combined Ehat = E/mu: EHM 3PN base + circular 4PN + 5PN non-spin."""
    nu = _nu(q)
    E = ehm.Ehat(q, x, e, chi1, chi2)
    if e == 0.0:
        E += energy_4pn_circular(nu, x) + energy_5pn_circular(nu, x)
    return E


def Lhat_combined(q, x, e=0.0, chi1=0.0, chi2=0.0):
    """Combined Lhat = L/mu: EHM 3PN base + circular 4PN + 5PN non-spin."""
    nu = _nu(q)
    L = ehm.Lhat(q, x, e, chi1, chi2)
    if e == 0.0:
        L += L_4pn_circular(nu, x) + L_5pn_circular(nu, x)
    return L


def E_binding(q, x, e=0.0, chi1=0.0, chi2=0.0):
    return _nu(q) * Ehat_combined(q, x, e, chi1, chi2)


def L_orbital(q, x, e=0.0, chi1=0.0, chi2=0.0):
    return _nu(q) * Lhat_combined(q, x, e, chi1, chi2)
