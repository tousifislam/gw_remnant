"""
Tests for the PN initial-condition expressions in ``InitialEnergyMomenta``.

The production defaults implement the SEOBNRv5EHM orbit-averaged energy/angular
momentum (3PN, eccentric + aligned spin) plus analytic-4PN / pseudo-5PN
non-spinning circular terms. Validation has three layers:

1. The independent SEOBNRv5 (precessing) transcription ``pn_reference_seobnrv5``
   (File 1) is self-checked via the first law and leading spin-orbit, then used
   as a *cross-paper* reference for the non-spinning sector.
2. The production code is checked against the independent EHM/combined recipe
   transcription ``combined_pn`` (port fidelity) across non-spin, aligned-spin
   and eccentric configurations.
3. Cross-paper sanity: the code's non-spinning energy/AM agree with File 1.

Sign bridge: code ``E_initial`` is the positive binding magnitude, so
-E_initial = E_bind; code ``L_initial`` = +L_orbital.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw_remnant.remnant_calculators.initial_energy_momenta import InitialEnergyMomenta
import pn_reference_seobnrv5 as f1     # File 1 (precessing) circular ground truth
import combined_pn as recipe           # EHM 3PN + 4PN/5PN non-spin recipe


# ----------------------------------------------------------------------------
# driver: build InitialEnergyMomenta with a prescribed PN parameter x0
# ----------------------------------------------------------------------------
def _make(q, x_target, chi1z=0.0, chi2z=0.0, ecc=0.0):
    omega = x_target**1.5
    t = np.linspace(0.0, 200.0, 8000)
    h22 = np.exp(-2j * omega * t)
    return InitialEnergyMomenta(
        t, {(2, 2): h22}, q,
        spin1_input=[0.0, 0.0, chi1z], spin2_input=[0.0, 0.0, chi2z],
        ecc_input=ecc,
    )


def _code_Ebind(q, x, chi1z=0.0, chi2z=0.0, ecc=0.0):
    return -_make(q, x, chi1z, chi2z, ecc).E_initial        # = E_binding


def _code_L(q, x, chi1z=0.0, chi2z=0.0, ecc=0.0):
    return _make(q, x, chi1z, chi2z, ecc).L_initial


QS = [1.0, 2.0, 4.0, 8.0]
SPINS = [(0.0, 0.0), (0.6, -0.3), (0.8, 0.5), (-0.4, 0.2)]


# ----------------------------------------------------------------------------
# 1. Validate the File-1 reference itself
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
def test_reference_satisfies_first_law_nonspinning(q):
    v0, dv = 0.25, 1e-6
    dE = (f1.binding_energy(q, v0 + dv) - f1.binding_energy(q, v0 - dv)) / (2 * dv)
    dL = (f1.orbital_angular_momentum(q, v0 + dv)
          - f1.orbital_angular_momentum(q, v0 - dv)) / (2 * dv)
    assert dE == pytest.approx(v0**3 * dL, rel=1e-8)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("chi1z, chi2z", [(0.6, -0.3), (0.8, 0.5)])
def test_reference_leading_spin_orbit(q, chi1z, chi2z):
    X1, X2, nu = q / (1 + q), 1 / (1 + q), (q / (1 + q)) * (1 / (1 + q))
    S1, S2 = chi1z * X1**2, chi2z * X2**2
    v = 1e-3
    E_SO = v**5 * (S1 * (-X2 - nu / 3) + S2 * (-X1 - nu / 3))
    L_SO = v**2 * (S1 * (-5 * X2 / 2 - 5 * nu / 6) + S2 * (-5 * X1 / 2 - 5 * nu / 6))
    dE = f1.binding_energy(q, v, chi1z, chi2z) - f1.binding_energy(q, v)
    dL = f1.orbital_angular_momentum(q, v, chi1z, chi2z) - f1.orbital_angular_momentum(q, v)
    assert dE == pytest.approx(E_SO, rel=1e-3)
    assert dL == pytest.approx(L_SO, rel=1e-3)


# ----------------------------------------------------------------------------
# 2. Production code reproduces the EHM/combined recipe it implements
#    (port fidelity: non-spin, aligned spin, and eccentric).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("chi1z, chi2z", SPINS)
def test_code_energy_matches_recipe_circular(q, chi1z, chi2z):
    x = 0.05
    assert _code_Ebind(q, x, chi1z, chi2z) == pytest.approx(
        recipe.E_binding(q, x, 0.0, chi1z, chi2z), rel=1e-6)


@pytest.mark.parametrize("q", QS)
@pytest.mark.parametrize("chi1z, chi2z", SPINS)
def test_code_L_matches_recipe_circular(q, chi1z, chi2z):
    x = 0.05
    assert _code_L(q, x, chi1z, chi2z) == pytest.approx(
        recipe.L_orbital(q, x, 0.0, chi1z, chi2z), rel=1e-6)


@pytest.mark.parametrize("q", [1.0, 3.0, 8.0])
@pytest.mark.parametrize("ecc", [0.1, 0.3])
@pytest.mark.parametrize("chi1z, chi2z", [(0.0, 0.0), (0.6, -0.3)])
def test_code_matches_recipe_eccentric(q, ecc, chi1z, chi2z):
    x = 0.05
    assert _code_Ebind(q, x, chi1z, chi2z, ecc) == pytest.approx(
        recipe.E_binding(q, x, ecc, chi1z, chi2z), rel=1e-6)
    assert _code_L(q, x, chi1z, chi2z, ecc) == pytest.approx(
        recipe.L_orbital(q, x, ecc, chi1z, chi2z), rel=1e-6)


# ----------------------------------------------------------------------------
# 3. Cross-paper sanity: non-spinning code vs File 1 (agree through 4PN)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("q", QS)
def test_code_nonspinning_matches_file1(q):
    x = 0.01
    v = np.sqrt(x)
    assert _code_Ebind(q, x) == pytest.approx(f1.binding_energy(q, v), rel=1e-7)
    assert _code_L(q, x) == pytest.approx(f1.orbital_angular_momentum(q, v), rel=1e-7)
