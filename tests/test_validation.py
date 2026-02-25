"""Tests for input validation in InitialEnergyMomenta._validate_inputs()."""

import numpy as np
import pytest

from gw_remnant.remnant_calculators.initial_energy_momenta import InitialEnergyMomenta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_args():
    """Return minimal valid arguments for _validate_inputs."""
    time = np.linspace(-100, 10, 1101)
    h22 = 0.1 * np.exp(-1j * 0.04 * time)
    hdict = {(2, 2): h22}
    return time, hdict, 2.0, None, None, None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTimeValidation:
    def test_time_too_short(self):
        time = np.array([0.0])
        _, hdict, q, s1, s2, ecc = _valid_args()
        hdict = {(2, 2): np.array([1.0 + 0j])}
        with pytest.raises(ValueError, match="at least 2 elements"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, s1, s2, ecc)

    def test_time_non_monotonic(self):
        time = np.array([0.0, 1.0, 0.5, 2.0])
        _, hdict, q, s1, s2, ecc = _valid_args()
        hdict = {(2, 2): np.ones(4, dtype=complex)}
        with pytest.raises(ValueError, match="monotonically increasing"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, s1, s2, ecc)


class TestModeValidation:
    def test_missing_22_mode(self):
        time, _, q, s1, s2, ecc = _valid_args()
        hdict = {(3, 3): np.ones(len(time), dtype=complex)}
        with pytest.raises(ValueError, match=r"\(2,\s*2\) mode"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, s1, s2, ecc)


class TestMassRatioValidation:
    def test_mass_ratio_below_one(self):
        time, hdict, _, s1, s2, ecc = _valid_args()
        with pytest.raises(ValueError, match="q must be >= 1"):
            InitialEnergyMomenta._validate_inputs(time, hdict, 0.5, s1, s2, ecc)


class TestSpinValidation:
    def test_spin_wrong_shape(self):
        time, hdict, q, _, _, ecc = _valid_args()
        bad_spin = np.array([0.1, 0.2])  # shape (2,) instead of (3,)
        with pytest.raises(ValueError, match="3-component vector"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, bad_spin, None, ecc)

    def test_spin_magnitude_exceeds_one(self):
        time, hdict, q, _, _, ecc = _valid_args()
        big_spin = np.array([0.0, 0.0, 1.1])
        with pytest.raises(ValueError, match="magnitude must be <= 1"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, big_spin, None, ecc)


class TestEccentricityValidation:
    def test_eccentricity_negative(self):
        time, hdict, q, s1, s2, _ = _valid_args()
        with pytest.raises(ValueError, match="0 <= e < 1"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, s1, s2, -0.1)

    def test_eccentricity_one(self):
        time, hdict, q, s1, s2, _ = _valid_args()
        with pytest.raises(ValueError, match="0 <= e < 1"):
            InitialEnergyMomenta._validate_inputs(time, hdict, q, s1, s2, 1.0)


class TestMInitialValidation:
    def test_negative_initial_mass(self):
        """M_initial <= 0 raises ValueError (validated in RemnantMassCalculator)."""
        from gw_remnant.gw_remnant_calculator import GWRemnantCalculator

        time, hdict, q = _valid_args()[:3]
        with pytest.raises(ValueError, match="M_initial must be positive"):
            GWRemnantCalculator(time, hdict, q, M_initial=-1.0)


class TestValidInputs:
    def test_valid_inputs_pass(self):
        """No error raised with valid inputs."""
        args = _valid_args()
        InitialEnergyMomenta._validate_inputs(*args)  # should not raise
