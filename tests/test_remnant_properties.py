"""Integration tests for remnant property calculations using q=8 NR data."""

import numpy as np
import pytest

from gw_remnant.gw_remnant_calculator import GWRemnantCalculator


# Reference values for q=8 non-spinning (from tutorials)
REF = {
    "remnant_mass": 0.98958938,
    "remnant_spin": 0.30795092,
    "E_rad": 0.01041062,
    "L_peak": 0.00012866,
    "remnant_kick": 0.00025646,
}


@pytest.fixture
def q8_calc(q8_nr_data):
    """Build GWRemnantCalculator from q=8 NR fixture."""
    time, hdict, q = q8_nr_data
    return GWRemnantCalculator(time=time, hdict=hdict, qinput=q)


# ---------------------------------------------------------------------------
# Individual property tests
# ---------------------------------------------------------------------------

class TestRemnantValues:
    def test_remnant_mass(self, q8_calc):
        assert q8_calc.remnant_mass == pytest.approx(REF["remnant_mass"], abs=1e-4)

    def test_remnant_spin(self, q8_calc):
        assert q8_calc.remnant_spin == pytest.approx(REF["remnant_spin"], abs=1e-4)

    def test_radiated_energy(self, q8_calc):
        assert q8_calc.E_rad == pytest.approx(REF["E_rad"], abs=1e-4)

    def test_peak_luminosity(self, q8_calc):
        assert q8_calc.L_peak == pytest.approx(REF["L_peak"], abs=1e-4)

    def test_kick_velocity(self, q8_calc):
        assert q8_calc.remnant_kick == pytest.approx(REF["remnant_kick"], abs=1e-4)


# ---------------------------------------------------------------------------
# API / structural tests
# ---------------------------------------------------------------------------

class TestAPI:
    EXPECTED_KEYS = {
        "mass_ratio", "M_initial", "E_rad", "L_peak",
        "remnant_mass", "remnant_spin",
        "remnant_spin_x", "remnant_spin_y", "remnant_spin_z",
        "remnant_kick", "remnant_kick_kmps", "peak_kick",
    }

    def test_get_remnant_properties_keys(self, q8_calc):
        props = q8_calc.get_remnant_properties()
        assert set(props.keys()) == self.EXPECTED_KEYS

    def test_print_remnants(self, q8_calc, capsys):
        q8_calc.print_remnants()
        captured = capsys.readouterr()
        assert "Remnant Properties Summary" in captured.out
        assert "Remnant mass" in captured.out


# ---------------------------------------------------------------------------
# Relative-mode test (E_initial=0, L_initial=0)
# ---------------------------------------------------------------------------

class TestSpinVector:
    def test_spin_vector_shape(self, q8_calc):
        """spin_vector_oft should be (3, N) and remnant_spin_vector should be (3,)."""
        assert q8_calc.spin_vector_oft.shape[0] == 3
        assert q8_calc.spin_vector_oft.shape[1] == len(q8_calc.time)
        assert q8_calc.remnant_spin_vector.shape == (3,)

    def test_spin_z_matches_scalar(self, q8_calc):
        """z-component of remnant_spin_vector should match scalar remnant_spin."""
        assert q8_calc.remnant_spin_vector[2] == pytest.approx(q8_calc.remnant_spin)

    def test_nonspinning_xy_near_zero(self, q8_calc):
        """For q=8 non-spinning, x and y spin components should be near zero."""
        assert q8_calc.remnant_spin_vector[0] == pytest.approx(0.0, abs=1e-3)
        assert q8_calc.remnant_spin_vector[1] == pytest.approx(0.0, abs=1e-3)

    def test_spin_z_matches_reference(self, q8_calc):
        """z-component should match the known reference value."""
        assert q8_calc.remnant_spin_vector[2] == pytest.approx(REF["remnant_spin"], abs=1e-4)


class TestRelativeMode:
    def test_relative_mode(self, q8_nr_data):
        """E_initial=0, L_initial=0 tracks changes relative to reference."""
        time, hdict, q = q8_nr_data
        calc = GWRemnantCalculator(
            time=time, hdict=hdict, qinput=q,
            E_initial=0, L_initial=0,
        )
        props = calc.get_remnant_properties()
        # All values should be finite
        for key, val in props.items():
            assert np.isfinite(val), f"{key} is not finite"
        # Remnant mass should still be positive
        assert props["remnant_mass"] > 0
