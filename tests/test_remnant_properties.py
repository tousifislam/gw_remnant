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
        "remnant_mass", "remnant_spin", "remnant_kick",
        "remnant_kick_kmps", "peak_kick",
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
