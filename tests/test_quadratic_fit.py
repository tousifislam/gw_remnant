"""Unit tests for the quadratic peak finder in LinearMomentumCalculator."""

import numpy as np
import pytest

from gw_remnant.remnant_calculators.kick_velocity_calculator import (
    LinearMomentumCalculator,
)


def _make_stub():
    """Create a minimal stub instance to access _get_peak_via_quadratic_fit."""
    stub = object.__new__(LinearMomentumCalculator)
    return stub


class TestQuadraticPeakFinder:
    def test_quadratic_peak_exact(self):
        """Known quadratic f(t) = -(t - 5)^2 + 10 should peak at t=5, f=10."""
        stub = _make_stub()
        t = np.linspace(0, 10, 101)
        func = -(t - 5.0) ** 2 + 10.0
        t_peak, f_peak = stub._get_peak_via_quadratic_fit(t, func)
        assert t_peak == pytest.approx(5.0, abs=1e-8)
        assert f_peak == pytest.approx(10.0, abs=1e-8)

    def test_lstsq_vs_analytic(self):
        """lstsq coefficients for f(t) = 3 + 2t - t^2 match expected [3, 2, -1]."""
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        func = 3.0 + 2.0 * t - t**2
        A = np.column_stack([np.ones(5), t, t**2])
        coefs, _, _, _ = np.linalg.lstsq(A, func, rcond=None)
        np.testing.assert_allclose(coefs, [3.0, 2.0, -1.0], atol=1e-12)
