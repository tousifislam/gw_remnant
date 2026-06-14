#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : trajectory_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes the center-of-mass trajectory of a binary black hole merger by
#   integrating the gravitational-wave recoil velocity in time.
#
#====================================================================================

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate

from .kick_velocity_calculator import LinearMomentumCalculator


class TrajectoryCalculator(LinearMomentumCalculator):
    """
    Calculator for the center-of-mass trajectory of a binary black hole merger.

    Integrates the recoil (center-of-mass) velocity v(t) in time to obtain the
    displacement x(t) = \\int v(t) dt. The binary trajectory is a coordinate-
    dependent notion, but the time integral of the linear momentum radiated in
    gravitational waves can be interpreted as the motion of the spacetime's
    center of mass as seen by an observer at future null infinity.

    All calculations are performed in geometric units where G=c=1, with positions
    in units of total mass M and velocities in units of speed of light c.

    Args:
        time (np.ndarray): Array of time values in geometric units (M)
        hdict (dict): Dictionary of complex waveform modes with (l,m) tuple keys,
            e.g., {(2,2): h_22(t), (3,3): h_33(t), ...}
        qinput (float): Mass ratio q = m1/m2, where m1 >= m2
        spin1_input (list or np.ndarray): Spin vector [sx, sy, sz] for primary black
            hole at the start of the waveform, in dimensionless units. Default is None
        spin2_input (list or np.ndarray): Spin vector [sx, sy, sz] for secondary black
            hole at the start of the waveform, in dimensionless units. Default is None
        ecc_input (float): Eccentricity at the reference time. User must provide
            accurate value; code does not validate. Default is None
        E_initial (float): Initial binding-energy magnitude in units of total mass M.
            If None, computed using PN expressions. Default is None
        L_initial (float): Initial angular momentum in units of M^2. If None,
            computed using PN expressions. Default is None
        M_initial (float): Initial total mass of the binary in units of M. Default is 1
        use_filter (bool): Whether to apply filtering to computed quantities.
            Default is False

    Attributes:
        xoft (np.ndarray): Center-of-mass displacement vector [N_times x 3] in units of M
        remnant_displacement (np.ndarray): Final center-of-mass displacement vector (3,)
            in units of M

    Inherits From:
        LinearMomentumCalculator: Provides the center-of-mass velocity voft.

    References:
        Trajectory from the radiated linear momentum: Eqs. (13)-(14) and the
        following discussion of arXiv:1802.04276.
        Interpretation as the center-of-mass motion seen at future null infinity:
        E. E. Flanagan and D. A. Nichols, PRD 95, 044002 (2017), arXiv:1510.03386.
    """

    def __init__(self, time: np.ndarray, hdict: dict[tuple[int, int], np.ndarray],
                 qinput: float, spin1_input: np.ndarray | list[float] | None = None,
                 spin2_input: np.ndarray | list[float] | None = None,
                 ecc_input: float | None = None, E_initial: float | None = None,
                 L_initial: float | None = None,
                 M_initial: float = 1, use_filter: bool = False) -> None:

        super().__init__(time, hdict, qinput, spin1_input, spin2_input,
                         ecc_input, E_initial, L_initial, M_initial, use_filter)

        self.xoft = self._compute_trajectory()
        self.remnant_displacement = self.xoft[-1]

    def _compute_trajectory(self):
        """
        Compute the center-of-mass trajectory.

        Integrates the recoil velocity v(t) in time, x(t) = \\int_0^t v(t') dt',
        using trapezoidal integration. See the discussion following Eq. (14) of
        arXiv:1802.04276.

        Returns:
            [np.ndarray]: Displacement vector [N_times x 3] in units of M.
        """
        v = np.transpose(self.voft)  # [3 x N_times]
        try:
            xoft = integrate.cumulative_trapezoid(v[0], self.time, initial=0.0)
            yoft = integrate.cumulative_trapezoid(v[1], self.time, initial=0.0)
            zoft = integrate.cumulative_trapezoid(v[2], self.time, initial=0.0)
        except AttributeError:
            # scipy < 1.14: cumulative_trapezoid was named cumtrapz
            xoft = integrate.cumtrapz(v[0], self.time, initial=0.0)
            yoft = integrate.cumtrapz(v[1], self.time, initial=0.0)
            zoft = integrate.cumtrapz(v[2], self.time, initial=0.0)
        return np.transpose(np.array([xoft, yoft, zoft]))
