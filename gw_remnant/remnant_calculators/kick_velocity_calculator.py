#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : kick_velocity_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes linear momentum flux and kick velocity of binary black hole mergers
#   from gravitational waveforms. Calculates time evolution of linear momentum,
#   velocity, and final recoil (kick) velocity of the remnant black hole.
#
#====================================================================================

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
import lal

from .remnant_mass_calculator import RemnantMassCalculator


class LinearMomentumCalculator(RemnantMassCalculator):
    """
    Calculator for linear momentum and kick velocity of binary black hole mergers.
    
    This class computes the linear momentum carried away by gravitational waves
    and the resulting recoil (kick) velocity imparted to the remnant black hole.
    The calculations use angular momentum flux formulas from gravitational wave
    multipoles.
    
    All calculations are performed in geometric units where G=c=1, with masses
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
        E_initial (float): Initial energy of the binary in units of total mass M.
            If None, computed using PN expressions. Set to 0 to inspect energy changes
            relative to reference. Default is None
        L_initial (float): Initial angular momentum of the binary in units of M^2.
            If None, computed using PN expressions. Set to 0 to inspect angular momentum
            changes relative to reference. Default is None
        M_initial (float): Initial total mass of the binary in units of M. Default is 1
        use_filter (bool): Whether to apply filtering to computed quantities.
            Default is False
    
    Attributes:
        lmax (int): Maximum l value of modes in the waveform
        P_dot (np.ndarray): Linear momentum flux vector [3 x N_times] in units of M
        Poft (np.ndarray): Cumulative radiated linear momentum [3 x N_times] in units of M
        voft (np.ndarray): Center of mass velocity vector [3 x N_times] in units of c
        kickoft (np.ndarray): Kick velocity magnitude as a function of time in units of c
        remnant_kick (float): Final kick velocity magnitude in units of c
        peak_kick (float): Peak kick velocity magnitude in units of c
    
    Inherits From:
        RemnantMassCalculator: Provides mass and energy calculations
    
    References:
        Kick velocity formulas from arXiv:1802.04276 and arXiv:0707.4654
    """
    
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None, 
                 M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                         ecc_input, E_initial, L_initial, M_initial, use_filter)
        
        self.lmax = self._get_lmax()
        self.P_dot = np.array([self._compute_dPxdt(), self._compute_dPydt(), 
                               self._compute_dPzdt()])
        self.Poft = self._compute_Poft()
        self.voft = self._compute_voft()
        self.kickoft = self._compute_kickoft()
        self.kickoft_kmps = self._compute_kickoft_in_kmps()
        self.remnant_kick = self._compute_remnant_kick()
        self.remnant_kick_kmps = self._compute_remnant_kick_in_kmps()
        self.peak_kick = self._compute_peak_kick()
    
    def _read_dhdt_dict(self, l, m):
        """
        Retrieve time derivative of a specific waveform mode.
        
        Returns the time derivative of the (l,m) mode if it exists in the
        waveform dictionary. For unphysical modes (e.g., l<2 or |m|>l) or
        modes not present in the waveform, returns zeros.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [np.ndarray]: Complex time derivative of h_lm, or zeros if mode
                is unavailable or unphysical.
        """
        if l < 2 or l > self.lmax:
            return np.zeros(len(self.time), dtype=complex)
        elif m < -l or m > l:
            return np.zeros(len(self.time), dtype=complex)
        else:
            if (l, m) in self.hdict.keys():
                return self.h_dot[l, m]
            else:
                return np.zeros(len(self.time), dtype=complex)
    
    def _get_lmax(self):
        """
        Determine maximum l value in waveform.
        
        Returns:
            [int]: Maximum spherical harmonic degree l present in the waveform.
        """
        return max([mode[0] for mode in self.hdict.keys()])
        
    def _coeffs_a(self, l, m):
        """
        Compute coefficient a(l,m) for linear momentum flux.
        
        See Eq. (3.16) of arXiv:0707.4654.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [float]: Coefficient value.
        """
        return ((l - m) * (l + m + 1))**0.5 / (l * (l + 1))

    def _coeffs_b(self, l, m):
        """
        Compute coefficient b(l,m) for linear momentum flux.
        
        See Eq. (3.17) of arXiv:0707.4654.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [float]: Coefficient value.
        """
        return (1 / (2 * l)) * (((l - 2) * (l + 2) * (l + m) * (l + m - 1)) / 
                                ((2 * l - 1) * (2 * l + 1)))**0.5

    def _coeffs_c(self, l, m):
        """
        Compute coefficient c(l,m) for linear momentum flux.
        
        See Eq. (3.18) of arXiv:0707.4654.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [float]: Coefficient value.
        """
        return 2 * m / (l * (l + 1))

    def _coeffs_d(self, l, m):
        """
        Compute coefficient d(l,m) for linear momentum flux.
        
        See Eq. (3.19) of arXiv:0707.4654.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [float]: Coefficient value.
        """
        return (1 / l) * (((l - 2) * (l + 2) * (l - m) * (l + m)) / 
                         ((2 * l - 1) * (2 * l + 1)))**0.5

    def _compute_dPxdt(self):
        """
        Compute x-component of linear momentum flux.
        
        Calculates the time derivative of the x-component of radiated linear
        momentum using gravitational wave multipole formulas.
        
        See Eq. (6) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dPx/dt as a function of time in units of M.
        """
        dPxdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dPxdt += (1 / (8 * np.pi)) * np.real(
                self.h_dot[(l, m)] * (
                    self._coeffs_a(l, m) * np.conj(self._read_dhdt_dict(l, m + 1)) + 
                    self._coeffs_b(l, -m) * np.conj(self._read_dhdt_dict(l - 1, m + 1)) - 
                    self._coeffs_b(l + 1, m + 1) * np.conj(self._read_dhdt_dict(l + 1, m + 1))
                )
            )
        return dPxdt
    
    def _compute_dPydt(self):
        """
        Compute y-component of linear momentum flux.
        
        Calculates the time derivative of the y-component of radiated linear
        momentum using gravitational wave multipole formulas.
        
        See Eq. (7) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dPy/dt as a function of time in units of M.
        """
        dPydt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dPydt += (1 / (8 * np.pi)) * np.imag(
                self.h_dot[(l, m)] * (
                    self._coeffs_a(l, m) * np.conj(self._read_dhdt_dict(l, m + 1)) + 
                    self._coeffs_b(l, -m) * np.conj(self._read_dhdt_dict(l - 1, m + 1)) - 
                    self._coeffs_b(l + 1, m + 1) * np.conj(self._read_dhdt_dict(l + 1, m + 1))
                )
            )
        return dPydt
    
    def _compute_dPzdt(self):
        """
        Compute z-component of linear momentum flux.
        
        Calculates the time derivative of the z-component of radiated linear
        momentum using gravitational wave multipole formulas.
        
        See Eq. (8) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dPz/dt as a function of time in units of M.
        """
        dPzdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dPzdt += (1 / (16 * np.pi)) * np.real(
                self.h_dot[(l, m)] * (
                    self._coeffs_c(l, m) * np.conj(self._read_dhdt_dict(l, m)) + 
                    self._coeffs_d(l, m) * np.conj(self._read_dhdt_dict(l - 1, m)) + 
                    self._coeffs_d(l + 1, m) * np.conj(self._read_dhdt_dict(l + 1, m))
                )
            )
        return dPzdt
    
    def _compute_Poft(self):
        """
        Compute cumulative radiated linear momentum vector.
        
        Integrates the linear momentum flux to obtain the total radiated linear
        momentum as a function of time. Uses trapezoidal integration.
        
        See Eqs. (6)-(8) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: Radiated linear momentum vector [3 x N_times] in units of M.
        """
        try:
            Pxoft = integrate.cumtrapz(self.P_dot[0], self.time, initial=0.0)
            Pyoft = integrate.cumtrapz(self.P_dot[1], self.time, initial=0.0)
            Pzoft = integrate.cumtrapz(self.P_dot[2], self.time, initial=0.0)
        except:
            Pxoft = integrate.cumulative_trapezoid(self.P_dot[0], self.time, initial=0.0)
            Pyoft = integrate.cumulative_trapezoid(self.P_dot[1], self.time, initial=0.0)
            Pzoft = integrate.cumulative_trapezoid(self.P_dot[2], self.time, initial=0.0)
        return np.array([Pxoft, Pyoft, Pzoft])
    
    def _compute_voft(self):
        """
        Compute center of mass velocity vector.
        
        Calculates the velocity of the center of mass of the system as a
        function of time using the radiated momentum and remaining mass.
        
        See Eq. (13) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: Velocity vector [N_times x 3] in units of c.
        """
        return np.transpose(self.Poft / self.Moft)
    
    def _compute_kickoft(self):
        """
        Compute kick velocity magnitude evolution.
        
        Calculates the magnitude of the recoil (kick) velocity imparted to
        the remnant as a function of time.
        
        Returns:
            [np.ndarray]: Kick velocity magnitude as a function of time in units of c.
        """
        return np.array([np.linalg.norm(self.voft[i]) for i in range(len(self.time))])

    def _compute_kickoft_in_kmps(self):
        """
        Compute kick velocity magnitude evolution in km/sec.
        
        Calculates the magnitude of the recoil (kick) velocity imparted to
        the remnant as a function of time.
        
        Returns:
            [np.ndarray]: Kick velocity magnitude as a function of time in units of c.
        """
        return self.kickoft * lal.C_SI * 1e-3

    def _get_peak_via_quadratic_fit(self, t, func):
        """
        Find peak of a function using quadratic interpolation.
        
        Fits a quadratic polynomial to the 5 points nearest to the maximum
        of the function to obtain a more accurate peak location and value.
        
        Args:
            t (np.ndarray): Array of time values
            func (np.ndarray): Array of function values
        
        Returns:
            [tuple]: (t_peak, f_peak) where t_peak is the time of the peak and
                f_peak is the peak value of the function.
        """
        # Find the time closest to the peak, ensuring room on either side
        index = np.argmax(func)
        index = max(2, min(len(t) - 3, index))
        
        # Quadratic fit to 5 points, subtracting t[index] for numerical stability
        testTimes = t[index - 2:index + 3] - t[index]
        testFuncs = func[index - 2:index + 3]
        xVecs = np.array([np.ones(5), testTimes, testTimes**2])
        invMat = np.linalg.inv(np.array([[v1.dot(v2) for v1 in xVecs] 
                                         for v2 in xVecs]))
        yVec = np.array([testFuncs.dot(v1) for v1 in xVecs])
        coefs = np.array([yVec.dot(v1) for v1 in invMat])
        
        return t[index] - coefs[1] / (2.0 * coefs[2]), coefs[0] - coefs[1]**2 / 4 / coefs[2]
    
    def _compute_peak_kick(self):
        """
        Compute peak kick velocity.
        
        Finds the maximum value of the kick velocity profile using quadratic
        interpolation for improved accuracy.
        
        Returns:
            [float]: Peak kick velocity magnitude in units of c.
        """
        return self._get_peak_via_quadratic_fit(self.time, self.kickoft)[1]
    
    def _compute_remnant_kick(self):
        """
        Compute final remnant kick velocity.
        
        Returns the final kick velocity of the remnant black hole, taken as
        the last value of the kick velocity time series.
        
        See Eq. (14) of arXiv:1802.04276.
        
        Returns:
            [float]: Final kick velocity magnitude in units of c.
        """
        return self.kickoft[-1]

    def _compute_remnant_kick_in_kmps(self):
        """
        Compute final remnant kick velocity in km/sec.
        
        Returns the final kick velocity of the remnant black hole, taken as
        the last value of the kick velocity time series.
        
        See Eq. (14) of arXiv:1802.04276.
        
        Returns:
            [float]: Final kick velocity magnitude in units of c.
        """
        return self.kickoft[-1] * lal.C_SI * 1e-3