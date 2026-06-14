#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : remnant_mass_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes remnant mass and energy evolution for binary black hole mergers
#   from gravitational waveforms. This is the base class for all remnant
#   property calculations in the package.
#
#====================================================================================

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from scipy import signal
from scipy.interpolate import splev, splrep
import gwtools

from .initial_energy_momenta import InitialEnergyMomenta


class RemnantMassCalculator(InitialEnergyMomenta):
    """
    Calculator for remnant mass and energy evolution of binary black holes.
    
    This is the base class for all remnant property calculations in the package.
    It computes the time evolution of energy flux, radiated energy, and mass,
    culminating in the final remnant black hole mass.
    
    The class computes energy loss through gravitational wave emission by
    integrating the flux across all waveform modes. It supports optional
    filtering for smoothing noisy numerical data.
    
    Args:
        time (np.ndarray): Array of time values in geometric units (M)
        hdict (dict): Dictionary of complex waveform modes with (l,m) tuple keys,
            e.g., {(2,2): h_22(t), (3,3): h_33(t), ...}
        q (float): Mass ratio q = m1/m2, where m1 >= m2
        chi1 (list or np.ndarray): Spin vector [sx, sy, sz] for primary black
            hole at the start of the waveform, in dimensionless units. Default is None
        chi2 (list or np.ndarray): Spin vector [sx, sy, sz] for secondary black
            hole at the start of the waveform, in dimensionless units. Default is None
        e_ref (float): Eccentricity at the reference time. User must provide
            accurate value; code does not validate. Default is None
        E_initial (float): Initial energy of the binary in units of total mass M.
            If None, computed using PN expressions. Set to 0 to inspect energy changes
            relative to reference. Default is None
        L_initial (float): Initial angular momentum of the binary in units of M^2.
            If None, computed using PN expressions. Set to 0 to inspect angular momentum
            changes relative to reference. Default is None
        M_initial (float): Initial total mass of the binary in units of M. Default is 1
        use_filter (bool): Whether to apply spline filtering to smooth computed flux.
            Useful for noisy numerical data. Default is False
    
    Attributes:
        use_filter (bool): Whether filtering is applied
        M_initial (float): Initial total mass
        h_dot (dict): Time derivatives of waveform modes
        E_dot (np.ndarray): Energy flux (luminosity) as a function of time
        Eoft (np.ndarray): Cumulative radiated energy as a function of time
        E_rad (float): Total radiated energy
        Moft (np.ndarray): Bondi mass (dynamic mass) as a function of time
        remnant_mass (float): Final remnant black hole mass
    
    Inherits From:
        InitialEnergyMomenta: Provides initial condition calculations
    
    References:
        Energy flux formulas from arXiv:1802.04276
        Remnant mass from arXiv:2301.07215
    """
    
    def __init__(self, time: np.ndarray, hdict: dict[tuple[int, int], np.ndarray],
                 q: float, chi1: np.ndarray | list[float] | None = None,
                 chi2: np.ndarray | list[float] | None = None,
                 e_ref: float | None = None, E_initial: float | None = None,
                 L_initial: float | None = None,
                 M_initial: float = 1, use_filter: bool = False) -> None:
        
        super().__init__(time, hdict, q, chi1, chi2,
                         e_ref, E_initial, L_initial)

        if M_initial <= 0:
            raise ValueError(f"M_initial must be positive, got {M_initial}.")

        self.use_filter = use_filter
        self.M_initial = M_initial
        self.h_dot = self._compute_dhdt()
        self.E_dot = self._compute_energy_flux_dEdt()
        self.Eoft = self._compute_radiated_energy()
        self.E_rad = self.Eoft[-1]
        self.Moft = self._compute_bondi_mass()
        self.remnant_mass = self._compute_remnant_mass()

    def _compute_dhdt(self):
        """
        Compute time derivatives of all waveform modes.
        
        Calculates dh_lm/dt for each mode in the waveform dictionary using
        second-order finite differences. If filtering is enabled, uses spline
        interpolation with mode-dependent smoothing parameters to reduce noise.
        
        Returns:
            [dict]: Dictionary of time derivatives {(l,m): dh_lm/dt}, where
                each value is a complex array.
        """
        hdot_dict = {mode: np.gradient(self.hdict[mode], edge_order=2) /
                     np.gradient(self.time, edge_order=2) 
                     for mode in self.hdict.keys()}
        
        if self.use_filter:
            # Mode-dependent smoothing parameters for spline filtering
            sdict = {(2, 2): 0.01, (2, 1): 0.001, (3, 1): 0.00008,
                     (3, 2): 0.00008, (3, 3): 0.001, (4, 2): 0.00001,
                     (4, 3): 0.00001, (4, 4): 0.00001}
            default_smoothing = 0.00001

            hdot_dict = {}
            for mode in self.hdict.keys():
                # Interpolate waveform with smoothing
                s_val = sdict.get((mode[0], abs(mode[1])), default_smoothing)
                tmp = gwtools.interpolant_h(self.time, self.hdict[mode], s=s_val)
                # Evaluate derivative of spline
                hdot_dict[mode] = (splev(self.time, tmp[0], der=1) + 
                                  1j * splev(self.time, tmp[1], der=1))
    
        return hdot_dict

    def _compute_energy_flux_dEdt(self):
        """
        Compute total energy flux from all waveform modes.
        
        Calculates the gravitational wave luminosity (dE/dt) by summing
        contributions from all modes. The flux from each mode is proportional
        to |dh_lm/dt|^2.
        
        If filtering is enabled, applies additional spline smoothing to the
        total flux to reduce high-frequency noise.
        
        See Eq. (2) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: Energy flux as a function of time in geometric units
                (dimensionless in units where M=1).
        """
        dEdt = 0.0
        for mode in self.h_dot.keys():
            # Individual mode contribution to flux
            dEdt_mode = (1 / (16 * np.pi)) * (np.abs(self.h_dot[mode])**2)
            dEdt += dEdt_mode
        
        # Optional smoothing of total flux
        if self.use_filter:
            tmp = gwtools.interpolant_h(self.time, dEdt, s=max(dEdt) * 0.0003)
            dEdt = splev(self.time, tmp, der=0)
        
        return dEdt
    
    def _compute_radiated_energy(self):
        """
        Compute cumulative radiated energy as a function of time.

        Integrates the energy flux to obtain the *pure* radiated gravitational-wave
        energy E_rad(t) = \\int dE/dt dt' using trapezoidal integration, starting
        from zero at the first sample. The initial binding energy ``E_initial`` is
        deliberately NOT folded in here so that ``Eoft``/``E_rad`` report the
        physical energy carried away by the waves (see ``_compute_bondi_mass`` for
        how the initial mass-energy enters the mass).

        Returns:
            [np.ndarray]: Radiated energy as a function of time in units of
                total mass M.
        """
        try:
            E_rad = integrate.cumulative_trapezoid(self.E_dot, self.time, initial=0.0)
        except AttributeError:
            # scipy < 1.14: cumulative_trapezoid was named cumtrapz
            E_rad = integrate.cumtrapz(self.E_dot, self.time, initial=0.0)
        return E_rad

    def _compute_bondi_mass(self):
        """
        Compute Bondi mass (dynamic mass) as a function of time.

        The Bondi mass is the total mass-energy of the system at each instant,
        decreasing monotonically as energy is radiated away:

            M(t) = M_ADM - E_rad(t),   with   M_ADM = M_initial - E_initial,

        where ``E_initial`` is the (positive) initial binding-energy magnitude, so
        that the curve starts at the ADM mass M(t_0) = M_initial - E_initial and
        its endpoint equals the remnant mass. With the default ``E_initial`` from
        the PN expressions, or with ``E_initial = M_initial - M_ADM`` supplied from
        NR metadata, this reproduces the correct mass balance.

        See Eq. (4) of arXiv:1802.04276.

        Returns:
            [np.ndarray]: Bondi mass M(t) as a function of time in units of
                initial total mass M.
        """
        return (self.M_initial - self.E_initial) - self.Eoft

    def _compute_remnant_mass(self):
        """
        Compute final remnant black hole mass.

        The remnant mass is the ADM mass-energy minus the total radiated energy,

            M_remnant = M_ADM - E_rad = (M_initial - E_initial) - E_rad,

        i.e. the endpoint of the Bondi-mass curve. This is consistent with
        ``_compute_bondi_mass`` and lets NR-provided initial energies be used
        directly via ``E_initial = M_initial - M_ADM``.

        See Eq. (9) of arXiv:2301.07215.

        Returns:
            [float]: Final remnant mass in units of initial total mass M.
        """
        return (self.M_initial - self.E_initial) - self.E_rad