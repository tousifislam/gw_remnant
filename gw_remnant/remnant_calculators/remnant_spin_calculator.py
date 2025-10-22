#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : remnant_spin_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes angular momentum flux and remnant spin of binary black hole mergers
#   from gravitational waveforms. Calculates time evolution of angular momentum
#   and final dimensionless spin of the remnant black hole.
#
#====================================================================================

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import splev, splrep
import gwtools

from .initial_energy_momenta import InitialEnergyMomenta
from .remnant_mass_calculator import RemnantMassCalculator
from .kick_velocity_calculator import LinearMomentumCalculator


class AngularMomentumCalculator(LinearMomentumCalculator, RemnantMassCalculator, InitialEnergyMomenta):
    """
    Calculator for angular momentum and remnant spin of binary black holes.
    
    This class computes the angular momentum carried away by gravitational waves
    and the resulting dimensionless spin of the remnant black hole. The calculations
    use angular momentum flux formulas from gravitational wave multipoles.
    
    All calculations are performed in geometric units where G=c=1, with angular
    momentum in units of M^2 and spin being dimensionless.
    
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
        J_dot (np.ndarray): Angular momentum flux vector [3 x N_times] in units of M^2
        Joft (np.ndarray): Cumulative radiated angular momentum [3 x N_times] in units of M^2
        spinoft (np.ndarray): Dimensionless spin magnitude as a function of time
        remnant_spin (float): Final remnant dimensionless spin magnitude
    
    Inherits From:
        LinearMomentumCalculator: Provides linear momentum calculations and _read_dhdt_dict method
        RemnantMassCalculator: Provides mass calculations
        InitialEnergyMomenta: Provides initial condition calculations
    
    References:
        Angular momentum flux formulas from arXiv:1802.04276 and arXiv:0707.4654
        Spin calculation from arXiv:2101.11015
    """
    
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None, 
                 M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                         ecc_input, E_initial, L_initial, M_initial, use_filter)
        
        self.J_dot = np.array([self._compute_dJxdt(), self._compute_dJydt(), 
                               self._compute_dJzdt()])
        self.Joft = self._compute_Joft()
        self.spinoft = self._compute_spin_evolution()
        self.remnant_spin = self._compute_remnant_spin()
        
    def _coeff_f(self, l, m):
        """
        Compute coefficient f(l,m) for angular momentum flux.
        
        See Eq. (3.25) of arXiv:0707.4654.
        
        Args:
            l (int): Spherical harmonic degree
            m (int): Spherical harmonic order
        
        Returns:
            [float]: Coefficient value.
        """
        return (l * (l + 1) - m * (m + 1))**0.5
    
    def _compute_dJxdt(self):
        """
        Compute x-component of angular momentum flux.
        
        Calculates the time derivative of the x-component of radiated angular
        momentum using gravitational wave multipole formulas.
        
        See Eq. (15) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dJx/dt as a function of time in units of M^2.
        """
        dJxdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dJxdt += (1 / (32 * np.pi)) * np.imag(
                self.hdict[(l, m)] * (
                    self._coeff_f(l, m) * np.conj(self._read_dhdt_dict(l, m + 1)) +
                    self._coeff_f(l, -m) * np.conj(self._read_dhdt_dict(l, m - 1))
                )
            )
        return dJxdt
    
    def _compute_dJydt(self):
        """
        Compute y-component of angular momentum flux.
        
        Calculates the time derivative of the y-component of radiated angular
        momentum using gravitational wave multipole formulas.
        
        See Eq. (16) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dJy/dt as a function of time in units of M^2.
        """
        dJydt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dJydt += (1 / (32 * np.pi)) * np.real(
                self.hdict[(l, m)] * (
                    self._coeff_f(l, m) * np.conj(self._read_dhdt_dict(l, m + 1)) +
                    self._coeff_f(l, -m) * np.conj(self._read_dhdt_dict(l, m - 1))
                )
            )
        return dJydt
    
    def _compute_dJzdt(self):
        """
        Compute z-component of angular momentum flux.
        
        Calculates the time derivative of the z-component of radiated angular
        momentum using gravitational wave multipole formulas.
        
        See Eq. (17) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: dJz/dt as a function of time in units of M^2.
        """
        dJzdt = np.zeros(len(self.time))
        for mode in self.hdict.keys():
            (l, m) = mode
            dJzdt += (1 / (16 * np.pi)) * m * np.imag(
                self.hdict[(l, m)] * np.conj(self.h_dot[(l, m)])
            )
        return dJzdt
    
    def _compute_Joft(self):
        """
        Compute cumulative radiated angular momentum vector.
        
        Integrates the angular momentum flux to obtain the total radiated angular
        momentum as a function of time. Uses trapezoidal integration.
        
        See Eqs. (15)-(17) of arXiv:1802.04276.
        
        Returns:
            [np.ndarray]: Radiated angular momentum vector [3 x N_times] in units of M^2.
        """
        try:
            Jxoft = integrate.cumtrapz(self.J_dot[0], self.time, initial=0.0)
            Jyoft = integrate.cumtrapz(self.J_dot[1], self.time, initial=0.0)
            Jzoft = integrate.cumtrapz(self.J_dot[2], self.time, initial=0.0)
        except:
            Jxoft = integrate.cumulative_trapezoid(self.J_dot[0], self.time, initial=0.0)
            Jyoft = integrate.cumulative_trapezoid(self.J_dot[1], self.time, initial=0.0)
            Jzoft = integrate.cumulative_trapezoid(self.J_dot[2], self.time, initial=0.0)
        return np.array([Jxoft, Jyoft, Jzoft])
    
    def _compute_spin_evolution(self):
        """
        Compute dimensionless spin evolution of the system.
        
        Calculates the dimensionless spin as a function of time by accounting for:
        1. Initial orbital angular momentum (L_initial)
        2. Initial spin angular momentum of both black holes (S1_z + S2_z)
        3. Angular momentum radiated away up to time t (J_rad,z(t))
        4. Time-dependent mass M(t)
        
        Formula: χ(t) = (L_initial + S1_z + S2_z - J_rad,z(t)) / M(t)^2
        
        See Eq. (20) of arXiv:2101.11015.
        
        Returns:
            [np.ndarray]: Dimensionless spin magnitude as a function of time.
        """
        # Get individual initial masses
        m1 = self.qinput / (1 + self.qinput)  # Primary mass
        m2 = 1 / (1 + self.qinput)  # Secondary mass
        
        # Handle spin input (could be scalar or array)
        chi1_z = np.linalg.norm(self.spin1_input)
        chi2_z = np.linalg.norm(self.spin2_input)
        
        # Convert dimensionless spins to angular momentum
        S1_z = chi1_z * m1**2
        S2_z = chi2_z * m2**2
        
        # Compute spin evolution at each time step
        spin_f = np.zeros(len(self.time))
        for i in range(len(spin_f)):
            # Total angular momentum at time t
            J_z_t = self.L_initial + S1_z + S2_z - self.Joft[2][i]
            
            # Dimensionless spin using time-dependent mass
            spin_f[i] = J_z_t / self.Moft[i]**2
        
        return spin_f
    
    def _compute_remnant_spin(self):
        """
        Compute final dimensionless spin of the remnant black hole.
        
        Calculates the final spin by accounting for:
        1. Initial orbital angular momentum (L_initial)
        2. Initial spin angular momentum of both black holes (S1_z + S2_z)
        3. Angular momentum radiated away (J_rad,z)
        
        Formula: χ_f = (L_initial + S1_z + S2_z - J_rad,z) / M_f^2
        
        where S1_z and S2_z are converted from dimensionless spins χ₁, χ₂
        using S_i = χ_i * m_i^2.
        
        See Eq. (20) of arXiv:2101.11015.
        
        Returns:
            [float]: Final remnant dimensionless spin magnitude.
        """
        # Get individual initial masses
        m1 = self.qinput / (1 + self.qinput)  # Primary mass
        m2 = 1 / (1 + self.qinput)  # Secondary mass
        
        # Handle spin input (could be scalar or array)
        chi1_z = np.linalg.norm(self.spin1_input)
        chi2_z = np.linalg.norm(self.spin2_input)
        
        # Convert dimensionless spins to angular momentum
        S1_z = chi1_z * m1**2
        S2_z = chi2_z * m2**2
        
        # Total angular momentum at final time
        J_final_z = self.L_initial + S1_z + S2_z - self.Joft[2][-1]
        
        # Dimensionless remnant spin
        remnant_spin = J_final_z / self.remnant_mass**2
        
        return remnant_spin