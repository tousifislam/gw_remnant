#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : gw_remnant_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes remnant properties of binary black hole mergers from gravitational
#   waveforms. Estimates energy flux, linear and angular momentum evolution,
#   and final remnant mass, spin, and kick velocity.
#
#====================================================================================

from __future__ import annotations
__author__ = "Tousif Islam"

from .remnant_calculators.initial_energy_momenta import InitialEnergyMomenta
from .remnant_calculators.peak_luminosity_calculator import PeakLuminosityCalculator
from .remnant_calculators.kick_velocity_calculator import LinearMomentumCalculator
from .remnant_calculators.remnant_mass_calculator import RemnantMassCalculator
from .remnant_calculators.remnant_spin_calculator import AngularMomentumCalculator
from .gw_utils.gw_plotter import GWPlotter


class GWRemnantCalculator(GWPlotter, PeakLuminosityCalculator, AngularMomentumCalculator,
                          LinearMomentumCalculator, RemnantMassCalculator, InitialEnergyMomenta):
    """
    Calculator for remnant properties of binary black hole mergers.
    
    This class computes remnant properties and time evolution of physical quantities
    from gravitational waveform data. It combines functionality from multiple
    calculator classes to provide comprehensive analysis of binary black hole mergers.
    
    Computed quantities include:
    - Energy flux and radiated energy
    - Linear momentum and kick velocity evolution
    - Angular momentum and spin evolution
    - Final remnant mass, spin, and kick velocity
    - Peak luminosity
    
    All calculations are performed in geometric units where G=c=1, with masses
    in units of total mass M and time in units of M.
    
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
            relative to reference. Set to known value (e.g., from NR simulation) for
            absolute tracking. Default is None
        L_initial (float): Initial angular momentum of the binary in units of M^2.
            If None, computed using PN expressions. Set to 0 to inspect angular momentum
            changes relative to reference. Set to known value (e.g., from NR simulation)
            for absolute tracking. Default is None
        M_initial (float): Initial total mass in units of M. Default is 1
        use_filter (bool): Whether to apply filtering to computed quantities.
            Default is False
    
    Attributes:
        h_dot (dict): Time derivative of waveform modes
        E_dot (np.ndarray): Energy flux as a function of time
        E_rad (float): Total radiated energy
        Eoft (np.ndarray): Cumulative radiated energy as a function of time
        Moft (np.ndarray): Binary mass as a function of time
        remnant_mass (float): Final remnant black hole mass
        P_dot (np.ndarray): Time derivative of linear momentum vector [3 x N_times]
        Poft (np.ndarray): Linear momentum vector as a function of time [3 x N_times]
        voft (np.ndarray): Center of mass velocity vector as a function of time [3 x N_times]
        kickoft (np.ndarray): Kick velocity magnitude as a function of time
        remnant_kick (float): Final kick velocity magnitude
        J_dot (np.ndarray): Time derivative of angular momentum vector [3 x N_times]
        Joft (np.ndarray): Angular momentum vector as a function of time [3 x N_times]
        spinoft (np.ndarray): Dimensionless spin magnitude as a function of time
        remnant_spin (float): Final remnant dimensionless spin magnitude
        L_peak (float): Peak luminosity
        peak_kick (float): Peak kick velocity
    
    Inherits From:
        GWPlotter: Plotting utilities for visualizing results
        PeakLuminosityCalculator: Peak luminosity calculations
        AngularMomentumCalculator: Angular momentum evolution
        LinearMomentumCalculator: Linear momentum and kick velocity
        RemnantMassCalculator: Mass and energy calculations
        InitialEnergyMomenta: Initial condition calculations
    
    Example:
        >>> import numpy as np
        >>> time = np.arange(-1000, 100, 0.1)
        >>> hdict = {(2,2): h_22_data, (3,3): h_33_data}
        >>> calc = GWRemnantCalculator(time, hdict, qinput=2.0,
        ...                           spin1_input=[0, 0, 0.5])
        >>> calc.print_remnants()
        >>> calc.plot_mass_energy()
    """
    
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None, 
                 M_initial=1, use_filter=False):
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                         ecc_input, E_initial, L_initial, M_initial, use_filter)
        
    def print_remnants(self):
        """
        Print summary of remnant properties.
        
        Displays key remnant quantities including mass ratio, initial mass,
        total radiated energy, peak luminosity, and final remnant mass, spin,
        and kick velocity. All quantities are printed in geometric units.
        """
        print("=" * 50)
        print("Remnant Properties Summary")
        print("=" * 50)
        print(f"Mass ratio                    : {self.qinput:.3f}")
        print(f"Initial mass                  : {self.M_initial:.8f} M")
        print(f"Total energy radiated         : {self.E_rad:.8f} M")
        print(f"Peak luminosity               : {self.L_peak:.8f}")
        print(f"Remnant mass                  : {self.remnant_mass:.8f} M")
        print(f"Remnant spin (dimensionless)  : {self.remnant_spin:.8f}")
        print(f"Remnant kick velocity         : {self.remnant_kick:.8f} c")
        print(f"Remnant kick velocity         : {self.remnant_kick_kmps:.2f} km/s")
        print("=" * 50)