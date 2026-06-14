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

import numpy as np

from .remnant_calculators.initial_energy_momenta import InitialEnergyMomenta
from .remnant_calculators.peak_luminosity_calculator import PeakLuminosityCalculator
from .remnant_calculators.kick_velocity_calculator import LinearMomentumCalculator
from .remnant_calculators.remnant_mass_calculator import RemnantMassCalculator
from .remnant_calculators.remnant_spin_calculator import AngularMomentumCalculator
from .remnant_calculators.trajectory_calculator import TrajectoryCalculator
from .gw_utils.gw_plotter import GWPlotter


class GWRemnantCalculator(GWPlotter, PeakLuminosityCalculator, AngularMomentumCalculator,
                          TrajectoryCalculator, LinearMomentumCalculator,
                          RemnantMassCalculator, InitialEnergyMomenta):
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
        q (float): Mass ratio q = m1/m2, where m1 >= m2
        chi1 (list or np.ndarray): Spin vector [sx, sy, sz] for primary black
            hole at the start of the waveform, in dimensionless units. Default is None
        chi2 (list or np.ndarray): Spin vector [sx, sy, sz] for secondary black
            hole at the start of the waveform, in dimensionless units. Default is None
        e_ref (float): Eccentricity at the reference time. User must provide
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
        spinoft (np.ndarray): Dimensionless spin z-component as a function of time
        remnant_spin (float): Final remnant dimensionless spin z-component
        spin_vector_oft (np.ndarray): Dimensionless spin vector [3 x N_times] (x, y, z)
        remnant_spin_vector (np.ndarray): Final remnant dimensionless spin vector (3,)
        L_peak (float): Peak luminosity
        peak_kick (float): Peak kick velocity
        xoft (np.ndarray): Center-of-mass displacement vector [N_times x 3] in units of M
        remnant_displacement (np.ndarray): Final center-of-mass displacement vector (3,)

    Inherits From:
        GWPlotter: Plotting utilities for visualizing results
        PeakLuminosityCalculator: Peak luminosity calculations
        AngularMomentumCalculator: Angular momentum evolution
        TrajectoryCalculator: Center-of-mass trajectory
        LinearMomentumCalculator: Linear momentum and kick velocity
        RemnantMassCalculator: Mass and energy calculations
        InitialEnergyMomenta: Initial condition calculations
    
    Example:
        >>> import numpy as np
        >>> time = np.arange(-1000, 100, 0.1)
        >>> hdict = {(2,2): h_22_data, (3,3): h_33_data}
        >>> calc = GWRemnantCalculator(time, hdict, q=2.0,
        ...                           chi1=[0, 0, 0.5])
        >>> calc.print_remnants()
        >>> calc.plot_mass_energy()
    """
    
    def __init__(self, time: np.ndarray, hdict: dict[tuple[int, int], np.ndarray],
                 q: float, chi1: np.ndarray | list[float] | None = None,
                 chi2: np.ndarray | list[float] | None = None,
                 e_ref: float | None = None, E_initial: float | None = None,
                 L_initial: float | None = None,
                 M_initial: float = 1, use_filter: bool = False) -> None:
        super().__init__(time, hdict, q, chi1, chi2,
                         e_ref, E_initial, L_initial, M_initial, use_filter)

    def get_remnant_properties(self) -> dict[str, float]:
        """
        Return remnant properties as a dictionary.

        Returns:
            dict: Dictionary containing:
                - 'mass_ratio': Mass ratio q = m1/m2
                - 'M_initial': Initial total mass
                - 'E_rad': Total radiated energy
                - 'L_peak': Peak luminosity
                - 'remnant_mass': Final remnant mass
                - 'remnant_spin': Final dimensionless spin (z-component)
                - 'remnant_spin_x': Final dimensionless spin x-component
                - 'remnant_spin_y': Final dimensionless spin y-component
                - 'remnant_spin_z': Final dimensionless spin z-component
                - 'remnant_kick': Final kick velocity in units of c
                - 'remnant_kick_kmps': Final kick velocity in km/s
                - 'peak_kick': Peak kick velocity in units of c
                - 'remnant_displacement_x/y/z': Final center-of-mass displacement
                    components in units of M
        """
        return {
            'mass_ratio': self.q,
            'M_initial': self.M_initial,
            'E_rad': self.E_rad,
            'L_peak': self.L_peak,
            'remnant_mass': self.remnant_mass,
            'remnant_spin': self.remnant_spin,
            'remnant_spin_x': self.remnant_spin_vector[0],
            'remnant_spin_y': self.remnant_spin_vector[1],
            'remnant_spin_z': self.remnant_spin_vector[2],
            'remnant_kick': self.remnant_kick,
            'remnant_kick_kmps': self.remnant_kick_kmps,
            'peak_kick': self.peak_kick,
            'remnant_displacement_x': self.remnant_displacement[0],
            'remnant_displacement_y': self.remnant_displacement[1],
            'remnant_displacement_z': self.remnant_displacement[2],
        }

    def print_remnants(self) -> None:
        """
        Print summary of remnant properties.

        Displays key remnant quantities including mass ratio, initial mass,
        total radiated energy, peak luminosity, and final remnant mass, spin,
        and kick velocity. All quantities are printed in geometric units.
        """
        print("=" * 50)
        print("Remnant Properties Summary")
        print("=" * 50)
        print(f"Mass ratio                    : {self.q:.3f}")
        print(f"Initial mass                  : {self.M_initial:.8f} M")
        print(f"Total energy radiated         : {self.E_rad:.8f} M")
        print(f"Peak luminosity               : {self.L_peak:.8f}")
        print(f"Remnant mass                  : {self.remnant_mass:.8f} M")
        print(f"Remnant spin (dimensionless)  : {self.remnant_spin:.8f}")
        print(f"Remnant spin vector (x,y,z)  : ({self.remnant_spin_vector[0]:.8f}, "
              f"{self.remnant_spin_vector[1]:.8f}, {self.remnant_spin_vector[2]:.8f})")
        print(f"Remnant kick velocity         : {self.remnant_kick:.8f} c")
        print(f"Remnant kick velocity         : {self.remnant_kick_kmps:.2f} km/s")
        print(f"Remnant displacement (x,y,z) : ({self.remnant_displacement[0]:.8f}, "
              f"{self.remnant_displacement[1]:.8f}, {self.remnant_displacement[2]:.8f}) M")
        print("=" * 50)