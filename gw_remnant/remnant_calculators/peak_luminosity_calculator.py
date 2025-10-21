#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : peak_luminosity_calculator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Computes peak gravitational wave luminosity of binary black hole mergers
#   using spline interpolation for accurate peak detection from discrete time
#   series data.
#
#====================================================================================

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

from .remnant_mass_calculator import RemnantMassCalculator


class PeakLuminosityCalculator(RemnantMassCalculator):
    """
    Calculator for peak gravitational wave luminosity.
    
    This class computes the peak luminosity (energy flux) of gravitational wave
    radiation from binary black hole mergers. It uses 4th order spline interpolation
    to find the maximum of the energy flux time series with improved accuracy
    beyond the discrete time resolution.
    
    The peak luminosity is an important characteristic of the merger, representing
    the maximum rate of energy emission in gravitational waves.
    
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
        L_peak (float): Peak luminosity in geometric units (dimensionless)
    
    Inherits From:
        RemnantMassCalculator: Provides energy flux (E_dot) calculations
    
    References:
        Peak luminosity definition from arXiv:2010.00120, Eq. (1)
    """
    
    def __init__(self, time, hdict, qinput, spin1_input=None, spin2_input=None, 
                 ecc_input=None, E_initial=None, L_initial=None, 
                 M_initial=1, use_filter=False):
        
        super().__init__(time, hdict, qinput, spin1_input, spin2_input, 
                         ecc_input, E_initial, L_initial, M_initial, use_filter)
        self.L_peak = self._compute_peak_luminosity()
    
    def _get_peaks_via_spline_fit(self, t, func):
        """
        Find peak of a function using 4th order spline interpolation.
        
        Fits the function to a 4th degree spline and finds the peak by locating
        the maximum among the critical points (roots of the derivative). This
        provides sub-grid resolution for peak detection.
        
        Args:
            t (np.ndarray): Array of time values
            func (np.ndarray): Array of function values corresponding to t
        
        Returns:
            [tuple]: (t_peak, f_peak) where t_peak is the time of the peak and
                f_peak is the peak value of the function.
        """
        # Use 4th degree spline for smooth interpolation
        spl = spline(t, func, k=4)
        
        # Find critical points from derivative roots
        cr_pts = spl.derivative().roots()
        
        # Include endpoints of the interval
        cr_pts = np.append(cr_pts, (t[0], t[-1]))
        
        # Evaluate spline at critical points
        cr_vals = spl(cr_pts)
        
        # Return the maximum
        max_index = np.argmax(cr_vals)
        return cr_pts[max_index], cr_vals[max_index]
    
    def _compute_peak_luminosity(self):
        """
        Compute peak gravitational wave luminosity.
        
        Identifies the peak of the energy flux (luminosity) time series using
        spline interpolation. First locates the discrete maximum, then fits
        a 4th order spline to ±10 points around this maximum to find the
        continuous peak with improved accuracy.
        
        See Eq. (1) of arXiv:2010.00120.
        
        Returns:
            [float]: Peak luminosity in geometric units (dimensionless). This is
                the maximum value of dE/dt.
        """
        # Find discrete maximum
        discrete_peak_index = np.argmax(self.E_dot)
        
        # Select ±10 points around the discrete peak
        indx_begin = discrete_peak_index - 10
        indx_end = discrete_peak_index + 10
        time_cut = self.time[indx_begin:indx_end]
        L_cut = self.E_dot[indx_begin:indx_end]
        
        # Find continuous peak using spline interpolation
        L_peak = self._get_peaks_via_spline_fit(time_cut, L_cut)[1]
        
        return L_peak