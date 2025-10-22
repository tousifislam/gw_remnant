#====================================================================================
#
#   File Information
#   ----------------
#   Filename    : waveform_generator.py
#   Author      : Tousif Islam
#   Created     : 2023-01-05
#   License     : MIT
#
#   Description
#   -----------
#   Generates gravitational waveforms using surrogate models and computes remnant
#   properties. Supports BHPTNRSur1dq1e4 and NRHybSur3dq8 waveform models, and
#   NRSur3dq8Remnant for remnant property predictions.
#
#====================================================================================

from __future__ import annotations

import sys
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline

import gwtools
import lal
import lalsimulation as lalsim
from lal import MSUN_SI, MTSUN_SI, PC_SI, C_SI

# Attempt to load BHPTNRSur1dq1e4
try:
    PATH_TO_BHPTNRSur = "/home/UMDAR.UMASSD.EDU/tislam/work/BHPTNRSurrogate"
    sys.path.append(PATH_TO_BHPTNRSur + "/surrogates")
    import BHPTNRSur1dq1e4 as bhptsur
    print("Loaded BHPTNRSur1dq1e4 model")
except:
    print("BHPTNRSur1dq1e4 cannot be loaded - check if you have BHPTNRSurrogate path correctly given!")

# Attempt to load NRHybSur3dq8
try:
    import gwsurrogate
    sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')
    print("Loaded NRHybSur3dq8 model")
except:
    print("NRHybSur3dq8 cannot be loaded - check if gwsurrogate is properly installed!")

# Attempt to load NRSur3dq8Remnant
try:
    import surfinBH
    fit_name = 'NRSur3dq8Remnant'
    fit = surfinBH.LoadFits(fit_name)
    print("Loaded NRSur3dq8Remnant fit.")
except:
    print("NRSur3dq8Remnant cannot be loaded - check if surfinBH is properly installed!")


class WaveformGenerator:
    """
    Generator for gravitational waveforms using surrogate models.
    
    This class generates gravitational waveforms for binary black hole mergers
    using surrogate models. Currently supports:
    - BHPTNRSur1dq1e4: Black hole perturbation theory + numerical relativity surrogate
    - NRHybSur3dq8: Numerical relativity hybrid surrogate
    
    Additionally computes remnant properties (final mass, spin, kick velocity) using:
    - NRSur3dq8Remnant: Remnant property surrogate (valid for q <= 10)
    
    All waveforms are aligned such that t=0 corresponds to the peak amplitude of
    the (2,2) mode.
    
    Args:
        mass_ratio (float): Mass ratio q = m1/m2, where m1 >= m2
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2),(2,1),(3,1),(3,2),(3,3),(4,2),(4,3),(4,4)]
        common_times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)
        f_low (float): Starting orbital frequency for NRHybSur3dq8 in geometric units.
            If None, defaults to 3e-3
        get_NRSur (bool): Whether to generate NRHybSur3dq8 waveform. Default is True
        get_BHPT (bool): Whether to generate BHPTNRSur1dq1e4 waveform. Default is True
    
    Attributes:
        qinput (float): Input mass ratio
        f_low (float): Starting frequency for NRHybSur3dq8
        common_times (np.ndarray): Common time grid for all waveforms
        modes (list): List of (l,m) mode tuples
        hnr (dict): NRHybSur3dq8 waveform modes
        hbhpt (dict): BHPTNRSur1dq1e4 waveform modes
        rem_sur (tuple): Remnant properties (mf, mf_err, chif, chif_err, vf, vf_err)
    """
    
    def __init__(self, mass_ratio, modes=None, common_times=None, f_low=None, 
                 get_NRSur=True, get_BHPT=True):
        
        # Mass ratio
        self.qinput = mass_ratio
        
        # Starting frequency for NRHybSur3dq8
        if f_low is None:
            self.f_low = 3e-3
        else:
            self.f_low = f_low
            
        # Common time grid for both waveforms
        if common_times is None:
            self.common_times = np.arange(-5000.0, 50.0, 0.1)
        else:
            self.common_times = common_times

        # Modes to generate
        if modes is None:
            self.modes = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
        else:
            self.modes = modes
            
        # Generate waveforms
        if get_NRSur:
            self.hnr = self._generate_nrsur()
        if get_BHPT:
            self.hbhpt = self._generate_bhptsur()
        print('final common time grid : [%.2f, %.2f]' % (self.common_times[0], self.common_times[-1]))
        
        # Generate NRSur remnant predictions (only valid for q <= 10)
        if self.qinput <= 10:
            self.rem_sur = self._NRSurRemnant_predictions()
        
    def _generate_nrsur(self):
        """
        Generate NRHybSur3dq8 waveform.
        
        Generates a non-precessing waveform using the NRHybSur3dq8 surrogate model
        for aligned-spin binary black hole mergers. The waveform is aligned such
        that t=0 corresponds to the peak of the (2,2) mode amplitude.
        
        Returns:
            [dict]: Dictionary of waveform modes {(l,m): h_lm(t)}, where h_lm is
                a complex time series on the common time grid. Both positive and
                negative m modes are included.
        """
        chiA = [0, 0, 0.0]
        chiB = [0, 0, 0.0]
        dt = 0.1  # Step size in units of M
        
        # Generate waveform
        # dyn stands for dynamics and is always None for this model
        t, h, dyn = sur(self.qinput, chiA, chiB, dt=dt, f_low=self.f_low) 
        
        # Align time so that t=0 is at the peak of (2,2) amplitude
        t_peak = self._peak_time(t, h[(2, 2)])
        t = t - t_peak
        print('NRSur original time grid : [%.2f, %.2f]' % (t[0], t[-1]))
        
        # Select and interpolate requested modes
        h_out = {}
        for mode in self.modes:
            # Interpolate to common time grid
            h_out[mode] = gwtools.gwtools.interpolate_h(t, h[mode], self.common_times)
            # Generate negative m modes using symmetry
            h_out[(mode[0], -mode[-1])] = ((-1)**mode[0]) * np.conjugate(h_out[mode])
        return h_out
    
    def _generate_bhptsur(self):
        """
        Generate BHPTNRSur1dq1e4 waveform.
        
        Generates a waveform using the BHPTNRSur1dq1e4 surrogate model, which
        combines black hole perturbation theory with numerical relativity for
        extreme mass ratio inspirals. The waveform is aligned such that t=0
        corresponds to the peak of the (2,2) mode amplitude.
        
        Returns:
            [dict]: Dictionary of waveform modes {(l,m): h_lm(t)}, where h_lm is
                a complex time series on the common time grid.
        """
        t, h = bhptsur.generate_surrogate(q=self.qinput, modes=self.modes)
        
        # Align time so that t=0 is at the peak of (2,2) amplitude
        t_peak = self._peak_time(t, h[(2, 2)])
        t = t - t_peak
        print('BHPTSur original time grid : [%.2f, %.2f]' % (t[0], t[-1]))
        
        # Interpolate all modes to common time grid
        for mode in h.keys():
            h[mode] = gwtools.gwtools.interpolate_h(t, h[mode], self.common_times)
        return h
    
    def _NRSurRemnant_predictions(self):
        """
        Compute remnant properties using NRSur3dq8Remnant surrogate.
        
        Predicts the final mass, dimensionless spin, and kick velocity of the
        remnant black hole after merger using the NRSur3dq8Remnant fit. This
        surrogate is valid for mass ratios q <= 10 and non-precessing spins.
        
        Returns:
            [tuple]: Tuple containing (mf, mf_err, chif_z, chif_z_err, |vf|, |vf_err|)
                where mf is final mass in units of total mass, chif_z is the z-component
                of final dimensionless spin, and |vf| is the magnitude of kick velocity
                in units of speed of light.
        """
        chiA = [0, 0, 0.0]   
        chiB = [0, 0, 0.0]
        
        # Get all remnant predictions with uncertainties
        mf, chif, vf, mf_err, chif_err, vf_err = fit.all(self.qinput, chiA, chiB)
        
        return mf, mf_err, chif[-1], chif_err[-1], np.linalg.norm(vf), np.linalg.norm(vf_err)

    def _get_peaks_via_spline_fit(self, t, func):
        """
        Find the peak of a function using spline interpolation.
        
        Fits the function to a 4th degree spline and finds its maximum by
        locating the roots of the derivative. This provides a more accurate
        peak location than simply using the maximum of the discrete samples.
        
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

    def _peak_time(self, t, mode):
        """
        Find the peak time of a waveform mode.
        
        Computes the time at which the amplitude squared of a waveform mode
        reaches its maximum using 4th order spline interpolation.
        
        Args:
            t (np.ndarray): Time array
            mode (np.ndarray): Complex waveform mode h_lm(t)
        
        Returns:
            [float]: Time at which |h_lm|^2 reaches its peak.
        """
        normSqrVsT = abs(mode)**2
        return self._get_peaks_via_spline_fit(t, normSqrVsT)[0]