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
#   Provides functions to generate gravitational waveforms using various surrogate
#   models. Users must import and provide surrogate modules in their own code,
#   giving them full control over dependencies and versions.
#
#====================================================================================

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import gwtools


def _get_peaks_via_spline_fit(t, func):
    """
    Find the peak of a function using spline interpolation.
    
    Fits the function to a 4th degree spline and finds its maximum by
    locating the roots of the derivative.
    
    Args:
        t (np.ndarray): Array of time values
        func (np.ndarray): Array of function values corresponding to t
    
    Returns:
        [tuple]: (t_peak, f_peak) where t_peak is the time of the peak and
            f_peak is the peak value of the function.
    """
    spl = spline(t, func, k=4)
    cr_pts = spl.derivative().roots()
    cr_pts = np.append(cr_pts, (t[0], t[-1]))
    cr_vals = spl(cr_pts)
    max_index = np.argmax(cr_vals)
    return cr_pts[max_index], cr_vals[max_index]


def _peak_time(t, mode):
    """
    Find the peak time of a waveform mode.
    
    Args:
        t (np.ndarray): Time array
        mode (np.ndarray): Complex waveform mode h_lm(t)
    
    Returns:
        [float]: Time at which |h_lm|^2 reaches its peak.
    """
    normSqrVsT = abs(mode)**2
    return _get_peaks_via_spline_fit(t, normSqrVsT)[0]


def generate_nrhybsur3dq8(gwsurrogate_module, mass_ratio, chi1=[0, 0, 0], 
                          chi2=[0, 0, 0], modes=None, times=None, 
                          f_low=3e-3, dt=0.1):
    """
    Generate NRHybSur3dq8 waveform.
    
    Generates gravitational waveform using the NRHybSur3dq8 surrogate model
    for aligned-spin binary black hole mergers. The waveform is aligned such
    that t=0 corresponds to the peak amplitude of the (2,2) mode.
    
    Args:
        gwsurrogate_module: The gwsurrogate module imported by the user
        mass_ratio (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10)
        chi1 (list): Dimensionless spin vector [sx, sy, sz] for primary BH.
            Default is [0, 0, 0]
        chi2 (list): Dimensionless spin vector [sx, sy, sz] for secondary BH.
            Default is [0, 0, 0]
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
        times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)
        f_low (float): Starting orbital frequency in geometric units.
            Default is 3e-3
        dt (float): Time step for waveform generation in units of M.
            Default is 0.1
    
    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms
    
    Raises:
        ValueError: If mass_ratio is outside valid range [1, 10]
    
    Example:
        >>> import gwsurrogate
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> times, h = wg.generate_nrhybsur3dq8(
        ...     gwsurrogate,
        ...     mass_ratio=3.0, 
        ...     chi1=[0, 0, 0.5],
        ...     modes=[(2,2), (3,3)]
        ... )
    """
    # Validate inputs
    if not 1 <= mass_ratio <= 10:
        raise ValueError(f"Mass ratio {mass_ratio} outside valid range [1, 10] "
                        f"for NRHybSur3dq8")
    
    # Set defaults
    if modes is None:
        modes = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
    
    if times is None:
        times = np.arange(-5000.0, 50.0, 0.1)
    
    # Generate waveform
    t, h, dyn = gwsurrogate_module(mass_ratio, chi1, chi2, dt=dt, f_low=f_low)
    
    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'NRHybSur3dq8 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')
    
    # Interpolate to requested time grid
    h_out = {}
    for mode in modes:
        h_out[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)
        # Add negative m modes using symmetry
        h_out[(mode[0], -mode[1])] = ((-1)**mode[0]) * np.conjugate(h_out[mode])
    
    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')
    
    return times, h_out


def generate_bhptnrsur1dq1e4(bhptsur_module, mass_ratio, modes=None, times=None):
    """
    Generate BHPTNRSur1dq1e4 waveform.
    
    Generates waveform using the BHPTNRSur1dq1e4 surrogate model, which combines
    black hole perturbation theory with numerical relativity for extreme and
    intermediate mass ratio inspirals.
    
    Args:
        bhptsur_module: The BHPTNRSur1dq1e4 module imported by the user
        mass_ratio (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10000)
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
        times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)
    
    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms
    
    Raises:
        ValueError: If mass_ratio is outside valid range
    
    Example:
        >>> import sys
        >>> sys.path.append('/path/to/BHPTNRSurrogate/surrogates')
        >>> import BHPTNRSur1dq1e4 as bhptsur
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> times, h = wg.generate_bhptnrsur1dq1e4(
        ...     bhptsur,
        ...     mass_ratio=100.0,
        ...     modes=[(2,2)]
        ... )
    """
    # Validate inputs
    if not 1 <= mass_ratio <= 10000:
        raise ValueError(f"Mass ratio {mass_ratio} outside typical range [1, 10000] "
                        f"for BHPTNRSur1dq1e4")
    
    # Set defaults
    if modes is None:
        modes = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
    
    if times is None:
        times = np.arange(-5000.0, 50.0, 0.1)
    
    # Generate waveform
    print("Generating BHPTNRSur1dq1e4 waveform...")
    t, h = bhptsur_module.generate_surrogate(q=mass_ratio, modes=modes, calibrated=True)
    
    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'BHPTNRSur1dq1e4 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')
    
    # Interpolate to requested time grid
    for mode in h.keys():
        h[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)
    
    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')
    
    return times, h


def generate_bhptnrsur2dq1e3(bhptsur_module, mass_ratio, spin, modes=None, times=None):
    """
    Generate BHPTNRSur1dq1e4 waveform.
    
    Generates waveform using the BHPTNRSur1dq1e4 surrogate model, which combines
    black hole perturbation theory with numerical relativity for extreme and
    intermediate mass ratio inspirals.
    
    Args:
        bhptsur_module: The BHPTNRSur1dq1e4 module imported by the user
        mass_ratio (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10000)
        spin (float) : Dimensionless spin of the primary black hole (-0.8 <= spin <= 0.8)
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
        times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)
    
    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms
    
    Raises:
        ValueError: If mass_ratio is outside valid range
    
    Example:
        >>> import sys
        >>> sys.path.append('/path/to/BHPTNRSurrogate/surrogates')
        >>> import BHPTNRSur1dq1e4 as bhptsur
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> times, h = wg.generate_bhptnrsur1dq1e4(
        ...     bhptsur,
        ...     mass_ratio=100.0,
        ...     modes=[(2,2)]
        ... )
    """
    # Validate inputs
    if not 1 <= mass_ratio <= 10000:
        raise ValueError(f"Mass ratio {mass_ratio} outside typical range [1, 10000] "
                        f"for BHPTNRSur1dq1e4")
    
    # Set defaults
    if modes is None:
        modes = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
    
    if times is None:
        times = np.arange(-5000.0, 50.0, 0.1)
    
    # Generate waveform
    print("Generating BHPTNRSur2dq1e3 waveform...")
    t, h = bhptsur_module.generate_surrogate(q=mass_ratio, spin1=spin, modes=modes, calibrated=True)
    
    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'BHPTNRSur1dq1e4 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')
    
    # Interpolate to requested time grid
    for mode in h.keys():
        h[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)
    
    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')
    
    return times, h


def compute_nrsur3dq8_remnant(surfinbh_module, mass_ratio, chi1=[0, 0, 0], 
                              chi2=[0, 0, 0], fit_name='NRSur3dq8Remnant',
                              print_output=True):
    """
    Compute remnant properties using NRSur3dq8Remnant surrogate.
    
    Predicts final mass, dimensionless spin, and kick velocity of the remnant
    black hole using the NRSur3dq8Remnant fit.
    
    Args:
        surfinbh_module: The surfinBH module imported by the user
        mass_ratio (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10)
        chi1 (list): Dimensionless spin vector [sx, sy, sz] for primary BH.
            Default is [0, 0, 0]
        chi2 (list): Dimensionless spin vector [sx, sy, sz] for secondary BH.
            Default is [0, 0, 0]
        fit_name (str): Name of the remnant fit to use. Default is 'NRSur3dq8Remnant'.
            Other options include 'NRSur7dq4Remnant' for precessing systems
        print_output (boolen): True
            Prints the final properties
    
    Returns:
        [dict]: Dictionary containing:
            - 'final_mass': Final mass in units of total mass M
            - 'final_mass_err': Uncertainty in final mass
            - 'final_spin': Final dimensionless spin magnitude
            - 'final_spin_z': z-component of final spin
            - 'final_spin_err': Uncertainty in final spin magnitude
            - 'kick_velocity': Kick velocity magnitude in units of c
            - 'kick_velocity_err': Uncertainty in kick velocity magnitude
            - 'final_spin_vector': Full spin vector [sx, sy, sz]
            - 'kick_velocity_vector': Full kick velocity vector [vx, vy, vz]
    
    Raises:
        ValueError: If mass_ratio is outside valid range [1, 10]
    
    Example:
        >>> import surfinBH
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> remnant = wg.compute_nrsur3dq8_remnant(
        ...     surfinBH,
        ...     mass_ratio=3.0, 
        ...     chi1=[0, 0, 0.7]
        ... )
        >>> print(f"Final mass: {remnant['final_mass']:.4f} M")
        >>> print(f"Final spin: {remnant['final_spin']:.4f}")
        >>> print(f"Kick velocity: {remnant['kick_velocity']*299792.458:.1f} km/s")
    """
    # Validate inputs
    if not 1 <= mass_ratio <= 10:
        raise ValueError(f"Mass ratio {mass_ratio} outside valid range [1, 10] "
                        f"for NRSur3dq8Remnant")
    
    # Get remnant predictions with uncertainties
    mf, chif, vf, mf_err, chif_err, vf_err = surfinbh_module.all(mass_ratio, chi1, chi2)
    
    # Package results in a dictionary
    remnant_properties = {
        'final_mass': mf,
        'final_mass_err': mf_err,
        'final_spin': np.linalg.norm(chif),
        'final_spin_z': chif[2],
        'final_spin_vector': chif,
        'final_spin_err': np.linalg.norm(chif_err),
        'kick_velocity': np.linalg.norm(vf),
        'kick_velocity_vector': vf,
        'kick_velocity_err': np.linalg.norm(vf_err),
    }
    
    if print_output:
        print("=" * 50)
        print("Remnant predictions from surfinBH")
        print("=" * 50)
        print(f"Final mass:      {mf:.6f} ± {mf_err:.6f} M")
        print(f"Final spin:      {np.linalg.norm(chif):.6f} ± {np.linalg.norm(chif_err):.6f}")
        print(f"Kick velocity:   {np.linalg.norm(vf):.6f} ± {np.linalg.norm(vf_err):.6f} c")
        print("=" * 50)
        
    return remnant_properties