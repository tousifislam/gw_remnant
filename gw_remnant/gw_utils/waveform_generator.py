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

# Default modes and time grid shared across waveform generators
DEFAULT_MODES = [(2, 2), (2, 1), (3, 1), (3, 2), (3, 3), (4, 2), (4, 3), (4, 4)]
DEFAULT_TIMES = np.arange(-5000.0, 50.0, 0.1)


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


def _print_remnant_summary(fit_name, mf, mf_err, chif, chif_err, vf, vf_err):
    """
    Print a surfinBH remnant-property summary, showing the final mass and both
    the magnitude and the full vector of the final spin and kick velocity.
    """
    print("=" * 60)
    print(f"Remnant predictions from {fit_name}")
    print("=" * 60)
    print(f"Final mass         : {mf:.6f} ± {mf_err:.6f} M")
    print(f"Final spin |chi|   : {np.linalg.norm(chif):.6f} ± {np.linalg.norm(chif_err):.6f}")
    print(f"Final spin vector  : ({chif[0]:.6f}, {chif[1]:.6f}, {chif[2]:.6f})")
    print(f"Kick |v|           : {np.linalg.norm(vf):.6f} ± {np.linalg.norm(vf_err):.6f} c")
    print(f"Kick vector        : ({vf[0]:.6e}, {vf[1]:.6e}, {vf[2]:.6e}) c")
    print("=" * 60)


def generate_nrhybsur3dq8(gwsurrogate_module, q: float,
                          chi1: list[float] = [0, 0, 0],
                          chi2: list[float] = [0, 0, 0],
                          modes: list[tuple[int, int]] | None = None,
                          times: np.ndarray | None = None,
                          f_low: float = 3e-3,
                          dt: float = 0.1) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """
    Generate NRHybSur3dq8 waveform.
    
    Generates gravitational waveform using the NRHybSur3dq8 surrogate model
    for aligned-spin binary black hole mergers. The waveform is aligned such
    that t=0 corresponds to the peak amplitude of the (2,2) mode.
    
    Args:
        gwsurrogate_module: The gwsurrogate module imported by the user
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10)
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
        ValueError: If q is outside valid range [1, 10]
    
    Example:
        >>> import gwsurrogate
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> times, h = wg.generate_nrhybsur3dq8(
        ...     gwsurrogate,
        ...     q=3.0, 
        ...     chi1=[0, 0, 0.5],
        ...     modes=[(2,2), (3,3)]
        ... )
    """
    # Validate inputs
    if not 1 <= q <= 10:
        raise ValueError(f"Mass ratio {q} outside valid range [1, 10] "
                        f"for NRHybSur3dq8")

    # Set defaults
    if modes is None:
        modes = list(DEFAULT_MODES)
    if times is None:
        times = DEFAULT_TIMES.copy()

    # Generate waveform
    t, h, dyn = gwsurrogate_module(q, chi1, chi2, dt=dt, f_low=f_low)
    
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


def generate_bhptnrsur1dq1e4(bhptsur_module, q: float,
                             modes: list[tuple[int, int]] | None = None,
                             times: np.ndarray | None = None) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """
    Generate BHPTNRSur1dq1e4 waveform.
    
    Generates waveform using the BHPTNRSur1dq1e4 surrogate model, which combines
    black hole perturbation theory with numerical relativity for extreme and
    intermediate mass ratio inspirals.
    
    Args:
        bhptsur_module: The BHPTNRSur1dq1e4 module imported by the user
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10000)
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
        times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)
    
    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms
    
    Raises:
        ValueError: If q is outside valid range
    
    Example:
        >>> import sys
        >>> sys.path.append('/path/to/BHPTNRSurrogate/surrogates')
        >>> import BHPTNRSur1dq1e4 as bhptsur
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> times, h = wg.generate_bhptnrsur1dq1e4(
        ...     bhptsur,
        ...     q=100.0,
        ...     modes=[(2,2)]
        ... )
    """
    # Validate inputs
    if not 1 <= q <= 10000:
        raise ValueError(f"Mass ratio {q} outside typical range [1, 10000] "
                        f"for BHPTNRSur1dq1e4")

    # Set defaults
    if modes is None:
        modes = list(DEFAULT_MODES)
    if times is None:
        times = DEFAULT_TIMES.copy()

    # Generate waveform
    print("Generating BHPTNRSur1dq1e4 waveform...")
    t, h = bhptsur_module.generate_surrogate(q=q, modes=modes, calibrated=True)
    
    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'BHPTNRSur1dq1e4 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')
    
    # Interpolate to requested time grid
    for mode in h.keys():
        h[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)
    
    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')
    
    return times, h


def generate_bhptnrsur2dq1e3(bhptsur_module, q: float, spin: float,
                             modes: list[tuple[int, int]] | None = None,
                             times: np.ndarray | None = None) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """
    Generate BHPTNRSur2dq1e3 waveform.

    Generates waveform using the BHPTNRSur2dq1e3 surrogate model, which combines
    black hole perturbation theory with numerical relativity for spinning
    intermediate and extreme mass ratio inspirals.

    Args:
        bhptsur_module: The BHPTNRSur2dq1e3 module imported by the user
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 1000)
        spin (float): Dimensionless spin of the primary black hole (-0.8 <= spin <= 0.8)
        modes (list): List of (l,m) mode tuples to generate. If None, defaults to
            [(2,2), (2,1), (3,1), (3,2), (3,3), (4,2), (4,3), (4,4)]
        times (np.ndarray): Time array in geometric units (M). If None,
            defaults to np.arange(-5000.0, 50.0, 0.1)

    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms

    Raises:
        ValueError: If q is outside valid range

    Example:
        >>> import sys
        >>> sys.path.append('/path/to/BHPTNRSurrogate/surrogates')
        >>> import BHPTNRSur2dq1e3 as bhptsur
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>>
        >>> times, h = wg.generate_bhptnrsur2dq1e3(
        ...     bhptsur,
        ...     q=100.0,
        ...     spin=0.5,
        ...     modes=[(2,2)]
        ... )
    """
    # Validate inputs
    if not 1 <= q <= 1000:
        raise ValueError(f"Mass ratio {q} outside typical range [1, 1000] "
                        f"for BHPTNRSur2dq1e3")

    # Set defaults
    if modes is None:
        modes = list(DEFAULT_MODES)
    if times is None:
        times = DEFAULT_TIMES.copy()

    # Generate waveform
    print("Generating BHPTNRSur2dq1e3 waveform...")
    t, h = bhptsur_module.generate_surrogate(q=q, spin1=spin, modes=modes, calibrated=True)

    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'BHPTNRSur2dq1e3 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')
    
    # Interpolate to requested time grid
    for mode in h.keys():
        h[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)
    
    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')
    
    return times, h


def generate_nrsur7dq4(gwsurrogate_module, q: float,
                       chi1: list[float] = [0, 0, 0],
                       chi2: list[float] = [0, 0, 0],
                       modes: list[tuple[int, int]] | None = None,
                       times: np.ndarray | None = None,
                       f_low: float = 0.0,
                       dt: float = 0.1) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """
    Generate NRSur7dq4 waveform.

    Generates a gravitational waveform using the NRSur7dq4 surrogate model for
    generically precessing binary black hole mergers. The waveform is in the
    inertial frame and aligned so that t=0 is the peak amplitude of the (2,2)
    mode. Because the binary precesses, all m modes are returned directly from
    the surrogate (the aligned-spin negative-m symmetry does not apply).

    Args:
        gwsurrogate_module: The loaded NRSur7dq4 gwsurrogate model
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 4)
        chi1 (list): Dimensionless spin vector [sx, sy, sz] for primary BH.
            Default is [0, 0, 0]
        chi2 (list): Dimensionless spin vector [sx, sy, sz] for secondary BH.
            Default is [0, 0, 0]
        modes (list): List of (l,m) mode tuples to keep. If None, all modes
            returned by the surrogate (up to l=4, all m) are kept.
        times (np.ndarray): Output time array in geometric units (M). If None,
            defaults to np.arange(-4000.0, 50.0, 0.1)
        f_low (float): Starting orbital frequency in geometric units; 0 uses the
            full surrogate length. Default is 0.0
        dt (float): Time step for waveform generation in units of M. Default is 0.1

    Returns:
        [tuple]: (times, waveform_dict) where:
            - times: Time array aligned so peak is at t=0
            - waveform_dict: Dictionary {(l,m): h_lm(t)} with complex waveforms

    Raises:
        ValueError: If q is outside valid range [1, 4]

    Example:
        >>> import gwsurrogate
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>>
        >>> sur = gwsurrogate.LoadSurrogate('NRSur7dq4')
        >>> times, h = wg.generate_nrsur7dq4(
        ...     sur,
        ...     q=3.0,
        ...     chi1=[0.5, 0.0, 0.3],
        ...     chi2=[0.0, 0.4, -0.2]
        ... )
    """
    # Validate inputs
    if not 1 <= q <= 4:
        raise ValueError(f"Mass ratio {q} outside valid range [1, 4] "
                        f"for NRSur7dq4")

    # Set defaults
    if times is None:
        times = np.arange(-4000.0, 50.0, 0.1)

    # Generate waveform (precessing; inertial-frame modes)
    t, h, dyn = gwsurrogate_module(q, chi1, chi2, dt=dt, f_low=f_low)

    # Align time so t=0 is at peak of (2,2) amplitude
    t_peak = _peak_time(t, h[(2, 2)])
    t = t - t_peak
    print(f'NRSur7dq4 time grid: [{t[0]:.2f}, {t[-1]:.2f}] M')

    # Interpolate the requested modes (all modes if None) to the output grid.
    # No negative-m symmetry is applied: precessing modes are independent.
    keep = list(h.keys()) if modes is None else modes
    h_out = {}
    for mode in keep:
        h_out[mode] = gwtools.gwtools.interpolate_h(t, h[mode], times)

    print(f'Output time grid: [{times[0]:.2f}, {times[-1]:.2f}] M')

    return times, h_out


def compute_nrsur3dq8_remnant(surfinbh_module, q: float,
                              chi1: list[float] = [0, 0, 0],
                              chi2: list[float] = [0, 0, 0],
                              fit_name: str = 'NRSur3dq8Remnant',
                              print_output: bool = True) -> dict[str, float | np.ndarray]:
    """
    Compute remnant properties using NRSur3dq8Remnant surrogate.
    
    Predicts final mass, dimensionless spin, and kick velocity of the remnant
    black hole using the NRSur3dq8Remnant fit.
    
    Args:
        surfinbh_module: The surfinBH module imported by the user
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 10)
        chi1 (list): Dimensionless spin vector [sx, sy, sz] for primary BH.
            Default is [0, 0, 0]
        chi2 (list): Dimensionless spin vector [sx, sy, sz] for secondary BH.
            Default is [0, 0, 0]
        fit_name (str): Name of the remnant fit to use. Default is 'NRSur3dq8Remnant'.
            Other options include 'NRSur7dq4Remnant' for precessing systems
        print_output (bool): Whether to print remnant properties. Default is True
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
        ValueError: If q is outside valid range [1, 10]
    
    Example:
        >>> import surfinBH
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>> 
        >>> remnant = wg.compute_nrsur3dq8_remnant(
        ...     surfinBH,
        ...     q=3.0, 
        ...     chi1=[0, 0, 0.7]
        ... )
        >>> print(f"Final mass: {remnant['final_mass']:.4f} M")
        >>> print(f"Final spin: {remnant['final_spin']:.4f}")
        >>> print(f"Kick velocity: {remnant['kick_velocity']*299792.458:.1f} km/s")
    """
    # Validate inputs
    if not 1 <= q <= 10:
        raise ValueError(f"Mass ratio {q} outside valid range [1, 10] "
                        f"for NRSur3dq8Remnant")
    
    # Get remnant predictions with uncertainties
    mf, chif, vf, mf_err, chif_err, vf_err = surfinbh_module.all(q, chi1, chi2)
    
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
        _print_remnant_summary(fit_name, mf, mf_err, chif, chif_err, vf, vf_err)

    return remnant_properties


def compute_nrsur7dq4_remnant(surfinbh_module, q: float,
                              chi1: list[float] = [0, 0, 0],
                              chi2: list[float] = [0, 0, 0],
                              fit_name: str = 'NRSur7dq4Remnant',
                              print_output: bool = True) -> dict[str, float | np.ndarray]:
    """
    Compute remnant properties using the NRSur7dq4Remnant surrogate.

    Predicts the final mass, dimensionless spin vector, and kick velocity vector
    of the remnant black hole for generically precessing binaries using the
    NRSur7dq4Remnant fit. The full spin and kick vectors are returned (and
    printed), which matters for precessing systems.

    Args:
        surfinbh_module: The loaded NRSur7dq4Remnant surfinBH fit
        q (float): Mass ratio q = m1/m2, where m1 >= m2 (1 <= q <= 4)
        chi1 (list): Dimensionless spin vector [sx, sy, sz] for primary BH.
            Default is [0, 0, 0]
        chi2 (list): Dimensionless spin vector [sx, sy, sz] for secondary BH.
            Default is [0, 0, 0]
        fit_name (str): Name of the remnant fit. Default is 'NRSur7dq4Remnant'
        print_output (bool): Whether to print the remnant summary. Default is True

    Returns:
        [dict]: Dictionary containing:
            - 'final_mass': Final mass in units of total mass M
            - 'final_mass_err': Uncertainty in final mass
            - 'final_spin': Final dimensionless spin magnitude
            - 'final_spin_z': z-component of final spin
            - 'final_spin_vector': Full spin vector [sx, sy, sz]
            - 'final_spin_err': Uncertainty in final spin magnitude
            - 'kick_velocity': Kick velocity magnitude in units of c
            - 'kick_velocity_vector': Full kick velocity vector [vx, vy, vz]
            - 'kick_velocity_err': Uncertainty in kick velocity magnitude

    Raises:
        ValueError: If q is outside valid range [1, 4]

    Example:
        >>> import surfinBH
        >>> from gw_remnant.gw_utils import waveform_generator as wg
        >>>
        >>> fit = surfinBH.LoadFits('NRSur7dq4Remnant')
        >>> remnant = wg.compute_nrsur7dq4_remnant(
        ...     fit, q=3.0,
        ...     chi1=[0.5, 0.0, 0.3], chi2=[0.0, 0.4, -0.2])
        >>> print(remnant['final_spin_vector'])
    """
    # Validate inputs
    if not 1 <= q <= 4:
        raise ValueError(f"Mass ratio {q} outside valid range [1, 4] "
                        f"for NRSur7dq4Remnant")

    # Get remnant predictions with uncertainties
    mf, chif, vf, mf_err, chif_err, vf_err = surfinbh_module.all(q, chi1, chi2)

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
        _print_remnant_summary(fit_name, mf, mf_err, chif, chif_err, vf, vf_err)

    return remnant_properties