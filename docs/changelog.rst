Changelog
=========

v0.4.0 (dev)
------------

- Added orbit-averaged frequency for robust PN initial conditions with
  eccentric waveforms
- Added NRSur7dq4 waveform generator and remnant helpers (final spin and
  kick vectors)
- Added ``pyseobnr`` as optional dependency (``pip install gw_remnant[eob]``)
- Added junk radiation warning printed on construction and unphysical-spin
  warning when ``|chi_f| > 1``
- Added center-of-mass trajectory calculator (``TrajectoryCalculator``,
  ``plot_trajectory``)
- Added seven tutorial notebooks covering surrogate, SXS, Maya, and pySEOBNR
  waveforms
- Removed old example notebooks

v0.3.0
------

- Added type hints to all public APIs
- Added 3D spin vector evolution tracking (``spin_vector_oft``,
  ``remnant_spin_vector``)
- Added spin vector diagnostic plot (``plot_spin_vector``)
- Added pytest suite for input validation, remnant properties, and peak finder
- Performance optimizations

v0.2.0
------

- Added ``GWRemnantCalculator`` as the main user-facing class
- Added diagnostic plotting utilities (``GWPlotter``)
- Added waveform generators for NRHybSur3dq8, BHPTNRSur1dq1e4, and
  BHPTNRSur2dq1e3
- Added ``compute_nrsur3dq8_remnant`` for surfinBH-based remnant predictions
- Support for user-provided custom waveforms

v0.1.0
------

- Initial release
- Core remnant mass, spin, kick velocity, and peak luminosity calculations
- Post-Newtonian initial conditions (non-spinning, spinning, eccentric)
