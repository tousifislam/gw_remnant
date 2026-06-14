# Overview of gw_remnant

## What it does

`gw_remnant` (v0.3.0) computes remnant black hole properties from gravitational waveforms of binary black hole mergers. Given a time-series of spherical-harmonic waveform modes h_lm(t), it extracts the final mass, spin, kick velocity, trajectory, and peak luminosity of the merger remnant.

## Architecture

The package uses a mixin-based class hierarchy that all composes into a single entry point:

```
GWRemnantCalculator   (gw_remnant_calculator.py)
 ├── GWPlotter                    — diagnostic plots (mass, momentum, spin, kick, trajectory)
 ├── PeakLuminosityCalculator     — peak dE/dt via spline interpolation
 ├── AngularMomentumCalculator    — J_dot, Joft, remnant spin (scalar + 3-vector)
 ├── TrajectoryCalculator         — center-of-mass displacement x(t) = ∫v dt
 ├── LinearMomentumCalculator     — P_dot, Poft, kick velocity v(t) = P/M(t)
 ├── RemnantMassCalculator        — dE/dt, E_rad(t), Bondi mass M(t), remnant mass
 └── InitialEnergyMomenta         — initial E, L from post-Newtonian (PN) expressions
```

Everything is computed eagerly in `__init__` — constructing a `GWRemnantCalculator` runs the full pipeline bottom-up.

## Key modules

| File | Role |
|------|------|
| `initial_energy_momenta.py` | Computes initial binding energy E₀ and angular momentum L₀ using PN theory (3PN eccentric+spin from SEOBNRv5EHM, plus 4PN/5PN non-spinning circular terms). Validates all inputs. |
| `remnant_mass_calculator.py` | Differentiates h_lm to get ḣ, computes energy flux dE/dt = Σ\|ḣ_lm\|²/(16π), integrates to get E_rad(t) and Bondi mass M(t) = M_ADM - E_rad(t). Optional spline filtering for noisy NR data. |
| `kick_velocity_calculator.py` | Computes linear momentum flux (Px, Py, Pz) from mode-coupling formulas (arXiv:0707.4654), integrates for P(t), divides by M(t) for velocity/kick. |
| `remnant_spin_calculator.py` | Computes angular momentum flux (Jx, Jy, Jz), integrates, and gets remnant spin χ = (L₀ + S₁ + S₂ - J_rad) / M_f². Supports full 3-vector spin for precessing systems. |
| `trajectory_calculator.py` | Integrates velocity v(t) to get center-of-mass displacement x(t). |
| `peak_luminosity_calculator.py` | Finds peak dE/dt via 4th-order spline interpolation. |
| `gw_plotter.py` | Split pre/post-merger time-series plots for all quantities, plus an orbital-plane trajectory plot. |
| `waveform_generator.py` | Helper functions to generate waveforms from surrogate models (NRHybSur3dq8, BHPTNRSur1dq1e4, BHPTNRSur2dq1e3, NRSur7dq4) and compute remnant properties from surfinBH fits. Users pass in the imported surrogate modules themselves. |

## Usage pattern

```python
calc = GWRemnantCalculator(time, h_dict, q=3.0, chi1=[0,0,0.5])
calc.print_remnants()      # summary table
calc.plot_mass_energy()    # diagnostic plots
props = calc.get_remnant_properties()  # dict of all results
```

## Dependencies

Core: `numpy`, `scipy`, `matplotlib`, `gwtools`. Optional: `gwsurrogate`, `surfinBH` (for waveform generation/remnant fits), `sxs`, `mayawaves` (for numerical relativity catalogs).
