# gw_remnant — Code Map

## Directory structure

```
gw_remnant/
├── pyproject.toml
├── __init__.py                          # top-level (re-exports package)
├── gw_remnant/
│   ├── __init__.py                      # v0.3.0, exports all submodules
│   ├── gw_remnant_calculator.py         # GWRemnantCalculator — main user-facing class
│   ├── gw_waveform_generator.py         # re-exports gw_utils.waveform_generator
│   ├── remnant_calculators/
│   │   ├── __init__.py
│   │   ├── initial_energy_momenta.py    # InitialEnergyMomenta — PN initial E, L (base class)
│   │   ├── remnant_mass_calculator.py   # RemnantMassCalculator — energy flux, radiated E, Bondi mass
│   │   ├── kick_velocity_calculator.py  # LinearMomentumCalculator — linear momentum flux, kick velocity
│   │   ├── trajectory_calculator.py     # TrajectoryCalculator — center-of-mass displacement
│   │   ├── remnant_spin_calculator.py   # AngularMomentumCalculator — angular momentum flux, remnant spin
│   │   └── peak_luminosity_calculator.py# PeakLuminosityCalculator — peak GW luminosity
│   └── gw_utils/
│       ├── __init__.py
│       ├── waveform_generator.py        # surrogate waveform generators + surfinBH remnant fits
│       └── gw_plotter.py               # GWPlotter — diagnostic time-series plots
├── tests/
│   ├── conftest.py
│   ├── test_remnant_properties.py       # end-to-end remnant property tests
│   ├── test_validation.py              # input validation tests
│   ├── test_quadratic_fit.py           # peak-finding tests
│   ├── test_pn_against_seobnrv5.py     # PN initial conditions vs SEOBNRv5 reference
│   ├── pn_reference_seobnrv5.py        # reference data generator
│   ├── ehm_reference.py               # EHM reference data
│   └── combined_pn.py                  # combined PN reference
├── tutorials/                           # 8 Jupyter notebooks (NR, surrogate, EOB sources)
└── docs/                                # Sphinx documentation (RST + autodoc)
```

## Module guide

### `gw_remnant_calculator.py` — main entry point
`GWRemnantCalculator` inherits from all calculator classes via diamond MRO. Instantiating it with `(time, h_dict, q)` computes everything: energy, mass, momentum, spin, kick, trajectory. Key methods: `get_remnant_properties()` returns a flat dict, `print_remnants()` prints a summary.

### `remnant_calculators/initial_energy_momenta.py` — base class
`InitialEnergyMomenta` validates inputs and computes initial orbital binding energy and angular momentum from PN expressions. Supports circular (to 5PN with NR-calibrated pseudo-5PN), aligned-spin (to 3PN), and eccentric (to 3PN) binaries. Uses orbit-averaged SEOBNRv5EHM expressions (arXiv:2412.12823) plus 4PN analytic (arXiv:2303.18143) and 5PN NR-calibrated (arXiv:1111.5378) terms.

### `remnant_calculators/remnant_mass_calculator.py`
`RemnantMassCalculator` computes dh/dt for all modes, energy flux dE/dt (Eq. 2 of arXiv:1802.04276), cumulative radiated energy, Bondi mass M(t), and remnant mass. Optional spline filtering for noisy data. This is the base for all other calculators.

### `remnant_calculators/kick_velocity_calculator.py`
`LinearMomentumCalculator` computes 3D linear momentum flux (dPx/dt, dPy/dt, dPz/dt) from multipole formulas (arXiv:0707.4654, arXiv:1802.04276), integrates to get P(t), divides by M(t) for velocity, and reports final/peak kick in both c and km/s.

### `remnant_calculators/trajectory_calculator.py`
`TrajectoryCalculator` integrates the recoil velocity v(t) to obtain center-of-mass displacement x(t). Reports final remnant displacement vector.

### `remnant_calculators/remnant_spin_calculator.py`
`AngularMomentumCalculator` computes 3D angular momentum flux (dJx/dt, dJy/dt, dJz/dt), integrates for J(t), and derives dimensionless spin chi(t) = (L_initial + S1 + S2 - J_rad) / M(t)^2. Supports both scalar L_initial (aligned) and vector L_initial (precessing).

### `remnant_calculators/peak_luminosity_calculator.py`
`PeakLuminosityCalculator` finds peak of energy flux using 4th-order spline interpolation around the discrete maximum.

### `gw_utils/waveform_generator.py`
Standalone functions to generate waveforms from surrogate models. The user passes the loaded module (dependency injection). Functions: `generate_nrhybsur3dq8()`, `generate_bhptnrsur1dq1e4()`, `generate_bhptnrsur2dq1e3()`, `generate_nrsur7dq4()`, `compute_nrsur3dq8_remnant()`, `compute_nrsur7dq4_remnant()`. All align t=0 to peak of (2,2) amplitude and interpolate to requested time grid.

### `gw_utils/gw_plotter.py`
`GWPlotter` provides diagnostic plots: `plot_mass_energy()`, `plot_linear_momentum()`, `plot_angular_momentum()`, `plot_kick_velocity()`, `plot_spin_vector()`, `plot_trajectory()`. All use a pre-merger / post-merger split layout (t <= -500M | t > -500M).

## Class hierarchy

```
InitialEnergyMomenta
  └── RemnantMassCalculator
        ├── PeakLuminosityCalculator
        └── LinearMomentumCalculator
              └── TrajectoryCalculator
              └── AngularMomentumCalculator
                    └── GWPlotter
                          └── GWRemnantCalculator  (user-facing, diamond MRO)
```

## Dependencies

**Hard:** numpy (>=1.20), scipy (>=1.7), matplotlib (>=3.3), gwtools

**Optional:**
- `surrogates`: gwsurrogate, surfinBH
- `nr`: sxs, mayawaves
- `eob`: pyseobnr

## Entry points

```python
from gw_remnant.gw_remnant_calculator import GWRemnantCalculator

calc = GWRemnantCalculator(time, h_dict, q=3.0, chi1=[0, 0, 0.5])
calc.print_remnants()
props = calc.get_remnant_properties()

# Waveform generation (pass your own loaded modules)
from gw_remnant.gw_utils.waveform_generator import generate_nrhybsur3dq8
t, h = generate_nrhybsur3dq8(loaded_surrogate, q=3.0)
```
