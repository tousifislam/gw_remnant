Physics Overview
================

``gw_remnant`` computes the properties of the remnant black hole formed after
a binary black hole merger by analysing the gravitational radiation.

How it works
------------

During a binary black hole merger, gravitational waves carry away energy,
linear momentum, and angular momentum from the system.  By integrating the
corresponding fluxes over time we can track how the binary evolves and what
the final remnant looks like.

The package computes:

1. **Energy flux** -- rate of energy loss through gravitational waves
2. **Radiated energy** -- cumulative energy emitted
3. **Remnant mass** -- initial mass minus radiated energy
4. **Linear momentum flux** -- asymmetric radiation carries net momentum
5. **Kick velocity** -- recoil velocity of the remnant
6. **Angular momentum flux** -- gravitational waves carry angular momentum
7. **Remnant spin** -- final dimensionless spin of the remnant
8. **Peak luminosity** -- maximum energy emission rate

Class hierarchy
---------------

.. inheritance-diagram:: gw_remnant.gw_remnant_calculator.GWRemnantCalculator
   :top-classes: gw_remnant.remnant_calculators.initial_energy_momenta.InitialEnergyMomenta
   :parts: 1

The main entry point is :class:`~gw_remnant.gw_remnant_calculator.GWRemnantCalculator`,
which inherits from several specialised calculators:

- :class:`~gw_remnant.remnant_calculators.initial_energy_momenta.InitialEnergyMomenta`
  -- post-Newtonian initial energy and angular momentum
- :class:`~gw_remnant.remnant_calculators.remnant_mass_calculator.RemnantMassCalculator`
  -- energy flux and remnant mass
- :class:`~gw_remnant.remnant_calculators.kick_velocity_calculator.LinearMomentumCalculator`
  -- linear momentum and kick velocity
- :class:`~gw_remnant.remnant_calculators.remnant_spin_calculator.AngularMomentumCalculator`
  -- angular momentum and remnant spin
- :class:`~gw_remnant.remnant_calculators.peak_luminosity_calculator.PeakLuminosityCalculator`
  -- peak luminosity detection
- :class:`~gw_remnant.gw_utils.gw_plotter.GWPlotter`
  -- diagnostic plotting utilities

All computation happens at construction time.  Once you create a
``GWRemnantCalculator`` instance the results are available as attributes.
