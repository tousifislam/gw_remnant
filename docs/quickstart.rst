Quick Start
===========

``gw_remnant`` computes remnant black hole properties from gravitational
waveform data.  You provide a time array and a dictionary of complex waveform
modes and the package does the rest.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from gw_remnant.gw_remnant_calculator import GWRemnantCalculator

   # time: 1D array in geometric units (M)
   # h_dict: dictionary of complex waveform modes, e.g. {(2,2): h22(t), ...}
   calc = GWRemnantCalculator(time, h_dict, q=2.0,
                              chi1=[0, 0, 0.5])

   # Print a summary table
   calc.print_remnants()

   # Access individual properties as a dictionary
   props = calc.get_remnant_properties()
   print(f"Remnant mass: {props['remnant_mass']:.6f} M")
   print(f"Remnant spin: {props['remnant_spin']:.6f}")

   # Diagnostic plots
   calc.plot_mass_energy(save_path="mass_energy.png")

Using built-in waveform generators
-----------------------------------

If you have ``gwsurrogate`` installed, you can generate waveforms directly:

.. code-block:: python

   import gwsurrogate
   from gw_remnant.gw_utils import waveform_generator as wg

   times, h_dict = wg.generate_nrhybsur3dq8(
       gwsurrogate,
       q=3.0,
       chi1=[0, 0, 0.5],
   )

   calc = GWRemnantCalculator(times, h_dict, q=3.0,
                              chi1=[0, 0, 0.5])
   calc.print_remnants()

Available plot methods
----------------------

.. code-block:: python

   calc.plot_mass_energy()        # Mass and energy evolution
   calc.plot_linear_momentum()    # Linear momentum components
   calc.plot_angular_momentum()   # Angular momentum components
   calc.plot_kick_velocity()      # Kick velocity magnitude
   calc.plot_spin_vector()        # Spin vector components

All plot methods accept an optional ``save_path`` argument.  When provided the
figure is saved to disk instead of being displayed interactively.
