Units and Conventions
=====================

Geometric units
---------------

All internal calculations use geometric units where *G* = *c* = 1:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Quantity
     - Unit
     - Symbol
   * - Mass
     - Total mass *M*
     - ``M``
   * - Time
     - *M*
     - ``t``
   * - Energy
     - *M*
     - ``E``
   * - Angular momentum
     - *M*\ :sup:`2`
     - ``J``
   * - Linear momentum
     - *M*
     - ``P``
   * - Velocity (kick)
     - *c*
     - ``v``
   * - Spin (dimensionless)
     - *S* / *M*\ :sup:`2`
     - ``chi``

Kick velocity in km/s
----------------------

The kick velocity can also be reported in km/s using the conversion
factor from ``lal`` (speed of light).  This requires ``lal`` to be installed:

.. code-block:: bash

   pip install gw_remnant[lal]

The converted value is available via:

.. code-block:: python

   props = calc.get_remnant_properties()
   print(props["remnant_kick_kmps"])  # km/s

Sign conventions
----------------

- **Mass ratio**: *q* = *m*\ :sub:`1` / *m*\ :sub:`2` with *m*\ :sub:`1` >= *m*\ :sub:`2`, so *q* >= 1.
- **Spin vectors**: Cartesian ``[sx, sy, sz]``, dimensionless.
  Aligned-spin systems have only a *z*-component.
- **Time alignment**: *t* = 0 corresponds to the peak amplitude of the
  (2, 2) waveform mode.
