gw_remnant
==========

**Remnant black hole properties from gravitational waveforms**

``gw_remnant`` is a Python package for extracting remnant mass, spin, peak
luminosity, and kick velocity of the remnant black hole directly from
gravitational radiation emitted during binary black hole mergers.

Energy and momenta carried away as gravitational waves determine the final
state of the remnant.  This package computes the full time evolution of these
quantities and reports the final remnant properties.

.. code-block:: python

   from gw_remnant.gw_remnant_calculator import GWRemnantCalculator

   calc = GWRemnantCalculator(time, h_dict, q=2.0,
                              chi1=[0, 0, 0.5])
   calc.print_remnants()
   calc.plot_mass_energy()

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: About

   citation
   changelog

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
