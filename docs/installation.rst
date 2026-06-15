Installation
============

From PyPI
---------

.. code-block:: bash

   pip install gw_remnant

With optional waveform surrogates:

.. code-block:: bash

   pip install gw_remnant[surrogates]

With numerical relativity catalog access:

.. code-block:: bash

   pip install gw_remnant[nr]

With effective-one-body waveforms:

.. code-block:: bash

   pip install gw_remnant[eob]

Install everything:

.. code-block:: bash

   pip install gw_remnant[all]

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/tousifislam/gw_remnant.git
   cd gw_remnant
   pip install -e .[dev]

Dependencies
------------

The following packages are required and will be installed automatically:

* `numpy <https://numpy.org/>`_ >= 1.20.0
* `scipy <https://scipy.org/>`_ >= 1.7.0
* `matplotlib <https://matplotlib.org/>`_ >= 3.3.0
* `gwtools <https://pypi.org/project/gwtools/>`_

Optional dependencies:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Install option
     - Packages
     - Purpose
   * - ``[surrogates]``
     - gwsurrogate, surfinBH
     - Waveform surrogate models and remnant fits
   * - ``[nr]``
     - sxs, mayawaves
     - Numerical relativity catalog access
   * - ``[eob]``
     - pyseobnr
     - Effective-one-body waveforms
   * - ``[all]``
     - All of the above
     - All non-dev optional dependencies

Building the documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   pip install gw_remnant[docs]
   cd docs
   make html

The built documentation will be in ``docs/_build/html/``.
