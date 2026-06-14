Installation
============

From PyPI
---------

.. code-block:: bash

   pip install gw_remnant

With optional waveform surrogates:

.. code-block:: bash

   pip install gw_remnant[surrogates]

Install everything:

.. code-block:: bash

   pip install gw_remnant[all]

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/tousifislam/gw_remnant.git
   cd gw_remnant
   pip install -e .[dev]

Requirements
------------

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- `gwtools <https://pypi.org/project/gwtools/>`_

Optional dependencies:

- ``gwsurrogate``, ``surfinBH`` -- for built-in waveform generation
