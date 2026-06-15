Contributing
============

Contributions to ``gw_remnant`` are welcome! Here's how you can help:

Reporting Issues
----------------

If you find a bug or have a feature request, please open an issue on the
`GitHub issue tracker <https://github.com/tousifislam/gw_remnant/issues>`_.

Development Setup
-----------------

1. Fork and clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/gw_remnant.git
      cd gw_remnant

2. Install in development mode:

   .. code-block:: bash

      pip install -e ".[dev]"

3. Run the test suite:

   .. code-block:: bash

      pytest tests/

4. If you modified documentation, build and check it:

   .. code-block:: bash

      pip install -e ".[docs]"
      cd docs
      make html

Pull Requests
-------------

1. Create a new branch for your feature or fix.
2. Make your changes with clear commit messages.
3. Ensure all tests pass before submitting.
4. Submit a pull request with a description of what you changed and why.

Code Style
----------

* Follow PEP 8 conventions.
* Use Google-style docstrings for functions and classes.
* Include type hints in function signatures.
