# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "gw_remnant"
copyright = "2025, Tousif Islam"
author = "Tousif Islam"
release = "0.3.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Mock imports for packages unavailable in CI ----------------------------

autodoc_mock_imports = [
    "gwtools",
    "lal",
    "lalsimulation",
    "gwsurrogate",
    "surfinBH",
]

# -- Autodoc settings -------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# -- Napoleon settings (Google-style docstrings) ----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- nbsphinx settings (render notebooks with stored outputs) ---------------

nbsphinx_execute = "never"

# -- Intersphinx mapping ----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output options -----------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

html_static_path = ["_static"]
html_title = "gw_remnant"

# -- Inheritance diagram options ---------------------------------------------

inheritance_graph_attrs = dict(rankdir="TB", size='"8.0, 12.0"')
inheritance_node_attrs = dict(shape="ellipse", fontsize=10)

# Use SVG for graphviz output; suppress warning if dot is missing
graphviz_output_format = "svg"
suppress_warnings = ["graphviz"]
