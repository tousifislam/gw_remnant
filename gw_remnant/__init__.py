"""
GW Remnant: Gravitational Wave Remnant Property Calculator
A package for calculating remnant properties of binary black hole mergers.
"""

__version__ = "0.1.0"
__author__ = "Tousif Islam"
__email__ = "tousifislam24@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025, Tousif Islam"
__credits__ = ["Tousif Islam", "Collaborator Name"]
__maintainer__ = "Tousif Islam"
__status__ = "Development" 

# Import main modules
from . import gw_remnant_calculator
from . import gw_waveform_generator

# Import submodules
from . import gw_utils
from . import remnant_calculators

__all__ = [
    'gw_remnant_calculator',
    'gw_waveform_generator',
    'gw_utils',
    'remnant_calculators',
]