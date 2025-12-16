"""
macecvs
=======

A Python package for MACE-based machine learning collective variables.

Modules:
--------
- `utils`: Utility functions and helper modules.
- `models`: Definitions of architectures like permutation invariant autoencoders.
- `training`: Scripts and utilities for training machine learning models.
- `calculations`: Modules for calculations and numerical operations.

Author: Thomas Pigeon
Email: thomas.pigeon@ifpen.fr
"""

# Import key submodules for easier package access
from .utils import *
from .models import *
from .training import *
from .calculations import *

# Define the package's version
__version__ = "0.1"

# Provide easy access to some core functions or classes
# Example: If there's a central function or class in your package
# from .models.permutation_invar_AE import PermutationInvariantAutoEncode
