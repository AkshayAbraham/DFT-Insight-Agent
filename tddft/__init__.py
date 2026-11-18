"""
tdft package
Exposes modular TD-DFT photophysics, UV-Vis spectral broadening, 
orbital transition character analysis, and validation utilities.
"""

from .uv_vis import parse_uv_vis_data, compute_uv_vis_spectrum
from .transitions import analyze_orbital_transitions, get_charge_transfer_warning

__all__ = [
    "parse_uv_vis_data",
    "compute_uv_vis_spectrum",
    "analyze_orbital_transitions",
    "get_charge_transfer_warning",
]