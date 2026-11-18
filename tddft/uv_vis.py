"""
tdft/uv_vis.py
Handles parsing TD-DFT excited states, identifying lambda_max, 
and computing Gaussian-broadened simulated UV-Vis absorption spectra.
"""

import numpy as np
import pandas as pd

# Conversion factor: eV to nanometers (hc in eV·nm)
EV_TO_NM = 1239.84193
CM1_TO_EV = 1.0 / 8065.54429  # Conversion from cm^-1 to eV


def parse_uv_vis_data(cclib_data) -> dict:
    """
    Extracts TD-DFT excitation energies, oscillator strengths, and wavelengths from cclib data.
    """
    if not hasattr(cclib_data, "etenergies") or cclib_data.etenergies is None:
        return {"has_tddft": False}

    # cclib parses etenergies in cm^-1 for Gaussian files. Convert to eV.
    raw_energies = np.array(cclib_data.etenergies)
    energies_ev = raw_energies * CM1_TO_EV

    osc_strengths = np.array(cclib_data.etosc) if hasattr(cclib_data, "etosc") and cclib_data.etosc is not None else np.zeros_like(energies_ev)
    
    # Convert eV energies to wavelengths in nm: lambda = 1239.84 / E(eV)
    wavelengths_nm = EV_TO_NM / energies_ev

    transitions = []
    for i in range(len(energies_ev)):
        transitions.append({
            "state": i + 1,
            "energy_ev": float(energies_ev[i]),
            "wavelength_nm": float(wavelengths_nm[i]),
            "oscillator_strength": float(osc_strengths[i])
        })

    # Find lambda_max (strongest peak in the optical range >= 150 nm)
    df_trans = pd.DataFrame(transitions)
    valid_trans = df_trans[df_trans["wavelength_nm"] >= 150]
    
    lambda_max_nm = None
    max_f = 0.0
    if not valid_trans.empty:
        max_idx = valid_trans["oscillator_strength"].idxmax()
        lambda_max_nm = float(valid_trans.loc[max_idx, "wavelength_nm"])
        max_f = float(valid_trans.loc[max_idx, "oscillator_strength"])

    return {
        "has_tddft": True,
        "total_states": len(energies_ev),
        "lambda_max_nm": lambda_max_nm,
        "max_oscillator_strength": max_f,
        "transitions": transitions
    }


def compute_uv_vis_spectrum(transitions: list, sigma_ev: float = 0.3, wavelength_grid=None) -> pd.DataFrame:
    """
    Generates a continuous UV-Vis absorption spectrum by applying Gaussian broadening 
    (default sigma ~ 0.3 eV) to discrete electronic transitions weighted by oscillator strength.
    """
    if not transitions:
        return pd.DataFrame()

    if wavelength_grid is None:
        wavelength_grid = np.linspace(200, 800, 600)  # Standard 200 nm to 800 nm grid

    absorbance = np.zeros_like(wavelength_grid, dtype=float)

    for t in transitions:
        e_center = t["energy_ev"]
        f_osc = t["oscillator_strength"]
        
        if f_osc <= 0:
            continue

        # Evaluate Gaussian in energy space and accumulate onto wavelength grid
        for idx, lam in enumerate(wavelength_grid):
            e_current = EV_TO_NM / lam
            gaussian_val = f_osc * np.exp(-((e_current - e_center) ** 2) / (2 * (sigma_ev ** 2)))
            absorbance[idx] += gaussian_val

    # Normalize absorbance between 0 and 1 for clean plotting
    max_abs = absorbance.max()
    if max_abs > 0:
        absorbance_norm = absorbance / max_abs
    else:
        absorbance_norm = absorbance

    df_spec = pd.DataFrame({
        "Wavelength_nm": wavelength_grid,
        "Absorbance": absorbance,
        "Normalized_Absorbance": absorbance_norm
    })

    return df_spec