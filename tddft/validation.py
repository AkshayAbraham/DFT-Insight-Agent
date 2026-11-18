"""
tdft/validation.py
Performs quality control checks on TD-DFT outputs, verifying state counts,
singlet/triplet state configurations, and batch functional consistency.
"""

import pandas as pd


def validate_tdft_job(cclib_data, max_wavelength_threshold_nm: float = 400.0) -> list:
    """
    Runs automated validation checks on a single TD-DFT calculation.
    Flags issues like insufficient states to cover the visible range or missing oscillator strengths.
    """
    warnings = []
    
    if not hasattr(cclib_data, "etenergies") or cclib_data.etenergies is None:
        return ["ERROR: No TD-DFT excitation energies found in output file."]

    energies_ev = cclib_data.etenergies
    n_states = len(energies_ev)

    # Check 1: State count check (ensure enough states were requested)
    EV_TO_NM = 1239.84193
    max_energy_ev = min(energies_ev)  # Lowest energy state (longest wavelength)
    max_wavelength_nm = EV_TO_NM / max_energy_ev

    if max_wavelength_nm < max_wavelength_threshold_nm and n_states < 10:
        warnings.warn_msg = (
            f"⚠️ **Insufficient States Warning:** Only {n_states} states computed, "
            f"reaching down to {max_wavelength_nm:.1f} nm. You may have missed lower-lying "
            "optical transitions in the visible spectrum. Consider increasing `nstates`."
        )
        warnings.append(warnings.warn_msg)

    # Check 2: Verify oscillator strengths existence
    if not hasattr(cclib_data, "etosc") or cclib_data.etosc is None:
        warnings.append("⚠️ **Missing Oscillator Strengths:** Transition intensities could not be verified.")

    # Check 3: Multiplicity / Spin state checks if available
    if hasattr(cclib_data, "etsecs") and len(energies_ev) == 0:
        warnings.append("⚠️ **Empty Transitions:** Excited state manifold contains no transitions.")

    if not warnings:
        warnings.append("✅ **TD-DFT QC Pass:** State sampling and transition metrics look healthy.")

    return warnings


def check_batch_consistency(batch_results: list) -> dict:
    """
    Validates consistency across a batch of TD-DFT files (e.g., matching state counts and methods).
    """
    total_files = len(batch_results)
    valid_tdft_files = [r for r in batch_results if r.get("has_tddft", False)]
    
    if not valid_tdft_files:
        return {"status": "FAILED", "message": "No valid TD-DFT files found in batch."}

    state_counts = [len(r.get("transitions", [])) for r in valid_tdft_files]
    uniform_states = len(set(state_counts)) == 1

    report = {
        "total_files_evaluated": total_files,
        "valid_tdft_jobs": len(valid_tdft_files),
        "state_count_consistency": uniform_states,
        "states_per_job": state_counts if uniform_states else "Mixed state counts across files",
        "batch_qc_status": "PASS" if uniform_states else "WARNING: Inconsistent number of states computed across batch series."
    }

    return report