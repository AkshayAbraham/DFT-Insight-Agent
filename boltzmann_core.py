"""
boltzmann_core.py
Handles conformer thermochemistry, energy window filtering, duplicate removal,
degeneracy weighting, Boltzmann populations, ensemble property averaging, and temperature scans.
"""

import numpy as np
import pandas as pd

HARTREE_TO_KCAL_MOL = 627.509474
KB = 0.0019872041  # Boltzmann constant in kcal/(mol·K)
DEFAULT_TEMP = 298.15


def process_conformer_ensemble(results: list, temperature: float = DEFAULT_TEMP, energy_window_kcal: float = 3.0, degeneracy_map: dict = None) -> tuple:
    """
    Processes a list of conformers, applying energy windows, duplicate filtering,
    symmetry/degeneracy weights, Boltzmann weights, and ensemble property averaging.
    """
    df = pd.DataFrame(results)
    valid_df = df[df["status"] == "SUCCESS"].copy()
    valid_df = valid_df[valid_df["free_energy_hartree"].notna()].copy()
    
    if valid_df.empty:
        df["delta_g_kcal_mol"] = float('inf')
        df["population_percent"] = 0.0
        return df, {}

    g_min = valid_df["free_energy_hartree"].min()
    valid_df["delta_g_kcal_mol"] = (valid_df["free_energy_hartree"] - g_min) * HARTREE_TO_KCAL_MOL

    # Energy Window & Duplicate Filtering
    filtered_df = valid_df[valid_df["delta_g_kcal_mol"] <= energy_window_kcal].copy()
    filtered_df = filtered_df.drop_duplicates(subset=["free_energy_hartree"], keep="first")

    if filtered_df.empty:
        filtered_df = valid_df.head(1).copy()

    # --- FEATURE #6: Degeneracy Weights (Symmetry/Enantiomers) ---
    if degeneracy_map is None:
        degeneracy_map = {}
    
    # Assign degeneracy (default to 1 if not specified)
    filtered_df["degeneracy"] = filtered_df["file_name"].map(lambda x: degeneracy_map.get(x, 1))

    # --- Boltzmann Populations with Degeneracy ---
    rt = KB * temperature
    # G includes entropy of mixing for degenerate states, or we weight the Boltzmann factor by degeneracy g_i * exp(-ΔG/RT)
    filtered_df["boltzmann_factor"] = filtered_df["degeneracy"] * np.exp(-filtered_df["delta_g_kcal_mol"] / rt)

    total_bf = filtered_df["boltzmann_factor"].sum()
    if total_bf > 0:
        filtered_df["population_percent"] = (filtered_df["boltzmann_factor"] / total_bf) * 100.0
    else:
        filtered_df["population_percent"] = 0.0

    ranked_df = filtered_df.sort_values(by="delta_g_kcal_mol", ascending=True).reset_index(drop=True)

    # --- Ensemble Averages (<P> = sum(p_i * P_i)) ---
    properties_to_average = ["homo_lumo_gap_ev", "dipole_moment_debye", "scf_energy_hartree", "homo_ev", "lumo_ev"]
    ensemble_averages = {}

    for prop in properties_to_average:
        if prop in ranked_df.columns:
            sub_df = ranked_df.dropna(subset=[prop])
            if not sub_df.empty:
                sub_weights = sub_df["boltzmann_factor"] / sub_df["boltzmann_factor"].sum()
                ensemble_averages[prop] = float(np.sum(sub_weights * sub_df[prop]))
            else:
                ensemble_averages[prop] = None

    return ranked_df, ensemble_averages


def calculate_temperature_scan(results: list, temp_range=range(200, 451, 10)) -> pd.DataFrame:
    """
    Calculates population shifts across a temperature range (Feature #2).
    """
    scan_data = []
    valid_results = [r for r in results if r.get("status") == "SUCCESS" and r.get("free_energy_hartree") is not None]
    
    if not valid_results:
        return pd.DataFrame()

    df_base = pd.DataFrame(valid_results)
    g_min = df_base["free_energy_hartree"].min()
    df_base["delta_g_kcal_mol"] = (df_base["free_energy_hartree"] - g_min) * HARTREE_TO_KCAL_MOL

    for t in temp_range:
        rt = KB * t
        df_base["bf"] = np.exp(-df_base["delta_g_kcal_mol"] / rt)
        total_bf = df_base["bf"].sum()
        
        for _, row in df_base.iterrows():
            pop = (row["bf"] / total_bf) * 100.0 if total_bf > 0 else 0.0
            scan_data.append({
                "Temperature_K": t,
                "file_name": row["file_name"],
                "population_percent": pop
            })

    return pd.DataFrame(scan_data)