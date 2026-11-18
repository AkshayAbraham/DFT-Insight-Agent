"""
tdft/transitions.py
Extracts molecular orbital contributions for excited states, classifies transition characters,
and flags charge-transfer states with scientific caveats for standard hybrid functionals.
"""

import numpy as np
import pandas as pd


def _extract_scalar(val) -> int:
    """Safely extracts an integer index from scalars, lists, or tuples (e.g., spin-orbitals)."""
    if isinstance(val, (int, float, np.number)):
        return int(val)
    elif isinstance(val, (tuple, list)) and len(val) > 0:
        return int(val[0])
    return int(val)


def analyze_orbital_transitions(cclib_data) -> list:
    """
    Extracts major molecular orbital contributions (e.g., HOMO->LUMO, percent contribution)
    from cclib parsed TD-DFT data and classifies transition character.
    """
    if not hasattr(cclib_data, "etsecs") or cclib_data.etsecs is None:
        return []

    transitions_breakdown = []
    
    homo_idx = None
    if hasattr(cclib_data, "homos") and cclib_data.homos is not None:
        h_val = cclib_data.homos[0]
        homo_idx = _extract_scalar(h_val)

    for state_idx, state_contribs in enumerate(cclib_data.etsecs):
        state_info = {
            "state": state_idx + 1,
            "dominant_transitions": [],
            "character_type": "Localized / Valence ($\pi \to \pi^* / n \to \pi$)",
            "is_charge_transfer": False
        }

        valid_contribs = []
        for contrib in state_contribs:
            if len(contrib) >= 3:
                start_mo = _extract_scalar(contrib[0])
                end_mo = _extract_scalar(contrib[1])
                amplitude = float(contrib[2])
                weight_percent = (amplitude ** 2) * 100.0
                
                if weight_percent >= 5.0:  # Filter minor noise below 5% contribution
                    from_label = f"HOMO{start_mo - homo_idx}" if homo_idx is not None else f"MO({start_mo})"
                    if start_mo == homo_idx:
                        from_label = "HOMO"
                    elif start_mo < homo_idx:
                        from_label = f"HOMO-{homo_idx - start_mo}"
                    
                    to_label = f"LUMO+{end_mo - (homo_idx + 1)}" if homo_idx is not None else f"MO({end_mo})"
                    if end_mo == homo_idx + 1:
                        to_label = "LUMO"

                    valid_contribs.append({
                        "from_mo": start_mo,
                        "to_mo": end_mo,
                        "from_label": from_label,
                        "to_label": to_label,
                        "weight_percent": round(weight_percent, 1)
                    })

        # Sort by contribution weight descending
        valid_contribs = sorted(valid_contribs, key=lambda x: x["weight_percent"], reverse=True)
        state_info["dominant_transitions"] = valid_contribs

        # Heuristic classification for transition character
        if valid_contribs:
            top_jump = valid_contribs[0]
            mo_jump_span = abs(top_jump["to_mo"] - top_jump["from_mo"])
            
            if mo_jump_span > 4:
                state_info["character_type"] = "Charge-Transfer (CT)"
                state_info["is_charge_transfer"] = True
            elif top_jump["from_label"].startswith("HOMO") and top_jump["to_label"].startswith("LUMO"):
                state_info["character_type"] = "Valence ($\pi \to \pi^*$)"
            else:
                state_info["character_type"] = "Mixed / Rydberg-like"

        transitions_breakdown.append(state_info)

    return transitions_breakdown


def get_charge_transfer_warning(transitions_breakdown: list) -> tuple:
    """
    Checks if any flagged charge-transfer states exist and returns a scientific warning banner.
    """
    ct_states = [t["state"] for t in transitions_breakdown if t["is_charge_transfer"]]
    
    if ct_states:
        warning_msg = (
            f"⚠️ **Charge-Transfer State Detected (States: {', '.join(map(str, ct_states))}):** "
            "Standard hybrid functionals like B3LYP systematically underestimate excitation energies "
            "for long-range charge-transfer transitions due to incorrect asymptotic exchange behavior. "
            "Consider verifying with range-separated hybrids (e.g., CAM-B3LYP, ωB97XD)."
        )
        return True, warning_msg
    
    return False, ""