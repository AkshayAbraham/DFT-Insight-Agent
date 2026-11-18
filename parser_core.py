"""
parser_core.py
Core chemistry logic and parsing engine using cclib, numpy, pandas, scipy, and scikit-learn.
Handles Gaussian .log/.out extraction, thermochemistry, frequency checks, 
conformer Boltzmann weighting, automated error diagnostics, and advanced batch cheminformatics analytics.
"""

import os
import numpy as np
import pandas as pd
import cclib
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Physical constants for unit conversions
HARTREE_TO_KCAL_MOL = 627.509474
HARTREE_TO_EV = 27.211386
KB = 0.0019872041  # Boltzmann constant in kcal/(mol·K)
DEFAULT_TEMP = 298.15  # Room temperature in Kelvin

# Common Gaussian error signatures and recommended fixes
ERROR_RULES = [
    {
        "pattern": "Convergence failure -- run terminated",
        "cause": "SCF electron density failed to converge within default maximum cycles.",
        "fix": "Add `SCF=(Xqc,MaxCycle=256)` to your route section to switch to quadratic convergence."
    },
    {
        "pattern": "Number of steps exceeded",
        "cause": "Geometry optimization reached the maximum step limit on a flat potential energy surface.",
        "fix": "Restart from the final geometry using `Opt=(CalcFC,MaxCycles=150)` or `Geom=AllCheck`."
    },
    {
        "pattern": "FormBX had a problem",
        "cause": "Near-linear bond angles (~180°) broke redundant internal coordinate calculations.",
        "fix": "Switch coordinate system by adding `Opt=Cartesian` or `Opt=GDIIS` to the route."
    },
    {
        "pattern": "Error termination via Lnk1e",
        "cause": "General unexpected runtime termination (check scratch disk space or memory limits %chk/%mem).",
        "fix": "Increase memory allocation (`%mem=8GB`) and check available disk space on the node."
    }
]


def diagnose_gaussian_error(file_path: str) -> dict:
    """Reads the tail of a failed Gaussian log file to diagnose root causes and suggest fixes."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        tail_text = "".join(lines[-40:])
        
        # Check if normal termination occurred
        if "Normal termination of Gaussian" in tail_text or any("Normal termination" in line for line in lines[-10:]):
            return {
                "status": "Success",
                "cause": None,
                "fix": None
            }
        
        # Scan for known error rules
        for rule in ERROR_RULES:
            if rule["pattern"] in tail_text:
                return {
                    "status": "Failed",
                    "cause": rule["cause"],
                    "fix": rule["fix"]
                }
                
        return {
            "status": "Failed",
            "cause": "Unknown error termination detected in log file footer.",
            "fix": "Inspect log file tail manually or check resource/memory limitations."
        }
    except Exception as e:
        return {
            "status": "Error",
            "cause": f"Could not read file: {str(e)}",
            "fix": "Ensure file path is valid and accessible."
        }


def parse_gaussian_log(file_path: str) -> dict:
    """
    Parses a single Gaussian output file using cclib and extracts 
    energies, HOMO/LUMO, dipole moments, and vibrational frequencies.
    """
    # 1. Check if the job crashed before trying to parse properties
    diag = diagnose_gaussian_error(file_path)
    if diag["status"] == "Failed":
        return {
            "file_name": os.path.basename(file_path),
            "status": "FAILED",
            "error_cause": diag["cause"],
            "suggested_fix": diag["fix"]
        }

    try:
        data = cclib.io.ccread(file_path)
        if data is None:
            return {"file_name": os.path.basename(file_path), "status": "ERROR", "message": "cclib failed to parse output."}

        # 2. Electronic Energy (Hartree)
        scf_energy_hartree = None
        if hasattr(data, "scfenergies") and len(data.scfenergies) > 0:
            scf_energy_hartree = float(data.scfenergies[-1]) / HARTREE_TO_EV

        # 3. Thermochemistry (ZPE, Enthalpy, Gibbs Free Energy)
        zpe = float(getattr(data, "zpve", 0.0))
        enthalpy = float(getattr(data, "enthalpy", 0.0))
        free_energy = float(getattr(data, "freeenergy", 0.0))

        # 4. HOMO / LUMO and Gap (eV)
        homo_energy = None
        lumo_energy = None
        gap_ev = None

        if hasattr(data, "homos") and hasattr(data, "moenergies"):
            homo_idx = data.homos[0]
            mo_energies = data.moenergies[0]
            if homo_idx < len(mo_energies) - 1:
                homo_energy = float(mo_energies[homo_idx])
                lumo_energy = float(mo_energies[homo_idx + 1])
                gap_ev = float(lumo_energy - homo_energy)

        # 5. Dipole Moment (Debye)
        total_dipole = None
        if hasattr(data, "moments") and len(data.moments) > 1:
            dipole_vector = data.moments[1]  # [x, y, z] dipole vector
            total_dipole = float(np.linalg.norm(dipole_vector))

        # 6. Vibrational Frequency Stability Check
        freq_status = "NOT CHECKED"
        imaginary_count = 0
        if hasattr(data, "vibfreqs") and data.vibfreqs is not None:
            freqs = data.vibfreqs
            imaginary_freqs = [f for f in freqs if f < 0]
            imaginary_count = len(imaginary_freqs)
            
            if imaginary_count == 0:
                freq_status = "PASS: True Local Minimum (0 imaginary frequencies)"
            elif imaginary_count == 1:
                freq_status = f"WARNING: Transition State / Saddle Point (1 imaginary freq: {imaginary_freqs[0]:.1f} cm^-1)"
            else:
                freq_status = f"FAIL: Higher-Order Saddle Point ({imaginary_count} imaginary freqs)"

        return {
            "file_name": os.path.basename(file_path),
            "status": "SUCCESS",
            "scf_energy_hartree": scf_energy_hartree,
            "zpe_hartree": zpe if zpe != 0.0 else None,
            "enthalpy_hartree": enthalpy if enthalpy != 0.0 else None,
            "free_energy_hartree": free_energy if free_energy != 0.0 else None,
            "homo_ev": homo_energy,
            "lumo_ev": lumo_energy,
            "homo_lumo_gap_ev": gap_ev,
            "dipole_moment_debye": total_dipole,
            "frequency_status": freq_status,
            "imaginary_freq_count": imaginary_count
        }

    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "status": "PARSER_ERROR",
            "message": str(e)
        }


def rank_conformers(file_paths: list, temperature: float = DEFAULT_TEMP) -> pd.DataFrame:
    """
    Parses a batch of conformer log files, computes relative Gibbs free energies (ΔG),
    Boltzmann populations at given temperature, and returns a sorted pandas DataFrame.
    """
    results = []
    for path in file_paths:
        res = parse_gaussian_log(path)
        if res.get("status") == "SUCCESS" and res.get("free_energy_hartree") is not None:
            results.append(res)
        else:
            results.append({
                **res,
                "free_energy_hartree": float('inf')
            })

    df = pd.DataFrame(results)
    valid_df = df[df["free_energy_hartree"] != float('inf')].copy()
    
    if valid_df.empty:
        return df

    # Find global minimum Gibbs Free Energy
    g_min = valid_df["free_energy_hartree"].min()

    # Calculate ΔG in kcal/mol: ΔG = (G_i - G_min) * 627.509474
    valid_df["delta_g_kcal_mol"] = (valid_df["free_energy_hartree"] - g_min) * HARTREE_TO_KCAL_MOL

    # Calculate Boltzmann factor: exp(-ΔG / RT)
    rt = KB * temperature  # RT in kcal/mol
    valid_df["boltzmann_factor"] = np.exp(-valid_df["delta_g_kcal_mol"] / rt)

    # Calculate percentage population weight
    total_bf = valid_df["boltzmann_factor"].sum()
    valid_df["population_percent"] = (valid_df["boltzmann_factor"] / total_bf) * 100.0

    # Sort ascending by ΔG (most stable conformation first)
    valid_df = valid_df.sort_values(by="delta_g_kcal_mol", ascending=True).reset_index(drop=True)

    return valid_df


def run_batch_cheminformatics_report(file_paths: list, n_clusters: int = 2) -> dict:
    """
    Executes advanced batch cheminformatics analytics across a list of Gaussian log files:
    1. Outlier Detection (Z-Score threshold > 2.5)
    2. Feature Correlation Matrix
    3. Principal Component Analysis (PCA) for structural variance reduction
    4. K-Means Unsupervised Clustering (Grouping electronic families)
    """
    results = [parse_gaussian_log(path) for path in file_paths]
    df = pd.DataFrame(results)
    
    successful_df = df[df["status"] == "SUCCESS"].copy()
    failed_df = df[df["status"] != "SUCCESS"].copy()

    # Dynamic minimum cluster check based on available successful logs
    min_required = max(2, n_clusters)
    if len(successful_df) < min_required:
        return {
            "error": f"Insufficient successful log files ({len(successful_df)}) for advanced statistical analysis. Minimum required: {min_required}.",
            "failed_runs_summary": failed_df[["file_name", "status"]].to_dict(orient="records") if not failed_df.empty else []
        }

    # Descriptors used for machine learning and statistical modeling
    feature_cols = ["homo_lumo_gap_ev", "dipole_moment_debye", "free_energy_hartree", "scf_energy_hartree"]
    
    # Filter out columns that might be entirely NaN or missing
    active_features = [col for col in feature_cols if col in successful_df.columns and successful_df[col].notna().sum() > 0]
    
    if len(active_features) < 2:
        return {"error": "Not enough valid numeric features extracted across logs to run multi-variable analytics."}

    X = successful_df[active_features].copy()
    for col in active_features:
        X[col] = X[col].fillna(X[col].mean())

    # Standardize data (mean=0, variance=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Outlier Detection (Z-Score > 2.5 with safety for zero-variance)
    z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    outlier_mask = (z_scores > 2.5).any(axis=1) if len(X) > 2 else np.zeros(len(X), dtype=bool)
    successful_df["is_outlier"] = outlier_mask
    outliers = successful_df[outlier_mask]["file_name"].tolist()

    # 2. Correlation Matrix
    corr_matrix = successful_df[active_features].corr().to_dict()

    # 3. Principal Component Analysis (PCA - max 2 components or number of features)
    n_components = min(2, len(active_features))
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(X_scaled)
    successful_df["pca_pc1"] = pca_coords[:, 0]
    successful_df["pca_pc2"] = pca_coords[:, 1] if n_components > 1 else 0.0
    explained_variance = pca.explained_variance_ratio_.tolist()

    # 4. K-Means Clustering (adjust clusters if batch size is small)
    actual_clusters = min(n_clusters, len(successful_df))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    successful_df["cluster_id"] = kmeans.fit_predict(X_scaled)

    return {
        "batch_summary": {
            "total_files_evaluated": len(file_paths),
            "successful_jobs": len(successful_df),
            "failed_jobs": len(failed_df)
        },
        "outlier_analysis": {
            "flagged_outliers": outliers,
            "detection_criteria": "Z-score magnitude > 2.5 across quantum descriptors."
        },
        "correlation_matrix": corr_matrix,
        "pca_analysis": {
            "explained_variance_ratio": {
                "PC1": round(explained_variance[0], 4),
                "PC2": round(explained_variance[1], 4) if len(explained_variance) > 1 else 0.0
            },
            "coordinates": successful_df[["file_name", "pca_pc1", "pca_pc2", "cluster_id"]].to_dict(orient="records")
        },
        "clustering_analysis": {
            "clusters": successful_df[["file_name", "cluster_id", "homo_lumo_gap_ev", "dipole_moment_debye"]].to_dict(orient="records")
        }
    }