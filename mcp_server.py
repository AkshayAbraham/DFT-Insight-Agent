"""
mcp_server.py
Exposes the DFT-Insight parser core functions and advanced cheminformatics 
analytics as Model Context Protocol (MCP) tools for Claude Desktop.
"""

import os
from mcp.server.fastmcp import FastMCP
from parser_core import parse_gaussian_log, rank_conformers, run_batch_cheminformatics_report

# Initialize the FastMCP server
mcp = FastMCP("DFT-Insight Agent")


@mcp.tool()
def analyze_gaussian_file(file_path: str) -> dict:
    """
    Parses a single Gaussian .log or .out output file. Extracts SCF energy,
    ZPE, Enthalpy, Gibbs Free Energy, HOMO/LUMO gap, dipole moment, 
    vibrational stability, and diagnoses failure reasons if the job crashed.
    
    Args:
        file_path: Absolute or relative path to the Gaussian .log file.
    """
    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}
    return parse_gaussian_log(file_path)


@mcp.tool()
def analyze_conformer_folder(folder_path: str = "sample_logs", temperature: float = 298.15) -> list:
    """
    Scans a folder containing multiple Gaussian log files, 
    ranks them by relative Gibbs Free Energy (ΔG in kcal/mol), and calculates 
    Boltzmann population weights at room temperature.
    
    Args:
        folder_path: Path to the directory containing Gaussian log files (defaults to 'sample_logs').
        temperature: Temperature in Kelvin (default 298.15).
    """
    if not os.path.exists(folder_path):
        return [{"error": f"Folder not found: {folder_path}"}]
    
    files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.endswith((".log", ".out"))
    ]
    
    if not files:
        return [{"error": f"No .log or .out files found in {folder_path}"}]
    
    df = rank_conformers(files, temperature=temperature)
    return df.to_dict(orient="records")


@mcp.tool()
def analyze_batch_cheminformatics(folder_path: str = "sample_logs", n_clusters: int = 2) -> dict:
    """
    Runs advanced batch cheminformatics analytics across a folder of Gaussian log files:
    Outlier detection (Z-score), feature correlation matrix, Principal Component 
    Analysis (PCA), and K-Means unsupervised clustering.
    
    Args:
        folder_path: Path to the directory containing Gaussian log files.
        n_clusters: Number of K-Means clusters to group molecules into (default 2).
    """
    if not os.path.exists(folder_path):
        return {"error": f"Folder not found: {folder_path}"}
    
    files = [
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.endswith((".log", ".out"))
    ]
    
    if not files:
        return {"error": f"No .log or .out files found in {folder_path}"}
    
    return run_batch_cheminformatics_report(files, n_clusters=n_clusters)


if __name__ == "__main__":
    mcp.run(transport='stdio')