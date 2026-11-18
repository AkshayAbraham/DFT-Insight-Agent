"""
test_run.py
Scans sample_logs, processes every available log file regardless of count,
prints individual details (including TD-DFT photophysics), generates a unified summary table, 
and executes the advanced cheminformatics machine learning report.
"""

import os
import json
from parser_core import parse_gaussian_log, rank_conformers, run_batch_cheminformatics_report

def run_tests():
    target_dir = "sample_logs"
    
    if not os.path.exists(target_dir):
        print(f"❌ Directory '{target_dir}' not found. Please create it and add your log files.")
        return

    # Gather all .log or .out files inside sample_logs
    file_paths = [
        os.path.join(target_dir, f) 
        for f in os.listdir(target_dir) 
        if f.endswith((".log", ".out"))
    ]

    if not file_paths:
        print(f"❌ No Gaussian log files found in '{target_dir}'.")
        return

    print(f"📁 Found {len(file_paths)} log file(s) in '{target_dir}'. Running analysis on all...\n")

    # 1. Print detailed property breakdown for every file (including TD-DFT check)
    for path in file_paths:
        print("=" * 60)
        print(f"📄 Detailed Breakdown for: {os.path.basename(path)}")
        print("=" * 60)
        result = parse_gaussian_log(path)
        for key, value in result.items():
            # Keep transitions breakdown compact for terminal readability if present
            if key == "transitions_breakdown" and value:
                print(f"  {key}: [{len(value)} excited states computed]")
            else:
                print(f"  {key}: {value}")
        print("\n")

    # 2. Generate unified summary & ranking table for all files
    print("=" * 60)
    print("📊 Unified Summary & Conformer Ranking Table:")
    print("=" * 60)
    df_ranked, ensemble_averages = rank_conformers(file_paths)
    
    cols_to_show = ["file_name", "status", "scf_energy_hartree", "delta_g_kcal_mol", "population_percent", "homo_lumo_gap_ev", "frequency_status", "lambda_max_nm"]
    available_cols = [c for c in cols_to_show if c in df_ranked.columns]
    
    print(df_ranked[available_cols].to_string(index=False))
    print("\n")

    # 3. Run and print the Advanced Cheminformatics Batch Report (PCA, Clustering, Outliers, Correlation)
    print("=" * 60)
    print("🧪 Advanced Cheminformatics Batch Report (ML & Statistics):")
    print("=" * 60)
    report = run_batch_cheminformatics_report(file_paths, n_clusters=2)
    print(json.dumps(report, indent=4))
    
    print("\n✅ All files and advanced analytics processed successfully!")

if __name__ == "__main__":
    run_tests()