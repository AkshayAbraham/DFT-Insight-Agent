"""
app.py
Streamlit web dashboard for the DFT-Insight Agent.
Supports single-file inspection, batch conformer ranking with Boltzmann ensembles,
energy window filtering, and advanced machine learning analytics.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from parser_core import parse_gaussian_log, rank_conformers, run_batch_cheminformatics_report

# Page configuration
st.set_page_config(
    page_title="DFT-Insight Agent",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 DFT-Insight Agent")
st.markdown("### AI-Powered Gaussian Output Analysis & Cheminformatics Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation & Options")
app_mode = st.sidebar.selectbox(
    "Choose Analysis Mode", 
    [
        "Single File Inspector", 
        "Batch Conformer Ranking & Boltzmann Analysis", 
        "Batch Cheminformatics & ML Insights"
    ]
)

# Ensure sample directory exists
os.makedirs("sample_logs", exist_ok=True)

# ---------------------------------------------------------
# MODE 1: SINGLE FILE INSPECTOR
# ---------------------------------------------------------
if app_mode == "Single File Inspector":
    st.header("📄 Single Simulation File Inspector")
    
    uploaded_file = st.file_uploader("Upload a single Gaussian .log / .out file", type=["log", "out"], key="single_uploader")
    
    if uploaded_file is not None:
        temp_path = os.path.join("sample_logs", uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        res = parse_gaussian_log(temp_path)
        
        if res["status"] == "FAILED":
            st.error(f"❌ Calculation Failed in {res['file_name']}")
            st.warning(f"**Root Cause:** {res['error_cause']}")
            st.info(f"💡 **Suggested Fix:** `{res['suggested_fix']}`")
        elif res["status"] == "SUCCESS":
            st.success(f"✅ Successfully parsed {res['file_name']}")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("SCF Energy", f"{res['scf_energy_hartree']:.4f} Ha" if res['scf_energy_hartree'] else "N/A")
            col2.metric("HOMO-LUMO Gap", f"{res['homo_lumo_gap_ev']:.2f} eV" if res['homo_lumo_gap_ev'] else "N/A")
            col3.metric("Dipole Moment", f"{res['dipole_moment_debye']:.2f} Debye" if res['dipole_moment_debye'] else "N/A")
            col4.metric("Imaginary Frequencies", res['imaginary_freq_count'])
            
            st.markdown("### Stability & Geometry Health")
            if "PASS" in res["frequency_status"]:
                st.success(res["frequency_status"])
            elif "WARNING" in res["frequency_status"]:
                st.warning(res["frequency_status"])
            else:
                st.info(res["frequency_status"])
                
            st.markdown("### Extracted Properties Breakdown")
            df_res = pd.DataFrame([res]).T.reset_index()
            df_res.columns = ["Property", "Value"]
            st.dataframe(df_res, use_container_width=True)

# ---------------------------------------------------------
# MODE 2: BATCH CONFORMER RANKING & BOLTZMANN ANALYSIS
# ---------------------------------------------------------
# Inside MODE 2 of app.py:

elif app_mode == "Batch Conformer Ranking & Boltzmann Analysis":
    st.header("📊 Batch Conformer Energy Ranking & Boltzmann Ensemble Analysis")
    st.markdown("Upload multiple Gaussian log files to filter by energy window, rank conformers, and compute Boltzmann population-weighted averages.")
    
    # Sidebar controls for Boltzmann analysis
    st.sidebar.markdown("---")
    st.sidebar.subheader("Boltzmann & Ensemble Settings")
    temperature = st.sidebar.number_input("Temperature (K)", min_value=100.0, max_value=500.0, value=298.15, step=5.0)
    energy_window = st.sidebar.slider("Energy Window (kcal/mol)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    
    uploaded_files = st.file_uploader(
        "Upload multiple Gaussian .log files", 
        type=["log", "out"], 
        accept_multiple_files=True, 
        key="conformer_batch_uploader"
    )
    
    if uploaded_files:
        temp_paths = []
        for uf in uploaded_files:
            t_path = os.path.join("sample_logs", uf.name)
            with open(t_path, "wb") as f:
                f.write(uf.getbuffer())
            temp_paths.append(t_path)
            
        st.info(f"📁 Successfully loaded {len(temp_paths)} file(s) for ensemble analysis.")
        
        # --- FEATURE #5: Scientific Sensitivity & DFT Warning Banner ---
        st.warning(
            "⚠️ **Scientific Caveat & DFT Sensitivity Warning:** "
            "At 298.15 K, a $\Delta G$ difference of **1.36 kcal/mol** translates to approximately a **10-fold shift** in population ratio. "
            "Standard Density Functional Theory (DFT) methods have intrinsic error bars of ~1 kcal/mol. Treat population percentages as qualitative thermodynamic trends rather than absolute ground truths."
        )
        
        # Optional Degeneracy Inputs
        with st.expander("⚙️ Advanced: Conformer Degeneracy / Symmetry Multiplicities"):
            st.markdown("Specify symmetry numbers or chiral multiplicities (e.g., enantiomeric pairs = 2) for specific log files:")
            deg_map = {}
            for uf in uploaded_files:
                deg_map[uf.name] = st.number_input(f"Multiplicity for {uf.name}", min_value=1, max_value=10, value=1, key=f"deg_{uf.name}")

        # Run batch ranking and temperature scan
        from boltzmann_core import process_conformer_ensemble, calculate_temperature_scan
        
        raw_results = [parse_gaussian_log(p) for p in temp_paths]
        df_ranked, ensemble_averages = process_conformer_ensemble(raw_results, temperature=temperature, energy_window_kcal=energy_window, degeneracy_map=deg_map)
        
        # --- DISPLAY ENSEMBLE METRICS ---
        st.markdown("### 🌡️ Population-Weighted Ensemble Averages ($\langle P \rangle$)")
        st.caption(f"Calculated at {temperature} K across conformers within {energy_window} kcal/mol of the global minimum.")
        
        e_col1, e_col2, e_col3 = st.columns(3)
        avg_gap = ensemble_averages.get("homo_lumo_gap_ev")
        avg_dipole = ensemble_averages.get("dipole_moment_debye")
        avg_scf = ensemble_averages.get("scf_energy_hartree")
        
        e_col1.metric("⟨HOMO-LUMO Gap⟩", f"{avg_gap:.2f} eV" if avg_gap else "N/A")
        e_col2.metric("⟨Dipole Moment⟩", f"{avg_dipole:.2f} Debye" if avg_dipole else "N/A")
        e_col3.metric("⟨SCF Energy⟩", f"{avg_scf:.4f} Ha" if avg_scf else "N/A")
        
        st.markdown("### 📋 Filtered & Ranked Conformer Table")
        st.dataframe(df_ranked, use_container_width=True)
        
        st.markdown("### 📈 Interactive Visual Insights")
        valid_df = df_ranked[df_ranked["delta_g_kcal_mol"] != float('inf')].copy()
        
        if not valid_df.empty:
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.markdown("#### Relative Free Energy ($\Delta G$ in kcal/mol)")
                fig_dg = px.bar(
                    valid_df, 
                    x="file_name", 
                    y="delta_g_kcal_mol",
                    text="delta_g_kcal_mol",
                    color="delta_g_kcal_mol",
                    color_continuous_scale="Viridis",
                    labels={"file_name": "Conformer", "delta_g_kcal_mol": "ΔG (kcal/mol)"}
                )
                fig_dg.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                st.plotly_chart(fig_dg, use_container_width=True)
                
            with col_g2:
                st.markdown("#### Boltzmann Population Weight (%)")
                fig_pop = px.pie(
                    valid_df, 
                    names="file_name", 
                    values="population_percent",
                    hole=0.4,
                    title=f"Equilibrium Distribution at {temperature} K"
                )
                st.plotly_chart(fig_pop, use_container_width=True)
                
            # --- FEATURE #2: Temperature Scan Plotly Line Chart ---
            st.markdown("### 🌡️ Conformer Population Shift Across Temperature (200K – 450K)")
            st.markdown("Observe how conformational equilibrium shifts under variable thermal energy (useful for variable-temperature NMR studies).")
            
            df_scan = calculate_temperature_scan(raw_results, temp_range=range(200, 451, 10))
            if not df_scan.empty:
                fig_scan = px.line(
                    df_scan,
                    x="Temperature_K",
                    y="population_percent",
                    color="file_name",
                    markers=True,
                    labels={"Temperature_K": "Temperature (K)", "population_percent": "Population Weight (%)", "file_name": "Conformer"},
                    title="Population Equilibrium vs. Temperature Scan"
                )
                st.plotly_chart(fig_scan, use_container_width=True)
        else:
            st.warning("Thermochemistry data required to generate energy graphs and temperature scans.")

# ---------------------------------------------------------
# MODE 3: BATCH CHEMINFORMATICS & ML INSIGHTS
# ---------------------------------------------------------
elif app_mode == "Batch Cheminformatics & ML Insights":
    st.header("🧪 Advanced Cheminformatics & Machine Learning Batch Report")
    st.markdown("Upload a batch of structural logs or analog series to run PCA, K-Means Clustering, Correlation Matrices, and Outlier Detection.")
    
    uploaded_ml_files = st.file_uploader(
        "Upload batch log files for ML analysis", 
        type=["log", "out"], 
        accept_multiple_files=True, 
        key="ml_batch_uploader"
    )
    
    if uploaded_ml_files:
        temp_paths = []
        for uf in uploaded_ml_files:
            t_path = os.path.join("sample_logs", uf.name)
            with open(t_path, "wb") as f:
                f.write(uf.getbuffer())
            temp_paths.append(t_path)
            
        n_clusters = st.sidebar.slider("Select K-Means Clusters", min_value=2, max_value=5, value=2)
        
        # Run advanced ML report
        report = run_batch_cheminformatics_report(temp_paths, n_clusters=n_clusters)
        
        if "error" in report:
            st.error(report["error"])
        else:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Files Evaluated", report["batch_summary"]["total_files_evaluated"])
            col_m2.metric("Successful Jobs", report["batch_summary"]["successful_jobs"])
            col_m3.metric("Flagged Outliers", len(report["outlier_analysis"]["flagged_outliers"]))
            
            if report["outlier_analysis"]["flagged_outliers"]:
                st.warning(f"⚠️ Outliers detected: {', '.join(report['outlier_analysis']['flagged_outliers'])}")
            else:
                st.success("✨ No statistical outliers detected in this batch.")
                
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.markdown("#### 📉 PCA Electronic Variance Scatter Plot")
                pca_data = report["pca_analysis"]["coordinates"]
                if pca_data:
                    df_pca = pd.DataFrame(pca_data)
                    evr = report["pca_analysis"]["explained_variance_ratio"]
                    fig_pca = px.scatter(
                        df_pca,
                        x="pca_pc1",
                        y="pca_pc2",
                        color="cluster_id",
                        text="file_name",
                        labels={
                            "pca_pc1": f"PC1 ({evr['PC1']*100:.1f}%)", 
                            "pca_pc2": f"PC2 ({evr['PC2']*100:.1f}%)",
                            "cluster_id": "Cluster"
                        },
                        title="Clustered Chemical Space (PCA)"
                    )
                    fig_pca.update_traces(textposition='top center')
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
            with col_v2:
                st.markdown("#### 🔬 Feature Correlation Matrix Heatmap")
                corr_dict = report["correlation_matrix"]
                if corr_dict:
                    df_corr = pd.DataFrame(corr_dict)
                    fig_corr = px.imshow(
                        df_corr,
                        text_auto=True,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1,
                        title="Quantum Property Correlations"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
            st.markdown("### 📋 Unsupervised Clustering Assignment Table")
            df_clusters = pd.DataFrame(report["clustering_analysis"]["clusters"])
            st.dataframe(df_clusters, use_container_width=True)