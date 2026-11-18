"""
app.py
Streamlit web dashboard for the DFT-Insight Agent.
Supports both single-file inspection and multi-file batch uploads for 
conformer ranking, comparisons, and machine learning analytics.
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
        "Batch Conformer Ranking & Comparison", 
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
# MODE 2: BATCH CONFORMER RANKING & COMPARISON
# ---------------------------------------------------------
elif app_mode == "Batch Conformer Ranking & Comparison":
    st.header("📊 Batch Conformer Energy Ranking & Boltzmann Analysis")
    st.markdown("Upload multiple Gaussian log files simultaneously to rank conformers and calculate Boltzmann weights.")
    
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
            
        st.info(f"📁 Successfully loaded {len(temp_paths)} file(s) for ranking.")
        
        # Run batch ranking
        df_ranked = rank_conformers(temp_paths)
        
        st.markdown("### Ranked Conformer Comparison Table")
        st.dataframe(df_ranked, use_container_width=True)
        
        st.markdown("### 📈 Interactive Visual Insights")
        valid_df = df_ranked[df_ranked["free_energy_hartree"] != float('inf')].copy()
        
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
                    title="Room Temperature Equilibrium Distribution"
                )
                st.plotly_chart(fig_pop, use_container_width=True)
        else:
            st.warning("Thermochemistry data (Gibbs Free Energy) required in log files to generate energy graphs.")

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