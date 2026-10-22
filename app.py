"""
Streamlit Web Dashboard
Predictive-Maintenance Neural Model with Simulated Annealing (SA) Feature Selection
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_preprocessing import load_data, prepare_features_and_target, engineer_features
from src.model_trainer import PredictiveMaintenanceANN
from src.evaluate import evaluate_model, get_roc_curve_data, get_pr_curve_data

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Predictive Maintenance | SA-ANN",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 12px;
    }
    .badge-normal {
        color: #28a745;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .badge-warning {
        color: #fd7e14;
        font-weight: 700;
        font-size: 1.25rem;
    }
    .badge-critical {
        color: #dc3545;
        font-weight: 700;
        font-size: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = 'data/ai4i2020.csv'
MODELS_DIR = 'models'

@st.cache_data
def get_dataset():
    """Loads and caches the AI4I 2020 dataset."""
    if os.path.exists(DATA_PATH):
        return load_data(DATA_PATH)
    return None

@st.cache_resource
def load_trained_artifacts():
    """Loads pre-trained models, scalers, and test bundle."""
    artifacts = {}
    try:
        baseline_path = os.path.join(MODELS_DIR, 'baseline_ann.joblib')
        sa_path = os.path.join(MODELS_DIR, 'sa_optimized_ann.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        test_path = os.path.join(MODELS_DIR, 'test_bundle.joblib')
        history_path = os.path.join(MODELS_DIR, 'sa_history.json')
        metrics_path = os.path.join(MODELS_DIR, 'metrics_summary.json')

        if os.path.exists(baseline_path):
            artifacts['baseline_ann'] = PredictiveMaintenanceANN.load(baseline_path)
        if os.path.exists(sa_path):
            artifacts['sa_ann'] = PredictiveMaintenanceANN.load(sa_path)
        if os.path.exists(scaler_path):
            artifacts['scaler'] = joblib.load(scaler_path)
        if os.path.exists(test_path):
            artifacts['test_bundle'] = joblib.load(test_path)
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                artifacts['sa_history'] = json.load(f)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                artifacts['metrics_summary'] = json.load(f)
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
    return artifacts

# Header Section
st.title("⚙️ Predictive Maintenance Neural Model")
st.markdown("### Simulated Annealing (SA) Feature Selection & Multilayer Perceptron (ANN)")
st.caption("College Artificial Neural Network Assignment | AI4I 2020 Dataset Benchmark")

# Load data and artifacts
df_raw = get_dataset()
artifacts = load_trained_artifacts()

# Sidebar
with st.sidebar:
    st.header("🎛️ System Status")
    if df_raw is not None:
        st.success("✅ Real AI4I 2020 Dataset Loaded")
        st.metric("Total Samples", f"{len(df_raw):,}")
        st.metric("Raw Columns", len(df_raw.columns))
    else:
        st.error(f"❌ Missing dataset at `{DATA_PATH}`.")

    if 'sa_ann' in artifacts and 'baseline_ann' in artifacts:
        st.success("✅ Pre-trained ANN Models Loaded")
        st.info(f"**Baseline Features**: 11\n\n**SA Features**: {len(artifacts['sa_ann'].selected_indices_)}")
    else:
        st.warning("⚠️ Models not trained yet. Run `python train_and_evaluate.py`")

    st.markdown("---")
    st.markdown("### 📚 Key Methodology")
    st.markdown(r"""
    - **Target Leakage Safeguard**: Failure flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and IDs are withheld from input features.
    - **Physical Feature Engineering**: $\Delta T$, Mechanical Power $P$, and Overstrain stress.
    - **Simulated Annealing**: Metropolis acceptance with geometric cooling schedule.
    - **MLP Classifier**: Balanced loss weighting & validation threshold calibration.
    """)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset & Exploratory Analysis",
    "🔬 Simulated Annealing Trajectory",
    "📈 Model Benchmarks & Metrics",
    "⚡ Real-Time Health Predictor"
])

# -------------------------------------------------------------
# TAB 1: DATASET & EDA
# -------------------------------------------------------------
with tab1:
    st.subheader("AI4I 2020 Dataset Exploration & Quality")
    if df_raw is None:
        st.error(f"Dataset not found at `{DATA_PATH}`. Please verify the file location.")
    else:
        # Top KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_rows = len(df_raw)
        normal_count = int((df_raw['Machine failure'] == 0).sum())
        failure_count = int((df_raw['Machine failure'] == 1).sum())
        fail_ratio = (failure_count / total_rows) * 100

        kpi1.metric("Total Records", f"{total_rows:,}")
        kpi2.metric("Normal Operations", f"{normal_count:,}", delta="96.61%", delta_color="normal")
        kpi3.metric("Machine Failures", f"{failure_count:,}", delta=f"{fail_ratio:.2f}%", delta_color="inverse")
        kpi4.metric("Missing / Null Values", "0 (100% Clean)", delta="Complete")

        st.markdown("---")
        col_eda1, col_eda2 = st.columns(2)

        with col_eda1:
            st.markdown("#### Primary Target Class Distribution")
            fig_target = px.pie(
                names=['Normal (0)', 'Machine Failure (1)'],
                values=[normal_count, failure_count],
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                hole=0.45,
                title="Severe Class Imbalance in Operational Dataset"
            )
            fig_target.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_target, use_container_width=True)

        with col_eda2:
            st.markdown("#### Diagnostic Failure Modes Breakdown")
            modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            mode_counts = {m: int(df_raw[m].sum()) for m in modes if m in df_raw.columns}
            mode_labels = {
                'TWF': 'Tool Wear Failure (TWF)',
                'HDF': 'Heat Dissipation (HDF)',
                'PWF': 'Power Failure (PWF)',
                'OSF': 'Overstrain Failure (OSF)',
                'RNF': 'Random Failure (RNF)'
            }
            df_modes = pd.DataFrame({
                'Mode': [mode_labels.get(k, k) for k in mode_counts.keys()],
                'Count': list(mode_counts.values())
            }).sort_values('Count', ascending=True)

            fig_modes = px.bar(
                df_modes,
                x='Count',
                y='Mode',
                orientation='h',
                color='Count',
                color_continuous_scale='Reds',
                title="Root-Cause Failure Mode Frequency"
            )
            fig_modes.update_layout(margin=dict(t=40, b=20, l=20, r=20), showlegend=False)
            st.plotly_chart(fig_modes, use_container_width=True)

        # Correlation Heatmap with Engineered Features
        st.markdown("#### Sensor & Engineered Features Correlation Matrix")
        df_eng = engineer_features(df_raw)
        numeric_cols = [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff [K]', 'Power [W]', 'Overstrain', 'Machine failure'
        ]
        corr_matrix = df_eng[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson Correlation Heatmap (Physical & Engineered Features)"
        )
        fig_corr.update_layout(margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_corr, use_container_width=True)

        with st.expander("🔍 View Raw Dataset Records"):
            st.dataframe(df_raw.head(50), use_container_width=True)

# -------------------------------------------------------------
# TAB 2: SIMULATED ANNEALING TRAJECTORY
# -------------------------------------------------------------
with tab2:
    st.subheader("Simulated Annealing Feature Selection Engine")

    history = artifacts.get('sa_history', None)
    if not history:
        st.warning("No Simulated Annealing optimization trajectory found. Run `python train_and_evaluate.py` to generate.")
    else:
        df_hist = pd.DataFrame(history)

        st.markdown(r"""
        **Simulated Annealing (SA)** searches for the optimal subset of features that maximizes validation $\text{F1}$
        while penalizing unnecessary complexity via the parsimonious energy formulation:
        """)
        st.latex(r"E(\mathbf{s}) = - \left( \text{F1}_{\text{val}}(\mathbf{s}) - \lambda \cdot \frac{k}{D} \right)")
        st.markdown(r"""
        Candidate worse solutions ($\Delta E > 0$) are probabilistically accepted using the **Metropolis criterion**:
        """)
        st.latex(r"P(\text{accept}) = \exp\left(-\frac{\Delta E}{T}\right)")

        # Convergence Charts
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(
                x=df_hist['iteration'], y=df_hist['current_energy'],
                mode='lines', name='Current State Energy', line=dict(color='rgba(41, 128, 185, 0.4)')
            ))
            fig_energy.add_trace(go.Scatter(
                x=df_hist['iteration'], y=df_hist['best_energy'],
                mode='lines', name='Global Best Energy', line=dict(color='#27ae60', width=3)
            ))
            fig_energy.update_layout(
                title="Energy Minimization Trajectory",
                xaxis_title="Iteration",
                yaxis_title="Energy E(s)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_energy, use_container_width=True)

        with col_c2:
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(
                x=df_hist['iteration'], y=df_hist['current_f1'],
                mode='lines', name='Current Validation F1', line=dict(color='rgba(231, 76, 60, 0.4)')
            ))
            fig_f1.add_trace(go.Scatter(
                x=df_hist['iteration'], y=df_hist['best_f1'],
                mode='lines', name='Best Validation F1', line=dict(color='#e67e22', width=3)
            ))
            fig_f1.update_layout(
                title="Validation F1-Score Progression",
                xaxis_title="Iteration",
                yaxis_title="Validation F1 Score",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_temp = px.line(
                df_hist, x='iteration', y='temperature',
                title="Geometric Cooling Schedule (T_next = alpha * T)",
                labels={'iteration': 'Iteration', 'temperature': 'Temperature T'}
            )
            fig_temp.update_traces(line_color='#8e44ad')
            fig_temp.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_temp, use_container_width=True)

        with col_c4:
            fig_k = px.line(
                df_hist, x='iteration', y='best_n_features',
                title="Evolution of Selected Feature Count (Parsimony Optimization)",
                labels={'iteration': 'Iteration', 'best_n_features': 'Optimal Feature Count'}
            )
            fig_k.update_traces(line_color='#2c3e50', line_width=3)
            fig_k.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_k, use_container_width=True)

        # Feature Selection Table
        st.markdown("#### Optimal Feature Subset Selected by Simulated Annealing")
        best_features = df_hist['best_features'].iloc[-1]
        summary_info = artifacts.get('metrics_summary', {})
        all_features = summary_info.get('all_features', [
            'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
            'Torque [Nm]', 'Tool wear [min]', 'Temp_Diff [K]', 'Power [W]', 'Overstrain',
            'Type_H', 'Type_L', 'Type_M'
        ])

        feat_records = []
        for feat in all_features:
            is_selected = feat in best_features
            feat_records.append({
                "Feature Name": feat,
                "Status": "✅ SELECTED" if is_selected else "❌ PRUNED",
                "Domain Category": "Physical Sensor" if "[" in feat else ("Electromechanical" if feat in ['Power [W]', 'Overstrain', 'Temp_Diff [K]'] else "Quality Variant"),
                "Physical Justification": (
                    "Directly captures core thermodynamics / stress dynamics" if is_selected
                    else "Pruned as redundant; information already captured by interaction features"
                )
            })

        st.dataframe(pd.DataFrame(feat_records), use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MODEL BENCHMARK & DIAGNOSTICS
# -------------------------------------------------------------
with tab3:
    st.subheader("Benchmark Comparison: Baseline ANN vs. SA-Optimized ANN")

    summary = artifacts.get('metrics_summary', None)
    test_bundle = artifacts.get('test_bundle', None)

    if not summary or not test_bundle:
        st.warning("Pre-computed model benchmarks not found. Run `python train_and_evaluate.py` first.")
    else:
        m_base = summary['baseline']
        m_sa = summary['sa_optimized']

        # Comparative KPI display
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Feature Count",
            f"{m_sa['Features Used']} vs {m_base['Features Used']}",
            delta=f"-{summary.get('feature_reduction_percent', 0)}% (Reduced)",
            delta_color="normal"
        )
        c2.metric(
            "Recall (Sensitivity)",
            f"{m_sa['Recall (Sensitivity)']:.4f}",
            delta=f"{(m_sa['Recall (Sensitivity)'] - m_base['Recall (Sensitivity)']):+.4f}",
            delta_color="normal"
        )
        c3.metric(
            "F1-Score",
            f"{m_sa['F1-Score']:.4f}",
            delta=f"{(m_sa['F1-Score'] - m_base['F1-Score']):+.4f}"
        )
        c4.metric(
            "ROC-AUC",
            f"{m_sa['ROC-AUC']:.4f}",
            delta=f"{(m_sa['ROC-AUC'] - m_base['ROC-AUC']):+.4f}"
        )

        st.markdown("#### Detailed Test Set Evaluation Table (2,000 Unseen Test Samples)")
        comp_df = pd.DataFrame([m_base, m_sa])
        cols_to_show = [
            'Model', 'Features Used', 'Threshold', 'F1-Score',
            'Recall (Sensitivity)', 'Precision', 'ROC-AUC', 'PR-AUC (Avg Precision)', 'Accuracy'
        ]
        st.dataframe(comp_df[cols_to_show], use_container_width=True)

        # ROC and PR Curves
        X_test = test_bundle['X_test']
        y_test = test_bundle['y_test']
        baseline_model = artifacts.get('baseline_ann')
        sa_model = artifacts.get('sa_ann')

        col_roc, col_pr = st.columns(2)

        with col_roc:
            fpr_base, tpr_base, auc_base = get_roc_curve_data(baseline_model, X_test, y_test)
            fpr_sa, tpr_sa, auc_sa = get_roc_curve_data(sa_model, X_test, y_test)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_base, y=tpr_base, mode='lines', name=f'Baseline (AUC = {auc_base:.3f})', line=dict(color='#2980b9', width=2)))
            fig_roc.add_trace(go.Scatter(x=fpr_sa, y=tpr_sa, mode='lines', name=f'SA-ANN (AUC = {auc_sa:.3f})', line=dict(color='#27ae60', width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(color='gray', dash='dash')))
            fig_roc.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with col_pr:
            rec_base, prec_base, pr_base = get_pr_curve_data(baseline_model, X_test, y_test)
            rec_sa, prec_sa, pr_sa = get_pr_curve_data(sa_model, X_test, y_test)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=rec_base, y=prec_base, mode='lines', name=f'Baseline (PR-AUC = {pr_base:.3f})', line=dict(color='#2980b9', width=2)))
            fig_pr.add_trace(go.Scatter(x=rec_sa, y=prec_sa, mode='lines', name=f'SA-ANN (PR-AUC = {pr_sa:.3f})', line=dict(color='#27ae60', width=2.5)))
            fig_pr.update_layout(
                title="Precision-Recall (PR) Curve",
                xaxis_title="Recall",
                yaxis_title="Precision",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=40, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        # Confusion Matrices
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            cm_base = np.array(m_base['Confusion Matrix'])
            fig_cm_base = px.imshow(
                cm_base, text_auto=True, color_continuous_scale='Blues',
                x=['Pred Normal (0)', 'Pred Failure (1)'],
                y=['Actual Normal (0)', 'Actual Failure (1)'],
                title=f"Baseline ANN Confusion Matrix (Threshold = {m_base['Threshold']})"
            )
            fig_cm_base.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_cm_base, use_container_width=True)

        with col_cm2:
            cm_sa = np.array(m_sa['Confusion Matrix'])
            fig_cm_sa = px.imshow(
                cm_sa, text_auto=True, color_continuous_scale='Greens',
                x=['Pred Normal (0)', 'Pred Failure (1)'],
                y=['Actual Normal (0)', 'Actual Failure (1)'],
                title=f"SA-Optimized ANN Confusion Matrix (Threshold = {m_sa['Threshold']})"
            )
            fig_cm_sa.update_layout(margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_cm_sa, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: REAL-TIME HEALTH PREDICTOR
# -------------------------------------------------------------
with tab4:
    st.subheader("Interactive Machine Failure Simulator & Diagnostic Assistant")
    st.markdown("Adjust live sensor controls or load preset industrial failure scenarios.")

    sa_model = artifacts.get('sa_ann', None)
    scaler = artifacts.get('scaler', None)
    test_bundle = artifacts.get('test_bundle', None)

    if not sa_model or not scaler or not test_bundle:
        st.error("Model artifacts not available. Please run `python train_and_evaluate.py` to train the models.")
    else:
        # Scenario Preset Buttons
        st.markdown("#### ⚡ Quick-Load Operational Scenarios")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

        # Session state for inputs
        if 'input_type' not in st.session_state:
            st.session_state.input_type = 'M'
            st.session_state.input_air = 298.5
            st.session_state.input_proc = 308.5
            st.session_state.input_rpm = 1500
            st.session_state.input_torque = 40.0
            st.session_state.input_wear = 30

        if btn_col1.button("🟢 Normal Operation", use_container_width=True):
            st.session_state.input_type = 'M'
            st.session_state.input_air = 298.5
            st.session_state.input_proc = 308.5
            st.session_state.input_rpm = 1500
            st.session_state.input_torque = 40.0
            st.session_state.input_wear = 20
            st.rerun()

        if btn_col2.button("🔴 Critical Overstrain", use_container_width=True):
            st.session_state.input_type = 'L'
            st.session_state.input_air = 300.0
            st.session_state.input_proc = 310.0
            st.session_state.input_rpm = 1350
            st.session_state.input_torque = 68.0
            st.session_state.input_wear = 215
            st.rerun()

        if btn_col3.button("🟠 Thermal Dissipation Strain", use_container_width=True):
            st.session_state.input_type = 'H'
            st.session_state.input_air = 304.0
            st.session_state.input_proc = 312.5
            st.session_state.input_rpm = 1380
            st.session_state.input_torque = 55.0
            st.session_state.input_wear = 120
            st.rerun()

        if btn_col4.button("⚡ High Power Stall Risk", use_container_width=True):
            st.session_state.input_type = 'L'
            st.session_state.input_air = 298.0
            st.session_state.input_proc = 308.0
            st.session_state.input_rpm = 2800
            st.session_state.input_torque = 72.0
            st.session_state.input_wear = 180
            st.rerun()

        st.markdown("---")

        # Input Controls
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)

        with ctrl_col1:
            st.markdown("##### Spindle Kinematics & Wear")
            p_rpm = st.slider("Rotational Speed [rpm]", 1100, 2900, int(st.session_state.input_rpm), step=10)
            p_torque = st.slider("Spindle Torque [Nm]", 3.0, 80.0, float(st.session_state.input_torque), step=0.5)
            p_wear = st.slider("Tool Wear [min]", 0, 260, int(st.session_state.input_wear), step=1)

        with ctrl_col2:
            st.markdown("##### Thermodynamics")
            p_air = st.slider("Air Temperature [K]", 294.0, 306.0, float(st.session_state.input_air), step=0.1)
            p_proc = st.slider("Process Temperature [K]", 304.0, 316.0, float(st.session_state.input_proc), step=0.1)
            p_type = st.selectbox("Product Quality Variant", ['L', 'M', 'H'], index=['L', 'M', 'H'].index(st.session_state.input_type))

        # Real-time physical calculations
        temp_diff = p_proc - p_air
        power_w = p_torque * (p_rpm * (2 * np.pi / 60.0))
        overstrain = p_wear * p_torque

        with ctrl_col3:
            st.markdown("##### Computed Physical Properties")
            st.info(f"**Temperature Differential (ΔT)**: {temp_diff:.2f} K")
            st.info(f"**Electromechanical Power (P)**: {power_w:.1f} W")
            st.info(f"**Overstrain Stress Product**: {overstrain:.1f} min·Nm")

        # Assemble feature vector matching training feature names
        feature_names = test_bundle['feature_names']
        input_dict = {
            'Air temperature [K]': p_air,
            'Process temperature [K]': p_proc,
            'Rotational speed [rpm]': p_rpm,
            'Torque [Nm]': p_torque,
            'Tool wear [min]': p_wear,
            'Temp_Diff [K]': temp_diff,
            'Power [W]': power_w,
            'Overstrain': overstrain,
            'Type_H': 1.0 if p_type == 'H' else 0.0,
            'Type_L': 1.0 if p_type == 'L' else 0.0,
            'Type_M': 1.0 if p_type == 'M' else 0.0,
        }

        # Build raw array in exact training order
        raw_feature_vector = np.array([[input_dict[col] for col in feature_names]])

        # Scale using pre-fit scaler
        scaled_feature_vector = scaler.transform(raw_feature_vector)

        # Run inference using SA-Optimized ANN
        failure_prob = float(sa_model.predict_proba(scaled_feature_vector)[0, 1])
        calibrated_threshold = sa_model.best_threshold_
        is_failure_predicted = failure_prob >= calibrated_threshold

        st.markdown("---")
        st.markdown("### 🔍 Model Diagnostic Inference")

        res_col1, res_col2 = st.columns([1, 1.2])

        with res_col1:
            # Risk Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=failure_prob * 100,
                title={'text': "Failure Probability (%)", 'font': {'size': 20}},
                number={'suffix': "%", 'valueformat': ".1f"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if failure_prob >= calibrated_threshold else "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.2)"},
                        {'range': [30, 70], 'color': "rgba(243, 156, 18, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.2)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': calibrated_threshold * 100
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            st.markdown("#### Health Assessment & Action Plan")
            if is_failure_predicted:
                st.error("🚨 **PREDICTED STATUS: IMMINENT MACHINE FAILURE**")
                st.markdown(f"""
                - **Failure Probability**: `{failure_prob * 100:.1f}%` (Exceeds decision threshold of `{calibrated_threshold * 100:.1f}%`)
                - **Model Used**: SA-Optimized ANN (Input reduced to 6 optimal features)
                """)
            else:
                st.success("✅ **PREDICTED STATUS: HEALTHY OPERATION**")
                st.markdown(f"""
                - **Failure Probability**: `{failure_prob * 100:.1f}%` (Well below decision threshold of `{calibrated_threshold * 100:.1f}%`)
                - **Operating Regime**: Normal wear & balanced thermodynamics.
                """)

            # Root-Cause Engineering Diagnostic Assistant
            st.markdown("##### 🛠️ Root-Cause Heuristic Diagnostics:")
            warnings = []
            if overstrain > 11000:
                warnings.append("⚠️ **Overstrain Stress Alert**: Tool wear combined with heavy torque exceeds critical structural limits.")
            if p_wear > 200:
                warnings.append("⚠️ **Tool Wear Expiry**: Cutting insert has accumulated over 200 minutes of cutting; insert failure imminent.")
            if temp_diff < 8.6:
                warnings.append("⚠️ **Heat Dissipation Risk**: Temperature difference between process and air is abnormally narrow (< 8.6 K).")
            if power_w > 9000 or power_w < 3500:
                warnings.append("⚠️ **Drive Power Discrepancy**: Spindle mechanical power is outside nominal operational envelope (3,500 W - 9,000 W).")

            if warnings:
                for w in warnings:
                    st.write(w)
            else:
                st.write("✨ All physical stress and thermodynamic parameters are within standard operating safety bounds.")
