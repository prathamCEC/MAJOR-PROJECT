"""
Streamlit Frontend Application for Multimodal Retinal AI System.

Provides an interactive clinical-style dashboard connecting to the FastAPI backend.
"""

from io import BytesIO
from pathlib import Path
import streamlit as st
from PIL import Image

from utils.api_client import RetinalAIClient

# Configure Page Settings
st.set_page_config(
    page_title="Multimodal Retinal AI — Stroke & Alzheimer's Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F7FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
    }
    .risk-high {
        color: #C53030;
        font-weight: 700;
    }
    .risk-mod {
        color: #D69E2E;
        font-weight: 700;
    }
    .risk-low {
        color: #2F855A;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_class(risk_str: str) -> str:
    if "HIGH" in risk_str:
        return "risk-high"
    elif "MODERATE" in risk_str:
        return "risk-mod"
    return "risk-low"


# -------------------------------------------------------------
# Sidebar: System & Backend Status
# -------------------------------------------------------------
st.sidebar.title("⚙️ System Status")
backend_url = st.sidebar.text_input("Backend REST API URL", value="http://127.0.0.1:8000")
client = RetinalAIClient(base_url=backend_url)

# Health Check
health_res = client.check_health()
if health_res.get("status") == "ok":
    st.sidebar.success(f"🟢 Backend Connected ({health_res.get('device', 'cpu').upper()})")
else:
    st.sidebar.error(f"🔴 Backend Offline: {health_res.get('message', 'Unreachable')}")
    st.sidebar.info("Tip: Start backend with `python -m backend.main`")

# Quick Demo Preset Loader
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Quick Demo Presets")
use_demo_data = st.sidebar.button("Load Pre-Populated Sample Patient", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Based Multimodal Retinal Analysis System v1.0.0 (Research Edition)")


# -------------------------------------------------------------
# Main Header
# -------------------------------------------------------------
st.markdown('<div class="main-header">👁️ AI-Based Multimodal Retinal Analysis System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Research-oriented prediction of Stroke & Alzheimer\'s Disease '
    'using Tripartite Retinal Scans (OCT-A, OCT-B, Fundus) and Patient Tabular Clinical Variables</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Section 1: Patient Clinical Health Variables
# -------------------------------------------------------------
st.subheader("1. Patient Clinical Profile (FT-Transformer Input)")

with st.expander("Enter Patient Clinical Biomarkers & History", expanded=True):
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        patient_id = st.text_input("Patient / Sample ID", value="PATIENT_NEW_01")
        old_groups = st.selectbox(
            "Age & Cognitive Group (Old groups)",
            options=["O_CD", "Y_CD", "O_CU", "Y_CU"],
            index=0 if not use_demo_data else 0,
            help="O_CD: Older Cognitive Decline, Y_CD: Younger Cognitive Decline, O_CU: Older Unimpaired, Y_CU: Younger Unimpaired",
        )
        gender = st.radio("Gender", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", horizontal=True)
        education = st.number_input("Formal Education (Years)", min_value=0.0, max_value=30.0, value=16.0, step=1.0)

    with col_c2:
        bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=70.0, value=27.2, step=0.1)
        obese = st.selectbox("Obesity Classification", options=[0.0, 1.0], format_func=lambda x: "Non-Obese (0)" if x == 0.0 else "Obese (1)")
        htn = st.selectbox("Hypertension (HTN)", options=[1, 0], format_func=lambda x: "Positive (1)" if x == 1 else "Negative (0)")
        dm2 = st.selectbox("Type 2 Diabetes (DM2)", options=[0, 1], format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")

    with col_c3:
        smoking_ever = st.selectbox("Smoking History (Ever)", options=[1, 0], format_func=lambda x: "History (1)" if x == 1 else "Never (0)")
        smoking_current = st.selectbox("Smoking Current", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        etoh_ever = st.selectbox("Alcohol Consumption (Ever)", options=[1, 0], format_func=lambda x: "History (1)" if x == 1 else "Never (0)")
        etoh_current = st.selectbox("Alcohol Consumption (Current)", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")


# -------------------------------------------------------------
# Section 2: Retinal Scans Upload
# -------------------------------------------------------------
st.subheader("2. Retinal Imaging Scans (OCT-A, OCT-B, Fundus)")

col_u1, col_u2, col_u3 = st.columns(3)

with col_u1:
    st.markdown("##### 🔬 OCT-A Retinal Scan")
    octa_file = st.file_uploader("Upload OCT-A Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="upload_octa")
    if octa_file:
        img_octa = Image.open(octa_file)
        st.image(img_octa, caption=f"OCT-A ({img_octa.size[0]}×{img_octa.size[1]})", use_container_width=True)

with col_u2:
    st.markdown("##### 🔬 OCT-B Retinal Scan")
    octb_file = st.file_uploader("Upload OCT-B Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="upload_octb")
    if octb_file:
        img_octb = Image.open(octb_file)
        st.image(img_octb, caption=f"OCT-B ({img_octb.size[0]}×{img_octb.size[1]})", use_container_width=True)

with col_u3:
    st.markdown("##### 🔬 Fundus Retinal Scan")
    fundus_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="upload_fundus")
    if fundus_file:
        img_fundus = Image.open(fundus_file)
        st.image(img_fundus, caption=f"Fundus ({img_fundus.size[0]}×{img_fundus.size[1]})", use_container_width=True)


# -------------------------------------------------------------
# Section 3: Analysis Execution Trigger
# -------------------------------------------------------------
st.markdown("---")
analyze_btn = st.button("🚀 ANALYZE PATIENT (RUN FULL PIPELINE)", type="primary", use_container_width=True)

if analyze_btn:
    # Prepare Clinical Dictionary
    clinical_payload = {
        "patient_id": patient_id,
        "Old_groups": old_groups,
        "Gender": str(gender),
        "Education": str(education),
        "BMI": str(bmi),
        "Obese": str(obese),
        "EtOH_ever": str(etoh_ever),
        "EtOH_current": str(etoh_current),
        "Smoking_ever": str(smoking_ever),
        "Smoking_current": str(smoking_current),
        "HTN": str(htn),
        "DM2": str(dm2),
    }

    # Prepare Image Tuples for multipart upload
    octa_tuple = (octa_file.name, octa_file.getvalue(), octa_file.type) if octa_file else None
    octb_tuple = (octb_file.name, octb_file.getvalue(), octb_file.type) if octb_file else None
    fundus_tuple = (fundus_file.name, fundus_file.getvalue(), fundus_file.type) if fundus_file else None

    with st.spinner("Executing Multimodal Deep Learning Inference (Phase 2 to 11)..."):
        res = client.run_analysis(
            clinical_data=clinical_payload,
            octa_file=octa_tuple,
            octb_file=octb_tuple,
            fundus_file=fundus_tuple,
        )

    if res.get("status") == "error":
        st.error(f"❌ Analysis Pipeline Failed: {res.get('detail')}")
    else:
        st.success(f"✅ Multimodal Assessment Completed Successfully (Report ID: {res['report_id']})")
        st.session_state["latest_result"] = res


# -------------------------------------------------------------
# Section 4: Results Dashboard
# -------------------------------------------------------------
if "latest_result" in st.session_state:
    data = st.session_state["latest_result"]
    st.markdown("## 📊 Assessment Results & Diagnostic Saliency Dashboard")

    # 1. Technical Image Quality (Phase 3)
    st.subheader("A. Retinal Image Technical Quality Assessment (Phase 3)")
    q_cols = st.columns(3)
    for idx, mod in enumerate(("octa", "octb", "fundus")):
        q_item = data["image_quality"].get(mod, {})
        with q_cols[idx]:
            st.markdown(f"**{mod.upper()} Scan Quality**")
            if q_item.get("available"):
                score = q_item.get("quality_score", 0.0)
                dec = q_item.get("decision", "ACCEPT")
                dec_color = "green" if dec == "ACCEPT" else "red"
                st.metric("Quality Score", f"{score:.1f} / 100", delta=dec, delta_color="normal" if dec == "ACCEPT" else "inverse")
            else:
                st.info("Not available for this session")

    # 2. Dual Multi-Task Predictions & Uncertainty (Phases 8 & 9)
    st.subheader("B. Multi-Task Disease Predictions & Uncertainty Analysis (Phases 8 & 9)")
    p_col1, p_col2 = st.columns(2)

    with p_col1:
        st_pred = data["stroke_prediction"]
        st_unc = data["stroke_uncertainty"]
        st.markdown(f"### 🧠 Stroke Target: <span class='{get_risk_class(st_pred['risk_category'])}'>{st_pred['risk_category']}</span>", unsafe_allow_html=True)
        st.progress(float(st_pred["probability"]))
        st.write(f"**Predicted Likelihood:** `{st_pred['probability']:.4f}` ({st_pred['class_label']})")
        st.write(f"**Model Confidence (Phase 9):** `{st_unc['confidence_percent']:.2f}%` ({st_unc['confidence_level']})")
        st.write(f"**Predictive Variance:** `σ² = {st_unc['predictive_variance']:.4f}` | **Entropy:** `H(p) = {st_unc['predictive_entropy']:.4f}`")
        st.caption(st_unc["statement"])

    with p_col2:
        al_pred = data["alzheimer_prediction"]
        al_unc = data["alzheimer_uncertainty"]
        st.markdown(f"### 🧬 Alzheimer's Target: <span class='{get_risk_class(al_pred['risk_category'])}'>{al_pred['risk_category']}</span>", unsafe_allow_html=True)
        st.progress(float(al_pred["probability"]))
        st.write(f"**Predicted Likelihood:** `{al_pred['probability']:.4f}` ({al_pred['class_label']})")
        st.write(f"**Model Confidence (Phase 9):** `{al_unc['confidence_percent']:.2f}%` ({al_unc['confidence_level']})")
        st.write(f"**Predictive Variance:** `σ² = {al_unc['predictive_variance']:.4f}` | **Entropy:** `H(p) = {al_unc['predictive_entropy']:.4f}`")
        st.caption(al_unc["statement"])

    # 3. Model Explainability (Phase 10 Grad-CAM & SHAP)
    st.subheader("C. Model Explainability & Saliency (Phase 10)")
    exp = data.get("explainability", {})

    tab_gcam, tab_shap = st.tabs(["🖼️ Retinal Swin Grad-CAM Heatmaps", "📈 Clinical Feature Attributions (SHAP)"])

    with tab_gcam:
        for disease_name, d_key in [("Stroke Target", "stroke"), ("Alzheimer's Target", "alzheimer")]:
            st.markdown(f"##### {disease_name} — Retinal Saliency Overlays")
            g_dict = exp.get(d_key, {}).get("gradcam", {})
            g_cols = st.columns(3)
            for idx, mod in enumerate(("octa", "octb", "fundus")):
                with g_cols[idx]:
                    m_item = g_dict.get(mod, {})
                    if m_item.get("status") == "SUCCESS" and m_item.get("panel_path") and Path(m_item["panel_path"]).exists():
                        st.image(m_item["panel_path"], caption=f"{mod.upper()} [Original | CAM | Overlay]", use_container_width=True)
                    else:
                        st.caption(f"{mod.upper()} visual overlay not available.")

    with tab_shap:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### Stroke Clinical SHAP Risk Factors")
            st_shap_plot = exp.get("stroke", {}).get("shap_plot_path")
            if st_shap_plot and Path(st_shap_plot).exists():
                st.image(st_shap_plot, caption="Stroke Feature Attributions", use_container_width=True)
            st_shap_items = exp.get("stroke", {}).get("shap_clinical", [])
            if st_shap_items:
                st.dataframe(st_shap_items, use_container_width=True)

        with col_s2:
            st.markdown("##### Alzheimer's Clinical SHAP Risk Factors")
            al_shap_plot = exp.get("alzheimer", {}).get("shap_plot_path")
            if al_shap_plot and Path(al_shap_plot).exists():
                st.image(al_shap_plot, caption="Alzheimer's Feature Attributions", use_container_width=True)
            al_shap_items = exp.get("alzheimer", {}).get("shap_clinical", [])
            if al_shap_items:
                st.dataframe(al_shap_items, use_container_width=True)

    # 4. Clinical Summary & PDF Report Download
    st.subheader("D. Clinical Summary & PDF Report Generation (Phase 11)")
    st.info(f"**Clinical Narrative Summary:**\n\n{data.get('clinical_summary', '')}")

    pdf_bytes = client.download_pdf_report(data["report_id"])
    if pdf_bytes:
        st.download_button(
            label="📄 DOWNLOAD CLINICAL ASSESSMENT REPORT (PDF)",
            data=pdf_bytes,
            file_name=f"clinical_report_{data['patient_id']}_{data['report_id']}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Mandatory Safety Disclaimer
    st.markdown("---")
    st.warning(f"⚠️ **RESEARCH NOTICE:** {data.get('disclaimer', '')}")
