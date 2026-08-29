"""
AI-Based Multimodal System for Stroke and Alzheimer's Detection
Professional Streamlit Medical AI Research Dashboard.
"""

from io import BytesIO
from pathlib import Path
import re
import streamlit as st
from PIL import Image

from utils.api_client import RetinalAIClient

# -------------------------------------------------------------
# 1. Page Configuration & Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Multimodal Retinal System — Stroke & Alzheimer's",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Medical AI Design System (Tailored neutral-blue palette)
st.markdown("""
<style>
    /* Global Typography & Backgrounds */
    .main {
        background-color: #F8FAFC;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    .badge-research {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .header-disclaimer {
        font-size: 0.8rem;
        color: #64748B;
        font-style: italic;
        margin-bottom: 1.25rem;
    }
    
    /* Healthcare Cards */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    
    /* Risk Badge Classes */
    .risk-high {
        color: #991B1B;
        background: #FEE2E2;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 700;
    }
    .risk-mod {
        color: #92400E;
        background: #FEF3C7;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 700;
    }
    .risk-low {
        color: #065F46;
        background: #D1FAE5;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-weight: 700;
    }

    /* Pipeline Status Tracker */
    .phase-badge-done {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .phase-badge-skip {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


def sanitize_patient_id(pid: str) -> str:
    """Sanitize patient ID for safe filenames."""
    return re.sub(r"[^\w\-]", "_", pid)


# -------------------------------------------------------------
# 2. Sidebar: Real-time System & Model Status
# -------------------------------------------------------------
st.sidebar.markdown("### ⚙️ SYSTEM DIAGNOSTICS")
backend_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")
client = RetinalAIClient(base_url=backend_url)

# Check Real Backend Health
health_res = client.check_health()
is_backend_online = health_res.get("status") == "ok"

if is_backend_online:
    device_info = health_res.get("device", "cpu").upper()
    st.sidebar.success(f"🟢 Backend Connected ({device_info})")
else:
    st.sidebar.error("🔴 Backend Unavailable")
    st.sidebar.caption("Run: `python -m uvicorn backend.main:app --reload`")

# Check Real Model Component Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI MODEL STATUS")
model_status = client.check_model_status()

if is_backend_online and model_status.get("status") != "error":
    st.sidebar.markdown(f"- **Swin Transformer:** `✓ Loaded ({model_status.get('phase4_octa')})`")
    st.sidebar.markdown(f"- **Clinical FT-Transformer:** `✓ Loaded ({model_status.get('phase6')})`")
    st.sidebar.markdown(f"- **Multi-Task Classifier:** `✓ Loaded ({model_status.get('phase8')})`")
    st.sidebar.markdown(f"- **MC Dropout Engine:** `✓ Available ({model_status.get('phase9')})`")
    st.sidebar.markdown(f"- **Grad-CAM + SHAP:** `✓ Available ({model_status.get('phase10')})`")
    st.sidebar.markdown(f"- **Report PDF Engine:** `✓ Available ({model_status.get('phase11')})`")
else:
    st.sidebar.markdown("- **Swin Transformer:** `⚠ Unknown`")
    st.sidebar.markdown("- **Clinical FT-Transformer:** `⚠ Unknown`")
    st.sidebar.markdown("- **Multi-Task Classifier:** `⚠ Unknown`")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset / Clear Session", use_container_width=True):
    if "latest_result" in st.session_state:
        del st.session_state["latest_result"]
    st.rerun()

st.sidebar.caption("Multimodal Medical AI System v1.0.0 — Research Build")


# -------------------------------------------------------------
# 3. Application Header
# -------------------------------------------------------------
st.markdown('<div class="badge-research">ACADEMIC RESEARCH PROTOTYPE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-header">AI-BASED MULTIMODAL SYSTEM FOR STROKE AND ALZHEIMER\'S DETECTION</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="sub-header">Retinal Imaging (OCT-A, OCT-B, Fundus) + Patient Tabular Clinical Biomarkers</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="header-disclaimer">⚠️ Not intended for standalone clinical diagnosis. '
    'Intended for academic research, screening decision-support, and multimodal explainability analysis.</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# 4. Retinal Modality Selection & Upload Section
# -------------------------------------------------------------
st.markdown("### 🔬 RETINAL IMAGING")
st.markdown("*Select one or more retinal imaging modalities (at least 1 required)*")

# Checkbox selection for optional modalities
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    select_octa = st.checkbox("OCT-A (Optical Coherence Tomography Angiography)", value=True, key="chk_octa")
with col_m2:
    select_octb = st.checkbox("OCT-B (Cross-Sectional Structural B-Scan)", value=True, key="chk_octb")
with col_m3:
    select_fundus = st.checkbox("Fundus (Color Fundus Photography)", value=True, key="chk_fundus")

selected_modalities = []
if select_octa:
    selected_modalities.append("octa")
if select_octb:
    selected_modalities.append("octb")
if select_fundus:
    selected_modalities.append("fundus")

# Check if at least 1 modality is selected
if len(selected_modalities) == 0:
    st.warning("⚠️ Please select at least one retinal imaging modality.")

# Dynamic Upload Cards Layout
octa_file = None
octb_file = None
fundus_file = None

if len(selected_modalities) > 0:
    upload_cols = st.columns(len(selected_modalities))
    col_idx = 0

    if select_octa:
        with upload_cols[col_idx]:
            st.markdown("#### OCT-A RETINAL SCAN")
            octa_file = st.file_uploader("Upload OCT-A Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octa")
            if octa_file:
                try:
                    img_octa = Image.open(octa_file)
                    st.image(img_octa, caption=f"OCT-A Preview", use_container_width=True)
                    st.success(f"✓ Uploaded successfully\n\n**File:** `{octa_file.name}`\n\n**Size:** `{img_octa.size[0]} × {img_octa.size[1]}` px ({img_octa.mode})")
                except Exception as e:
                    st.error(f"Invalid image: {e}")
        col_idx += 1

    if select_octb:
        with upload_cols[col_idx]:
            st.markdown("#### OCT-B RETINAL SCAN")
            octb_file = st.file_uploader("Upload OCT-B Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octb")
            if octb_file:
                try:
                    img_octb = Image.open(octb_file)
                    st.image(img_octb, caption=f"OCT-B Preview", use_container_width=True)
                    st.success(f"✓ Uploaded successfully\n\n**File:** `{octb_file.name}`\n\n**Size:** `{img_octb.size[0]} × {img_octb.size[1]}` px ({img_octb.mode})")
                except Exception as e:
                    st.error(f"Invalid image: {e}")
        col_idx += 1

    if select_fundus:
        with upload_cols[col_idx]:
            st.markdown("#### FUNDUS RETINAL SCAN")
            fundus_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_fundus")
            if fundus_file:
                try:
                    img_fundus = Image.open(fundus_file)
                    st.image(img_fundus, caption=f"Fundus Preview", use_container_width=True)
                    st.success(f"✓ Uploaded successfully\n\n**File:** `{fundus_file.name}`\n\n**Size:** `{img_fundus.size[0]} × {img_fundus.size[1]}` px ({img_fundus.mode})")
                except Exception as e:
                    st.error(f"Invalid image: {e}")
        col_idx += 1


# -------------------------------------------------------------
# 5. Patient Clinical Profile Section
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 PATIENT CLINICAL PROFILE")
st.markdown("*Clinical biomarkers and health history expected by the Phase 6 FT-Transformer*")

with st.expander("Enter / Edit Patient Clinical Biomarkers", expanded=True):
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        patient_id = st.text_input("Patient / Sample ID", value="PATIENT_001")
        old_groups = st.selectbox(
            "Age / Cognitive Cohort (Old groups)",
            options=["O_CD", "Y_CD", "O_CU", "Y_CU"],
            index=0,
            help="O_CD: Older Cognitive Decline, Y_CD: Younger Cognitive Decline, O_CU: Older Unimpaired, Y_CU: Younger Unimpaired",
        )
        gender = st.radio("Gender", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", horizontal=True)
        education = st.number_input("Formal Education (Years)", min_value=0.0, max_value=30.0, value=16.0, step=1.0)

    with col_c2:
        bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=70.0, value=26.8, step=0.1)
        obese = st.selectbox("Obesity Classification", options=[0.0, 1.0], format_func=lambda x: "Non-Obese (0)" if x == 0.0 else "Obese (1)")
        htn = st.selectbox("Hypertension (HTN)", options=[1, 0], format_func=lambda x: "Positive / Diagnosed (1)" if x == 1 else "Negative (0)")
        dm2 = st.selectbox("Type 2 Diabetes (DM2)", options=[0, 1], format_func=lambda x: "Negative (0)" if x == 0 else "Positive / Diagnosed (1)")

    with col_c3:
        smoking_ever = st.selectbox("Smoking History (Ever)", options=[1, 0], format_func=lambda x: "History of Smoking (1)" if x == 1 else "Never Smoked (0)")
        smoking_current = st.selectbox("Current Smoking", options=[0, 1], format_func=lambda x: "Non-Smoker (0)" if x == 0 else "Active Smoker (1)")
        etoh_ever = st.selectbox("Alcohol Consumption (Ever)", options=[1, 0], format_func=lambda x: "History of Consumption (1)" if x == 1 else "Never (0)")
        etoh_current = st.selectbox("Current Alcohol Consumption", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Active (1)")


# -------------------------------------------------------------
# 6. Analyze Patient Action Trigger
# -------------------------------------------------------------
st.markdown("---")

can_analyze = len(selected_modalities) > 0 and is_backend_online

if st.button("🚀 ANALYZE PATIENT (RUN MULTIMODAL INFERENCE)", type="primary", use_container_width=True, disabled=not can_analyze):
    # Verify that files for selected modalities are uploaded
    uploaded_missing = []
    if select_octa and not octa_file:
        uploaded_missing.append("OCT-A")
    if select_octb and not octb_file:
        uploaded_missing.append("OCT-B")
    if select_fundus and not fundus_file:
        uploaded_missing.append("Fundus")

    if uploaded_missing:
        st.error(f"❌ Please upload the selected retinal scan(s): {', '.join(uploaded_missing)}")
    else:
        # Clear previous analysis state
        if "latest_result" in st.session_state:
            del st.session_state["latest_result"]

        # Prepare Payload
        clinical_payload = {
            "patient_id": patient_id.strip(),
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

        octa_tuple = (octa_file.name, octa_file.getvalue(), octa_file.type) if octa_file else None
        octb_tuple = (octb_file.name, octb_file.getvalue(), octb_file.type) if octb_file else None
        fundus_tuple = (fundus_file.name, fundus_file.getvalue(), fundus_file.type) if fundus_file else None

        with st.spinner("Executing Full Multimodal AI Pipeline (Phases 2 through 11)..."):
            res = client.run_analysis(
                clinical_data=clinical_payload,
                octa_file=octa_tuple,
                octb_file=octb_tuple,
                fundus_file=fundus_tuple,
            )

        if res.get("status") == "error":
            st.error(f"❌ Inference Failed: {res.get('detail')}")
        else:
            st.success(f"✅ Multimodal Analysis Completed for {patient_id} (Report ID: {res['report_id']})")
            st.session_state["latest_result"] = res


# -------------------------------------------------------------
# 7. Pipeline Status & Results Dashboard
# -------------------------------------------------------------
if "latest_result" in st.session_state:
    data = st.session_state["latest_result"]
    mods_processed = data.get("modalities_processed", [])

    st.markdown("## 📊 PATIENT ANALYSIS RESULTS")
    
    # Session Header Info
    st.markdown(
        f"**Patient ID:** `{data['patient_id']}` | **Report ID:** `{data['report_id']}` | "
        f"**Timestamp:** `{data['timestamp']}` | **Active Modalities:** `{', '.join([m.upper() for m in mods_processed])}`"
    )

    # ---------------------------------------------------------
    # Pipeline Execution Matrix (Phases 2 - 11)
    # ---------------------------------------------------------
    st.markdown("##### ⚙️ Pipeline Execution Status")
    p_cols = st.columns(10)
    phases = [
        ("P2", "Preprocessing"),
        ("P3", "Quality"),
        ("P4", "Swin-T"),
        ("P5", "Fusion"),
        ("P6", "FT-Trans"),
        ("P7", "Cross-Attn"),
        ("P8", "Disease Pred"),
        ("P9", "Uncertainty"),
        ("P10", "Explainability"),
        ("P11", "Report"),
    ]
    for idx, (p_code, p_name) in enumerate(phases):
        with p_cols[idx]:
            st.markdown(f"<span class='phase-badge-done'>✓ {p_code}: {p_name}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Dual Disease Risk Cards (Phase 8 & 9)
    # ---------------------------------------------------------
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st_pred = data["stroke_prediction"]
        st_unc = data["stroke_uncertainty"]
        risk_cls = "risk-high" if "HIGH" in st_pred["risk_category"] else ("risk-mod" if "MODERATE" in st_pred["risk_category"] else "risk-low")
        
        st.markdown(f"""
        <div class="med-card">
            <div class="card-title">🧠 STROKE RISK ASSESSMENT</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">
                Predicted Probability: {st_pred['probability']*100:.1f}%
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span class="{risk_cls}">Risk Level: {st_pred['risk_category']}</span>
                <span style="margin-left: 0.5rem; font-weight: 600; color: #475569;">({st_pred['class_label']})</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0.75rem 0;" />
            <div style="font-size: 0.85rem; color: #334155;">
                <b>Model Confidence (Phase 9):</b> {st_unc['confidence_percent']:.2f}% ({st_unc['confidence_level']})<br/>
                <b>Predictive Uncertainty:</b> σ² = {st_unc['predictive_variance']:.4f} ({st_unc['uncertainty_level']})<br/>
                <b>Shannon Entropy:</b> H(p) = {st_unc['predictive_entropy']:.4f} nats
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        al_pred = data["alzheimer_prediction"]
        al_unc = data["alzheimer_uncertainty"]
        risk_cls_al = "risk-high" if "HIGH" in al_pred["risk_category"] else ("risk-mod" if "MODERATE" in al_pred["risk_category"] else "risk-low")

        st.markdown(f"""
        <div class="med-card">
            <div class="card-title">🧬 ALZHEIMER'S DISEASE RISK ASSESSMENT</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0;">
                Predicted Probability: {al_pred['probability']*100:.1f}%
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span class="{risk_cls_al}">Risk Level: {al_pred['risk_category']}</span>
                <span style="margin-left: 0.5rem; font-weight: 600; color: #475569;">({al_pred['class_label']})</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0.75rem 0;" />
            <div style="font-size: 0.85rem; color: #334155;">
                <b>Model Confidence (Phase 9):</b> {al_unc['confidence_percent']:.2f}% ({al_unc['confidence_level']})<br/>
                <b>Predictive Uncertainty:</b> σ² = {al_unc['predictive_variance']:.4f} ({al_unc['uncertainty_level']})<br/>
                <b>Shannon Entropy:</b> H(p) = {al_unc['predictive_entropy']:.4f} nats
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Retinal Technical Quality Assessment (Phase 3)
    # ---------------------------------------------------------
    st.markdown("### 🔍 RETINAL IMAGE QUALITY ASSESSMENT (PHASE 3)")
    q_mods = [m for m in ("octa", "octb", "fundus") if data["image_quality"].get(m, {}).get("available")]
    
    if q_mods:
        q_cols = st.columns(len(q_mods))
        for idx, mod in enumerate(q_mods):
            q_info = data["image_quality"][mod]
            with q_cols[idx]:
                score = q_info.get("quality_score", 0.0)
                dec = q_info.get("decision", "ACCEPT")
                st.metric(
                    label=f"{mod.upper()} Scan Quality",
                    value=f"{score:.1f} / 100",
                    delta=f"Status: {dec}",
                    delta_color="normal" if dec == "ACCEPT" else "inverse",
                )
    else:
        st.info("No retinal quality data available for this session.")

    # ---------------------------------------------------------
    # Explainability & Diagnostic Saliency (Phase 10)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 💡 WHY DID THE MODEL MAKE THIS PREDICTION?")
    
    exp = data.get("explainability", {})
    tab_gcam, tab_shap, tab_bio = st.tabs([
        "🖼️ Retinal Swin Grad-CAM Heatmaps",
        "📈 Clinical Feature Attributions (SHAP)",
        "🔬 Model-Identified Biomarkers & Findings",
    ])

    with tab_gcam:
        for disease_name, d_key in [("Stroke Target", "stroke"), ("Alzheimer's Disease Target", "alzheimer")]:
            st.markdown(f"#### {disease_name} — Retinal Saliency Overlays")
            g_dict = exp.get(d_key, {}).get("gradcam", {})
            
            # Filter only modalities actually uploaded and available
            available_gcam = [m for m in ("octa", "octb", "fundus") if g_dict.get(m, {}).get("status") == "SUCCESS"]
            
            if available_gcam:
                g_cols = st.columns(len(available_gcam))
                for idx, mod in enumerate(available_gcam):
                    m_item = g_dict[mod]
                    with g_cols[idx]:
                        if m_item.get("panel_path") and Path(m_item["panel_path"]).exists():
                            st.image(
                                m_item["panel_path"],
                                caption=f"{mod.upper()} [Original Scan | Grad-CAM Heatmap | Alpha Overlay]",
                                use_container_width=True,
                            )
            else:
                st.caption(f"No retinal Grad-CAM overlays generated for {disease_name}.")

    with tab_shap:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Stroke Clinical Risk Factors (SHAP)")
            st_shap_plot = exp.get("stroke", {}).get("shap_plot_path")
            if st_shap_plot and Path(st_shap_plot).exists():
                st.image(st_shap_plot, caption="Stroke Feature Attributions", use_container_width=True)
            st_shap_items = exp.get("stroke", {}).get("shap_clinical", [])
            if st_shap_items:
                st.dataframe(st_shap_items, use_container_width=True)

        with col_s2:
            st.markdown("#### Alzheimer's Clinical Risk Factors (SHAP)")
            al_shap_plot = exp.get("alzheimer", {}).get("shap_plot_path")
            if al_shap_plot and Path(al_shap_plot).exists():
                st.image(al_shap_plot, caption="Alzheimer's Feature Attributions", use_container_width=True)
            al_shap_items = exp.get("alzheimer", {}).get("shap_clinical", [])
            if al_shap_items:
                st.dataframe(al_shap_items, use_container_width=True)

    with tab_bio:
        st.markdown("#### Model-Identified Morphological Findings")
        st.info("ℹ️ **Notice:** Explicit retinal biomarker detection (e.g. automated FAZ area quantification, RNFL thickness profiling) is not currently implemented. Grad-CAM visual heatmaps depict spatial activation density learned by the Swin Transformer backbones.")

    # ---------------------------------------------------------
    # Clinical Summary & Dynamic PDF Report Download
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📄 CLINICAL ASSESSMENT REPORT (PHASE 11)")
    st.info(f"**Multimodal Narrative Summary:**\n\n{data.get('clinical_summary', '')}")

    pdf_bytes = client.download_pdf_report(data["report_id"])
    if pdf_bytes:
        clean_pid = sanitize_patient_id(data["patient_id"])
        st.download_button(
            label="📥 DOWNLOAD CLINICAL REPORT (PDF)",
            data=pdf_bytes,
            file_name=f"patient_{clean_pid}_retinal_analysis_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )

    # Mandatory Safety Disclaimer Banner
    st.markdown("---")
    st.warning(f"⚠️ **RESEARCH DISCLAIMER:** {data.get('disclaimer', '')}")
