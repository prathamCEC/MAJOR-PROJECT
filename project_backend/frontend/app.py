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

# Professional Medical AI Design System
st.markdown("""
<style>
    /* Global Backgrounds & Layout */
    .main {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Typography */
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.05rem;
        font-weight: 500;
        color: #334155;
        margin-bottom: 0.5rem;
    }
    .badge-research {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    .header-disclaimer {
        font-size: 0.82rem;
        color: #64748B;
        font-style: italic;
        margin-bottom: 1.25rem;
        border-left: 3px solid #CBD5E1;
        padding-left: 0.75rem;
    }
    
    /* Medical Dashboard Cards */
    .med-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03), 0 4px 6px -2px rgba(0,0,0,0.02);
        margin-bottom: 1.25rem;
    }
    .med-card-highlight {
        background: #FFFFFF;
        border: 1.5px solid #3B82F6;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08);
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    /* Risk Badges */
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .risk-high {
        color: #991B1B;
        background: #FEE2E2;
        border: 1px solid #FECACA;
    }
    .risk-mod {
        color: #92400E;
        background: #FEF3C7;
        border: 1px solid #FDE68A;
    }
    .risk-low {
        color: #065F46;
        background: #D1FAE5;
        border: 1px solid #A7F3D0;
    }
    
    /* Overall Risk Callout Banner */
    .overall-risk-box {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .overall-risk-high {
        background-color: #FEF2F2;
        border: 1.5px solid #F87171;
        color: #991B1B;
    }
    .overall-risk-mod {
        background-color: #FFFBEB;
        border: 1.5px solid #FBBF24;
        color: #92400E;
    }
    .overall-risk-low {
        background-color: #ECFDF5;
        border: 1.5px solid #34D399;
        color: #065F46;
    }

    /* Pipeline Status Matrix */
    .phase-badge-done {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.3rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
        text-align: center;
        width: 100%;
    }
    .phase-badge-skip {
        background: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
        padding: 0.3rem 0.5rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
        text-align: center;
        width: 100%;
    }

    /* Attention Color Legend Bar */
    .attention-legend-bar {
        height: 14px;
        border-radius: 7px;
        background: linear-gradient(to right, #000080, #0055ff, #00ffff, #00ff00, #ffff00, #ff5500, #ff0000);
        margin: 0.4rem 0;
        border: 1px solid #CBD5E1;
    }
    .attention-legend-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 0.03em;
    }

    /* Step Indicator Badges */
    .step-badge {
        display: inline-block;
        background: #0F172A;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.55rem;
        border-radius: 4px;
        margin-right: 0.4rem;
    }
    
    /* Result Section Separator */
    .results-hero-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    .results-hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 0.3rem;
        color: #F8FAFC;
    }
    .results-hero-meta {
        font-size: 0.88rem;
        color: #94A3B8;
        display: flex;
        flex-wrap: wrap;
        gap: 1.25rem;
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
    st.sidebar.success(f"🟢 Backend Online ({device_info})")
else:
    st.sidebar.error("🔴 Backend Unavailable")
    st.sidebar.caption("Run: `python -m uvicorn backend.main:app --reload`")

# Check Real Model Component Status
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 AI MODEL STATUS")
model_status = client.check_model_status()

if is_backend_online and model_status.get("status") != "error":
    st.sidebar.markdown(f"- **Swin Transformer:** `✓ ({model_status.get('phase4_octa')})`")
    st.sidebar.markdown(f"- **Clinical FT-Transformer:** `✓ ({model_status.get('phase6')})`")
    st.sidebar.markdown(f"- **Multi-Task Classifier:** `✓ ({model_status.get('phase8')})`")
    st.sidebar.markdown(f"- **MC Dropout Engine:** `✓ ({model_status.get('phase9')})`")
    st.sidebar.markdown(f"- **Grad-CAM + SHAP:** `✓ ({model_status.get('phase10')})`")
    st.sidebar.markdown(f"- **Report PDF Engine:** `✓ ({model_status.get('phase11')})`")
else:
    st.sidebar.markdown("- **Swin Transformer:** `⚠ Unknown`")
    st.sidebar.markdown("- **Clinical FT-Transformer:** `⚠ Unknown`")
    st.sidebar.markdown("- **Multi-Task Classifier:** `⚠ Unknown`")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Analysis / Clear State", use_container_width=True):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
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
    '<div class="header-disclaimer">⚠️ <b>Research Decision-Support Notice:</b> This AI prototype evaluates multimodal retinal imaging and clinical biomarkers for investigative research. It does not provide an independent clinical diagnosis.</div>',
    unsafe_allow_html=True,
)


# -------------------------------------------------------------
# 4. Step 1: Patient Clinical Profile
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### <span class='step-badge'>STEP 1</span> PATIENT CLINICAL PROFILE", unsafe_allow_html=True)
st.markdown("*Enter clinical risk factors and biomarkers for the FT-Transformer encoder*")

with st.expander("📝 Patient Clinical Factors Form", expanded=True):
    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        patient_id = st.text_input("Patient / Sample Identifier", value="PATIENT_001", key="in_pid")
        old_groups = st.selectbox(
            "Age / Cognitive Cohort (Old groups)",
            options=["O_CD", "Y_CD", "O_CU", "Y_CU"],
            index=0,
            help="O_CD: Older Cognitive Decline, Y_CD: Younger Cognitive Decline, O_CU: Older Unimpaired, Y_CU: Younger Unimpaired",
            key="in_old_groups",
        )
        gender = st.radio("Biological Gender", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", horizontal=True, key="in_gender")
        education = st.number_input("Formal Education (Years)", min_value=0.0, max_value=30.0, value=16.0, step=1.0, key="in_edu")

    with col_c2:
        bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=70.0, value=26.8, step=0.1, key="in_bmi")
        obese = st.selectbox("Obesity Classification", options=[0.0, 1.0], format_func=lambda x: "Non-Obese (0)" if x == 0.0 else "Obese (1)", key="in_obese")
        htn = st.selectbox("Hypertension (HTN)", options=[1, 0], format_func=lambda x: "Positive / Diagnosed (1)" if x == 1 else "Negative (0)", key="in_htn")
        dm2 = st.selectbox("Type 2 Diabetes (DM2)", options=[0, 1], format_func=lambda x: "Negative (0)" if x == 0 else "Positive / Diagnosed (1)", key="in_dm2")

    with col_c3:
        smoking_ever = st.selectbox("Smoking History (Ever)", options=[1, 0], format_func=lambda x: "History of Smoking (1)" if x == 1 else "Never Smoked (0)", key="in_smk_ever")
        smoking_current = st.selectbox("Current Smoking Status", options=[0, 1], format_func=lambda x: "Non-Smoker (0)" if x == 0 else "Active Smoker (1)", key="in_smk_curr")
        etoh_ever = st.selectbox("Alcohol Consumption (Ever)", options=[1, 0], format_func=lambda x: "History of Consumption (1)" if x == 1 else "Never (0)", key="in_etoh_ever")
        etoh_current = st.selectbox("Current Alcohol Consumption", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Active (1)", key="in_etoh_curr")


# -------------------------------------------------------------
# 5. Step 2 & 3: Retinal Modality Selection & Upload
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### <span class='step-badge'>STEP 2 & 3</span> RETINAL IMAGING SELECTION & UPLOAD", unsafe_allow_html=True)
st.markdown("*Select one or more modalities to analyze (at least 1 required: OCT-A, OCT-B, and/or Fundus)*")

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    select_octa = st.checkbox("🔬 OCT-A (Angiography)", value=True, key="chk_octa")
with col_m2:
    select_octb = st.checkbox("🔬 OCT-B (Structural B-Scan)", value=True, key="chk_octb")
with col_m3:
    select_fundus = st.checkbox("👁️ Fundus (Color Retinal Photo)", value=True, key="chk_fundus")

selected_modalities = []
if select_octa:
    selected_modalities.append("octa")
if select_octb:
    selected_modalities.append("octb")
if select_fundus:
    selected_modalities.append("fundus")

octa_file = None
octb_file = None
fundus_file = None

if len(selected_modalities) == 0:
    st.warning("⚠️ Please select at least one retinal imaging modality.")
else:
    upload_cols = st.columns(len(selected_modalities))
    col_idx = 0

    if select_octa:
        with upload_cols[col_idx]:
            st.markdown("#### OCT-A SCAN")
            octa_file = st.file_uploader("Upload OCT-A Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octa")
            if octa_file:
                try:
                    img_octa = Image.open(octa_file)
                    st.image(img_octa, caption=f"OCT-A Preview: {octa_file.name}", use_container_width=True)
                    st.success(f"✓ **OCT-A Ready** ({img_octa.size[0]}×{img_octa.size[1]} px, {img_octa.mode})")
                except Exception as e:
                    st.error(f"Invalid image format: {e}")
        col_idx += 1

    if select_octb:
        with upload_cols[col_idx]:
            st.markdown("#### OCT-B SCAN")
            octb_file = st.file_uploader("Upload OCT-B Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octb")
            if octb_file:
                try:
                    img_octb = Image.open(octb_file)
                    st.image(img_octb, caption=f"OCT-B Preview: {octb_file.name}", use_container_width=True)
                    st.success(f"✓ **OCT-B Ready** ({img_octb.size[0]}×{img_octb.size[1]} px, {img_octb.mode})")
                except Exception as e:
                    st.error(f"Invalid image format: {e}")
        col_idx += 1

    if select_fundus:
        with upload_cols[col_idx]:
            st.markdown("#### FUNDUS SCAN")
            fundus_file = st.file_uploader("Upload Fundus Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_fundus")
            if fundus_file:
                try:
                    img_fundus = Image.open(fundus_file)
                    st.image(img_fundus, caption=f"Fundus Preview: {fundus_file.name}", use_container_width=True)
                    st.success(f"✓ **Fundus Ready** ({img_fundus.size[0]}×{img_fundus.size[1]} px, {img_fundus.mode})")
                except Exception as e:
                    st.error(f"Invalid image format: {e}")
        col_idx += 1


# -------------------------------------------------------------
# 6. Step 4: Input Summary & Review
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### <span class='step-badge'>STEP 4</span> INPUT SUMMARY & VERIFICATION", unsafe_allow_html=True)

# Build Modality Status Strings
mod_status_items = []
if select_octa:
    mod_status_items.append("✓ **OCT-A:** Uploaded" if octa_file else "⏳ **OCT-A:** Selected (File Needed)")
else:
    mod_status_items.append("✗ **OCT-A:** Not Provided")

if select_octb:
    mod_status_items.append("✓ **OCT-B:** Uploaded" if octb_file else "⏳ **OCT-B:** Selected (File Needed)")
else:
    mod_status_items.append("✗ **OCT-B:** Not Provided")

if select_fundus:
    mod_status_items.append("✓ **Fundus:** Uploaded" if fundus_file else "⏳ **Fundus:** Selected (File Needed)")
else:
    mod_status_items.append("✗ **Fundus:** Not Provided")

col_sum1, col_sum2, col_sum3 = st.columns(3)
with col_sum1:
    st.markdown(f"**Patient ID:** `{patient_id.strip() or 'PATIENT_001'}`")
    st.markdown(f"**Cohort:** `{old_groups}` | **Gender:** `{'Male' if gender==1 else 'Female'}`")
with col_sum2:
    st.markdown(f"**BMI:** `{bmi:.1f}` | **Obese:** `{'Yes' if obese==1.0 else 'No'}`")
    st.markdown(f"**HTN:** `{'Positive' if htn==1 else 'Negative'}` | **DM2:** `{'Positive' if dm2==1 else 'Negative'}`")
with col_sum3:
    st.markdown(" • " + "<br/> • ".join(mod_status_items), unsafe_allow_html=True)


# -------------------------------------------------------------
# 7. Step 5: Analyze Patient Action Trigger
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### <span class='step-badge'>STEP 5</span> EXECUTE MULTIMODAL INFERENCE", unsafe_allow_html=True)

# Check validity
missing_required_files = []
if select_octa and not octa_file:
    missing_required_files.append("OCT-A")
if select_octb and not octb_file:
    missing_required_files.append("OCT-B")
if select_fundus and not fundus_file:
    missing_required_files.append("Fundus")

can_analyze = len(selected_modalities) > 0 and len(missing_required_files) == 0 and is_backend_online

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    btn_analyze = st.button(
        "🔬 ANALYZE PATIENT (RUN FULL MULTIMODAL PIPELINE)",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
    )
with col_btn2:
    if st.button("↻ Reset Form", use_container_width=True):
        if "latest_result" in st.session_state:
            del st.session_state["latest_result"]
        st.rerun()

if not can_analyze and is_backend_online and len(missing_required_files) > 0:
    st.info(f"ℹ️ Please upload image(s) for selected modality: **{', '.join(missing_required_files)}** to enable analysis.")

if btn_analyze:
    # Prepare Payload
    clinical_payload = {
        "patient_id": patient_id.strip() or "PATIENT_001",
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

    # Step-by-step progress simulation display
    progress_placeholder = st.empty()
    with progress_placeholder.container():
        st.info("⏳ **Running Multimodal Deep Learning Pipeline...**\n\n"
                "✓ Validating patient record & staging raw retinal scans\n"
                "✓ Phase 2 Preprocessing & Phase 3 Image Quality Assessment\n"
                "● Extracting Swin Transformer spatial representations\n"
                "○ Fusing cross-modal embeddings (DMRA & FT-Transformer)\n"
                "○ Evaluating Multi-Task disease prediction & MC-Dropout uncertainty\n"
                "○ Generating Grad-CAM spatial heatmaps & SHAP attributions\n"
                "○ Compiling clean clinical assessment report")

    res = client.run_analysis(
        clinical_data=clinical_payload,
        octa_file=octa_tuple,
        octb_file=octb_tuple,
        fundus_file=fundus_tuple,
    )
    progress_placeholder.empty()

    if res.get("status") == "error":
        st.error(f"❌ Analysis Execution Failed: {res.get('detail')}")
    else:
        st.success(f"✅ Multimodal Analysis Completed Successfully (Report ID: {res.get('report_id')})")
        st.session_state["latest_result"] = res


# -------------------------------------------------------------
# 8. Step 6, 7, 8: Results Dashboard, Interactive Grad-CAM & PDF Report
# -------------------------------------------------------------
if "latest_result" in st.session_state:
    data = st.session_state["latest_result"]
    mods_processed = data.get("modalities_processed", [])
    exp = data.get("explainability", {})

    # Dedicated Results Hero Header
    st.markdown(f"""
    <div class="results-hero-header">
        <div class="results-hero-title">📊 PATIENT ANALYSIS COMPLETE</div>
        <div class="results-hero-meta">
            <span><b>Patient ID:</b> {data['patient_id']}</span>
            <span><b>Report ID:</b> {data['report_id']}</span>
            <span><b>Timestamp:</b> {data['timestamp']}</span>
            <span><b>Modalities Analyzed:</b> {', '.join([m.upper() for m in mods_processed]) or 'Clinical Only'}</span>
            <span><b>Analysis Status:</b> <span style="color: #34D399; font-weight: 700;">✓ Completed</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Overall Evaluation Risk Level
    # ---------------------------------------------------------
    overall_risk = data.get("overall_risk_level", "LOW").upper()
    if "HIGH" in overall_risk:
        risk_box_cls = "overall-risk-high"
        risk_icon = "🚨"
        risk_desc = "High multimodal risk indicator detected across disease heads. Prompt clinical evaluation recommended."
    elif "MODERATE" in overall_risk:
        risk_box_cls = "overall-risk-mod"
        risk_icon = "⚠️"
        risk_desc = "Moderate risk indicator observed. Secondary follow-up and monitoring suggested."
    else:
        risk_box_cls = "overall-risk-low"
        risk_icon = "🛡️"
        risk_desc = "Low overall risk indicators detected based on submitted retinal imaging and clinical profile."

    st.markdown(f"""
    <div class="overall-risk-box {risk_box_cls}">
        <div>
            <span style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.02em;">{risk_icon} OVERALL RISK LEVEL: {overall_risk} RISK</span>
            <div style="font-size: 0.85rem; margin-top: 0.2rem;">{risk_desc}</div>
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; text-align: right;">
            Deterministic Multi-Task Synthesis
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Multi-Task Prediction Cards (Stroke & Alzheimer's)
    # ---------------------------------------------------------
    st.markdown("### <span class='step-badge'>STEP 6</span> DISEASE PREDICTION & UNCERTAINTY SCORECARDS", unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st_pred = data["stroke_prediction"]
        st_unc = data["stroke_uncertainty"]
        st_risk_cat = st_pred["risk_category"].upper()
        risk_cls_st = "risk-high" if "HIGH" in st_risk_cat else ("risk-mod" if "MODERATE" in st_risk_cat else "risk-low")
        pred_label_st = "Positive (Risk Indicator Present)" if st_pred["predicted_class"] == 1 else "Negative (No Indicator)"

        st.markdown(f"""
        <div class="med-card-highlight">
            <div class="card-title">🧠 STROKE RISK ASSESSMENT</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">
                {st_pred['probability']*100:.1f}% <span style="font-size: 1rem; font-weight: 600; color: #64748B;">Probability</span>
            </div>
            <div style="margin-bottom: 0.85rem;">
                <span class="risk-badge {risk_cls_st}">{st_risk_cat}</span>
                <span style="margin-left: 0.5rem; font-weight: 600; font-size: 0.9rem; color: #334155;">Prediction: {pred_label_st}</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0.75rem 0;" />
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <b>Model Confidence (Phase 9):</b> {st_unc['confidence_percent']:.2f}% ({st_unc['confidence_level']})<br/>
                <b>Predictive Uncertainty:</b> σ² = {st_unc['predictive_variance']:.4f} ({st_unc['uncertainty_level']})<br/>
                <b>Predictive Shannon Entropy:</b> H(p) = {st_unc['predictive_entropy']:.4f} nats
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_p2:
        al_pred = data["alzheimer_prediction"]
        al_unc = data["alzheimer_uncertainty"]
        al_risk_cat = al_pred["risk_category"].upper()
        risk_cls_al = "risk-high" if "HIGH" in al_risk_cat else ("risk-mod" if "MODERATE" in al_risk_cat else "risk-low")
        pred_label_al = "Positive (Risk Indicator Present)" if al_pred["predicted_class"] == 1 else "Negative (No Indicator)"

        st.markdown(f"""
        <div class="med-card-highlight">
            <div class="card-title">🧬 ALZHEIMER'S DISEASE RISK ASSESSMENT</div>
            <div style="font-size: 2.1rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">
                {al_pred['probability']*100:.1f}% <span style="font-size: 1rem; font-weight: 600; color: #64748B;">Probability</span>
            </div>
            <div style="margin-bottom: 0.85rem;">
                <span class="risk-badge {risk_cls_al}">{al_risk_cat}</span>
                <span style="margin-left: 0.5rem; font-weight: 600; font-size: 0.9rem; color: #334155;">Prediction: {pred_label_al}</span>
            </div>
            <hr style="border: 0; border-top: 1px solid #E2E8F0; margin: 0.75rem 0;" />
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <b>Model Confidence (Phase 9):</b> {al_unc['confidence_percent']:.2f}% ({al_unc['confidence_level']})<br/>
                <b>Predictive Uncertainty:</b> σ² = {al_unc['predictive_variance']:.4f} ({al_unc['uncertainty_level']})<br/>
                <b>Predictive Shannon Entropy:</b> H(p) = {al_unc['predictive_entropy']:.4f} nats
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Retinal Image Quality Assessment (Phase 3)
    # ---------------------------------------------------------
    st.markdown("### 🔍 RETINAL IMAGE QUALITY ASSESSMENT (PHASE 3)")
    q_mods = [m for m in ("octa", "octb", "fundus") if data.get("image_quality", {}).get(m, {}).get("available")]

    if q_mods:
        q_cols = st.columns(len(q_mods))
        for idx, mod in enumerate(q_mods):
            q_info = data["image_quality"][mod]
            with q_cols[idx]:
                score = q_info.get("quality_score", 0.0)
                dec = q_info.get("decision", "ACCEPT")
                metrics = q_info.get("metrics", {})
                
                st.markdown(f"""
                <div class="med-card">
                    <div class="card-title">🔬 {mod.upper()} SCAN QUALITY</div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #0F172A;">
                        {score:.1f} <span style="font-size: 0.85rem; color: #64748B;">/ 100</span>
                    </div>
                    <div style="margin: 0.4rem 0;">
                        <span class="risk-badge {'risk-low' if dec=='ACCEPT' else 'risk-high'}">Decision: {dec}</span>
                    </div>
                    <div style="font-size: 0.78rem; color: #475569; margin-top: 0.4rem;">
                        <b>Sharpness/Blur:</b> {metrics.get('sharpness', metrics.get('blur', '—'))}<br/>
                        <b>Brightness:</b> {metrics.get('brightness', '—')}<br/>
                        <b>Contrast:</b> {metrics.get('contrast', '—')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No retinal scan quality data available for this session.")

    # ---------------------------------------------------------
    # Pipeline Execution Matrix (Phases 2 - 11)
    # ---------------------------------------------------------
    st.markdown("##### ⚙️ Pipeline Execution Verification (Phases 2 through 11)")
    p_cols = st.columns(10)
    phases = [
        ("P2", "Preprocessing"),
        ("P3", "Image Quality"),
        ("P4", "Swin-T"),
        ("P5", "Retinal Fusion"),
        ("P6", "Clinical Trans"),
        ("P7", "Cross-Attn"),
        ("P8", "Disease Pred"),
        ("P9", "Uncertainty"),
        ("P10", "Explainability"),
        ("P11", "Report Synthesis"),
    ]
    for idx, (p_code, p_name) in enumerate(phases):
        with p_cols[idx]:
            st.markdown(f"<div class='phase-badge-done'>✓ {p_code}<br/><span style='font-size:0.65rem;'>{p_name}</span></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Step 7: Interactive Model Explainability (Grad-CAM + SHAP)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### <span class='step-badge'>STEP 7</span> MODEL EXPLAINABILITY", unsafe_allow_html=True)
    st.markdown("#### 🖼️ RETINAL HEATMAP / GRAD-CAM")
    st.markdown("*Highlighted regions show areas that contributed to the model prediction.*")

    # Attention Legend Bar
    st.markdown("""
    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 0.85rem 1.25rem; margin-bottom: 1.25rem;">
        <div style="font-size: 0.82rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem;">
            MODEL ATTENTION INTENSITY SPECTRUM
        </div>
        <div class="attention-legend-bar"></div>
        <div class="attention-legend-labels">
            <span>LOW MODEL ATTENTION</span>
            <span>MODERATE ATTENTION</span>
            <span>HIGH MODEL ATTENTION</span>
        </div>
        <div style="font-size: 0.76rem; color: #64748B; margin-top: 0.4rem; font-style: italic;">
            ℹ️ Highlighted regions represent areas receiving higher model attention/contribution according to the Grad-CAM method.
            AI-generated explainability visualization — does not independently diagnose disease.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for Disease Targets
    tab_exp_stroke, tab_exp_alz, tab_shap = st.tabs([
        "🧠 Stroke Saliency (Grad-CAM)",
        "🧬 Alzheimer's Saliency (Grad-CAM)",
        "📈 Clinical Feature Importance (SHAP)",
    ])

    with tab_exp_stroke:
        st_gcam_data = exp.get("stroke", {}).get("gradcam", {})
        available_stroke_mods = [m for m in ("octa", "octb", "fundus") if st_gcam_data.get(m, {}).get("status") == "SUCCESS"]

        if available_stroke_mods:
            for mod in available_stroke_mods:
                m_info = st_gcam_data[mod]
                st.markdown(f"#### {mod.upper()} Scan Attribution (Stroke Model Logits)")

                # View Mode Selector Controls
                view_mode = st.radio(
                    f"Select {mod.upper()} Visualization View:",
                    options=["Side-by-Side (Original vs Overlay)", "Original Retinal Scan", "Grad-CAM Heatmap", "Alpha-Blended Overlay", "3-Panel Overview"],
                    horizontal=True,
                    key=f"vm_st_{mod}",
                )

                orig_p = m_info.get("original_path")
                heat_p = m_info.get("heatmap_path")
                over_p = m_info.get("overlay_path")
                panel_p = m_info.get("panel_path")

                if view_mode == "Side-by-Side (Original vs Overlay)":
                    c_s1, c_s2 = st.columns(2)
                    with c_s1:
                        if orig_p and Path(orig_p).exists():
                            st.image(orig_p, caption=f"Original {mod.upper()} Scan", use_container_width=True)
                        elif panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption="Original Scan (From Panel)", use_container_width=True)
                    with c_s2:
                        if over_p and Path(over_p).exists():
                            st.image(over_p, caption=f"{mod.upper()} Grad-CAM Overlay — Model Attention", use_container_width=True)
                        elif panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption="Overlay (From Panel)", use_container_width=True)

                elif view_mode == "Original Retinal Scan":
                    if orig_p and Path(orig_p).exists():
                        st.image(orig_p, caption=f"Original {mod.upper()} Retinal Scan", use_container_width=True)
                    else:
                        st.info("Original image component not separately cached.")

                elif view_mode == "Grad-CAM Heatmap":
                    if heat_p and Path(heat_p).exists():
                        st.image(heat_p, caption=f"{mod.upper()} Grad-CAM Activation Heatmap", use_container_width=True)
                    else:
                        st.info("Heatmap component not separately cached.")

                elif view_mode == "Alpha-Blended Overlay":
                    if over_p and Path(over_p).exists():
                        st.image(over_p, caption=f"{mod.upper()} Grad-CAM Overlay — Attentional Highlights", use_container_width=True)
                    else:
                        st.info("Overlay component not separately cached.")

                elif view_mode == "3-Panel Overview":
                    if panel_p and Path(panel_p).exists():
                        st.image(panel_p, caption=f"{mod.upper()} [Original Scan | Grad-CAM Heatmap | Alpha Overlay]", use_container_width=True)
                    else:
                        st.info("3-panel figure not available.")

                st.caption(f"ℹ️ {mod.upper()} Model Attention Visualization — Highlights retinal regions contributing to Stroke prediction.")
                st.markdown("---")
        else:
            st.info("No retinal Grad-CAM overlays generated for Stroke target.")

    with tab_exp_alz:
        al_gcam_data = exp.get("alzheimer", {}).get("gradcam", {})
        available_alz_mods = [m for m in ("octa", "octb", "fundus") if al_gcam_data.get(m, {}).get("status") == "SUCCESS"]

        if available_alz_mods:
            for mod in available_alz_mods:
                m_info = al_gcam_data[mod]
                st.markdown(f"#### {mod.upper()} Scan Attribution (Alzheimer's Model Logits)")

                # View Mode Selector Controls
                view_mode_al = st.radio(
                    f"Select {mod.upper()} Visualization View:",
                    options=["Side-by-Side (Original vs Overlay)", "Original Retinal Scan", "Grad-CAM Heatmap", "Alpha-Blended Overlay", "3-Panel Overview"],
                    horizontal=True,
                    key=f"vm_al_{mod}",
                )

                orig_p = m_info.get("original_path")
                heat_p = m_info.get("heatmap_path")
                over_p = m_info.get("overlay_path")
                panel_p = m_info.get("panel_path")

                if view_mode_al == "Side-by-Side (Original vs Overlay)":
                    c_a1, c_a2 = st.columns(2)
                    with c_a1:
                        if orig_p and Path(orig_p).exists():
                            st.image(orig_p, caption=f"Original {mod.upper()} Scan", use_container_width=True)
                        elif panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption="Original Scan (From Panel)", use_container_width=True)
                    with c_a2:
                        if over_p and Path(over_p).exists():
                            st.image(over_p, caption=f"{mod.upper()} Grad-CAM Overlay — Model Attention", use_container_width=True)
                        elif panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption="Overlay (From Panel)", use_container_width=True)

                elif view_mode_al == "Original Retinal Scan":
                    if orig_p and Path(orig_p).exists():
                        st.image(orig_p, caption=f"Original {mod.upper()} Retinal Scan", use_container_width=True)
                    else:
                        st.info("Original image component not separately cached.")

                elif view_mode_al == "Grad-CAM Heatmap":
                    if heat_p and Path(heat_p).exists():
                        st.image(heat_p, caption=f"{mod.upper()} Grad-CAM Activation Heatmap", use_container_width=True)
                    else:
                        st.info("Heatmap component not separately cached.")

                elif view_mode_al == "Alpha-Blended Overlay":
                    if over_p and Path(over_p).exists():
                        st.image(over_p, caption=f"{mod.upper()} Grad-CAM Overlay — Attentional Highlights", use_container_width=True)
                    else:
                        st.info("Overlay component not separately cached.")

                elif view_mode_al == "3-Panel Overview":
                    if panel_p and Path(panel_p).exists():
                        st.image(panel_p, caption=f"{mod.upper()} [Original Scan | Grad-CAM Heatmap | Alpha Overlay]", use_container_width=True)
                    else:
                        st.info("3-panel figure not available.")

                st.caption(f"ℹ️ {mod.upper()} Model Attention Visualization — Highlights retinal regions contributing to Alzheimer's prediction.")
                st.markdown("---")
        else:
            st.info("No retinal Grad-CAM overlays generated for Alzheimer's target.")

    with tab_shap:
        st.markdown("#### CLINICAL FEATURE IMPORTANCE (GAME-THEORETIC SHAP)")
        st.markdown("*Quantifies marginal contributions of tabular clinical biomarkers to model log-odds.*")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("##### Stroke Clinical Risk Factors")
            st_shap_plot = exp.get("stroke", {}).get("shap_plot_path")
            if st_shap_plot and Path(st_shap_plot).exists():
                st.image(st_shap_plot, caption="Stroke SHAP Attributions Bar Chart", use_container_width=True)
            st_shap_items = exp.get("stroke", {}).get("shap_clinical", [])
            if st_shap_items:
                st.dataframe(st_shap_items, use_container_width=True)

        with col_s2:
            st.markdown("##### Alzheimer's Clinical Risk Factors")
            al_shap_plot = exp.get("alzheimer", {}).get("shap_plot_path")
            if al_shap_plot and Path(al_shap_plot).exists():
                st.image(al_shap_plot, caption="Alzheimer's SHAP Attributions Bar Chart", use_container_width=True)
            al_shap_items = exp.get("alzheimer", {}).get("shap_clinical", [])
            if al_shap_items:
                st.dataframe(al_shap_items, use_container_width=True)

    # ---------------------------------------------------------
    # Step 8: Clinical Assessment Report (Phase 11) & PDF Download
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### <span class='step-badge'>STEP 8</span> CLINICAL ASSESSMENT REPORT (PHASE 11)", unsafe_allow_html=True)
    st.info(f"**Multimodal Narrative Summary:**\n\n{data.get('clinical_summary', '')}")

    pdf_bytes = client.download_pdf_report(data["report_id"])
    if pdf_bytes:
        clean_pid = sanitize_patient_id(data["patient_id"])
        st.download_button(
            label="📄 DOWNLOAD CLINICAL REPORT (PDF — NO HEATMAPS)",
            data=pdf_bytes,
            file_name=f"patient_{clean_pid}_clinical_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        st.caption("ℹ️ *Note: Per clinical documentation standards, the generated PDF report contains structured clinical scorecards, SHAP factor attributions, and synthesis, without embedding raw Grad-CAM heatmaps.*")

    # Mandatory Safety Disclaimer Banner
    st.markdown("---")
    st.warning(f"⚠️ **RESEARCH DISCLAIMER:** {data.get('disclaimer', '')}")
