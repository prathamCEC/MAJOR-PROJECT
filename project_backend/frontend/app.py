"""
AI-Based Multimodal System for Stroke and Alzheimer's Detection.
Professional Medical AI Research Web Application.

Enforces strict authentication routing:
LOGIN -> DASHBOARD -> PATIENTS -> CLINICAL DATA -> UPLOAD -> ANALYSE -> RESULTS -> REPORT
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import re
import streamlit as st
from PIL import Image

from utils.api_client import RetinalAIClient

# -------------------------------------------------------------
# 1. Page Configuration & Professional Medical AI Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Retinal AI — Multimodal Stroke & Alzheimer's System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Global Layout & Medical Neutral Palette */
    .main {
        background-color: #F8FAFC;
    }
    .block-container {
        padding-top: 1.8rem;
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
    
    /* Medical Cards & Containers */
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
    
    /* Metric KPI Banners */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.2rem;
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

    /* Pipeline Status Badges */
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

    /* Clinician User Card */
    .user-profile-box {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def sanitize_patient_id(pid: str) -> str:
    """Sanitize patient ID for safe filenames."""
    return re.sub(r"[^\w\-]", "_", pid)


# -------------------------------------------------------------
# 2. State & Client Initialization
# -------------------------------------------------------------
if "auth_token" not in st.session_state:
    st.session_state["auth_token"] = None
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Dashboard"
if "selected_patient_code" not in st.session_state:
    st.session_state["selected_patient_code"] = None

# Backend Connection
backend_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")
client = RetinalAIClient(base_url=backend_url)

health_res = client.check_health()
is_backend_online = health_res.get("status") == "ok"


# -------------------------------------------------------------
# 3. Session Validation & Auto-Logout on Token Expiration
# -------------------------------------------------------------
if st.session_state["auth_token"]:
    me_res = client.get_me(st.session_state["auth_token"])
    if me_res.get("status") == "unauthorized":
        st.session_state["auth_token"] = None
        st.session_state["current_user"] = None
        st.warning("⚠️ Your clinical session has expired. Please log in again to continue.")
        st.rerun()
    elif me_res.get("status") == "ok":
        st.session_state["current_user"] = me_res.get("data")


# =============================================================
# 4. UNAUTHENTICATED FLOW: LOGIN / REGISTER (Section 18)
# When unauthenticated: LOGIN PAGE MUST APPEAR FIRST.
# Redirect all protected views to LOGIN.
# =============================================================
if not st.session_state["auth_token"]:
    st.markdown('<div class="badge-research">SECURE CLINICAL AUTHENTICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">RETINAL AI RESEARCH PORTAL</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Multimodal Diagnostic Decision Support for Stroke & Alzheimer\'s Disease</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-disclaimer">🔒 <b>Protected Access:</b> This research decision-support system requires authorized clinician authentication. Please sign in or register to access patient records and diagnostic pipelines.</div>',
        unsafe_allow_html=True,
    )

    if not is_backend_online:
        st.error("🔴 **FastAPI Backend Server is Offline.** Please ensure the backend is running (`python -m uvicorn backend.main:app --reload`).")

    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        tab_login, tab_register = st.tabs(["🔐 Sign In", "📝 Register New Clinician"])

        with tab_login:
            st.markdown("#### Clinician Authentication")
            login_ident = st.text_input("Username or Email", key="login_username")
            login_pass = st.text_input("Password", type="password", key="login_password")

            if st.button("Sign In to Portal", type="primary", use_container_width=True, disabled=not is_backend_online):
                if not login_ident or not login_pass:
                    st.warning("Please enter your username/email and password.")
                else:
                    with st.spinner("Verifying credentials..."):
                        res = client.login(username_or_email=login_ident, password=login_pass)

                    if res.get("status") == "ok":
                        st.session_state["auth_token"] = res["data"]["access_token"]
                        # Fetch profile
                        prof = client.get_me(st.session_state["auth_token"])
                        if prof.get("status") == "ok":
                            st.session_state["current_user"] = prof["data"]
                        st.session_state["active_nav"] = "Dashboard"
                        st.success("✓ Authentication successful! Redirecting to Clinical Dashboard...")
                        st.rerun()
                    elif res.get("status") == "rate_limited":
                        st.error(f"🚨 {res.get('detail')}")
                    else:
                        st.error(f"❌ {res.get('detail')}")

        with tab_register:
            st.markdown("#### New Account Registration")
            reg_name = st.text_input("Full Name (e.g. Dr. Jane Doe)", key="reg_name")
            reg_email = st.text_input("Email Address", key="reg_email")
            reg_username = st.text_input("Desired Username", key="reg_username")
            reg_pass = st.text_input("Password (min 8 chars, mixed case, numbers, special symbol)", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

            if st.button("Register Clinician Account", use_container_width=True, disabled=not is_backend_online):
                if not all([reg_name, reg_email, reg_username, reg_pass]):
                    st.warning("Please fill in all registration fields.")
                elif reg_pass != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    with st.spinner("Registering account..."):
                        reg_res = client.register(
                            email=reg_email,
                            username=reg_username,
                            password=reg_pass,
                            full_name=reg_name,
                        )
                    if reg_res.get("status") == "ok":
                        st.success("✅ Account registered successfully! You may now sign in using your credentials.")
                    elif reg_res.get("status") == "rate_limited":
                        st.error(f"🚨 {reg_res.get('detail')}")
                    else:
                        st.error(f"❌ {reg_res.get('detail')}")

    # Prevent rendering of any protected views
    st.stop()


# =============================================================
# 5. AUTHENTICATED USER NAVIGATION & SIDEBAR
# =============================================================
token = st.session_state["auth_token"]
current_user = st.session_state["current_user"] or {}

# Sidebar Clinician Profile Card
st.sidebar.markdown(f"""
<div class="user-profile-box">
    <div style="font-size: 0.78rem; font-weight: 700; color: #64748B;">AUTHENTICATED CLINICIAN</div>
    <div style="font-size: 1rem; font-weight: 800; color: #0F172A;">{current_user.get('full_name', 'Clinician')}</div>
    <div style="font-size: 0.78rem; color: #475569;">@{current_user.get('username', 'user')} • <b>{current_user.get('role', 'USER')}</b></div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    options=["Dashboard", "Patients", "New Analysis", "Results", "Reports", "Profile & Security"],
    index=["Dashboard", "Patients", "New Analysis", "Results", "Reports", "Profile & Security"].index(
        st.session_state.get("active_nav", "Dashboard")
    ),
    key="nav_radio",
)
st.session_state["active_nav"] = nav_selection

# Logout Action
if st.sidebar.button("🚪 Logout / Sign Out", use_container_width=True):
    client.logout(token)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
# Backend & AI Model Diagnostics
st.sidebar.markdown("### ⚙️ SYSTEM DIAGNOSTICS")
if is_backend_online:
    device_info = health_res.get("device", "cpu").upper()
    st.sidebar.success(f"🟢 Backend Online ({device_info})")
else:
    st.sidebar.error("🔴 Backend Unavailable")

model_status = client.check_model_status()
if is_backend_online and model_status.get("status") != "error":
    st.sidebar.markdown(f"- **Swin Transformer:** `✓ ({model_status.get('phase4_octa')})`")
    st.sidebar.markdown(f"- **Clinical FT-Transformer:** `✓ ({model_status.get('phase6')})`")
    st.sidebar.markdown(f"- **Multi-Task Classifier:** `✓ ({model_status.get('phase8')})`")
    st.sidebar.markdown(f"- **MC Dropout Engine:** `✓ ({model_status.get('phase9')})`")
    st.sidebar.markdown(f"- **Grad-CAM + SHAP:** `✓ ({model_status.get('phase10')})`")
    st.sidebar.markdown(f"- **Report PDF Engine:** `✓ ({model_status.get('phase11')})`")

st.sidebar.caption("Multimodal Medical AI System v1.0.0 — Research Build")


# =============================================================
# 6. VIEW 1: CLINICAL DASHBOARD (Section 19)
# =============================================================
if nav_selection == "Dashboard":
    st.markdown('<div class="badge-research">CLINICAL OVERVIEW DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">DIAGNOSTIC SURVEILLANCE DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Assisted Stroke and Alzheimer\'s Disease Retinal Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-disclaimer">⚠️ <b>Research Decision-Support Notice:</b> This AI prototype evaluates multimodal retinal imaging and clinical biomarkers for investigative research. It does not provide an independent clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )

    # Fetch Real Counts
    all_patients = client.list_patients(token)
    all_analyses = client.list_analyses(token)
    all_reports = client.list_reports(token)

    high_risk_count = sum(1 for a in all_analyses if a.get("overall_risk_level") == "HIGH")

    # KPI Metrics Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value">{len(all_patients)}</div>
            <div class="kpi-label">Registered Patients</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value">{len(all_analyses)}</div>
            <div class="kpi-label">AI Analyses Completed</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value" style="color: #DC2626;">{high_risk_count}</div>
            <div class="kpi-label">High-Risk Flagged Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-value" style="color: #2563EB;">{len(all_reports)}</div>
            <div class="kpi-label">Generated Reports (PDF)</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Action Buttons
    st.markdown("---")
    st.markdown("### ⚡ QUICK CLINICAL ACTIONS")
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("🔬 START NEW ANALYSIS", type="primary", use_container_width=True):
            st.session_state["active_nav"] = "New Analysis"
            st.rerun()
    with qa2:
        if st.button("🩺 REGISTER NEW PATIENT", use_container_width=True):
            st.session_state["active_nav"] = "Patients"
            st.rerun()
    with qa3:
        if st.button("📋 VIEW CLINICAL REPORTS", use_container_width=True):
            st.session_state["active_nav"] = "Reports"
            st.rerun()

    # Recent Analyses Table
    st.markdown("---")
    st.markdown("### 📊 RECENT DIAGNOSTIC SESSIONS")
    if all_analyses:
        table_rows = []
        for a in all_analyses[:15]:
            risk = a.get("overall_risk_level", "N/A")
            st_prob = f"{a['stroke_probability']*100:.1f}%" if a.get("stroke_probability") is not None else "—"
            al_prob = f"{a['alzheimer_probability']*100:.1f}%" if a.get("alzheimer_probability") is not None else "—"
            created = a.get("created_at", "")[:19].replace("T", " ")
            table_rows.append({
                "Session UUID": a.get("session_uuid", "")[:8] + "...",
                "Patient Code": a.get("patient_code", "UNKNOWN"),
                "Modalities": a.get("modalities_requested", "").upper(),
                "Overall Risk": risk,
                "Stroke Prob": st_prob,
                "Alzheimer Prob": al_prob,
                "Timestamp": created,
                "Report ID": a.get("report_id", "—"),
            })
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.info("No prior analysis sessions found. Click 'Start New Analysis' to run your first evaluation.")


# =============================================================
# 7. VIEW 2: PATIENTS DIRECTORY (Section 20 requirement 1)
# =============================================================
elif nav_selection == "Patients":
    st.markdown('<div class="badge-research">PATIENT DIRECTORY & HEALTH RECORDS</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">PATIENT CLINICAL MANAGEMENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Register, search, and maintain patient biomarker profiles</div>', unsafe_allow_html=True)

    # Form to register new patient
    with st.expander("➕ Register New Patient Profile", expanded=False):
        with st.form("form_create_patient", clear_on_submit=True):
            cp1, cp2, cp3 = st.columns(3)
            with cp1:
                new_pcode = st.text_input("Patient Code / Identifier*", value="PATIENT_NEW_01")
                new_name = st.text_input("Full Name", value="")
                new_cohort = st.selectbox("Age / Cognitive Cohort", options=["O_CD", "Y_CD", "O_CU", "Y_CU"])
                new_gender = st.radio("Biological Gender", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", horizontal=True)
            with cp2:
                new_edu = st.number_input("Formal Education (Years)", min_value=0.0, max_value=30.0, value=16.0, step=1.0)
                new_bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=70.0, value=26.5, step=0.1)
                new_obese = st.selectbox("Obesity Classification", options=[0.0, 1.0], format_func=lambda x: "Non-Obese (0)" if x == 0.0 else "Obese (1)")
                new_htn = st.selectbox("Hypertension (HTN)", options=[1, 0], format_func=lambda x: "Positive (1)" if x == 1 else "Negative (0)")
            with cp3:
                new_dm2 = st.selectbox("Type 2 Diabetes (DM2)", options=[0, 1], format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")
                new_smk_ev = st.selectbox("Smoking History (Ever)", options=[1, 0], format_func=lambda x: "Ever Smoked (1)" if x == 1 else "Never (0)")
                new_smk_cur = st.selectbox("Current Smoker", options=[0, 1], format_func=lambda x: "Non-Smoker (0)" if x == 0 else "Active Smoker (1)")
                new_et_ev = st.selectbox("Alcohol Consumption (Ever)", options=[1, 0], format_func=lambda x: "Ever (1)" if x == 1 else "Never (0)")
                new_et_cur = st.selectbox("Current Alcohol", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Active (1)")

            submit_patient = st.form_submit_button("Save Patient Profile", type="primary", use_container_width=True)
            if submit_patient:
                if not new_pcode.strip():
                    st.warning("Patient code is required.")
                else:
                    payload = {
                        "patient_code": new_pcode.strip().upper(),
                        "full_name": new_name.strip() or None,
                        "age_group": new_cohort,
                        "gender": new_gender,
                        "education_years": new_edu,
                        "bmi": new_bmi,
                        "obese": new_obese,
                        "hypertension": new_htn,
                        "diabetes_type2": new_dm2,
                        "smoking_ever": new_smk_ev,
                        "smoking_current": new_smk_cur,
                        "alcohol_ever": new_et_ev,
                        "alcohol_current": new_et_cur,
                    }
                    p_res = client.create_patient(token, payload)
                    if p_res.get("status") == "ok":
                        st.success(f"✓ Patient '{new_pcode.strip().upper()}' registered successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ {p_res.get('detail')}")

    # Search and Directory
    search_q = st.text_input("🔍 Search Patient by Code or Name", "")
    patients_list = client.list_patients(token, search=search_q if search_q else None)

    if patients_list:
        st.markdown(f"**Found {len(patients_list)} patient record(s):**")
        for p in patients_list:
            with st.container():
                pc1, pc2, pc3, pc4 = st.columns([2, 3, 3, 2])
                with pc1:
                    st.markdown(f"**`{p['patient_code']}`**")
                    st.caption(p.get("full_name") or "No name specified")
                with pc2:
                    st.markdown(f"**Cohort:** {p['age_group']} | **Gender:** {'Male' if p['gender']==1 else 'Female'}")
                    st.caption(f"BMI: {p['bmi']:.1f} • Education: {p['education_years']} yrs")
                with pc3:
                    st.markdown(f"**HTN:** {'Positive' if p['hypertension']==1 else 'Negative'} | **DM2:** {'Positive' if p['diabetes_type2']==1 else 'Negative'}")
                    st.caption(f"Smoking: {'Active' if p['smoking_current']==1 else 'No'} • Alcohol: {'Active' if p['alcohol_current']==1 else 'No'}")
                with pc4:
                    if st.button(f"🔬 Analyze `{p['patient_code']}`", key=f"btn_an_{p['patient_code']}", use_container_width=True):
                        st.session_state["selected_patient_code"] = p["patient_code"]
                        st.session_state["active_nav"] = "New Analysis"
                        st.rerun()
                st.markdown("<hr style='margin:0.5rem 0; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    else:
        st.info("No patient records match the search criteria.")


# =============================================================
# 8. VIEW 3: NEW ANALYSIS PAGE (Section 20)
# Select patient -> Tabular clinical data -> Select modality (OCT-A, OCT-B, Fundus)
# Single modality supported -> Upload & Validate -> Execute Real AI Pipeline
# =============================================================
elif nav_selection == "New Analysis":
    st.markdown('<div class="badge-research">MULTIMODAL DIAGNOSTIC WORKFLOW</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">MULTIMODAL RETINAL INFERENCE PIPELINE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select patient, configure clinical biomarkers, select imaging modality, and run deep learning inference</div>', unsafe_allow_html=True)

    # Step 1: Select / Create Patient
    st.markdown("### <span class='step-badge'>STEP 1</span> SELECT OR SPECIFY PATIENT", unsafe_allow_html=True)

    patients_list = client.list_patients(token)
    p_codes = [p["patient_code"] for p in patients_list]

    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        if p_codes:
            default_idx = 0
            if st.session_state.get("selected_patient_code") in p_codes:
                default_idx = p_codes.index(st.session_state["selected_patient_code"])
            chosen_patient = st.selectbox("Select Existing Patient Record", options=p_codes, index=default_idx)
            patient_id = chosen_patient
        else:
            patient_id = st.text_input("Patient Identifier", value="PATIENT_001")

    # Pre-fill patient variables if found
    patient_record = next((p for p in patients_list if p["patient_code"] == patient_id), None) if p_codes else None

    # Step 2: Clinical Factors Form
    st.markdown("---")
    st.markdown("### <span class='step-badge'>STEP 2</span> CLINICAL BIOMARKERS & RISK FACTORS", unsafe_allow_html=True)
    st.markdown("*Adjust clinical risk factors for the FT-Transformer clinical encoder*")

    with st.expander("📝 Tabular Clinical Variables", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            old_groups = st.selectbox(
                "Age / Cognitive Cohort (Old groups)",
                options=["O_CD", "Y_CD", "O_CU", "Y_CU"],
                index=["O_CD", "Y_CD", "O_CU", "Y_CU"].index(patient_record["age_group"]) if patient_record and patient_record["age_group"] in ["O_CD", "Y_CD", "O_CU", "Y_CU"] else 0,
            )
            gender_val = patient_record["gender"] if patient_record else 1
            gender = st.radio("Biological Gender", options=[1, 0], index=0 if gender_val==1 else 1, format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", horizontal=True)
            education = st.number_input("Formal Education (Years)", min_value=0.0, max_value=30.0, value=float(patient_record["education_years"]) if patient_record else 16.0, step=1.0)
        with col_c2:
            bmi = st.number_input("Body Mass Index (BMI kg/m²)", min_value=10.0, max_value=70.0, value=float(patient_record["bmi"]) if patient_record else 26.5, step=0.1)
            obese_val = float(patient_record["obese"]) if patient_record else 0.0
            obese = st.selectbox("Obesity Classification", options=[0.0, 1.0], index=0 if obese_val==0.0 else 1, format_func=lambda x: "Non-Obese (0)" if x == 0.0 else "Obese (1)")
            htn_val = patient_record["hypertension"] if patient_record else 1
            htn = st.selectbox("Hypertension (HTN)", options=[1, 0], index=0 if htn_val==1 else 1, format_func=lambda x: "Positive (1)" if x == 1 else "Negative (0)")
            dm2_val = patient_record["diabetes_type2"] if patient_record else 0
            dm2 = st.selectbox("Type 2 Diabetes (DM2)", options=[0, 1], index=0 if dm2_val==0 else 1, format_func=lambda x: "Negative (0)" if x == 0 else "Positive (1)")
        with col_c3:
            smk_ev_val = patient_record["smoking_ever"] if patient_record else 1
            smoking_ever = st.selectbox("Smoking History (Ever)", options=[1, 0], index=0 if smk_ev_val==1 else 1, format_func=lambda x: "Ever (1)" if x == 1 else "Never (0)")
            smk_cur_val = patient_record["smoking_current"] if patient_record else 0
            smoking_current = st.selectbox("Current Smoking", options=[0, 1], index=0 if smk_cur_val==0 else 1, format_func=lambda x: "Non-Smoker (0)" if x == 0 else "Active Smoker (1)")
            et_ev_val = patient_record["alcohol_ever"] if patient_record else 1
            etoh_ever = st.selectbox("Alcohol Consumption (Ever)", options=[1, 0], index=0 if et_ev_val==1 else 1, format_func=lambda x: "Ever (1)" if x == 1 else "Never (0)")
            et_cur_val = patient_record["alcohol_current"] if patient_record else 0
            etoh_current = st.selectbox("Current Alcohol", options=[0, 1], index=0 if et_cur_val==0 else 1, format_func=lambda x: "No (0)" if x == 0 else "Active (1)")

    # Step 3: Modality Selection & Upload (Section 20: Select one or more modalities; single modality MUST work!)
    st.markdown("---")
    st.markdown("### <span class='step-badge'>STEP 3</span> RETINAL MODALITY SELECTION & UPLOAD", unsafe_allow_html=True)
    st.info("ℹ️ **Flexible Modality Support:** You may select **one** or more retinal modalities (OCT-A, OCT-B, and/or Fundus). Single-modality evaluation is fully supported; uploading all three is NOT required.")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        select_octa = st.checkbox("🔬 OCT-A (Angiography)", value=True, key="chk_octa")
    with col_m2:
        select_octb = st.checkbox("🔬 OCT-B (Structural B-Scan)", value=False, key="chk_octb")
    with col_m3:
        select_fundus = st.checkbox("👁️ Fundus (Color Retinal Photo)", value=False, key="chk_fundus")

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
        st.warning("⚠️ Please select at least one retinal imaging modality to proceed.")
    else:
        st.markdown(f"**Selected Modality Subsets:** `{', '.join([m.upper() for m in selected_modalities])}`")
        upload_cols = st.columns(len(selected_modalities))
        col_idx = 0

        if select_octa:
            with upload_cols[col_idx]:
                st.markdown("#### OCT-A SCAN")
                octa_file = st.file_uploader("Upload OCT-A Scan", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octa")
                if octa_file:
                    try:
                        img_octa = Image.open(octa_file)
                        st.image(img_octa, caption=f"OCT-A: {octa_file.name}", use_container_width=True)
                        st.success(f"✓ Valid: {img_octa.size[0]}×{img_octa.size[1]} px, {img_octa.mode}")
                    except Exception as e:
                        st.error(f"Invalid image format: {e}")
            col_idx += 1

        if select_octb:
            with upload_cols[col_idx]:
                st.markdown("#### OCT-B SCAN")
                octb_file = st.file_uploader("Upload OCT-B Scan", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_octb")
                if octb_file:
                    try:
                        img_octb = Image.open(octb_file)
                        st.image(img_octb, caption=f"OCT-B: {octb_file.name}", use_container_width=True)
                        st.success(f"✓ Valid: {img_octb.size[0]}×{img_octb.size[1]} px, {img_octb.mode}")
                    except Exception as e:
                        st.error(f"Invalid image format: {e}")
            col_idx += 1

        if select_fundus:
            with upload_cols[col_idx]:
                st.markdown("#### FUNDUS SCAN")
                fundus_file = st.file_uploader("Upload Fundus Scan", type=["png", "jpg", "jpeg", "tif", "tiff"], key="up_fundus")
                if fundus_file:
                    try:
                        img_fundus = Image.open(fundus_file)
                        st.image(img_fundus, caption=f"Fundus: {fundus_file.name}", use_container_width=True)
                        st.success(f"✓ Valid: {img_fundus.size[0]}×{img_fundus.size[1]} px, {img_fundus.mode}")
                    except Exception as e:
                        st.error(f"Invalid image format: {e}")
            col_idx += 1

    # Step 4: Verification & Execution Trigger
    st.markdown("---")
    st.markdown("### <span class='step-badge'>STEP 4</span> EXECUTE MULTIMODAL INFERENCE", unsafe_allow_html=True)

    missing_files = []
    if select_octa and not octa_file:
        missing_files.append("OCT-A")
    if select_octb and not octb_file:
        missing_files.append("OCT-B")
    if select_fundus and not fundus_file:
        missing_files.append("Fundus")

    can_analyze = len(selected_modalities) > 0 and len(missing_files) == 0 and is_backend_online

    btn_analyze = st.button(
        "🔬 START DIAGNOSTIC INFERENCE",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
    )

    if not can_analyze and is_backend_online and len(missing_files) > 0:
        st.info(f"ℹ️ Please upload an image file for selected modality: **{', '.join(missing_files)}** to enable inference.")

    if btn_analyze:
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

        progress_box = st.empty()
        with progress_box.container():
            st.info("⏳ **Executing Multimodal Pipeline...**\n\n"
                    "✓ Staging uploaded retinal scan(s)\n"
                    "✓ Running Phase 2 Image Preprocessing & Phase 3 Quality Assessment\n"
                    "● Computing Swin Transformer patch embeddings\n"
                    "○ Fusing tabular clinical biomarkers via FT-Transformer\n"
                    "○ Evaluating Multi-Task disease prediction & MC-Dropout uncertainty\n"
                    "○ Generating Grad-CAM spatial heatmaps & SHAP attributions\n"
                    "○ Compiling clinical assessment report (PDF & JSON)")

        res = client.run_analysis(
            clinical_data=clinical_payload,
            octa_file=octa_tuple,
            octb_file=octb_tuple,
            fundus_file=fundus_tuple,
            token=token,
        )
        progress_box.empty()

        if res.get("status") == "error":
            st.error(f"❌ Analysis Execution Failed: {res.get('detail')}")
        elif res.get("status") == "unauthorized":
            st.error("Session expired. Please log in again.")
            st.session_state["auth_token"] = None
            st.rerun()
        else:
            st.success(f"✅ Multimodal Analysis Completed Successfully (Report ID: {res.get('report_id')})")
            st.session_state["latest_result"] = res
            st.session_state["active_nav"] = "Results"
            st.rerun()


# =============================================================
# 9. VIEW 4: RESULTS PAGE & WEB GRAD-CAM (Section 21)
# CRITICAL: Grad-CAM heatmaps MUST appear on Web Results Page.
# DO NOT embed Grad-CAM in PDF report.
# =============================================================
elif nav_selection == "Results":
    st.markdown('<div class="badge-research">DIAGNOSTIC OUTCOMES & EXPLAINABILITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">MULTIMODAL DIAGNOSTIC ASSESSMENT RESULTS</div>', unsafe_allow_html=True)

    if "latest_result" not in st.session_state or not st.session_state["latest_result"]:
        st.info("ℹ️ No active diagnostic result in session. Please run a new analysis from the 'New Analysis' page or select a prior session from Dashboard.")
    else:
        data = st.session_state["latest_result"]
        mods_processed = data.get("modalities_processed", [])
        exp = data.get("explainability", {})

        # Results Hero Header
        st.markdown(f"""
        <div class="results-hero-header">
            <div class="results-hero-title">📊 EVALUATION COMPLETE</div>
            <div class="results-hero-meta">
                <span><b>Patient ID:</b> {data.get('patient_id')}</span>
                <span><b>Report ID:</b> {data.get('report_id')}</span>
                <span><b>Session UUID:</b> {data.get('session_id')}</span>
                <span><b>Timestamp:</b> {data.get('timestamp')}</span>
                <span><b>Modalities Processed:</b> {', '.join([m.upper() for m in mods_processed]) or 'Clinical Tabular Only'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Overall Multi-Task Risk Level Callout
        overall_risk = data.get("overall_risk_level", "LOW").upper()
        if "HIGH" in overall_risk:
            risk_box_cls = "overall-risk-high"
            risk_icon = "🚨"
            risk_desc = "High multimodal risk detected across disease heads. Clinical review recommended."
        elif "MODERATE" in overall_risk:
            risk_box_cls = "overall-risk-mod"
            risk_icon = "⚠️"
            risk_desc = "Moderate risk indicators observed. Secondary diagnostic follow-up suggested."
        else:
            risk_box_cls = "overall-risk-low"
            risk_icon = "🛡️"
            risk_desc = "Low overall risk indicators detected based on submitted imaging and clinical factors."

        st.markdown(f"""
        <div class="overall-risk-box {risk_box_cls}">
            <div>
                <span style="font-size: 1.25rem; font-weight: 800; letter-spacing: 0.02em;">{risk_icon} OVERALL MULTI-TASK RISK: {overall_risk}</span>
                <div style="font-size: 0.85rem; margin-top: 0.2rem;">{risk_desc}</div>
            </div>
            <div style="font-size: 0.95rem; font-weight: 700; text-align: right;">
                Deterministic Dual Head Evaluation
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Scorecards: Stroke & Alzheimer's
        st.markdown("### 🎯 DUAL-TARGET PREDICTION SCORECARDS & UNCERTAINTY")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st_pred = data["stroke_prediction"]
            st_unc = data["stroke_uncertainty"]
            st_risk_cat = st_pred["risk_category"].upper()
            risk_cls_st = "risk-high" if "HIGH" in st_risk_cat else ("risk-mod" if "MODERATE" in st_risk_cat else "risk-low")
            pred_label_st = "Positive (Risk Present)" if st_pred["predicted_class"] == 1 else "Negative (No Indicator)"

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
                    <b>Predictive Variance (Uncertainty):</b> σ² = {st_unc['variance'] if 'variance' in st_unc else st_unc.get('predictive_variance', 0.0):.4f}<br/>
                    <b>Predictive Shannon Entropy:</b> H(p) = {st_unc['entropy'] if 'entropy' in st_unc else st_unc.get('predictive_entropy', 0.0):.4f} nats
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            al_pred = data["alzheimer_prediction"]
            al_unc = data["alzheimer_uncertainty"]
            al_risk_cat = al_pred["risk_category"].upper()
            risk_cls_al = "risk-high" if "HIGH" in al_risk_cat else ("risk-mod" if "MODERATE" in al_risk_cat else "risk-low")
            pred_label_al = "Positive (Risk Present)" if al_pred["predicted_class"] == 1 else "Negative (No Indicator)"

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
                    <b>Predictive Variance (Uncertainty):</b> σ² = {al_unc['variance'] if 'variance' in al_unc else al_unc.get('predictive_variance', 0.0):.4f}<br/>
                    <b>Predictive Shannon Entropy:</b> H(p) = {al_unc['entropy'] if 'entropy' in al_unc else al_unc.get('predictive_entropy', 0.0):.4f} nats
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Retinal Image Quality Assessment (Phase 3)
        st.markdown("---")
        st.markdown("### 🔍 RETINAL SCAN TECHNICAL QUALITY (PHASE 3)")
        q_data = data.get("image_quality", {})
        avail_q_mods = [m for m in ("octa", "octb", "fundus") if q_data.get(m, {}).get("available")]
        if avail_q_mods:
            q_cols = st.columns(len(avail_q_mods))
            for idx, mod in enumerate(avail_q_mods):
                q_info = q_data[mod]
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
                            <b>Sharpness:</b> {metrics.get('sharpness', metrics.get('blur', '—'))}<br/>
                            <b>Brightness:</b> {metrics.get('brightness', '—')}<br/>
                            <b>Contrast:</b> {metrics.get('contrast', '—')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No retinal scan quality assessment metrics available.")

        # Pipeline Verification Badges (Phases 2-11)
        st.markdown("##### ⚙️ Pipeline Verification (Phases 2 through 11)")
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
        # Interactive Model Explainability (Grad-CAM + SHAP)
        # CRITICAL: Grad-CAM heatmaps rendered on Web Results Page
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🖼️ INTERACTIVE MODEL EXPLAINABILITY (GRAD-CAM ON WEB)")
        st.markdown("*Highlighted regions show retinal features contributing to deep neural network logits.*")

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
                ℹ️ Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes features influencing Swin Transformer predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_gcam_st, tab_gcam_al, tab_shap = st.tabs([
            "🧠 Stroke Saliency (Grad-CAM)",
            "🧬 Alzheimer's Saliency (Grad-CAM)",
            "📈 Clinical Feature Importance (SHAP)",
        ])

        with tab_gcam_st:
            st_gcam_data = exp.get("stroke", {}).get("gradcam", {})
            available_st_mods = [m for m in ("octa", "octb", "fundus") if st_gcam_data.get(m, {}).get("status") == "SUCCESS"]

            if available_st_mods:
                for mod in available_st_mods:
                    m_info = st_gcam_data[mod]
                    st.markdown(f"#### {mod.upper()} Retinal Scan Attribution — Stroke Target")

                    v_mode = st.radio(
                        f"Visualization View for {mod.upper()}:",
                        options=["Side-by-Side (Original vs Overlay)", "Original Scan", "Heatmap Only", "Overlay Only", "3-Panel Composite"],
                        horizontal=True,
                        key=f"res_vm_st_{mod}",
                    )
                    orig_p = m_info.get("original_path")
                    heat_p = m_info.get("heatmap_path")
                    over_p = m_info.get("overlay_path")
                    panel_p = m_info.get("panel_path")

                    if v_mode == "Side-by-Side (Original vs Overlay)":
                        cs1, cs2 = st.columns(2)
                        with cs1:
                            if orig_p and Path(orig_p).exists():
                                st.image(orig_p, caption=f"Original {mod.upper()} Scan", use_container_width=True)
                            elif panel_p and Path(panel_p).exists():
                                st.image(panel_p, caption="Original (Panel)", use_container_width=True)
                        with cs2:
                            if over_p and Path(over_p).exists():
                                st.image(over_p, caption=f"{mod.upper()} Grad-CAM Attention Overlay", use_container_width=True)
                            elif panel_p and Path(panel_p).exists():
                                st.image(panel_p, caption="Overlay (Panel)", use_container_width=True)
                    elif v_mode == "Original Scan":
                        if orig_p and Path(orig_p).exists():
                            st.image(orig_p, caption=f"Original {mod.upper()} Retinal Scan", use_container_width=True)
                    elif v_mode == "Heatmap Only":
                        if heat_p and Path(heat_p).exists():
                            st.image(heat_p, caption=f"{mod.upper()} Grad-CAM Activation Heatmap", use_container_width=True)
                    elif v_mode == "Overlay Only":
                        if over_p and Path(over_p).exists():
                            st.image(over_p, caption=f"{mod.upper()} Attention Overlay", use_container_width=True)
                    elif v_mode == "3-Panel Composite":
                        if panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption=f"{mod.upper()} Composite [Original | Heatmap | Overlay]", use_container_width=True)

                    st.markdown("---")
            else:
                st.info("No retinal Grad-CAM overlays generated for Stroke target.")

        with tab_gcam_al:
            al_gcam_data = exp.get("alzheimer", {}).get("gradcam", {})
            available_al_mods = [m for m in ("octa", "octb", "fundus") if al_gcam_data.get(m, {}).get("status") == "SUCCESS"]

            if available_al_mods:
                for mod in available_al_mods:
                    m_info = al_gcam_data[mod]
                    st.markdown(f"#### {mod.upper()} Retinal Scan Attribution — Alzheimer's Target")

                    v_mode_al = st.radio(
                        f"Visualization View for {mod.upper()}:",
                        options=["Side-by-Side (Original vs Overlay)", "Original Scan", "Heatmap Only", "Overlay Only", "3-Panel Composite"],
                        horizontal=True,
                        key=f"res_vm_al_{mod}",
                    )
                    orig_p = m_info.get("original_path")
                    heat_p = m_info.get("heatmap_path")
                    over_p = m_info.get("overlay_path")
                    panel_p = m_info.get("panel_path")

                    if v_mode_al == "Side-by-Side (Original vs Overlay)":
                        ca1, ca2 = st.columns(2)
                        with ca1:
                            if orig_p and Path(orig_p).exists():
                                st.image(orig_p, caption=f"Original {mod.upper()} Scan", use_container_width=True)
                            elif panel_p and Path(panel_p).exists():
                                st.image(panel_p, caption="Original (Panel)", use_container_width=True)
                        with ca2:
                            if over_p and Path(over_p).exists():
                                st.image(over_p, caption=f"{mod.upper()} Grad-CAM Attention Overlay", use_container_width=True)
                            elif panel_p and Path(panel_p).exists():
                                st.image(panel_p, caption="Overlay (Panel)", use_container_width=True)
                    elif v_mode_al == "Original Scan":
                        if orig_p and Path(orig_p).exists():
                            st.image(orig_p, caption=f"Original {mod.upper()} Retinal Scan", use_container_width=True)
                    elif v_mode_al == "Heatmap Only":
                        if heat_p and Path(heat_p).exists():
                            st.image(heat_p, caption=f"{mod.upper()} Grad-CAM Activation Heatmap", use_container_width=True)
                    elif v_mode_al == "Overlay Only":
                        if over_p and Path(over_p).exists():
                            st.image(over_p, caption=f"{mod.upper()} Attention Overlay", use_container_width=True)
                    elif v_mode_al == "3-Panel Composite":
                        if panel_p and Path(panel_p).exists():
                            st.image(panel_p, caption=f"{mod.upper()} Composite [Original | Heatmap | Overlay]", use_container_width=True)

                    st.markdown("---")
            else:
                st.info("No retinal Grad-CAM overlays generated for Alzheimer's target.")

        with tab_shap:
            st.markdown("#### CLINICAL BIOMARKER ATTRIBUTIONS (SHAP)")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("##### Stroke Biomarker Importance")
                st_plot = exp.get("stroke", {}).get("shap_plot_path")
                if st_plot and Path(st_plot).exists():
                    st.image(st_plot, caption="Stroke Clinical Factor SHAP Values", use_container_width=True)
                st_items = exp.get("stroke", {}).get("shap_clinical", [])
                if st_items:
                    st.dataframe(st_items, use_container_width=True)
            with col_s2:
                st.markdown("##### Alzheimer's Biomarker Importance")
                al_plot = exp.get("alzheimer", {}).get("shap_plot_path")
                if al_plot and Path(al_plot).exists():
                    st.image(al_plot, caption="Alzheimer's Clinical Factor SHAP Values", use_container_width=True)
                al_items = exp.get("alzheimer", {}).get("shap_clinical", [])
                if al_items:
                    st.dataframe(al_items, use_container_width=True)

        # ---------------------------------------------------------
        # PDF Report Download (Section 22)
        # CRITICAL: DO NOT PUT GRAD-CAM/HEATMAP IN THE PDF REPORT
        # ---------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📄 GENERATE & DOWNLOAD CLINICAL REPORT (PDF)")
        st.info(f"**Clinical Narrative Summary:**\n\n{data.get('clinical_summary', '')}")

        pdf_bytes = client.download_pdf_report(data["report_id"], token=token)
        if pdf_bytes:
            clean_pid = sanitize_patient_id(data.get("patient_id", "patient"))
            st.download_button(
                label="📥 DOWNLOAD CLINICAL REPORT (PDF — NO HEATMAPS)",
                data=pdf_bytes,
                file_name=f"patient_{clean_pid}_{data['report_id']}_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.caption("ℹ️ *Per clinical documentation specifications, the generated PDF report contains validated scorecards, risk assessments, and SHAP factors, strictly without embedding raw Grad-CAM heatmaps.*")

        st.warning(f"⚠️ **RESEARCH DISCLAIMER:** {data.get('disclaimer', 'This software is an investigational academic prototype. Not for primary clinical diagnosis.')}")


# =============================================================
# 10. VIEW 5: REPORTS REPOSITORY
# =============================================================
elif nav_selection == "Reports":
    st.markdown('<div class="badge-research">CLINICAL DOCUMENTATION ARCHIVE</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">GENERATED CLINICAL REPORTS REPOSITORY</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Review, download, and audit past clinical assessment reports</div>', unsafe_allow_html=True)

    reports_list = client.list_reports(token)
    if reports_list:
        st.markdown(f"**Found {len(reports_list)} generated report(s):**")
        for rep in reports_list:
            with st.container():
                rc1, rc2, rc3, rc4 = st.columns([2, 3, 2, 3])
                with rc1:
                    st.markdown(f"**`{rep['report_id']}`**")
                    st.caption(f"Patient: `{rep['patient_code']}`")
                with rc2:
                    risk = rep.get("overall_risk_level", "LOW")
                    risk_cls = "risk-high" if "HIGH" in risk else ("risk-mod" if "MODERATE" in risk else "risk-low")
                    st.markdown(f"<span class='risk-badge {risk_cls}'>{risk} RISK</span>", unsafe_allow_html=True)
                    st.caption(f"Date: {rep.get('created_at', '')[:19].replace('T', ' ')}")
                with rc3:
                    st_p = f"{rep['stroke_probability']*100:.1f}%" if rep.get("stroke_probability") is not None else "—"
                    al_p = f"{rep['alzheimer_probability']*100:.1f}%" if rep.get("alzheimer_probability") is not None else "—"
                    st.caption(f"Stroke: {st_p} • AD: {al_p}")
                with rc4:
                    # Download PDF button
                    pdf_data = client.download_pdf_report(rep["report_id"], token=token)
                    if pdf_data:
                        st.download_button(
                            label=f"📄 Download PDF",
                            data=pdf_data,
                            file_name=f"report_{rep['patient_code']}_{rep['report_id']}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_{rep['report_id']}",
                            use_container_width=True,
                        )
                st.markdown("<hr style='margin:0.5rem 0; border-top:1px solid #E2E8F0;'/>", unsafe_allow_html=True)
    else:
        st.info("No generated clinical reports available yet.")


# =============================================================
# 11. VIEW 6: CLINICIAN PROFILE & SECURITY
# =============================================================
elif nav_selection == "Profile & Security":
    st.markdown('<div class="badge-research">CLINICIAN PROFILE & SECURITY AUDITING</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">CLINICIAN PROFILE & SECURITY STATUS</div>', unsafe_allow_html=True)

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-title">👤 CLINICIAN IDENTITY</div>
            <b>Full Name:</b> {current_user.get('full_name', 'Clinician')}<br/>
            <b>Username:</b> {current_user.get('username', 'user')}<br/>
            <b>Email:</b> {current_user.get('email', 'user@hospital.org')}<br/>
            <b>System Role:</b> <span class="badge-research">{current_user.get('role', 'USER')}</span><br/>
            <b>Account Status:</b> {'Active' if current_user.get('is_active') else 'Inactive'}<br/>
            <b>Registered Since:</b> {current_user.get('created_at', '')[:10]}
        </div>
        """, unsafe_allow_html=True)

    with c_p2:
        st.markdown("""
        <div class="med-card">
            <div class="card-title">🔒 SECURITY & DATA PROTECTION</div>
            <b>Authentication Mechanism:</b> Argon2id Hashing + HS256 JWT Bearer Tokens<br/>
            <b>Session Inactivity Timeout:</b> 60 minutes<br/>
            <b>Rate Limiting:</b> Sliding Window Protection Active<br/>
            <b>Audit Trail:</b> Security & Clinical Activity Logged to Relational Database<br/>
            <b>Security Headers:</b> X-Frame-Options: DENY, X-Content-Type-Options: nosniff
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚠️ ACADEMIC & RESEARCH PROTOCOL DISCLAIMER")
    st.warning(
        "This software is an investigational academic prototype developed for multimodal biomarker and retinal imaging research. "
        "It is NOT certified as a medical device by the FDA or CE and must NOT be used as the sole basis for clinical diagnosis or treatment planning. "
        "Always combine computational predictions with comprehensive clinical diagnostic evaluations."
    )
