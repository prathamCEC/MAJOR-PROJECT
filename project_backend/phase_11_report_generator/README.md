# Phase 11 — Clinical-Style Assessment Report Generator & PDF Generation

## 1. Overview & Purpose
Phase 11 is the **Final Reporting & Document Generation Layer** of the Multimodal Retinal AI System. It ingests the validated multimodal predictions from **Phase 8**, uncertainty/confidence estimations from **Phase 9**, and visual/tabular attributions from **Phase 10**, synthesizing them into a multi-page **Clinical-Style PDF Report** alongside a complete machine-readable **JSON artifact** and aggregate research CSV summary.

> [!WARNING]
> **RESEARCH PROTOTYPE NOTICE**:
> All generated reports, risk categories, and visual attributions are experimental AI research outputs intended solely for investigational and decision-support purposes. They **DO NOT** constitute a confirmed clinical diagnosis and must not replace evaluation by a licensed healthcare professional.

```text
       Phase 8 (Predictions) ──┐
       Phase 9 (Uncertainty) ──┼──→ Phase 11 Report Generator ──┬──→ PDF Report (Multi-page)
       Phase 10 (Grad-CAM)   ──┤                                 ├──→ JSON Schema (API/Web)
       Phase 10 (SHAP)       ──┘                                 └──→ Summary CSV Log
```

---

## 2. Core Architectural Components

### A. Report Schema & Data Container (`report_data.py`)
- Standardizes patient demographics, technical image quality results (Phase 3), dual multi-task disease evaluations (Stroke and Alzheimer's Disease), and explainability visualizations into a structured `ClinicalReportData` dataclass.
- Performs mathematical validation on probabilities ($p \in [0, 1]$), confidence scores, and finite variance/entropy metrics.

### B. Research Risk Categorization & Confidence Mapping (`risk_calculator.py`)
- Maps model predicted probabilities to configurable research risk categories:
  - **LOW RISK**: $p < 0.35$
  - **MODERATE RISK**: $0.35 \le p < 0.65$
  - **HIGH RISK**: $p \ge 0.65$
- Evaluates confidence categories: `HIGH CONFIDENCE` ($\ge 80\%$), `MODERATE CONFIDENCE` ($60\text{--}80\%$), `LOW CONFIDENCE` ($< 60\%$).

### C. Clinical PDF Report Generator (`pdf_generator.py`)
- Uses ReportLab Platypus to render multi-page documents with running headers, dynamic "Page X of Y" pagination, color-coded scorecard tables, scaled Grad-CAM heatmaps, and SHAP bar charts.

### D. JSON & Summary Exporter (`json_generator.py`)
- Exports exact machine-readable JSON files (`report_<patient_id>_<report_id>.json`) for downstream web frontend consumption.

---

## 3. How to Run Phase 11

Execute from `project_backend/`:

### 1. View Report Configuration Summary:
```powershell
python -m phase_11_report_generator.main summary
```

### 2. Generate Patient Report (PDF + JSON):
```powershell
python -m phase_11_report_generator.main generate `
    --patient-id PATIENT_01 `
    --octa datasets/approved/octa/octa_sample_1_processed.png
```

---

## 4. Running Automated Tests

Run the complete Phase 11 test suite:
```powershell
python -m pytest phase_11_report_generator/tests/ -v
```
