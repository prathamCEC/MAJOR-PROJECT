# Phase 10 — Multimodal Explainability (Swin Grad-CAM + SHAP)

## 1. Overview & Purpose
Phase 10 implements the **Multimodal Explainability Suite** providing interpretable visual and game-theoretic attributions for the **Phase 8 Multi-Task Disease Predictions** (**Stroke** and **Alzheimer's Disease**), integrated with **Phase 9 Monte Carlo Dropout uncertainty and confidence estimates**.

> [!WARNING]
> **RESEARCH PROTOTYPE NOTICE**:
> This software is an academic/research explainability engine. Generated heatmaps and SHAP values indicate computational feature contributions to model logits and **MUST NOT** be interpreted as clinical proof of causal disease etiology or medical diagnostic claims.

```text
                                  PATIENT INPUT
                 ┌──────────────────────┴──────────────────────┐
                 ↓                                             ↓
       RETINAL SCANS (OCT-A/B/Fundus)               CLINICAL VARIABLES (Tabular)
                 │                                             │
                 ▼                                             ▼
          SWIN TRANSFORMER                               FT-TRANSFORMER
                 │                                             │
                 ├──────────────────────────────┐              │
                 ▼                              ▼              ▼
           SWIN GRAD-CAM                  CLINICAL SHAP (Feature Attributions)
      (Spatial Retinal Heatmaps)           (Age, BMI, HTN, DM2, Smoking, etc.)
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        │
                                        ▼
                      MULTIMODAL EXPLAINABILITY REPORT
                         (Integrated with Phase 9)
                   ┌─────────────────────────────────────────┐
                   │ • Disease Predictions (Stroke / AD)     │
                   │ • MC-Dropout Uncertainty & Confidence % │
                   │ • Visual Grad-CAM Overlays              │
                   │ • Directional SHAP Feature Importance   │
                   │ • Modality Attribution Percentages      │
                   └─────────────────────────────────────────┘
```

---

## 2. Core Architectural Components

### A. Swin Transformer Grad-CAM (`swin_gradcam.py`)
- Adapts Grad-CAM to hierarchical Swin Transformers by reshaping 1D spatial token sequences $[B, 49, 768] \to [B, 7, 7, 768] \to [B, 768, 7, 7]$ for Stage 4 deep activations.
- Computes gradient-weighted channel combinations $\text{CAM} = \text{ReLU}\left(\sum_k \alpha_k A_k\right)$, normalizes to $[0, 1]$, and bilinearly interpolates to $(224, 224)$.
- Decoupled computations for **Stroke** and **Alzheimer's Disease** targets.
- Guaranteed hook registration and removal lifecycles to prevent GPU memory leaks.

### B. Multimodal Clinical SHAP Explainer (`shap_explainer.py`)
- Quantifies marginal Shapley contributions $\phi_i$ of individual patient clinical variables (BMI, Education, Hypertension, Diabetes, Smoking, Alcohol, etc.).
- Categorizes features into `INCREASES_RISK` ($\phi_i > 0$) vs `DECREASES_RISK` ($\phi_i < 0$).
- Computes high-level modality attribution percentages across OCT-A, OCT-B, Fundus, and Clinical pathways.

### C. Visualizer & Panel Generator (`visualization.py`)
- Produces 3-panel publication-ready figures: `[Original Retinal Scan | Grad-CAM Heatmap | Alpha-Blended Overlay]`.
- Produces horizontal feature importance bar charts for Stroke and Alzheimer's disease.

---

## 3. How to Run Phase 10

Execute from `project_backend/`:

### 1. View Explainability Configuration Summary:
```powershell
python -m phase_10_explainability.main summary
```

### 2. Generate Patient Explainability Suite:
```powershell
python -m phase_10_explainability.main explain \
    --patient-id PATIENT_01 \
    --octa datasets/approved/octa/octa_sample_1_processed.png \
    --output phase_10_explainability/outputs
```

---

## 4. Running Automated Tests

Run the complete Phase 10 test suite:
```powershell
python -m pytest phase_10_explainability/tests/ -v
```
