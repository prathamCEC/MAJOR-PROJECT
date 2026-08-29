# Retinal Disease AI Backend — Phase 2 to Phase 8 Architecture

## Project Overview
This repository contains the backend architecture for research-oriented multimodal retinal image and clinical data analysis across **OCT-A**, **OCT-B**, and **Fundus** imaging modalities alongside structured patient health variables.

The pipeline investigates biomarkers related to:
- **Stroke**
- **Alzheimer's Disease**

The complete modular system consists of:
1. **Phase 2 — Retinal Image Preprocessing**: Standardizes dimensions ($224 \times 224$), reduces noise, enhances contrast, and handles borders non-destructively.
2. **Phase 3 — Image Quality Assessment**: Evaluates technical image suitability (sharpness, illumination, contrast, noise, clipping, content, color) and routes data to `datasets/approved/` or `datasets/rejected/`.
3. **Phase 4 — Swin Transformer Deep Learning**: Modality-dedicated Swin Transformer architectures (`swin_tiny_patch4_window7_224`) for feature extraction and supervised disease classification.
4. **Phase 5 — Dynamic Modality Reliability Attention & Cross-Attention Fusion**: Combines modality-specific representations into a single **Unified Retinal Representation (URR)** with learned reliability weights and robust missing-modality handling.
5. **Phase 6 — FT-Transformer for Structured Clinical Data**: Transforms patient tabular health variables (BMI, hypertension, diabetes, smoking, demographics) into a normalized **Clinical Representation (CR)** vector.
6. **Phase 7 — Retina–Clinical Cross-Attention Fusion & Unified Patient Representation (UPR)**: Fuses Retinal (URR) and Clinical (CR) vectors via bidirectional cross-attention and learnable gated multimodal fusion into a **Unified Patient Representation (UPR)** vector ($[B, 512]$).
7. **Phase 8 — Multi-Task Disease Prediction Network**: Consumes the UPR to perform decoupled, simultaneous binary predictions for **Stroke** and **Alzheimer's Disease** with masked multi-task loss and class imbalance handling.
8. **Integration Layers**: Bridges Phases 2 through 8 into automated end-to-end workflows.

---

## Architecture

```text
                     [RAW RETINAL IMAGES]                     [PATIENT CLINICAL DATA]
                   (datasets/raw/<modality>/)                 (5_ASSOCIATED DATA.xlsx)
                               │                                         │
                               ▼                                         ▼
                            PHASE 2                                   PHASE 6
                  RETINAL IMAGE PREPROCESSING                   FT-TRANSFORMER (TABULAR)
                               │                                         │
                               ▼                               ┌─────────┴─────────┐
                       [PROCESSED IMAGES]                      ↓                   ↓
                (datasets/processed/<modality>/)           Numerical          Categorical
                               │                          (BMI, Educ.)       (HTN, DM2, etc.)
                               ▼                               ↓                   ↓
                            PHASE 3                     Preprocessing       Tokenizer
                    IMAGE QUALITY ASSESSMENT                   └─────────┬─────────┘
                               │                                         ↓
                               ▼                                 [CLS] + Tokens
                            PHASE 4                                      ↓
                        SWIN TRANSFORMER                        FT-Transformer Stack
                     (OCT-A, OCT-B, Fundus)                              ↓
                               │                               Clinical Representation
                               ▼                                    [B, CR_dim=512]
                            PHASE 5                                      │
                MULTIMODAL RETINAL FUSION (DMRA)                         │
                               │                                         │
                               ▼                                         │
             UNIFIED RETINAL REPRESENTATION (URR)                        │
                       [B, URR_dim=512]                                  │
                               │                                         │
                               └────────────────────┬────────────────────┘
                                                    │
                                                    ▼
                                                 PHASE 7
                                     RETINA–CLINICAL CROSS-ATTENTION
                                      (Bidirectional Transformer)
                                                    │
                                                    ▼
                                         GATED MULTIMODAL FUSION
                                                    │
                                                    ▼
                                      UNIFIED PATIENT REPRESENTATION (UPR)
                                               [B, UPR_dim=512]
                                                    │
                                                    ▼
                                                 PHASE 8
                                  MULTI-TASK DISEASE PREDICTION NETWORK
                                        (Shared Prediction Trunk)
                                                    │
                                     ┌──────────────┴──────────────┐
                                     ↓                             ↓
                            STROKE PREDICTION             ALZHEIMER'S PREDICTION
                            (Logit, Prob, Class)           (Logit, Prob, Class)
```

---

## Directory Structure

```text
project_backend/
│
├── phase_2_image_preprocessing/          # Phase 2 Preprocessing Package (41 tests)
├── phase_3_image_quality_assessment/     # Phase 3 Quality Assessment Package (39 tests)
├── phase_4_swin_transformer/             # Phase 4 Swin Transformer Package (22 tests)
├── phase_5_retinal_fusion/               # Phase 5 Multimodal Retinal Fusion Package (16 tests)
├── phase_6_clinical_transformer/         # Phase 6 Clinical FT-Transformer Package (18 tests)
├── phase_7_retina_clinical_fusion/       # Phase 7 Retina-Clinical Cross-Attention Package (18 tests)
│
├── phase_8_multitask_prediction/         # Phase 8 Multi-Task Disease Prediction Package (17 tests)
│   ├── config.py                         # Shared hidden dims, loss weights, thresholds
│   ├── shared_network.py                 # Shared prediction representation trunk
│   ├── prediction_heads.py               # Stroke and Alzheimer's binary classification heads
│   ├── loss.py                           # Masked multi-task loss (missing label safe)
│   ├── model.py                          # MultiTaskDiseasePredictionNetwork (MC-Dropout ready)
│   ├── metrics.py                        # Multi-task accuracy, sensitivity, specificity, AUC
│   ├── trainer.py                        # Supervised multi-task training & validation loop
│   ├── validation.py                     # Tensor dimension & finite numerical audits
│   ├── inference.py                      # End-to-end multi-task inference pipeline
│   ├── checkpoint.py                     # Checkpoint manager
│   ├── main.py                           # CLI entry point for summary and prediction
│   ├── tests/                            # Comprehensive unit & integration tests
│   ├── README.md                         # Phase 8 documentation
│   └── requirements.txt
│
├── integration/                          # Integration Layer
│   ├── config.py                         # Routing configuration & dataset paths
│   ├── phase2_phase3_pipeline.py         # Phase 2 -> Phase 3 pipeline runner
│   ├── phase3_phase4_pipeline.py         # Phase 3 -> Phase 4 pipeline runner
│   ├── phase4_phase5_pipeline.py         # Phase 4 -> Phase 5 pipeline runner
│   └── phase5_phase6_interface.py        # Phase 5 (URR) + Phase 6 (CR) compatibility bridge
│
├── datasets/
│   ├── raw/                              # Ingested raw images (octa, octb, fundus)
│   ├── processed/                        # Preprocessed 224x224 images
│   ├── approved/                         # Technical quality approved (consumed by Phase 4)
│   ├── rejected/                         # Technical quality rejected images
│   └── splits/                           # Partitioned train, val, and test manifests
│
└── logs/
    ├── phase2_failed_images.txt          # Phase 2 error log
    ├── phase3_quality_results.csv        # Phase 3 quality evaluation table
    ├── phase3_quality_results.json       # Phase 3 detailed evaluation logs
    └── integrated_pipeline_results.csv   # End-to-end pipeline report
```

---

## How to Run

All commands should be executed from the `project_backend/` directory.

### Mode 1 — Phase 2 Preprocessing
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality all
```

### Mode 2 — Phase 3 Quality Assessment
```bash
python -m phase_3_image_quality_assessment.src.main --modality all
```

### Mode 3 — Phase 4 Swin Model Training
```bash
python -m phase_4_swin_transformer.main train --modality octa --task alzheimers
```

### Mode 4 — Phase 5 Retinal Fusion (DMRA + URR)
```bash
python -m phase_5_retinal_fusion.main fuse \
    --octa datasets/approved/octa/octa_sample_1_processed.png \
    --output phase_5_retinal_fusion/outputs/patient_urr.pt
```

### Mode 5 — Phase 6 Clinical FT-Transformer
```bash
python -m phase_6_clinical_transformer.main extract \
    --data "../5_ASSOCIATED DATA.xlsx" \
    --output phase_6_clinical_transformer/outputs/clinical_representations.pt
```

### Mode 6 — Phase 7 Retina-Clinical Cross-Attention & UPR Fusion
```bash
python -m phase_7_retina_clinical_fusion.main fuse \
    --retinal phase_5_retinal_fusion/outputs/patient_urr.pt \
    --clinical phase_6_clinical_transformer/outputs/clinical_representations.pt \
    --output phase_7_retina_clinical_fusion/outputs/unified_patient_representation.pt
```

### Mode 7 — Phase 8 Multi-Task Disease Prediction
```bash
# Architecture summary
python -m phase_8_multitask_prediction.main summary

# Predict Stroke & Alzheimer's outcomes from UPR
python -m phase_8_multitask_prediction.main predict \
    --upr phase_7_retina_clinical_fusion/outputs/unified_patient_representation.pt \
    --output phase_8_multitask_prediction/outputs/multitask_predictions.pt
```

---

## Running Automated Tests

Run the complete test suite across Phase 2 through Phase 8 + Integration:
```bash
python -m pytest -v
```
