# Retinal Disease AI Backend — Phase 2 to Phase 5 Architecture

## Project Overview
This repository contains the backend architecture for research-oriented multimodal retinal image analysis across **OCT-A**, **OCT-B**, and **Fundus** imaging modalities.

The pipeline investigates retinal imaging biomarkers related to:
- **Stroke**
- **Alzheimer's Disease**

The complete modular system consists of:
1. **Phase 2 — Retinal Image Preprocessing**: Standardizes dimensions ($224 \times 224$), reduces noise, enhances contrast, and handles borders non-destructively.
2. **Phase 3 — Image Quality Assessment**: Evaluates technical image suitability (sharpness, illumination, contrast, noise, clipping, content, color) and routes data to `datasets/approved/` or `datasets/rejected/`.
3. **Phase 4 — Swin Transformer Deep Learning**: Modality-dedicated Swin Transformer architectures (`swin_tiny_patch4_window7_224`) for feature extraction and supervised disease classification.
4. **Phase 5 — Dynamic Modality Reliability Attention & Cross-Attention Fusion**: Combines modality-specific representations into a single **Unified Retinal Representation (URR)** with learned reliability weights and robust missing-modality handling.
5. **Integration Layers**: Bridges Phases 2 $\rightarrow$ 3, Phases 3 $\rightarrow$ 4, and Phases 4 $\rightarrow$ 5 into automated end-to-end workflows.

---

## Architecture

```text
                     [RAW RETINAL IMAGES]
                   (datasets/raw/<modality>/)
                               │
                               ▼
                            PHASE 2
                  RETINAL IMAGE PREPROCESSING
                               │
                               ▼
                       [PROCESSED IMAGES]
                (datasets/processed/<modality>/)
                               │
                               ▼
                    INTEGRATION LAYER (2->3)
                               │
                               ▼
                            PHASE 3
                    IMAGE QUALITY ASSESSMENT
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
              ACCEPT        WARNING         REJECT
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   DATASET ROUTING       │
                  ├────────────┬────────────┤
                  │  APPROVED  │  REJECTED  │
                  └────────────┴────────────┘
                               │
                               ▼
                     [APPROVED DATASETS]
                 (datasets/approved/<modality>/)
                               │
                               ▼
                    INTEGRATION LAYER (3->4)
                               │
                               ▼
                            PHASE 4
                        SWIN TRANSFORMER
              (OCT-A, OCT-B, and Fundus Models)
                               │
                     [Modality Features]
            (OCT-A: 768, OCT-B: 768, Fundus: 768)
                               │
                               ▼
                    INTEGRATION LAYER (4->5)
                               │
                               ▼
                            PHASE 5
                MULTIMODAL RETINAL FUSION (DMRA)
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
            Modality        Dynamic         Cross-
           Projection     Reliability     Attention
           (512-dim)     Weights (w_m)      Fusion
                │              │              │
                └──────────────┼──────────────┘
                               ▼
             UNIFIED RETINAL REPRESENTATION (URR)
                       [B, urr_dim=512]
                               │
                               ▼
                            PHASE 6
              (FT-Transformer for Clinical Data)
```

---

## Directory Structure

```text
project_backend/
│
├── phase_2_image_preprocessing/          # Phase 2 Preprocessing Package
│   ├── src/                              # Loader, CLAHE, Gaussian/Median, Resize, Normalize
│   ├── tests/                            # 41 Unit and integration tests
│   ├── README.md                         # Phase 2 documentation
│   └── requirements.txt
│
├── phase_3_image_quality_assessment/     # Phase 3 Quality Assessment Package
│   ├── src/                              # Blur, Brightness, Contrast, Noise, Clipping, Content, Color
│   ├── tests/                            # 39 Unit and integration tests
│   ├── README.md                         # Phase 3 documentation
│   └── requirements.txt
│
├── phase_4_swin_transformer/             # Phase 4 Swin Transformer Package
│   ├── models/                           # Swin factory, OCT-A, OCT-B, and Fundus models
│   ├── train.py                          # Training loop with AdamW, CosineAnnealing, AMP
│   ├── evaluate.py                       # Clinical metrics, confusion matrix, ROC/PR curves
│   ├── inference.py                      # Single-image & batch inference engine
│   ├── explainability.py                 # Attention activation heatmaps
│   ├── validation.py                     # Dataset validation & label verification
│   ├── leakage_check.py                  # Zero-leakage audit
│   ├── split_dataset.py                  # Stratified & patient-isolated partitioning
│   ├── tests/                            # 22 unit tests
│   ├── README.md                         # Phase 4 documentation
│   └── requirements.txt
│
├── phase_5_retinal_fusion/               # Phase 5 Multimodal Retinal Fusion Package
│   ├── modality_projection.py            # Modality feature projection to common embedding
│   ├── reliability_attention.py          # Dynamic Modality Reliability Attention (DMRA)
│   ├── cross_attention.py                # Multi-head Transformer cross-attention
│   ├── urr.py                            # Unified Retinal Representation (URR) Head
│   ├── fusion_model.py                   # End-to-end multimodal fusion model
│   ├── feature_loader.py                 # Phase 4 Swin feature extractor adapter
│   ├── validation.py                     # Tensor validation and NaN/Inf audits
│   ├── main.py                           # CLI entry point for fusion & summary
│   ├── tests/                            # Comprehensive Phase 5 unit & integration tests
│   ├── README.md                         # Phase 5 documentation
│   └── requirements.txt
│
├── integration/                          # Integration Layer
│   ├── config.py                         # Routing configuration & dataset paths
│   ├── phase2_phase3_pipeline.py         # Phase 2 -> Phase 3 pipeline runner
│   ├── phase3_phase4_pipeline.py         # Phase 3 -> Phase 4 pipeline runner
│   └── phase4_phase5_pipeline.py         # Phase 4 -> Phase 5 pipeline runner
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

### Mode 1 — Phase 2 Only (Preprocessing)
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality all
```

### Mode 2 — Phase 3 Only (Quality Assessment)
```bash
python -m phase_3_image_quality_assessment.src.main --modality all
```

### Mode 3 — Phase 2 $\rightarrow$ Phase 3 Integrated Workflow
```bash
python -m integration.phase2_phase3_pipeline --modality all
```

### Mode 4 — Phase 3 $\rightarrow$ Phase 4 Dataset Preparation
```bash
python -m integration.phase3_phase4_pipeline --modality all --task alzheimers
```

### Mode 5 — Phase 4 Swin Transformer Model Training & Evaluation
```bash
# Train OCT-A Swin model
python -m phase_4_swin_transformer.main train --modality octa --task alzheimers

# Evaluate saved checkpoint
python -m phase_4_swin_transformer.main evaluate --checkpoint phase_4_swin_transformer/outputs/octa/experiment_001/best_model.pth --modality octa
```

### Mode 6 — Phase 5 Multimodal Retinal Fusion (DMRA + URR)
```bash
# Display fusion architecture summary
python -m phase_5_retinal_fusion.main summary

# Fuse patient scans into Unified Retinal Representation (URR)
python -m phase_5_retinal_fusion.main fuse \
    --octa datasets/approved/octa/octa_sample_1_processed.png \
    --output phase_5_retinal_fusion/outputs/patient_urr.pt
```

---

## Running Automated Tests

Run the complete test suite across Phase 2, Phase 3, Phase 4, Phase 5, and Integration:
```bash
python -m pytest -v
```
