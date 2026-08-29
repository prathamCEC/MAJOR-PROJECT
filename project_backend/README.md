# Retinal Disease AI Backend — Phase 2 & Phase 3 Architecture

## Project Overview
This repository contains the backend architecture for research-oriented multimodal retinal image analysis across **OCT-A**, **OCT-B**, and **Fundus** imaging modalities.

The pipeline prepares clinical data for downstream AI modeling (Phase 4 — Swin Transformer) through two specialized stages:
1. **Phase 2 — Retinal Image Preprocessing**: Standardizes dimensions ($224 \times 224$), reduces noise, enhances contrast, and handles borders non-destructively.
2. **Phase 3 — Image Quality Assessment**: Evaluates technical image suitability (sharpness, illumination, contrast, noise, clipping, content, color) and routes data to `datasets/approved/` or `datasets/rejected/`.
3. **Integration Layer**: Connects Phase 2 and Phase 3 in an automated workflow.

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
                       INTEGRATION LAYER
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
                            PHASE 4
                       SWIN TRANSFORMER
                          (FUTURE)
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
├── integration/                          # Phase 2 -> Phase 3 Integration Layer
│   ├── config.py                         # Routing configuration & dataset paths
│   └── phase2_phase3_pipeline.py         # End-to-end pipeline runner
│
├── datasets/
│   ├── raw/                              # Ingested raw images (octa, octb, fundus)
│   ├── processed/                        # Preprocessed 224x224 images
│   ├── approved/                         # Technical quality approved (ready for Phase 4)
│   └── rejected/                         # Technical quality rejected images
│
└── logs/
    ├── phase2_failed_images.txt          # Phase 2 error log
    ├── phase3_quality_results.csv        # Phase 3 quality evaluation table
    ├── phase3_quality_results.json       # Phase 3 detailed evaluation logs
    └── integrated_pipeline_results.csv   # End-to-end pipeline report
```

---

## Three Execution Modes

All commands should be executed from the `project_backend/` directory.

### Mode 1 — Phase 2 Only (Preprocessing)
```bash
# Run all modalities
python -m phase_2_image_preprocessing.src.batch_processor --modality all

# Run specific modality
python -m phase_2_image_preprocessing.src.batch_processor --modality octa
```

### Mode 2 — Phase 3 Only (Quality Assessment on Processed Images)
```bash
# Run all modalities
python -m phase_3_image_quality_assessment.src.main --modality all

# Assess a single image
python -m phase_3_image_quality_assessment.src.main \
    --image datasets/processed/octa/patient001_processed.png \
    --modality octa
```

### Mode 3 — Phase 2 $\rightarrow$ Phase 3 (End-to-End Integrated Workflow)
```bash
# Run end-to-end integration for all modalities
python -m integration.phase2_phase3_pipeline --modality all

# Run specific modality
python -m integration.phase2_phase3_pipeline --modality fundus
```

---

## Running Automated Tests

Run the complete 80-test suite across Phase 2, Phase 3, and Integration:
```bash
python -m pytest phase_2_image_preprocessing/tests/ phase_3_image_quality_assessment/tests/ -v
```
