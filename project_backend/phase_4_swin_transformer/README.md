# Phase 4 — Swin Transformer for Retinal Disease Analysis (OCT-A, OCT-B, Fundus)

## 1. Overview & Purpose
Phase 4 implements deep learning classification architectures using **Swin Transformer** (`swin_tiny_patch4_window7_224`) tailored for retinal imaging across **OCT-A**, **OCT-B**, and **Fundus** modalities.

The system investigates retinal imaging biomarkers related to:
- **Stroke**
- **Alzheimer's Disease**

> [!IMPORTANT]
> **Research Use Only / Non-Clinical Disclaimer**:
> Phase 4 outputs represent deep learning model probabilities. The model output is strictly formatted as **"Predicted Class"**, **"Prediction Probability"**, and **"Model Confidence"**. Phase 4 software does not establish clinical validity or provide clinical diagnoses.

---

## 2. Modality Architecture & Channel Handling

Each imaging modality possesses dedicated data adapters and independent model checkpoints:

```
        ┌─────────────┐               ┌─────────────┐               ┌─────────────┐
        │    OCT-A    │               │    OCT-B    │               │   FUNDUS    │
        └──────┬──────┘               └──────┬──────┘               └──────┬──────┘
               │                             │                             │
               ▼                             ▼                             ▼
       [Grayscale -> 3ch]            [Grayscale -> 3ch]             [True RGB Color]
               │                             │                             │
               ▼                             ▼                             ▼
        Swin Transformer              Swin Transformer              Swin Transformer
        (OCT-A Checkpoint)            (OCT-B Checkpoint)           (Fundus Checkpoint)
               │                             │                             │
               ▼                             ▼                             ▼
       Predicted Disease             Predicted Disease             Predicted Disease
          Confidence                    Confidence                    Confidence
```

### Channel Specifics:
1. **OCT-A (Optical Coherence Tomography Angiography)**:
   - Preserves grayscale vascular / capillary network details.
   - Replicated across 3 channels ($[R, G, B]$) for compatibility with standard ImageNet-pretrained Swin tokenizers without fabricating false color.
2. **OCT-B (Cross-Sectional Structural OCT)**:
   - Preserves grayscale cross-sectional layer boundaries (RNFL, photoreceptors, choroid).
   - Replicated across 3 channels for Swin backbone.
3. **Fundus (Color Retinal Photography)**:
   - Preserves genuine 3-channel RGB spectral information required for macula, optic cup/disc, and vessel color analysis.

---

## 3. Directory & Checkpoint Structure

```
phase_4_swin_transformer/
├── config.py              # Centralized hyperparameters & modality configs
├── enums.py               # Modality, DiseaseTask, and SplitType enums
├── dataset.py             # RetinalDataset (PyTorch Dataset) & DataLoader factory
├── transforms.py          # Modality-aware data augmentations & ImageNet normalizers
├── validation.py          # Dataset audit engine (missing files, labels, patients)
├── leakage_check.py       # Zero-leakage audit (hash collisions, patient overlaps)
├── split_dataset.py       # Stratified 70/15/15 train/val/test splitting
│
├── models/
│   ├── swin_factory.py    # Factory creating Swin Transformer with custom heads
│   ├── octa_model.py      # OCT-A specialized Swin wrapper
│   ├── octb_model.py      # OCT-B specialized Swin wrapper
│   └── fundus_model.py    # Fundus specialized Swin wrapper
│
├── metrics.py             # Accuracy, F1, Sensitivity, Specificity, ROC/PR curves
├── checkpoint.py          # Atomic CheckpointManager (best_model.pth & last_model.pth)
├── train.py               # Complete training loop with AdamW, CosineAnnealing, AMP
├── evaluate.py            # Test set evaluation & publication curve plotting
├── inference.py           # Single-image & batch inference engine
├── explainability.py      # Attention / feature activation heatmap visualizer
├── utils.py               # Seed initialization & device detection (CUDA/CPU)
├── main.py                # Unified CLI entry point
│
├── outputs/               # Isolated experiment outputs
│   ├── octa/experiment_001/
│   ├── octb/experiment_001/
│   └── fundus/experiment_001/
│
└── tests/                 # Complete automated pytest test suite
```

---

## 4. How to Run Phase 4

Execute all commands from `project_backend/`:

### 1. Dataset Validation & Audit:
```powershell
python -m phase_4_swin_transformer.main validate --modality octa --task alzheimers
```

### 2. Dataset Splitting & Leakage Audit:
```powershell
python -m phase_4_swin_transformer.main split --modality octa --task alzheimers --data datasets/splits/octa_alzheimers_manifest.csv
```

### 3. Model Training:
```powershell
# Train OCT-A Swin model
python -m phase_4_swin_transformer.main train --modality octa --task alzheimers --epochs 20 --batch-size 8

# Train OCT-B Swin model
python -m phase_4_swin_transformer.main train --modality octb --task alzheimers --epochs 20 --batch-size 8

# Train Fundus Swin model
python -m phase_4_swin_transformer.main train --modality fundus --task alzheimers --epochs 20 --batch-size 8
```

### 4. Test Set Evaluation:
```powershell
python -m phase_4_swin_transformer.main evaluate --checkpoint phase_4_swin_transformer/outputs/octa/experiment_001/best_model.pth --modality octa
```

### 5. Single-Image & Batch Inference:
```powershell
# Single image prediction
python -m phase_4_swin_transformer.main inference --checkpoint phase_4_swin_transformer/outputs/octa/experiment_001/best_model.pth --modality octa --image datasets/approved/octa/sample_01.png

# Batch prediction
python -m phase_4_swin_transformer.main inference --checkpoint phase_4_swin_transformer/outputs/octa/experiment_001/best_model.pth --modality octa --input datasets/approved/octa/
```

### 6. Attention / Feature Map Explainability:
```powershell
python -m phase_4_swin_transformer.main explain --checkpoint phase_4_swin_transformer/outputs/octa/experiment_001/best_model.pth --modality octa --image datasets/approved/octa/sample_01.png
```

---

## 5. Phase 3 $\rightarrow$ Phase 4 Integration

Phase 4 consumes data strictly from the **Phase 3 Approved** directory (`datasets/approved/`).
Run the integration bridge:
```powershell
python -m integration.phase3_phase4_pipeline --modality all --task alzheimers
```
This routine scans approved scans, maps clinical annotations from `5_ASSOCIATED DATA.xlsx`, and generates partitioned manifests (`train.csv`, `val.csv`, `test.csv`) with zero patient leakage.
