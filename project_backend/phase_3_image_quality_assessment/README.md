# Phase 3 — Retinal Image Quality Assessment

## 1. Overview & Purpose
Phase 3 performs rigorous, non-destructive technical image quality assessment (IQA) for retinal imaging across **OCT-A**, **OCT-B**, and **Fundus** modalities.

The system evaluates whether preprocessed images from Phase 2 meet diagnostic and computational quality standards before they can be routed to the approved dataset for downstream Swin Transformer AI modeling (Phase 4).

> [!IMPORTANT]
> **Medical Disclaimer & Scope**:
> Phase 3 evaluates **technical image quality only** (focus, illumination, contrast, noise, clipping, content integrity, and color fidelity). **Phase 3 does not diagnose disease** (such as stroke or Alzheimer's disease), assess patient health, or establish clinical validity.

---

## 2. Non-Destructive Quality Assessment Philosophy
Phase 3 strictly operates in an analytical, non-modifying capacity. It adheres to the execution pipeline:

$$\text{READ} \longrightarrow \text{VALIDATE} \longrightarrow \text{MEASURE} \longrightarrow \text{NORMALIZE} \longrightarrow \text{SCORE} \longrightarrow \text{DECIDE} \longrightarrow \text{REPORT}$$

Phase 3 **never** sharpens, resizes, denoises, color-shifts, normalizes pixel values, or overwrites image tensors.

---

## 3. Supported Modalities
The system provides tailored, modality-aware quality assessment for three imaging modalities:
- **`octa` (OCT-A / Optical Coherence Tomography Angiography)**: Vascular networks, capillary densities, and foveal avascular zone (FAZ) definition.
- **`octb` (OCT-B / Structural Cross-Sectional OCT)**: Retinal layer contrast (RNFL, ganglion cell complex, choroid) and speckle noise levels.
- **`fundus` (Color Retinal Fundus Photography)**: True RGB color fidelity, macula/optic disc illumination, and peripheral field exposure.

---

## 4. Technical Quality Metrics

### 1. Blur & Sharpness (`blur_detection.py`)
- **Metric**: Modified Laplacian variance and Tenengrad Sobel gradient energy.
- **Goal**: Quantifies edge sharpness across vessels and layer boundaries. Normalized using logarithmic scaling ($0 - 100$).

### 2. Brightness & Illumination (`brightness.py`)
- **Metric**: Mean and median intensity (grayscale for OCT; CIE LAB $L$-channel luminance for Fundus).
- **Goal**: Identifies underexposed shadows ($< 15$) and overexposed washed-out frames ($> 240$) using a calibrated trapezoidal response curve.

### 3. Contrast (`contrast.py`)
- **Metric**: Root-mean-square (RMS) standard deviation and $5^{\text{th}}\text{-to-}95^{\text{th}}$ percentile dynamic range.
- **Goal**: Ensures sufficient dynamic range across structural layers and vessels without excessive saturation.

### 4. Noise Level (`noise.py`)
- **Metric**: Spatial high-frequency residual standard deviation ($R = I - \text{GaussianBlur}(I)$) and estimated SNR in dB.
- **Goal**: Penalizes high sensor noise while preserving clean diagnostic textures.

### 5. Clipping & Saturation (`clipping.py`)
- **Metric**: Percentage of active pixels clipped at minimum intensity ($\le 2$) or maximum intensity ($\ge 253$).
- **Goal**: Excludes natural outer black padding and evaluates clipping within active retinal tissue.

### 6. Content Integrity (`content_quality.py`)
- **Metric**: Shannon information entropy ($H = -\sum p_i \log_2 p_i$) and foreground structure ratio.
- **Goal**: Immediately disqualifies flat, empty, or corrupted frames with near-zero diagnostic entropy.

### 7. Color Quality (`color_quality.py` — Fundus Only)
- **Metric**: Inter-channel chromatic disparity and HSV saturation mean.
- **Goal**: Detects monochrome/grayscale fundus uploads and unnatural single-channel blowout without forcing artificial RGB equality on biological tissue.

---

## 5. Modality Weighting & Scoring

Every sub-metric is mapped to a standardized $0 - 100$ scale before calculating the weighted composite score:

$$\text{Overall Score} = \sum_{i} w_i \cdot \text{Score}_i$$

| Dimension | OCT-A Weight | OCT-B Weight | Fundus Weight |
| :--- | :---: | :---: | :---: |
| **Blur / Sharpness** | 0.25 | 0.20 | 0.20 |
| **Brightness / Illumination** | 0.15 | 0.15 | 0.15 |
| **Contrast** | 0.20 | 0.25 | 0.15 |
| **Noise Level** | 0.15 | 0.15 | 0.10 |
| **Clipping / Saturation** | 0.10 | 0.10 | 0.10 |
| **Content Integrity** | 0.15 | 0.15 | 0.15 |
| **Color Fidelity** | 0.00 | 0.00 | 0.15 |
| **Total** | **1.00** | **1.00** | **1.00** |

---

## 6. Decision Logic & Warning Policy

- **`ACCEPT`**: $\text{Overall Score} \ge 65.0$ AND all critical hard failures pass $\longrightarrow$ **Approved for AI**.
- **`WARNING`**: $50.0 \le \text{Overall Score} < 65.0 \longrightarrow$ Borderline technical quality.
  - Under `warning_policy = "approve"` (default): Routed to approved dataset with warning status recorded in metadata.
  - Under `warning_policy = "reject"`: Routed to rejected dataset.
- **`REJECT`**: $\text{Overall Score} < 50.0$ OR triggered critical hard failure (e.g. flat image, severe defocus, extreme clipping) $\longrightarrow$ **Rejected**.

---

## 7. Directory Architecture

```
project_backend/
│
├── phase_3_image_quality_assessment/
│   ├── src/
│   │   ├── config.py              # Centralized weights, thresholds, and paths
│   │   ├── image_loader.py        # Non-destructive image reader
│   │   ├── validation.py          # Pre-assessment image checks & exceptions
│   │   ├── blur_detection.py      # Sharpness / Laplacian variance extractor
│   │   ├── brightness.py          # Illumination / luminance extractor
│   │   ├── contrast.py            # RMS contrast extractor
│   │   ├── noise.py               # Spatial residual noise estimator
│   │   ├── clipping.py            # Under/over-saturation detector
│   │   ├── color_quality.py       # Fundus chromatic fidelity checker
│   │   ├── content_quality.py     # Shannon entropy & structural checker
│   │   ├── normalization.py       # 0-100 metric normalization
│   │   ├── quality_score.py       # Composite score calculator
│   │   ├── decision.py            # Quality decision engine
│   │   ├── pipeline.py            # End-to-end QualityAssessmentPipeline
│   │   ├── batch_processor.py     # Batch folder processor with CSV/JSON logs
│   │   └── main.py                # Standalone CLI entry point
│   │
│   ├── tests/                     # 39 automated pytest test suites
│   ├── README.md                  # This documentation file
│   └── requirements.txt           # Minimal dependencies
│
├── datasets/
│   ├── processed/                 # Phase 2 output images (consumed by Phase 3)
│   ├── approved/                  # Approved dataset for downstream Phase 4
│   └── rejected/                  # Rejected images
│
└── logs/
    ├── phase3_quality_results.csv # Tabular metric scores and decisions
    └── phase3_quality_results.json# Detailed JSON metric logs
```

---

## 8. How to Run Phase 3

Execute from `project_backend/`:

### Single Image Assessment:
```bash
python -m phase_3_image_quality_assessment.src.main \
    --image datasets/processed/octa/patient001_processed.png \
    --modality octa
```

### Batch Dataset Assessment:
```bash
# Assess all modalities in datasets/processed/
python -m phase_3_image_quality_assessment.src.main --modality all

# Assess OCT-A only
python -m phase_3_image_quality_assessment.src.main --modality octa

# Assess OCT-B only
python -m phase_3_image_quality_assessment.src.main --modality octb

# Assess Fundus only
python -m phase_3_image_quality_assessment.src.main --modality fundus
```

### Run Automated Tests:
```bash
python -m pytest phase_3_image_quality_assessment/tests/ -v
```

---

## 9. Programmatic Python API

```python
from pathlib import Path
from phase_3_image_quality_assessment import (
    assess_image_file,
    assess_image,
    QualityAssessmentPipeline,
)

# Assess single image file
result = assess_image_file(
    image_path="datasets/processed/octa/patient001_processed.png",
    modality="octa",
)

print(f"Overall Quality Score: {result.overall_score}/100")
print(f"Decision: {result.decision} (Approved for AI: {result.is_approved_for_ai})")
print(f"Detailed Dimension Scores: {result.scores}")
```
