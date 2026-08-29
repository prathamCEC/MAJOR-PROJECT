# Phase 2 — Retinal Image Preprocessing

## 1. Overview & Purpose
Phase 2 is the standardized image preprocessing module for a research-oriented AI system analyzing multimodal retinal imaging. The pipeline ingests raw retinal images across three distinct modalities and transforms them into standardized, clean, artifact-free, machine-learning-ready tensors tailored for downstream Image Quality Assessment (Phase 3) and Vision/Swin Transformer architectures (Phase 4).

> [!IMPORTANT]
> **Medical Disclaimer & Safety**:
> Phase 2 is strictly an image preprocessing module for research and educational purposes. **Phase 2 does not diagnose disease** (such as Stroke or Alzheimer's disease), infer clinical status, or produce diagnostic predictions.

---

## 2. Why Retinal Image Preprocessing is Required
Raw retinal images collected from clinical imaging devices exhibit substantial technical variance:
- **Artificial margins and black borders** introduced by imaging sensors and circular camera apertures.
- **Illumination irregularities and low contrast** between delicate anatomical structures and the retinal background.
- **High-frequency sensor noise and speckle artifacts** that can distort feature representation.
- **Heterogeneous image dimensions and aspect ratios** across different imaging hardware.

Phase 2 standardizes all inputs to fixed dimensions ($224 \times 224$), uniform dynamic range, and consistent channel representations while strictly adhering to the **Quality Preservation Principle**: never over-process, preserve all fine capillary networks, retinal layer boundaries, optic disc, and macular landmarks.

---

## 3. Supported Modalities
The system provides tailored, modality-aware preprocessing for three imaging modalities:

1. **`octa` (OCT-A / Optical Coherence Tomography Angiography)**: Captures microvascular blood flow and capillary networks.
2. **`octb` (OCT-B / Structural Cross-Sectional OCT)**: Captures structural retinal layers (RNFL, ganglion cell complex, choroid, layer thickness).
3. **`fundus` (Color Retinal Fundus Photography)**: Captures widefield color photographs displaying the optic disc, macula, blood vessels, and retinal lesions.

---

## 4. Modality-Specific Strategies

### OCT-A Preprocessing Strategy
- **Structure Goal**: Preserve fine capillary networks, vessel density, vessel calibers, and vessel boundaries.
- **Channel Handling**: Loaded as single-channel grayscale intensity. Replicated to 3 channels post-processing for Swin Transformer compatibility.
- **Contrast**: CLAHE applied directly to grayscale intensity (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
- **Noise Reduction**: Conservative Gaussian smoothing ($3 \times 3$, $\sigma=0.0$). Median filtering is disabled by default to prevent eroding thin capillary lines.

### OCT-B Preprocessing Strategy
- **Structure Goal**: Preserve cross-sectional retinal layer transitions, layer thickness, and RNFL boundaries.
- **Channel Handling**: Loaded as single-channel grayscale intensity. Replicated to 3 channels post-processing.
- **Contrast**: CLAHE applied directly to grayscale intensity (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
- **Noise Reduction**: Conservative Gaussian smoothing ($3 \times 3$) combined with conservative Median filtering ($k=3$) to suppress speckle noise without blurring cross-layer transitions.

### Fundus Preprocessing Strategy
- **Structure Goal**: Preserve true color information, optic disc, macula, vascular trees, and lesion pigmentation.
- **Channel Handling**: Maintained strictly in 3-channel color (BGR in OpenCV / RGB). Never converted permanently to grayscale.
- **Contrast**: Modality-specific **LAB Luminance CLAHE**. Converts BGR $\rightarrow$ CIE LAB, applies CLAHE exclusively to the L (luminance) channel, and converts back to BGR. This avoids RGB channel distortion.
- **Noise Reduction**: Conservative Gaussian smoothing ($3 \times 3$, $\sigma=0.0$).

---

## 5. Preprocessing Pipeline Execution Order

```
                    [RAW RETINAL IMAGE]
                             │
                             ▼
                    1. IMAGE LOADER
             (Format check, modality-aware read)
                             │
                             ▼
                    2. RAW VALIDATION
             (Finite check, dimensions, dtypes)
                             │
                             ▼
                   3. SAFE BORDER CROP
             (Content bbox, safety margin, skip if <2%)
                             │
                             ▼
                  4. MODALITY-AWARE CLAHE
             (Grayscale for OCT; LAB L-channel for Fundus)
                             │
                             ▼
                   5. GAUSSIAN FILTER
             (Conservative 3x3 smoothing, sigma=0)
                             │
                             ▼
                    6. MEDIAN FILTER
             (Speckle suppression for OCT-B; k=3)
                             │
                             ▼
                 7. NORMALIZATION & CONVERSION
             (Float32 [0.0, 1.0] -> Uint8 [0, 255])
                             │
                             ▼
               8. ASPECT-RATIO RESIZE & PADDING
             (Distortion-free scaling + symmetric pad to 224x224)
                             │
                             ▼
               9. 3-CHANNEL STANDARDIZATION
             (Replicate single-channel -> 3-ch for Swin-T)
                             │
                             ▼
                  10. FINAL VALIDATION
             (Check shape (224,224,3), uint8, range [0, 255])
                             │
                             ▼
                  11. SAVE & DISK VERIFY
             (Write to processed/ and verify reload)
```

---

## 6. Directory Structure

```
project_backend/
│
├── phase_2_image_preprocessing/
│   │
│   ├── __init__.py
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py              # Central configuration & ModalityConfig dataclasses
│   │   ├── image_loader.py        # Modality-aware OpenCV image loader
│   │   ├── validation.py          # Strict raw/processed validation & custom exceptions
│   │   ├── border_crop.py         # Non-destructive black border detection & cropping
│   │   ├── clahe.py               # Grayscale & LAB color space CLAHE
│   │   ├── gaussian_filter.py     # Conservative Gaussian blur
│   │   ├── median_filter.py       # Conservative Median filter
│   │   ├── normalization.py       # Float32 normalization & uint8 quantization
│   │   ├── resizing.py            # Aspect-ratio-preserving resize + symmetric padding
│   │   ├── pipeline.py            # PreprocessPipeline class & standalone API
│   │   ├── batch_processor.py     # CLI batch runner with stats and failure logger
│   │   └── utils.py               # File discovery, naming, reload verification
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_image_loader.py   # Loader & corrupt file tests
│   │   ├── test_validation.py     # Validation, NaN/Inf checks
│   │   ├── test_border_crop.py    # Safe border detection tests
│   │   ├── test_preprocessing.py  # Filter unit tests
│   │   ├── test_normalization.py  # Float/uint8 scaling tests
│   │   ├── test_resizing.py       # Resizing and padding tests
│   │   └── test_pipeline.py       # End-to-end integration & batch tests
│   │
│   ├── README.md                  # This documentation file
│   └── requirements.txt           # Minimal dependencies
│
├── datasets/
│   ├── raw/
│   │   ├── octa/                  # Raw OCT-A images (read-only)
│   │   ├── octb/                  # Raw OCT-B images (read-only)
│   │   └── fundus/                # Raw Fundus images (read-only)
│   │
│   └── processed/
│       ├── octa/                  # Standardized 224x224 OCT-A outputs
│       ├── octb/                  # Standardized 224x224 OCT-B outputs
│       └── fundus/                # Standardized 224x224 Fundus outputs
│
├── logs/
│   └── phase2_failed_images.txt   # Execution failure log
│
└── README.md                      # Root project overview
```

---

## 7. Configuration Parameters

Centralized in `phase_2_image_preprocessing/src/config.py`:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `TARGET_WIDTH` | `224` | Target output width |
| `TARGET_HEIGHT` | `224` | Target output height |
| `SUPPORTED_MODALITIES` | `("octa", "octb", "fundus")` | Supported modalities |
| `SUPPORTED_IMAGE_EXTENSIONS` | `(.png, .jpg, .jpeg, .tif, .tiff, .bmp, .ppm)` | Supported file extensions |
| `CLAHE_CLIP_LIMIT` | `2.0` | Contrast limitation threshold |
| `CLAHE_TILE_GRID_SIZE` | `(8, 8)` | Local equalization tile grid |
| `GAUSSIAN_KERNEL_SIZE` | `(3, 3)` | Smoothing kernel dimensions |
| `GAUSSIAN_SIGMA` | `0.0` | Smoothing standard deviation |
| `MEDIAN_KERNEL_SIZE` | `3` | Aperture linear size for median filter |
| `CROP_MARGIN` | `2` | Safety margin (px) around detected content |
| `MIN_BORDER_RATIO` | `0.02` | Minimum border area (2%) required to trigger crop |
| `CONVERT_TO_3_CHANNEL` | `True` | Standardize single-channel images to 3 channels |

---

## 8. How to Run Phase 2

All commands should be executed from the `project_backend/` directory.

### 1. Run Preprocessing on All Modalities
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality all
```

### 2. Run Preprocessing on OCT-A Only
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality octa
```

### 3. Run Preprocessing on OCT-B Only
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality octb
```

### 4. Run Preprocessing on Fundus Only
```bash
python -m phase_2_image_preprocessing.src.batch_processor --modality fundus
```

### Optional CLI Arguments
- `--overwrite`: Force reprocessing of images that already have a corresponding `_processed.png` file.
- `--raw-dir <path>`: Custom raw dataset root directory.
- `--processed-dir <path>`: Custom output directory.
- `--log-file <path>`: Custom path for the failure log file.

---

## 9. How to Run Automated Tests

Execute pytest from `project_backend/`:
```bash
pytest phase_2_image_preprocessing/tests/ -v
```

---

## 10. Programmatic Python API

For downstream Phase 3 (Image Quality Assessment) and custom workflows:

```python
from pathlib import Path
from phase_2_image_preprocessing import preprocess_image, PreprocessPipeline

# High-level single image API
processed_array, saved_path = preprocess_image(
    input_path=Path("datasets/raw/octa/patient001.png"),
    output_path=Path("datasets/processed/octa/patient001_processed.png"),
    modality="octa",
)

# Pipeline object API
pipeline = PreprocessPipeline(modality="fundus")
processed_array, _ = pipeline.process(
    input_path="datasets/raw/fundus/patient002.png"
)
```

---

## 11. Error Handling & Rerun Safety
- **Fault Isolation**: Batch processing runs in a per-image `try/except` block. A corrupted image or read error will not interrupt processing of subsequent images.
- **Failure Logging**: Any failed image is recorded with a timestamp, modality, file path, and exact error message in `logs/phase2_failed_images.txt`.
- **Rerun Idempotence**: If `patient001_processed.png` already exists in `datasets/processed/<modality>/`, it is safely skipped by default unless `--overwrite` is specified.
- **Reload Verification**: Every processed image written to disk is immediately reopened and checked to verify zero write-corruption.

---

## 12. Limitations & Design Boundaries
- Phase 2 does not assess image quality scores (reserved for Phase 3).
- Phase 2 does not extract feature vectors or run Swin Transformer inferences (reserved for Phase 4).
- Phase 2 does not infer clinical disease phenotypes.
