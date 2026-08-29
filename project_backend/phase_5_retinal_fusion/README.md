# Phase 5 — Dynamic Modality Reliability Attention (DMRA), Cross-Attention Fusion & Unified Retinal Representation (URR)

## 1. Overview & Purpose
Phase 5 implements a multimodal retinal feature fusion architecture that combines modality-specific deep representations (OCT-A, OCT-B, and Fundus) extracted by Phase 4 Swin Transformer backbones into a single, robust **Unified Retinal Representation (URR)**.

The resulting URR vector ($[B, D_{\text{urr}}]$) serves as the official retinal feature representation passed to downstream clinical reasoning modules:
- **Phase 6**: FT-Transformer for Tabular Clinical Data
- **Phase 7**: Retina–Clinical Cross-Attention (Unified Patient Representation)
- **Phase 8**: Multi-Task Disease Prediction (Alzheimer's Disease & Stroke)

```text
                               PHASE 4
                     ┌────────────┼────────────┐
                     ↓            ↓            ↓
                   OCT-A        OCT-B       FUNDUS
                     ↓            ↓            ↓
                [B, 49, 768] [B, 49, 768] [B, 49, 768]
                     │            │            │
                     └────────────┼────────────┘
                                  ↓
                       MODALITY PROJECTION
                      (Linear + LayerNorm)
                                  ↓
                        [B, N, embed_dim=512]
                                  │
                                  ▼
                   DYNAMIC MODALITY RELIABILITY ATTENTION (DMRA)
                 ┌──────────────────────────────────────────────┐
                 │ 1. Average Pool over tokens                  │
                 │ 2. Reliability Scorer (MLP + GELU + Dropout) │
                 │ 3. Masked Softmax (Handles missing modalities│
                 │ 4. Output Modality Weights (w_m >= 0, Σw = 1)│
                 └──────────────────────┬───────────────────────┘
                                        ↓
                           RELIABILITY-WEIGHTED TOKENS
                                        ↓
                        TRANSFORMER CROSS-ATTENTION FUSION
                 ┌──────────────────────────────────────────────┐
                 │ 1. Add Modality Type Embeddings              │
                 │ 2. Multi-Head Cross-Attention (8 heads)      │
                 │ 3. Feed-Forward Network (FFN) + Residuals    │
                 └──────────────────────┬───────────────────────┘
                                        ↓
                       UNIFIED RETINAL REPRESENTATION (URR)
                 ┌──────────────────────────────────────────────┐
                 │ 1. Attentive Pooling / Summary Head          │
                 │ 2. Linear Projection + LayerNorm             │
                 │ 3. Fixed Output Vector [B, urr_dim=512]      │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                                     PHASE 6
                        (Clinical Data Integration)
```

---

## 2. Core Architectural Components

### A. Modality Feature Projections (`modality_projection.py`)
- Maps heterogeneous input feature dimensions ($D_{\text{in}} = 768$) to a unified embedding space ($D_{\text{embed}} = 512$).
- Layer structure: `Linear(768, 512) -> LayerNorm -> GELU -> Dropout(0.1) -> Linear(512, 512) -> LayerNorm`.
- Handles pooled vectors ($[B, D]$), spatial token sequences ($[B, N, D]$), or feature maps ($[B, H, W, D]$).

### B. Dynamic Modality Reliability Attention (DMRA) (`reliability_attention.py`)
- The system dynamically learns per-sample reliability weights ($w_{\text{octa}}, w_{\text{octb}}, w_{\text{fundus}}$) from the actual features rather than using fixed constants.
- Non-linear scoring network per modality:
  $$\bar{X}_m = \text{Mean}(X_m) \in [B, D]$$
  $$s_m = \text{Linear}(\text{GELU}(\text{Linear}(\text{LN}(\bar{X}_m)))) \in [B, 1]$$
- **Masked Softmax**: Automatically handles missing modalities by masking out unavailable channels, guaranteeing:
  $$\sum_{m \in \text{Available}} w_m = 1.0, \quad w_{\text{missing}} = 0.0$$
- Modulates feature tokens: $\tilde{X}_m = w_m \cdot X_m$.

### C. Transformer Cross-Attention Fusion (`cross_attention.py`)
- Genuine multi-head cross-attention mechanism with Query, Key, and Value projections.
- Adds learnable **Modality Type Embeddings** to preserve modality origin.
- Multi-layer Transformer blocks (`Pre-LayerNorm -> MultiheadAttention -> Dropout -> Residual -> FFN -> Residual`).

### D. Unified Retinal Representation (URR) Head (`urr.py`)
- Aggregates the cross-attended token streams into a fixed-dimensional vector $[B, D_{\text{urr}}]$ (default: 512).
- Uses **Learned Attentive Pooling** (`Tanh` scoring with softmax weights) ensuring constant dimensional output regardless of whether 1, 2, or 3 modalities were provided.

---

## 3. Missing Modality Handling

Phase 5 is resilient to incomplete clinical imaging protocols:
- **Case 1**: OCT-A + OCT-B + Fundus (3 modalities)
- **Case 2**: OCT-A + OCT-B (missing Fundus)
- **Case 3**: OCT-A + Fundus (missing OCT-B)
- **Case 4**: OCT-B + Fundus (missing OCT-A)
- **Case 5**: Single Modality (Only OCT-A, Only OCT-B, or Only Fundus)

In all cases, the output URR vector dimension is strictly preserved as $[B, 512]$ with zero runtime failures and zero fabricated random noise.

---

## 4. Phase 4 Connection & Feature Extraction

Phase 5 directly consumes features extracted by the Phase 4 Swin Transformer backbones:
- `Phase4FeatureExtractor` loads trained or pretrained Swin models (`swin_tiny_patch4_window7_224`).
- Calls `model.extract_features(tensor_img, pool=False)` to yield $[B, 49, 768]$ spatial token sequences.
- Feeds extracted tensors directly into the fusion pipeline.

---

## 5. How to Run Phase 5

Execute from `project_backend/`:

### 1. View Model Architecture Summary:
```powershell
python -m phase_5_retinal_fusion.main summary
```

### 2. Fuse Multimodal Retinal Scans for a Patient:
```powershell
python -m phase_5_retinal_fusion.main fuse \
    --octa datasets/approved/octa/sample_01.png \
    --octb datasets/approved/octb/sample_01.png \
    --fundus datasets/approved/fundus/sample_01.png \
    --output phase_5_retinal_fusion/outputs/patient01_urr.pt
```

### 3. Run Phase 4 $\rightarrow$ Phase 5 Integration Pipeline:
```powershell
python -m integration.phase4_phase5_pipeline \
    --octa datasets/approved/octa/octa_sample_1_processed.png \
    --patient-id PATIENT_001
```

---

## 6. Running Automated Tests

Run the complete test suite:
```powershell
python -m pytest phase_5_retinal_fusion/tests/ -v
```
