# Phase 7 — Retina–Clinical Cross-Attention Fusion & Unified Patient Representation (UPR)

## 1. Overview & Purpose
Phase 7 implements deep multimodal fusion that unites:
1. **Unified Retinal Representation (URR)** produced by **Phase 5** ($[B, 512]$ or $[B, N, 512]$)
2. **Clinical Representation (CR)** produced by **Phase 6** ($[B, 512]$ or $[B, M, 512]$)

Using **Bidirectional Multi-Head Cross-Attention** and a **Learnable Gated Multimodal Fusion Mechanism**, Phase 7 generates the **Unified Patient Representation (UPR)** ($[B, 512]$).

> [!IMPORTANT]
> Phase 7 does **NOT** perform disease classification (Stroke or Alzheimer's). It produces a unified representation vector specifically structured for consumption by **Phase 8 (Multi-Task Disease Prediction Network)**.

```text
               PHASE 5 URR                            PHASE 6 CR
          (Retinal Representation)              (Clinical Representation)
               [B, D_ret=512]                        [B, D_clin=512]
                     │                                     │
                     ▼                                     ▼
           Retinal Projection                     Clinical Projection
          [B, N, D_common=512]                   [B, M, D_common=512]
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                  BIDIRECTIONAL RETINA–CLINICAL CROSS-ATTENTION
                 ┌──────────────────────────────────────────────┐
                 │ 1. Retina Queries  <- Clinical Keys/Values   │
                 │    (Visual biomarkers attend to history)     │
                 │ 2. Clinical Queries <- Retina Keys/Values    │
                 │    (Clinical history attends to microvasc.)  │
                 │ • Pre-LayerNorm & MultiheadAttention (8 hds) │
                 │ • Residual Connections & FFN (1024-dim)      │
                 └──────────────────────┬───────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ↓                             ↓
               Enhanced Retinal Tokens       Enhanced Clinical Tokens
                 [B, N, D_common=512]          [B, M, D_common=512]
                         ↓                             ↓
                 Attentive Pooling             Attentive Pooling
                         ↓                             ↓
               Retinal Global Vector         Clinical Global Vector
                 [B, D_common=512]             [B, D_common=512]
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         GATED MULTIMODAL FUSION
                 ┌──────────────────────────────────────────────┐
                 │ gate = sigmoid(Linear([v_ret || v_clin]))    │
                 │ v_fused = gate * v_ret + (1 - gate) * v_clin │
                 └──────────────────────┬───────────────────────┘
                                        │
                                        ▼
                         UPR PROJECTION & LAYERNORM
                                        │
                                        ▼
                    UNIFIED PATIENT REPRESENTATION (UPR)
                               [B, UPR_dim=512]
                                        │
                                        ▼
                                     PHASE 8
                        (Multi-Task Disease Prediction)
```

---

## 2. Core Architectural Components

### A. Multimodal Projection Layer (`projection.py`)
- Standardizes both Retinal and Clinical representations into common embedding dimension $D_{\text{common}} = 512$.
- Supports single global vectors $[B, D]$ and multi-token sequences $[B, N, D]$ seamlessly.

### B. Bidirectional Cross-Attention Transformer (`cross_attention.py`)
- **Retina $\leftarrow$ Clinical**: Retinal queries attend to clinical variables (hypertension, diabetes, BMI, age) to highlight clinically vulnerable microvasculature.
- **Clinical $\leftarrow$ Retina**: Clinical queries attend to retinal biomarkers to contextualize systemic findings.
- **Transformer Architecture**: Pre-LayerNorm, 8 heads, GELU activations, dropout, and residual skip connections.

### C. Attentive Sequence Pooling (`pooling.py`)
- Collapses enhanced token sequences into compact global feature vectors $[B, D_{\text{common}}]$.

### D. Gated Multimodal Fusion (`fusion.py`)
- Dynamic gating network computes sample-adaptive balance between retinal and clinical streams:
  $$\text{gate} = \sigma(W_{\text{gate}} [\mathbf{v}_{\text{ret}} \,\|\, \mathbf{v}_{\text{clin}}] + b_{\text{gate}})$$
  $$\mathbf{v}_{\text{fused}} = \text{gate} \odot \mathbf{v}_{\text{ret}} + (1 - \text{gate}) \odot \mathbf{v}_{\text{clin}}$$
- Deep projection head produces the standardized $[B, 512]$ UPR vector.

### E. Numerical Safety & Verification (`validation.py`)
- Verifies batch size alignment, tensor rank, head divisibility, and guards against NaN/Inf values.

---

## 3. How to Run Phase 7

Execute from `project_backend/`:

### 1. View Architecture Summary:
```powershell
python -m phase_7_retina_clinical_fusion.main summary
```

### 2. Fuse Retinal and Clinical Tensors:
```powershell
python -m phase_7_retina_clinical_fusion.main fuse \
    --retinal phase_5_retinal_fusion/outputs/patient_urr.pt \
    --clinical phase_6_clinical_transformer/outputs/clinical_representations.pt \
    --output phase_7_retina_clinical_fusion/outputs/unified_patient_representation.pt
```

---

## 4. Running Automated Tests

Run the complete test suite:
```powershell
python -m pytest phase_7_retina_clinical_fusion/tests/ -v
```
