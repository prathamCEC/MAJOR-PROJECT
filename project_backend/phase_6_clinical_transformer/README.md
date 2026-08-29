# Phase 6 — FT-Transformer for Structured Clinical Data & Clinical Representation (CR)

## 1. Overview & Purpose
Phase 6 implements a **Feature Tokenizer Transformer (FT-Transformer)** architecture tailored for structured patient clinical attributes. It transforms heterogeneous tabular clinical variables (continuous measurements, categorical factors, binary health markers) into an information-rich, fixed-dimensional **Clinical Representation (CR)** vector ($[B, D_{\text{cr}}]$).

The resulting Clinical Representation is designed specifically to interface with:
- **Phase 5**: Unified Retinal Representation (URR, $[B, 512]$)
- **Phase 7**: Retina–Clinical Cross-Attention (Unified Patient Representation)
- **Phase 8**: Multi-Task Disease Prediction (Stroke & Alzheimer's Disease)

```text
                           CLINICAL DATA
                     (5_ASSOCIATED DATA.xlsx)
                                 │
                 ┌───────────────┴───────────────┐
                 ↓                               ↓
       Numerical Features              Categorical Features
        (BMI, Education)           (Old groups, Gender, Obese,
                                    EtOH, Smoking, HTN, DM2)
                 ↓                               ↓
       Median Imputation &             Vocabulary Mapping &
       Standard Normalization           Unknown Category Handling
                 ↓                               ↓
       Numerical Tokenizer             Categorical Tokenizer
        (e_j = x_j * W_j + b_j)         (Embedding Lookup)
                 └───────────────┬───────────────┘
                                 ↓
                         Feature Tokens
                     [B, N_features, D=256]
                                 ↓
                         Prepend [CLS] Token
                     [B, 1 + N_features, D=256]
                                 │
                                 ▼
                     FT-TRANSFORMER BACKBONE
                 ┌──────────────────────────────────────┐
                 │ • Pre-LayerNorm                      │
                 │ • Multi-Head Self-Attention (8 heads)│
                 │ • Feed-Forward Network (512-dim)     │
                 │ • Stack of 3 Transformer Blocks      │
                 └──────────────────┬───────────────────┘
                                    ↓
                         Extracted [CLS] Token
                                    ↓
                     CLINICAL REPRESENTATION HEAD
                   (LayerNorm + Linear Projection)
                                    │
                                    ▼
                      CLINICAL REPRESENTATION (CR)
                            [B, D_cr = 512]
                                    │
                                    ▼
                                 PHASE 7
                    (Retina–Clinical Cross-Attention)
```

---

## 2. Core Architectural Components

### A. Clinical Schema System (`schema.py`)
- Standardizes feature roles without hard-coding assumptions.
- Default schema configured for retinal clinical cohort (`5_ASSOCIATED DATA.xlsx`):
  - **Numerical**: `BMI`, `Education`
  - **Categorical**: `Old groups`, `Gender`, `Obese`
  - **Binary**: `EtOH_ever`, `EtOH_current`, `Smoking_ever`, `Smoking_current`, `HTN`, `DM2`
  - **Metadata (Excluded from Model)**: `ID#` (Patient identifier), `AD` (Target label).

### B. Preprocessing & Leakage Prevention (`preprocessing.py`)
- **Zero Leakage**: Scaler statistics (means, standard deviations) and imputation medians are computed **strictly on the training split**.
- **Missing Value Handling**: Numerical missing values are replaced by training medians. Categorical missing values are mapped to index `0` (`<UNK>`).
- **Safe Inference**: Unseen categories at inference time automatically map to `<UNK>` without runtime failures.

### C. Feature Tokenizer (`feature_tokenizer.py`)
- Implements authentic FT-Transformer tokenization (Gorishniy et al., NeurIPS 2021):
  - Numerical features: $e_j = x_j \cdot W_j + b_j \in \mathbb{R}^D$ where $W_j \in \mathbb{R}^D, b_j \in \mathbb{R}^D$ are learnable.
  - Categorical features: $e_k = \text{Embedding}_k(c_k) \in \mathbb{R}^D$.
- Prepends a learnable `[CLS]` token $\in \mathbb{R}^{1 \times 1 \times D}$, outputting $[B, 1 + N_{\text{features}}, D]$.

### D. FT-Transformer Backbone (`ft_transformer.py`)
- Multi-layer Pre-LayerNorm Transformer blocks:
  - Multi-Head Self-Attention (8 heads, embedding dimension 256).
  - Feed-Forward Network with GELU activations and dropout.
  - Residual skip connections.

### E. Clinical Representation Head (`clinical_representation.py`)
- Extracts the `[CLS]` token from the Transformer sequence.
- Projects to a 512-dimensional output vector matching Phase 5 URR.

---

## 3. Patient Privacy & Identifier Handling

Patient identifiers (`ID#`) are strictly treated as **metadata**:
- Never passed into the FT-Transformer as feature tokens.
- Preserved separately alongside tensors to link patient retinal representations with clinical representations.
- `patient_level_split()` enforces strict group partitioning to ensure zero patient overlap between train, validation, and test splits.

---

## 4. How to Run Phase 6

Execute from `project_backend/`:

### 1. View Architecture & Schema Summary:
```powershell
python -m phase_6_clinical_transformer.main summary
```

### 2. Audit Clinical Dataset File:
```powershell
python -m phase_6_clinical_transformer.main validate --data ../5_ASSOCIATED DATA.xlsx
```

### 3. Extract Clinical Representations:
```powershell
python -m phase_6_clinical_transformer.main extract \
    --data ../5_ASSOCIATED DATA.xlsx \
    --output phase_6_clinical_transformer/outputs/clinical_representations.pt
```

---

## 5. Running Automated Tests

Run the complete test suite:
```powershell
python -m pytest phase_6_clinical_transformer/tests/ -v
```
