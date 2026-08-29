# Phase 9 — Monte Carlo Dropout & Confidence/Uncertainty Estimation Engine

## 1. Overview & Purpose
Phase 9 implements **Monte Carlo Dropout (MC-Dropout)** uncertainty quantification for the **Phase 8 Multi-Task Disease Prediction Network**. By performing $T$ independent stochastic forward passes with dropout layers activated at test time, Phase 9 estimates the epistemic and predictive uncertainty for **Stroke** and **Alzheimer's Disease** classifications.

> [!WARNING]
> **RESEARCH PROTOTYPE NOTICE**:
> This software is an academic/research model. Uncertainty and confidence scores represent statistical dispersion across model sub-networks and **MUST NOT** be interpreted as clinically validated probabilities of correctness or medical diagnostic claims.

```text
                  UNIFIED PATIENT REPRESENTATION (UPR)
                             [B, UPR_dim=512]
                                    │
                                    ▼
                 PHASE 8 MULTI-TASK NETWORK (MC-MODE)
             ┌──────────────────────────────────────────┐
             │ • model.eval() with DROPOUT ACTIVE       │
             │ • T Stochastic Forward Passes (default 30)│
             └──────────────────────┬───────────────────┘
                                    │
                  Stochastic Predictions [B, T]
                                    │
                   ┌────────────────┴────────────────┐
                   ↓                                 ↓
            STROKE UNCERTAINTY              ALZHEIMER'S UNCERTAINTY
        ┌────────────────────────────┐    ┌────────────────────────────┐
        │ • Predictive Mean (mu)     │    │ • Predictive Mean (mu)     │
        │ • Variance (sigma^2)       │    │ • Variance (sigma^2)       │
        │ • Std Dev (sigma)          │    │ • Std Dev (sigma)          │
        │ • Shannon Entropy H(p)     │    │ • Shannon Entropy H(p)     │
        │ • Confidence Score (%)     │    │ • Confidence Score (%)     │
        │ • Thresholded Class Pred   │    │ • Thresholded Class Pred   │
        └────────────────────────────┘    └────────────────────────────┘
```

---

## 2. Theoretical Formulation

### A. Fine-Grained MC-Dropout Activation (`enable_mc_dropout`)
Instead of calling `model.train()` (which inappropriately activates training behaviors in LayerNorm and BatchNorm), `enable_mc_dropout(model)` sets the model to `eval()` and selectively enables `train(True)` **only** on dropout modules (`nn.Dropout`, `nn.AlphaDropout`, etc.).

### B. Predictive Mean & Variance
For $T$ stochastic probability draws $p_1, p_2, \dots, p_T$:
- **Predictive Mean**:
  $$\mu = \frac{1}{T}\sum_{t=1}^T p_t \in [0, 1]$$
- **Predictive Variance (Sample Variance)**:
  $$\sigma^2 = \frac{1}{T-1}\sum_{t=1}^T (p_t - \mu)^2 \ge 0$$
- **Predictive Standard Deviation**:
  $$\sigma = \sqrt{\sigma^2} \ge 0$$

### C. Shannon Predictive Entropy
Quantifies prediction dispersion in information-theoretic nats:
$$H(p) = -p \ln(p) - (1-p) \ln(1-p)$$
where $p = \text{clamp}(\mu, \epsilon, 1-\epsilon)$ with $\epsilon = 10^{-7}$.

### D. Bounded Research Confidence Metric
Theoretical maximum Bernoulli variance is $\sigma_{\max}^2 = 0.25$ (when $p=0.5$):
$$\text{normalized\_uncertainty} = \text{clamp}\left(\frac{\sigma^2}{0.25}, 0.0, 1.0\right)$$
$$\text{confidence} = 1.0 - \text{normalized\_uncertainty} \in [0.0, 1.0]$$
$$\text{confidence\_percent} = \text{confidence} \times 100.0 \in [0.0, 100.0]$$

---

## 3. How to Run Phase 9

Execute from `project_backend/`:

### 1. View Uncertainty Configuration Summary:
```powershell
python -m phase_9_uncertainty.main summary
```

### 2. Estimate Stroke & Alzheimer's Uncertainty from UPR Tensor:
```powershell
python -m phase_9_uncertainty.main estimate \
    --upr phase_7_retina_clinical_fusion/outputs/unified_patient_representation.pt \
    --samples 30 \
    --threshold 0.5 \
    --output phase_9_uncertainty/outputs/uncertainty_estimates.pt
```

---

## 4. Running Automated Tests

Run the complete Phase 9 test suite:
```powershell
python -m pytest phase_9_uncertainty/tests/ -v
```
