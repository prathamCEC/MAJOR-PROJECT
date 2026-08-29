"""
Tests for Multimodal SHAP feature attributions.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from phase_10_explainability.shap_explainer import MultimodalSHAPExplainer


def test_shap_clinical_feature_attributions():
    explainer = MultimodalSHAPExplainer(background_samples=10, random_seed=42)

    patient_record = {
        "ID#": "PATIENT_SHAP_01",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 16.0,
        "BMI": 28.5,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 1,
    }

    # Linear mock prediction function
    def mock_predict_fn(df: pd.DataFrame):
        n = len(df)
        st_logits = np.array([float(r.get("HTN", 0) * 0.5 + r.get("DM2", 0) * 0.4) for _, r in df.iterrows()])
        al_logits = np.array([float(r.get("Education", 12) * -0.1) for _, r in df.iterrows()])
        return {"stroke_logits": st_logits, "alzheimer_logits": al_logits}

    shap_res = explainer.explain_clinical_features(
        patient_record=patient_record,
        predict_fn=mock_predict_fn,
        n_permutations=15,
    )

    assert "stroke" in shap_res and "alzheimer" in shap_res
    st_shap = shap_res["stroke"]
    al_shap = shap_res["alzheimer"]

    assert len(st_shap["feature_names"]) > 0
    assert len(st_shap["shap_values"]) == len(st_shap["feature_names"])
    assert np.isfinite(st_shap["shap_values"]).all()

    # HTN should increase stroke risk (positive SHAP) in mock
    summary_dict = {item["feature"]: item for item in st_shap["summary"]}
    assert "HTN" in summary_dict
    assert summary_dict["HTN"]["shap_value"] >= 0.0


def test_modality_importance_attribution():
    explainer = MultimodalSHAPExplainer()

    ret_weights = {"octa": 0.5, "octb": 0.3, "fundus": 0.2}
    gate_weights = torch.tensor([[0.6]])  # 60% retinal, 40% clinical

    mod_attr = explainer.explain_modality_importance(ret_weights, gate_weights)

    assert "octa_attribution_percent" in mod_attr
    assert "clinical_attribution_percent" in mod_attr
    total_pct = sum(mod_attr.values())
    assert np.isclose(total_pct, 100.0, atol=1e-1)
