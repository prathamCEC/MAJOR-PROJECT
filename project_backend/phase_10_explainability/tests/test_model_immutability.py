"""
Tests verifying model parameter immutability during Grad-CAM and SHAP explainability passes.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_10_explainability.explainability_engine import MultimodalExplainabilityEngine


def test_model_parameters_unchanged_after_explanation(tmp_path: Path):
    model = MultiTaskDiseasePredictionNetwork()

    # Capture deep copy of weights before explanation
    params_before = {name: param.clone() for name, param in model.named_parameters()}

    # Create mock scan
    img_path = tmp_path / "test_scan.png"
    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(img_path)

    clinical_record = {
        "ID#": "PATIENT_IMMUTABLE_TEST",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 14.0,
        "BMI": 25.0,
        "Obese": 0.0,
        "EtOH_ever": 0,
        "EtOH_current": 0,
        "Smoking_ever": 0,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }

    engine = MultimodalExplainabilityEngine(multitask_model=model, device="cpu")
    engine.explain_patient(
        patient_id="PATIENT_IMMUTABLE_TEST",
        retinal_scans={"octa": img_path},
        clinical_record=clinical_record,
        save_plots=False,
    )

    # Verify every weight tensor is bit-identical
    for name, param in model.named_parameters():
        assert torch.equal(params_before[name], param), f"Parameter '{name}' was modified during explainability!"
