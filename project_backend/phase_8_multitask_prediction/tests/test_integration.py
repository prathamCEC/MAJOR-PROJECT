"""
Integration tests for Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from phase_8_multitask_prediction.config import get_project_root
from phase_8_multitask_prediction.inference import EndToEndDiseasePredictor


def test_full_pipeline_patient_disease_prediction(tmp_path: Path):
    """
    Test end-to-end execution of full pipeline:
    Phase 4 (Swin) -> Phase 5 (DMRA) -> Phase 6 (FT-Transformer) -> Phase 7 (UPR) -> Phase 8 (Disease Predictions).
    """
    # 1. Generate mock retinal scans
    p_octa = tmp_path / "scan_octa.png"
    p_octb = tmp_path / "scan_octb.png"
    p_fundus = tmp_path / "scan_fundus.png"

    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(p_octa)
    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(p_octb)
    Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)).save(p_fundus)

    # 2. Patient Clinical Record
    clinical_record = {
        "ID#": "PATIENT_FULL_EVAL",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 12,
        "BMI": 26.2,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 1,
    }

    # 3. Run EndToEndDiseasePredictor
    predictor = EndToEndDiseasePredictor(device="cpu")
    res = predictor.predict_patient(
        patient_id="PATIENT_FULL_EVAL",
        retinal_scans={"octa": p_octa, "octb": p_octb, "fundus": p_fundus},
        clinical_record=clinical_record,
    )

    # 4. Verify Outputs
    assert res["patient_id"] == "PATIENT_FULL_EVAL"
    assert "stroke" in res and "alzheimer" in res
    assert 0.0 <= res["stroke"]["probability"] <= 1.0
    assert res["stroke"]["predicted_class"] in (0, 1)
    assert 0.0 <= res["alzheimer"]["probability"] <= 1.0
    assert res["alzheimer"]["predicted_class"] in (0, 1)
    assert res["upr"].shape == (1, 512)
    assert "NOT CLINICAL DIAGNOSIS" in res["disclaimer"]


def test_predict_from_saved_phase7_upr_tensor():
    """
    Test direct disease prediction from disk-saved Phase 7 UPR tensor.
    """
    root = get_project_root()
    upr_path = root / "phase_7_retina_clinical_fusion" / "outputs" / "unified_patient_representation.pt"

    if not upr_path.exists():
        pytest.skip(f"Phase 7 UPR file not found at {upr_path}; skipping disk test.")

    data = torch.load(str(upr_path), map_location="cpu", weights_only=False)
    upr_tensor = data["upr"]

    predictor = EndToEndDiseasePredictor(device="cpu")
    res = predictor.predict_from_upr(upr_tensor)

    assert "stroke" in res and "alzheimer" in res
    assert res["stroke"]["probability"].shape == (upr_tensor.shape[0], 1)
    assert res["alzheimer"]["probability"].shape == (upr_tensor.shape[0], 1)
