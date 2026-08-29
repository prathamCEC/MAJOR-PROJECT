"""
Integration tests for Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8 -> Phase 9.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from phase_9_uncertainty.config import get_project_root
from phase_9_uncertainty.engine import MCDropoutUncertaintyEngine
from phase_9_uncertainty.pipeline import EndToEndUncertaintyPredictor


def test_full_pipeline_patient_uncertainty_evaluation(tmp_path: Path):
    """
    Test end-to-end execution of full pipeline:
    Phase 4 (Swin) -> Phase 5 (DMRA) -> Phase 6 (FT-Transformer) -> Phase 7 (UPR)
    -> Phase 8 (Multi-Task) -> Phase 9 (MC Dropout Uncertainty & Confidence).
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
        "ID#": "PATIENT_UNCERTAINTY_TEST",
        "Old groups": "O_CD",
        "Gender": 0,
        "Education": 16,
        "BMI": 24.5,
        "Obese": 0.0,
        "EtOH_ever": 0,
        "EtOH_current": 0,
        "Smoking_ever": 0,
        "Smoking_current": 0,
        "HTN": 0,
        "DM2": 0,
    }

    # 3. Run EndToEndUncertaintyPredictor
    predictor = EndToEndUncertaintyPredictor(device="cpu")
    res = predictor.evaluate_patient(
        patient_id="PATIENT_UNCERTAINTY_TEST",
        retinal_scans={"octa": p_octa, "octb": p_octb, "fundus": p_fundus},
        clinical_record=clinical_record,
        mc_samples=5,
    )

    # 4. Verify Outputs
    assert res["patient_id"] == "PATIENT_UNCERTAINTY_TEST"
    assert "stroke" in res and "alzheimer" in res
    assert 0.0 <= res["stroke"]["mc_mean_probability"] <= 1.0
    assert 0.0 <= res["alzheimer"]["mc_mean_probability"] <= 1.0
    assert res["stroke"]["mc_variance"] >= 0.0
    assert res["alzheimer"]["mc_variance"] >= 0.0
    assert 0.0 <= res["stroke"]["confidence_percent"] <= 100.0
    assert 0.0 <= res["alzheimer"]["confidence_percent"] <= 100.0
    assert res["stroke"]["predicted_class"] in (0, 1)
    assert res["alzheimer"]["predicted_class"] in (0, 1)
    assert "NOT A CLINICALLY CALIBRATED DIAGNOSIS" in res["disclaimer"]


def test_uncertainty_from_saved_phase7_upr_tensor():
    """
    Test direct uncertainty estimation on disk-saved Phase 7 UPR tensor.
    """
    root = get_project_root()
    upr_path = root / "phase_7_retina_clinical_fusion" / "outputs" / "unified_patient_representation.pt"

    if not upr_path.exists():
        pytest.skip(f"Phase 7 UPR file not found at {upr_path}; skipping disk test.")

    data = torch.load(str(upr_path), map_location="cpu", weights_only=False)
    upr_tensor = data["upr"]

    engine = MCDropoutUncertaintyEngine(device="cpu")
    res = engine.estimate_uncertainty(upr=upr_tensor, mc_samples=10)

    assert "stroke" in res and "alzheimer" in res
    assert res["stroke"]["mc_mean_probability"].shape == (upr_tensor.shape[0],)
    assert res["alzheimer"]["mc_mean_probability"].shape == (upr_tensor.shape[0],)
    assert res["stroke"]["confidence_percent"].shape == (upr_tensor.shape[0],)
