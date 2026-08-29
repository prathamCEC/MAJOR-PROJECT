"""
Integration tests for Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 -> Phase 8 -> Phase 9 -> Phase 10.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_10_explainability.config import ExplainabilityConfig
from phase_10_explainability.explainability_engine import MultimodalExplainabilityEngine


def test_full_pipeline_patient_multimodal_explainability(tmp_path: Path):
    """
    Test full end-to-end multimodal explainability:
    Phase 4-7 (UPR) -> Phase 8 (Disease Classification) -> Phase 9 (MC Dropout) -> Phase 10 (Grad-CAM + SHAP).
    """
    # 1. Create mock retinal scans
    p_octa = tmp_path / "scan_octa.png"
    p_octb = tmp_path / "scan_octb.png"
    p_fundus = tmp_path / "scan_fundus.png"

    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(p_octa)
    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(p_octb)
    Image.fromarray(np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)).save(p_fundus)

    # 2. Patient Clinical Record
    clinical_record = {
        "ID#": "PATIENT_FULL_EXPLAIN_TEST",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 16,
        "BMI": 26.2,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }

    cfg = ExplainabilityConfig(
        output_dir=str(tmp_path / "outputs"),
        save_visualizations=True,
        shap_background_samples=10,
        mc_samples=5,
    )
    engine = MultimodalExplainabilityEngine(config=cfg, device="cpu")

    res = engine.explain_patient(
        patient_id="PATIENT_FULL_EXPLAIN_TEST",
        retinal_scans={"octa": p_octa, "octb": p_octb, "fundus": p_fundus},
        clinical_record=clinical_record,
        save_plots=True,
    )

    # Verify Output Structure
    assert res["patient_id"] == "PATIENT_FULL_EXPLAIN_TEST"
    assert "stroke" in res and "alzheimer" in res

    st = res["stroke"]
    al = res["alzheimer"]

    # Verify Predictions & Uncertainty
    assert 0.0 <= st["probability"] <= 1.0
    assert 0.0 <= al["probability"] <= 1.0
    assert 0.0 <= st["uncertainty"]["confidence_percent"] <= 100.0
    assert 0.0 <= al["uncertainty"]["confidence_percent"] <= 100.0

    # Verify Grad-CAM
    assert "octa" in st["gradcam"] and "status" in st["gradcam"]["octa"]
    assert st["gradcam"]["octa"]["status"] == "SUCCESS"
    assert st["gradcam"]["octa"]["cam_heatmap"].shape == (224, 224)

    # Verify SHAP
    assert "summary" in st["shap_clinical"]
    assert len(st["shap_clinical"]["summary"]) > 0

    # Verify Modality Attributions
    assert "octa_attribution_percent" in res["modality_attribution"]
    assert "clinical_attribution_percent" in res["modality_attribution"]

    # Verify Disclaimer
    assert "RESEARCH EXPLANATIONS ONLY" in res["disclaimer"]

    # Verify saved figures
    patient_out_dir = tmp_path / "outputs" / "PATIENT_FULL_EXPLAIN_TEST"
    assert (patient_out_dir / "gradcam_stroke_octa.png").exists()
    assert (patient_out_dir / "shap_clinical_stroke.png").exists()
