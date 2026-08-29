"""
Integration tests for Phase 4 -> Phase 5 -> Phase 6 -> Phase 7.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from phase_7_retina_clinical_fusion.feature_loader import PatientMultimodalPipeline
from phase_7_retina_clinical_fusion.config import get_default_retina_clinical_config, get_project_root


def test_full_pipeline_patient_upr_extraction(tmp_path: Path):
    """
    Test end-to-end execution of full pipeline:
    Phase 4 (Swin) -> Phase 5 (DMRA + URR) -> Phase 6 (Clinical FT-Transformer) -> Phase 7 (UPR).
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
        "ID#": "PATIENT_TEST_01",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 12,
        "BMI": 25.4,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }

    # 3. Run Pipeline Orchestrator
    pipeline = PatientMultimodalPipeline(device="cpu")
    result = pipeline.extract_patient_upr(
        patient_id="PATIENT_TEST_01",
        retinal_scans={
            "octa": p_octa,
            "octb": p_octb,
            "fundus": p_fundus,
        },
        clinical_record=clinical_record,
    )

    # 4. Verify Outputs
    assert result["patient_id"] == "PATIENT_TEST_01"
    assert result["upr"].shape == (1, 512)
    assert result["retinal_urr"].shape == (1, 512)
    assert result["clinical_cr"].shape == (1, 512)
    assert result["gate_weights"].shape == (1, 512)
    assert torch.isfinite(result["upr"]).all()


def test_real_clinical_and_retinal_representation_fusion():
    """
    Test fusion on pre-generated real clinical representations and retinal URR representations.
    """
    root = get_project_root()
    ret_out = root / "phase_5_retinal_fusion" / "outputs" / "patient_urr.pt"
    clin_out = root / "phase_6_clinical_transformer" / "outputs" / "clinical_representations.pt"

    if not ret_out.exists() or not clin_out.exists():
        pytest.skip("Pre-generated phase 5 or phase 6 outputs not found; skipping disk tensor test.")

    ret_data = torch.load(str(ret_out), map_location="cpu", weights_only=False)
    ret_tensor = ret_data["urr"]  # [1, 512]

    clin_data = torch.load(str(clin_out), map_location="cpu", weights_only=False)
    clin_tensor = clin_data["clinical_representations"]  # [28, 512]

    # Align single patient retinal with first clinical patient
    ret_single = ret_tensor[0:1]
    clin_single = clin_tensor[0:1]

    from phase_7_retina_clinical_fusion.fusion_model import RetinaClinicalFusionModel
    model = RetinaClinicalFusionModel()
    model.eval()

    with torch.no_grad():
        res = model(ret_single, clin_single)

    assert res["upr"].shape == (1, 512)
    assert torch.isfinite(res["upr"]).all()
