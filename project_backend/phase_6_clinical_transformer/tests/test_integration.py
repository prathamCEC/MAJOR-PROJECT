"""
Integration tests for Phase 6: Clinical Dataset -> FT-Transformer -> Clinical Representation (CR) -> Phase 7 Interface.
"""

from pathlib import Path
import pandas as pd
import pytest
import torch

from phase_6_clinical_transformer.schema import get_default_retinal_clinical_schema
from phase_6_clinical_transformer.config import get_default_clinical_config, get_project_root
from phase_6_clinical_transformer.validation import ClinicalDataValidator, validate_clinical_representation_output
from phase_6_clinical_transformer.feature_loader import ClinicalFeatureExtractor
from integration.phase5_phase6_interface import MultimodalPatientRepresentationBridge


def test_real_clinical_dataset_extraction():
    """
    Test extraction of Clinical Representations directly from real clinical data (5_ASSOCIATED DATA.xlsx).
    """
    root = get_project_root().parent  # MAJOR-PROJECT root
    excel_path = root / "5_ASSOCIATED DATA.xlsx"

    if not excel_path.exists():
        pytest.skip(f"Real clinical dataset not found at {excel_path}; skipping real data test.")

    df = pd.read_excel(excel_path)
    schema = get_default_retinal_clinical_schema()

    # 1. Audit dataset
    validator = ClinicalDataValidator(schema=schema)
    report = validator.audit_dataframe(df)
    assert report.is_valid, f"Clinical dataset audit failed with errors: {report.errors}"
    assert report.total_records == 28

    # 2. Extract Clinical Representations
    extractor = ClinicalFeatureExtractor()
    extractor.fit_and_initialize(df)
    results = extractor.extract_representations(df, batch_size=8)

    cr = results["clinical_representations"]
    pids = results["patient_ids"]

    # 3. Validate output shape and values
    validate_clinical_representation_output(
        {"clinical_representation": cr},
        expected_batch_size=28,
        expected_dim=512,
    )
    assert len(pids) == 28
    assert pids[0] == "N6A_L"


def test_phase5_phase6_interface_compatibility(tmp_path: Path):
    """
    Verify that Phase 5 Retinal Representation (URR) and Phase 6 Clinical Representation (CR)
    are compatible in dimension (512-dim) and ready for Phase 7 Retina-Clinical Cross-Attention.
    """
    import numpy as np
    from PIL import Image

    # Mock retinal scans
    p_octa = tmp_path / "scan_octa.png"
    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(p_octa)

    clinical_record = {
        "ID#": "PATIENT_01",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 12,
        "BMI": 24.5,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }

    bridge = MultimodalPatientRepresentationBridge(device="cpu")
    out = bridge.extract_unified_patient_inputs(
        patient_id="PATIENT_01",
        retinal_scans={"octa": p_octa},
        clinical_record=clinical_record,
    )

    assert out["is_phase7_compatible"] is True
    assert out["retinal_representation"].shape == (1, 512)
    assert out["clinical_representation"].shape == (1, 512)
