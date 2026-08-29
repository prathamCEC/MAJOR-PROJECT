"""
End-to-End Integration tests: Phase 4 Feature Extraction -> DMRA -> Cross-Attention -> URR.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from integration.phase4_phase5_pipeline import Phase4ToPhase5Integrator
from phase_5_retinal_fusion.config import FusionConfig
from phase_5_retinal_fusion.validation import (
    validate_input_features,
    validate_modality_mask,
    validate_urr_output,
)


def test_full_phase4_to_phase5_integration(tmp_path: Path):
    """
    Test real end-to-end integration:
    Creates 3 retinal modality scans -> Runs Phase 4 Swin Encoders -> Fuses in Phase 5 -> Validates URR output.
    """
    # Create sample scans for OCT-A, OCT-B, and Fundus
    p_octa = tmp_path / "patient_octa.png"
    p_octb = tmp_path / "patient_octb.png"
    p_fundus = tmp_path / "patient_fundus.png"

    Image.fromarray(np.random.randint(40, 220, (224, 224), dtype=np.uint8)).save(p_octa)
    Image.fromarray(np.random.randint(40, 220, (224, 224), dtype=np.uint8)).save(p_octb)
    Image.fromarray(np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)).save(p_fundus)

    scans = {
        "octa": p_octa,
        "octb": p_octb,
        "fundus": p_fundus,
    }

    cfg = FusionConfig(embed_dim=256, urr_dim=512, device="cpu")
    integrator = Phase4ToPhase5Integrator(fusion_config=cfg, device="cpu")

    result = integrator.fuse_patient_scans("PAT_001", scans)

    assert result["patient_id"] == "PAT_001"
    assert result["urr"].shape == (1, 512)
    assert len(result["modality_weights"]) == 3
    assert set(result["active_modalities"]) == {"octa", "octb", "fundus"}

    # Verify weights sum to 1.0
    total_w = sum(result["modality_weights"].values())
    assert abs(total_w - 1.0) < 1e-5


def test_validation_utilities():
    feats = {
        "octa": torch.randn(2, 49, 768),
        "octb": torch.randn(2, 49, 768),
    }
    # Should pass without error
    validate_input_features(feats, expected_input_dims={"octa": 768, "octb": 768})

    # Test error on NaN
    feats_nan = {"octa": torch.tensor([[float("nan")] * 768])}
    with pytest.raises(ValueError, match="contains NaN"):
        validate_input_features(feats_nan)
