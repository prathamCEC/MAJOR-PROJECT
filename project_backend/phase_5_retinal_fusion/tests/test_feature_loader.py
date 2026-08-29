"""
Tests for Phase4FeatureExtractor adapter.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_5_retinal_fusion.feature_loader import Phase4FeatureExtractor


def test_feature_extractor_on_mock_scans(tmp_path: Path):
    extractor = Phase4FeatureExtractor(device="cpu", pretrained_backbone=False)

    # Create mock scan images
    img_octa = tmp_path / "mock_octa.png"
    img_fundus = tmp_path / "mock_fundus.png"

    Image.fromarray(np.random.randint(50, 200, (64, 64), dtype=np.uint8)).save(img_octa)
    Image.fromarray(np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)).save(img_fundus)

    # 1. Single scan extraction
    feat_octa = extractor.extract_from_image_path(img_octa, modality="octa", pool=False)
    assert feat_octa.shape == (1, 49, 768)

    feat_fundus_pooled = extractor.extract_from_image_path(img_fundus, modality="fundus", pool=True)
    assert feat_fundus_pooled.shape == (1, 768)

    # 2. Patient scan extraction
    scans = {"octa": img_octa, "fundus": img_fundus}
    feats, masks = extractor.extract_multimodal_patient_features(scans, pool=False)

    assert "octa" in feats and feats["octa"].shape == (1, 49, 768)
    assert "fundus" in feats and feats["fundus"].shape == (1, 49, 768)
    assert masks["octa"].item() == 1.0
    assert masks["octb"].item() == 0.0
    assert masks["fundus"].item() == 1.0
