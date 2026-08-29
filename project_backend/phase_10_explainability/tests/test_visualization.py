"""
Tests for visualization panel generation and saving.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from phase_10_explainability.visualization import save_gradcam_panel, save_shap_bar_chart, overlay_cam_on_image


def test_overlay_and_save_gradcam_panel(tmp_path: Path):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cam = np.random.uniform(0.0, 1.0, (224, 224)).astype(np.float32)

    colored_cam, overlay = overlay_cam_on_image(img, cam, alpha=0.5)
    assert colored_cam.shape == (224, 224, 3)
    assert overlay.shape == (224, 224, 3)

    out_file = tmp_path / "test_panel.png"
    save_gradcam_panel(
        original_img=img,
        cam=cam,
        output_path=out_file,
        modality="octa",
        disease_target="Stroke",
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_save_shap_bar_chart(tmp_path: Path):
    summary = [
        {"feature": "HTN", "value": 1, "shap_value": 0.45, "direction": "INCREASES_RISK"},
        {"feature": "Education", "value": 16, "shap_value": -0.32, "direction": "DECREASES_RISK"},
        {"feature": "BMI", "value": 28.5, "shap_value": 0.12, "direction": "INCREASES_RISK"},
    ]

    out_file = tmp_path / "test_shap.png"
    save_shap_bar_chart(
        shap_summary=summary,
        output_path=out_file,
        disease_target="Stroke",
        base_value=0.1,
    )

    assert out_file.exists()
    assert out_file.stat().st_size > 0
