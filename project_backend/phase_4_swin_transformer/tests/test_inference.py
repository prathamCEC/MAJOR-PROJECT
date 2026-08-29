"""
Tests for SwinInferenceEngine.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.models.swin_factory import create_swin_model
from phase_4_swin_transformer.checkpoint import CheckpointManager
from phase_4_swin_transformer.inference import SwinInferenceEngine


@pytest.fixture
def mock_checkpoint(tmp_path: Path) -> Path:
    model = create_swin_model(Modality.OCTA, num_classes=2, pretrained=False)
    opt = torch.optim.AdamW(model.parameters())
    mgr = CheckpointManager(tmp_path)
    ckpt_p = mgr.save_checkpoint(
        model=model,
        optimizer=opt,
        scheduler=None,
        epoch=1,
        best_metric=0.9,
        class_mapping={"normal": 0, "disease": 1},
        modality=Modality.OCTA,
        task=DiseaseTask.ALZHEIMERS,
        is_best=True,
    )
    return ckpt_p


def test_inference_single_and_batch(mock_checkpoint: Path, tmp_path: Path):
    # Create test images
    img_dir = tmp_path / "test_imgs"
    img_dir.mkdir()
    img1 = img_dir / "sample1.png"
    img2 = img_dir / "sample2.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(img1)
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(img2)

    engine = SwinInferenceEngine(checkpoint_path=mock_checkpoint, modality=Modality.OCTA)

    # Single prediction
    pred = engine.predict_image(img1)
    assert pred.image_name == "sample1.png"
    assert pred.predicted_class in ("normal", "disease")
    assert 0.0 <= pred.confidence <= 1.0
    assert len(pred.probabilities) == 2

    # Batch prediction
    csv_out = tmp_path / "preds.csv"
    batch_preds = engine.predict_batch(img_dir, output_csv_path=csv_out)
    assert len(batch_preds) == 2
    assert csv_out.exists()
