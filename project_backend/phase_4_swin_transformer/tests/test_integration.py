"""
Tests for Phase 3 -> Phase 4 integration and end-to-end smoke tests.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from phase_4_swin_transformer.enums import DiseaseTask, Modality
from phase_4_swin_transformer.train import train_swin
from phase_4_swin_transformer.config import ModalityTrainingConfig
from integration.phase3_phase4_pipeline import Phase3ToPhase4Integrator


def test_integration_manifest_building(tmp_path: Path):
    approved_dir = tmp_path / "approved"
    octa_dir = approved_dir / "octa"
    octa_dir.mkdir(parents=True)

    # Create dummy approved images
    img1 = octa_dir / "N6A_L_processed.png"
    img2 = octa_dir / "N11C_R_processed.png"
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(img1)
    Image.fromarray(np.random.randint(0, 255, (64, 64), dtype=np.uint8)).save(img2)

    integrator = Phase3ToPhase4Integrator(
        approved_base_dir=approved_dir,
        splits_dir=tmp_path / "splits",
    )
    manifest = integrator.build_modality_manifest(Modality.OCTA, task=DiseaseTask.ALZHEIMERS)
    assert len(manifest) == 2
    assert "image_path" in manifest.columns
    assert "label" in manifest.columns


def test_train_smoke_test(tmp_path: Path):
    """
    Smoke test running a tiny 1-epoch training and evaluation flow to verify pipeline correctness.
    """
    # Create tiny 8-sample dataset (4 normal, 4 disease)
    data = []
    for i in range(8):
        lbl = i % 2
        p = tmp_path / f"img_{i}.png"
        Image.fromarray(np.random.randint(50, 200, (64, 64), dtype=np.uint8)).save(p)
        data.append({
            "image_path": str(p),
            "modality": "octa",
            "label": lbl,
            "class_name": "normal" if lbl == 0 else "disease",
            "patient_id": f"P{i:02d}",
        })
    import pandas as pd
    df = pd.DataFrame(data)

    cfg = ModalityTrainingConfig(
        modality=Modality.OCTA,
        is_color=False,
        image_size=224,
        batch_size=2,
        epochs=1,
        learning_rate=1e-4,
        pretrained=False,
        mixed_precision=False,
        device="cpu",
    )

    exp_dir = tmp_path / "exp_smoke"
    result = train_swin(
        modality=Modality.OCTA,
        data_source=df,
        task=DiseaseTask.ALZHEIMERS,
        config=cfg,
        experiment_dir=exp_dir,
    )

    assert Path(result["best_model_path"]).exists()
    assert (exp_dir / "training_history.csv").exists()
    assert (exp_dir / "metrics.json").exists()
    assert (exp_dir / "confusion_matrix.png").exists()
