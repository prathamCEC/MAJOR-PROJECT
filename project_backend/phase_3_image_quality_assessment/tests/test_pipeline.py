"""
End-to-End Pipeline Tests for Phase 3 Image Quality Assessment.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from phase_3_image_quality_assessment.src.pipeline import assess_image, assess_image_file, QualityAssessmentPipeline
from phase_3_image_quality_assessment.src.batch_processor import Phase3BatchProcessor


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create sample processed images under octa, octb, fundus."""
    octa_dir = tmp_path / "octa"
    octb_dir = tmp_path / "octb"
    fundus_dir = tmp_path / "fundus"

    octa_dir.mkdir(parents=True)
    octb_dir.mkdir(parents=True)
    fundus_dir.mkdir(parents=True)

    # 1. High-quality OCT-A
    octa = np.zeros((224, 224, 3), dtype=np.uint8)
    octa[40:180, 40:180] = 80
    cv2.line(octa, (50, 50), (170, 170), (220, 220, 220), 2)
    cv2.imwrite(str(octa_dir / "good_octa.png"), octa)

    # 2. High-quality Fundus
    fundus = np.zeros((224, 224, 3), dtype=np.uint8)
    fundus[30:190, 30:190, 2] = 180  # R
    fundus[30:190, 30:190, 1] = 80   # G
    fundus[30:190, 30:190, 0] = 30   # B
    cv2.circle(fundus, (112, 112), 25, (60, 190, 230), -1)
    cv2.imwrite(str(fundus_dir / "good_fundus.png"), fundus)

    # 3. Defective low-contrast / blurred OCT-B
    octb = np.ones((224, 224, 3), dtype=np.uint8) * 128
    octb[0, 0] = 129
    cv2.imwrite(str(octb_dir / "bad_octb.png"), octb)

    return tmp_path


def test_assess_octa_file(sample_dataset: Path) -> None:
    img_path = sample_dataset / "octa" / "good_octa.png"
    result = assess_image_file(img_path, modality="octa")

    assert result.modality == "octa"
    assert result.overall_score > 0.0
    assert result.decision in ("ACCEPT", "WARNING")


def test_assess_fundus_file(sample_dataset: Path) -> None:
    img_path = sample_dataset / "fundus" / "good_fundus.png"
    result = assess_image_file(img_path, modality="fundus")

    assert result.modality == "fundus"
    assert "color_score" in result.scores
    assert result.scores["color_score"] > 50.0


def test_batch_processor(sample_dataset: Path, tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    processor = Phase3BatchProcessor(input_dir=sample_dataset, log_dir=log_dir)
    summary = processor.run(modality_filter="all")

    assert summary.total_images == 3
    assert processor.csv_path.exists()
    assert processor.json_path.exists()
