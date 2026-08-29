"""
End-to-End Pipeline and Batch Processor Integration Tests.
"""

from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pytest

from phase_2_image_preprocessing.src.pipeline import PreprocessPipeline, preprocess_image
from phase_2_image_preprocessing.src.batch_processor import BatchProcessor
from phase_2_image_preprocessing.src.utils import verify_saved_image


@pytest.fixture
def synthetic_retinal_environment(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create a temporary dataset environment with synthetic OCT-A, OCT-B, and Fundus images."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    log_file = tmp_path / "logs" / "phase2_failed_images.txt"

    raw_octa = raw_dir / "octa"
    raw_octb = raw_dir / "octb"
    raw_fundus = raw_dir / "fundus"

    raw_octa.mkdir(parents=True)
    raw_octb.mkdir(parents=True)
    raw_fundus.mkdir(parents=True)

    # 1. Synthetic OCT-A image with fine vessel-like lines
    octa_img = np.zeros((300, 300), dtype=np.uint8)
    octa_img[50:250, 50:250] = 50
    cv2.line(octa_img, (60, 60), (240, 240), 200, 2)
    cv2.imwrite(str(raw_octa / "patient_octa_001.png"), octa_img)

    # 2. Synthetic OCT-B image with horizontal layer bands
    octb_img = np.zeros((200, 400), dtype=np.uint8)
    octb_img[40:70, :] = 120
    octb_img[80:120, :] = 180
    octb_img[130:160, :] = 90
    cv2.imwrite(str(raw_octb / "patient_octb_001.png"), octb_img)

    # 3. Synthetic Fundus image with optic disc and background color
    fundus_img = np.zeros((350, 450, 3), dtype=np.uint8)
    fundus_img[:, :, 0] = 20   # B
    fundus_img[:, :, 1] = 60   # G
    fundus_img[:, :, 2] = 190  # R
    cv2.circle(fundus_img, (150, 175), 40, (80, 200, 240), -1)  # Optic disc
    cv2.imwrite(str(raw_fundus / "patient_fundus_001.png"), fundus_img)

    # 4. A corrupted image file to test failure isolation
    corrupt_file = raw_octa / "patient_octa_corrupt.png"
    corrupt_file.write_bytes(b"CORRUPTED_FILE_DATA_HEADER")

    return raw_dir, processed_dir, log_file


def test_pipeline_octa_end_to_end(synthetic_retinal_environment) -> None:
    """Test OCT-A pipeline execution, shape standardization, and save integrity."""
    raw_dir, processed_dir, _ = synthetic_retinal_environment
    in_path = raw_dir / "octa" / "patient_octa_001.png"
    out_path = processed_dir / "octa" / "patient_octa_001_processed.png"

    pipeline = PreprocessPipeline(modality="octa")
    processed_arr, saved_path = pipeline.process(input_path=in_path, output_path=out_path)

    assert processed_arr.shape == (224, 224, 3)
    assert processed_arr.dtype == np.uint8
    assert saved_path == out_path
    assert out_path.exists()
    assert verify_saved_image(out_path, expected_shape=(224, 224), expected_channels=3)


def test_pipeline_octb_end_to_end(synthetic_retinal_environment) -> None:
    """Test OCT-B pipeline execution with structural layer preservation."""
    raw_dir, processed_dir, _ = synthetic_retinal_environment
    in_path = raw_dir / "octb" / "patient_octb_001.png"
    out_path = processed_dir / "octb" / "patient_octb_001_processed.png"

    processed_arr, saved_path = preprocess_image(
        input_path=in_path, output_path=out_path, modality="octb"
    )

    assert processed_arr.shape == (224, 224, 3)
    assert processed_arr.dtype == np.uint8
    assert saved_path == out_path
    assert out_path.exists()


def test_pipeline_fundus_end_to_end(synthetic_retinal_environment) -> None:
    """Test Fundus pipeline execution with color preservation."""
    raw_dir, processed_dir, _ = synthetic_retinal_environment
    in_path = raw_dir / "fundus" / "patient_fundus_001.png"
    out_path = processed_dir / "fundus" / "patient_fundus_001_processed.png"

    pipeline = PreprocessPipeline(modality="fundus")
    processed_arr, saved_path = pipeline.process(input_path=in_path, output_path=out_path)

    assert processed_arr.shape == (224, 224, 3)
    assert processed_arr.dtype == np.uint8
    # Ensure red channel dominates (color preservation)
    assert np.mean(processed_arr[:, :, 2]) > np.mean(processed_arr[:, :, 0])


def test_raw_data_never_modified(synthetic_retinal_environment) -> None:
    """Test that raw files remain strictly byte-identical after processing."""
    raw_dir, processed_dir, _ = synthetic_retinal_environment
    in_path = raw_dir / "fundus" / "patient_fundus_001.png"
    raw_bytes_before = in_path.read_bytes()

    out_path = processed_dir / "fundus" / "patient_fundus_001_processed.png"
    preprocess_image(input_path=in_path, output_path=out_path, modality="fundus")

    raw_bytes_after = in_path.read_bytes()
    assert raw_bytes_before == raw_bytes_after


def test_batch_processor_execution_and_rerun_safety(synthetic_retinal_environment) -> None:
    """Test batch processing across all modalities, error recovery, and rerun skipping."""
    raw_dir, processed_dir, log_file = synthetic_retinal_environment

    processor = BatchProcessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        log_file=log_file,
        overwrite=False,
    )

    # First Run
    summary1 = processor.run(modality_filter="all")

    # OCT-A had 1 valid and 1 corrupt
    assert summary1.modality_stats["octa"].total == 2
    assert summary1.modality_stats["octa"].successful == 1
    assert summary1.modality_stats["octa"].failed == 1

    # OCT-B had 1 valid
    assert summary1.modality_stats["octb"].total == 1
    assert summary1.modality_stats["octb"].successful == 1
    assert summary1.modality_stats["octb"].failed == 0

    # Fundus had 1 valid
    assert summary1.modality_stats["fundus"].total == 1
    assert summary1.modality_stats["fundus"].successful == 1
    assert summary1.modality_stats["fundus"].failed == 0

    # Overall
    assert summary1.total_images == 4
    assert summary1.total_successful == 3
    assert summary1.total_failed == 1
    assert summary1.total_skipped == 0

    # Check log file contains failure record
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "patient_octa_corrupt.png" in log_content

    # Second Run without overwrite -> should skip previously processed files
    summary2 = processor.run(modality_filter="all")
    assert summary2.total_skipped == 3
    assert summary2.total_successful == 0
    assert summary2.total_failed == 1  # Corrupt image still fails when retried

    # Third Run with overwrite=True -> should re-process instead of skipping
    processor_overwrite = BatchProcessor(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        log_file=log_file,
        overwrite=True,
    )
    summary3 = processor_overwrite.run(modality_filter="all")
    assert summary3.total_successful == 3
    assert summary3.total_skipped == 0
