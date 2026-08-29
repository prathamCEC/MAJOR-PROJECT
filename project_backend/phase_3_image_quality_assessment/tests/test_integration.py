"""
Integration Tests for Phase 2 -> Phase 3 End-to-End Workflow.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from integration.phase2_phase3_pipeline import IntegratedPipeline


@pytest.fixture
def integrated_test_environment(tmp_path: Path):
    """Setup a full test environment with raw images and routing directories."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    approved_dir = tmp_path / "approved"
    rejected_dir = tmp_path / "rejected"
    log_dir = tmp_path / "logs"

    raw_octa = raw_dir / "octa"
    raw_octb = raw_dir / "octb"
    raw_fundus = raw_dir / "fundus"

    raw_octa.mkdir(parents=True)
    raw_octb.mkdir(parents=True)
    raw_fundus.mkdir(parents=True)

    # 1. Valid raw OCT-A image
    octa = np.zeros((300, 300), dtype=np.uint8)
    octa[50:250, 50:250] = 70
    cv2.line(octa, (60, 60), (240, 240), 220, 2)
    cv2.imwrite(str(raw_octa / "patient_octa.png"), octa)

    # 2. Valid raw Fundus image
    fundus = np.zeros((350, 450, 3), dtype=np.uint8)
    fundus[:, :, 2] = 180  # R
    fundus[:, :, 1] = 70   # G
    fundus[:, :, 0] = 20   # B
    cv2.circle(fundus, (150, 175), 40, (80, 200, 240), -1)
    cv2.imwrite(str(raw_fundus / "patient_fundus.png"), fundus)

    # 3. Corrupt raw file (Phase 2 failure check)
    corrupt = raw_octb / "patient_corrupt.png"
    corrupt.write_bytes(b"CORRUPT_BYTES_DATA")

    return raw_dir, processed_dir, approved_dir, rejected_dir, log_dir


def test_integrated_pipeline_end_to_end(integrated_test_environment) -> None:
    raw_dir, processed_dir, approved_dir, rejected_dir, log_dir = integrated_test_environment

    pipeline = IntegratedPipeline(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        approved_dir=approved_dir,
        rejected_dir=rejected_dir,
        log_dir=log_dir,
    )

    summary = pipeline.run(modality_filter="all")

    assert summary.total_images == 3
    assert summary.modality_stats["octa"].phase2_success == 1
    assert summary.modality_stats["octa"].phase3_assessed == 1

    # Check corrupt file was recorded as Phase 2 failure
    assert summary.modality_stats["octb"].phase2_failed == 1

    # Check output directories contain routed files
    approved_octa = approved_dir / "octa" / "patient_octa_processed.png"
    assert approved_octa.exists() or (rejected_dir / "octa" / "patient_octa_processed.png").exists()

    # Check report files generated
    assert (log_dir / "integrated_pipeline_results.csv").exists()
    assert (log_dir / "integrated_pipeline_results.json").exists()
