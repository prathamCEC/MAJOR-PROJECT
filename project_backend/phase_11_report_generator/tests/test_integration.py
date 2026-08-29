"""
Integration test for full multimodal pipeline -> Phase 11 Clinical Report Generation.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from phase_10_explainability.config import ExplainabilityConfig
from phase_10_explainability.explainability_engine import MultimodalExplainabilityEngine
from phase_11_report_generator.config import ReportConfig
from phase_11_report_generator.pipeline import EndToEndReportPipeline


def test_full_pipeline_patient_report_generation(tmp_path: Path):
    # 1. Create mock retinal scan
    scan_p = tmp_path / "scan_octa.png"
    Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8)).save(scan_p)

    # 2. Patient Clinical Record
    clinical_rec = {
        "ID#": "PATIENT_P11_INTEG",
        "Old groups": "O_CD",
        "Gender": 1,
        "Education": 16.0,
        "BMI": 27.2,
        "Obese": 0.0,
        "EtOH_ever": 1,
        "EtOH_current": 0,
        "Smoking_ever": 1,
        "Smoking_current": 0,
        "HTN": 1,
        "DM2": 0,
    }

    rep_cfg = ReportConfig(output_dir=str(tmp_path / "reports"))
    exp_cfg = ExplainabilityConfig(
        output_dir=str(tmp_path / "exp_outputs"),
        save_visualizations=True,
        shap_background_samples=5,
        mc_samples=5,
    )
    exp_engine = MultimodalExplainabilityEngine(config=exp_cfg, device="cpu")

    pipeline = EndToEndReportPipeline(
        report_config=rep_cfg,
        explainability_engine=exp_engine,
    )

    out = pipeline.process_patient_and_generate_report(
        patient_id="PATIENT_P11_INTEG",
        retinal_scans={"octa": scan_p},
        clinical_record=clinical_rec,
    )

    # Verify Output Keys
    assert "pdf_path" in out and "json_path" in out and "report_data" in out
    pdf_p = Path(out["pdf_path"])
    json_p = Path(out["json_path"])

    assert pdf_p.exists() and pdf_p.stat().st_size > 1000
    assert json_p.exists() and json_p.stat().st_size > 100

    report_data = out["report_data"]
    assert report_data.patient_id == "PATIENT_P11_INTEG"
    assert report_data.stroke_assessment.probability >= 0.0
    assert report_data.alzheimer_assessment.probability >= 0.0
