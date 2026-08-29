"""
End-to-End Report Generation Pipeline.

Coordinates the complete pipeline flow:
Retinal Scans + Clinical Data -> Phase 10 Multimodal Explainability Engine
(includes Phase 4 Swin, Phase 5 DMRA, Phase 6 FT-Transformer, Phase 7 UPR,
Phase 8 Predictions, Phase 9 MC-Dropout) -> Phase 11 Clinical Report (PDF + JSON).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from phase_10_explainability.config import get_default_explainability_config
from phase_10_explainability.explainability_engine import MultimodalExplainabilityEngine
from .config import ReportConfig, get_default_report_config
from .report_generator import ClinicalReportGenerator


class EndToEndReportPipeline:
    """
    Seamless high-level orchestrator generating clinical-style assessment PDF & JSON reports.
    """

    def __init__(
        self,
        report_config: Optional[ReportConfig] = None,
        explainability_engine: Optional[MultimodalExplainabilityEngine] = None,
        device: str = "auto",
    ):
        self.report_config = report_config or get_default_report_config()
        self.explainability_engine = explainability_engine or MultimodalExplainabilityEngine(
            config=get_default_explainability_config(),
            device=device,
        )
        self.report_generator = ClinicalReportGenerator(config=self.report_config)

    def process_patient_and_generate_report(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
        phase3_quality: Optional[Dict[str, Any]] = None,
        report_id: Optional[str] = None,
        pdf_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full patient evaluation and output generated PDF/JSON report paths.
        """
        # Step 1: Execute Phase 10 Explainability (triggers Phases 4-9)
        p10_res = self.explainability_engine.explain_patient(
            patient_id=patient_id,
            retinal_scans=retinal_scans,
            clinical_record=clinical_record,
            save_plots=True,
        )

        st_info = p10_res["stroke"]
        al_info = p10_res["alzheimer"]

        phase8_predictions = {
            "stroke_prediction": st_info["predicted_class"],
            "stroke_probability": st_info["probability"],
            "alzheimer_prediction": al_info["predicted_class"],
            "alzheimer_probability": al_info["probability"],
        }

        phase9_uncertainty = {
            "stroke": st_info["uncertainty"],
            "alzheimer": al_info["uncertainty"],
        }

        present_mods = [k for k, v in retinal_scans.items() if v and Path(v).exists()]

        # Step 2: Generate Report
        report_out = self.report_generator.generate_full_report(
            patient_id=patient_id,
            phase8_predictions=phase8_predictions,
            phase9_uncertainty=phase9_uncertainty,
            phase10_explainability=p10_res,
            phase3_quality=phase3_quality,
            clinical_record=clinical_record,
            modalities_present=present_mods,
            report_id=report_id,
            pdf_path=pdf_path,
            json_path=json_path,
        )

        return report_out
