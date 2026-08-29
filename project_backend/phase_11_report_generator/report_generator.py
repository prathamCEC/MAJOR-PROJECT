"""
Master Clinical Report Generator Engine.

Integrates upstream outputs from:
- Phase 3: Image Quality Assessment
- Phase 8: Multi-Task Disease Predictions (Stroke + Alzheimer's)
- Phase 9: Monte Carlo Dropout Uncertainty & Confidence Metrics
- Phase 10: Swin Grad-CAM Heatmaps & Clinical SHAP Attributions

Synthesizes validated report data and generates both PDF and JSON outputs.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from .config import ReportConfig, get_default_report_config
from .report_data import (
    ClinicalReportData,
    PatientDemographics,
    ImageQualityItem,
    DiseaseAssessmentItem,
    GradCAMItem,
    ClinicalSHAPItem,
    ExplainabilitySummary,
)
from .risk_calculator import (
    calculate_risk_category,
    calculate_confidence_category,
    get_uncertainty_statement,
)
from .summary_builder import (
    build_clinical_narrative_summary,
    build_multimodal_structural_summary,
    get_default_limitations_text,
)
from .pdf_generator import ClinicalPDFReportGenerator
from .json_generator import JSONReportExporter


class ClinicalReportGenerator:
    """
    Master report orchestration engine for Phase 11.
    """

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or get_default_report_config()
        self.pdf_generator = ClinicalPDFReportGenerator(config=self.config)
        self.json_exporter = JSONReportExporter(config=self.config)

    def build_report_data(
        self,
        patient_id: str,
        phase8_predictions: Dict[str, Any],
        phase9_uncertainty: Optional[Dict[str, Any]] = None,
        phase10_explainability: Optional[Dict[str, Any]] = None,
        phase3_quality: Optional[Dict[str, Any]] = None,
        clinical_record: Optional[Dict[str, Any]] = None,
        modalities_present: Optional[List[str]] = None,
        report_id: Optional[str] = None,
    ) -> ClinicalReportData:
        """
        Assemble and validate unified ClinicalReportData from phase outputs.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rep_id = report_id or f"REP-{patient_id[:12]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 1. Modalities availability
        present_set = set(modalities_present or ["octa", "octb", "fundus"])
        modalities_avail = {
            "octa": "octa" in present_set,
            "octb": "octb" in present_set,
            "fundus": "fundus" in present_set,
        }

        # 2. Patient Demographics & Clinical Profile
        c_rec = clinical_record or {}
        demographics = PatientDemographics(
            patient_id=patient_id,
            age_group=str(c_rec.get("Old groups", "Not provided")),
            gender="Male (1)" if str(c_rec.get("Gender")) == "1" else ("Female (0)" if str(c_rec.get("Gender")) == "0" else "Not provided"),
            education_years=str(c_rec.get("Education", "Not provided")),
            bmi=str(c_rec.get("BMI", "Not provided")),
            hypertension="Positive (1)" if str(c_rec.get("HTN")) == "1" else ("Negative (0)" if str(c_rec.get("HTN")) == "0" else "Not provided"),
            diabetes_type2="Positive (1)" if str(c_rec.get("DM2")) == "1" else ("Negative (0)" if str(c_rec.get("DM2")) == "0" else "Not provided"),
            smoking_status="History (1)" if str(c_rec.get("Smoking_ever")) == "1" else "None (0)",
            alcohol_status="History (1)" if str(c_rec.get("EtOH_ever")) == "1" else "None (0)",
            raw_clinical_record=c_rec,
        )

        # 3. Image Quality (Phase 3)
        quality_map = {}
        q_src = phase3_quality or {}
        for mod in ("octa", "octb", "fundus"):
            if mod in q_src:
                q_item = q_src[mod]
                quality_map[mod] = ImageQualityItem(
                    modality=mod,
                    available=True,
                    quality_score=q_item.get("quality_score"),
                    decision=q_item.get("decision", "ACCEPT"),
                    details=q_item.get("metrics", {}),
                )
            else:
                quality_map[mod] = ImageQualityItem(
                    modality=mod,
                    available=modalities_avail.get(mod, False),
                    quality_score=None,
                    decision="Not available" if not modalities_avail.get(mod, False) else "ACCEPT (Unchecked)",
                )

        # 4. Phase 8 Multi-Task Predictions
        st_pred = int(phase8_predictions.get("stroke_prediction", 0))
        st_prob = float(phase8_predictions.get("stroke_probability", 0.0))
        al_pred = int(phase8_predictions.get("alzheimer_prediction", 0))
        al_prob = float(phase8_predictions.get("alzheimer_probability", 0.0))

        # 5. Phase 9 Uncertainty / Confidence Integration
        p9_st = (phase9_uncertainty or {}).get("stroke", {})
        p9_al = (phase9_uncertainty or {}).get("alzheimer", {})

        st_conf = float(p9_st.get("confidence_percent", 95.0))
        st_unc_lvl = str(p9_st.get("uncertainty_level", "LOW"))
        st_conf_lvl = str(p9_st.get("confidence_level", calculate_confidence_category(st_conf, self.config)))
        st_var = float(p9_st.get("predictive_variance", 0.001))
        st_ent = float(p9_st.get("predictive_entropy", 0.1))
        st_is_elevated = bool(p9_st.get("is_elevated_uncertainty", False))

        al_conf = float(p9_al.get("confidence_percent", 95.0))
        al_unc_lvl = str(p9_al.get("uncertainty_level", "LOW"))
        al_conf_lvl = str(p9_al.get("confidence_level", calculate_confidence_category(al_conf, self.config)))
        al_var = float(p9_al.get("predictive_variance", 0.001))
        al_ent = float(p9_al.get("predictive_entropy", 0.1))
        al_is_elevated = bool(p9_al.get("is_elevated_uncertainty", False))

        st_assessment = DiseaseAssessmentItem(
            disease_name="Stroke",
            predicted_class=st_pred,
            probability=st_prob,
            confidence_percent=st_conf,
            uncertainty_level=st_unc_lvl,
            confidence_level=st_conf_lvl,
            predictive_variance=st_var,
            predictive_entropy=st_ent,
            risk_category=calculate_risk_category(st_prob, self.config),
            is_elevated_uncertainty=st_is_elevated,
        )

        al_assessment = DiseaseAssessmentItem(
            disease_name="Alzheimer's Disease",
            predicted_class=al_pred,
            probability=al_prob,
            confidence_percent=al_conf,
            uncertainty_level=al_unc_lvl,
            confidence_level=al_conf_lvl,
            predictive_variance=al_var,
            predictive_entropy=al_ent,
            risk_category=calculate_risk_category(al_prob, self.config),
            is_elevated_uncertainty=al_is_elevated,
        )

        # 6. Phase 10 Explainability Integration
        p10 = phase10_explainability or {}
        exp_st = p10.get("stroke", {})
        exp_al = p10.get("alzheimer", {})

        # Grad-CAM items
        st_gcam = {}
        for mod, item in exp_st.get("gradcam", {}).items():
            st_gcam[mod] = GradCAMItem(
                modality=mod,
                status=item.get("status", "UNAVAILABLE"),
                panel_path=item.get("panel_path"),
            )

        al_gcam = {}
        for mod, item in exp_al.get("gradcam", {}).items():
            al_gcam[mod] = GradCAMItem(
                modality=mod,
                status=item.get("status", "UNAVAILABLE"),
                panel_path=item.get("panel_path"),
            )

        # SHAP items
        st_shap_list = []
        for s_item in exp_st.get("shap_clinical", {}).get("summary", []):
            st_shap_list.append(ClinicalSHAPItem(
                feature_name=s_item.get("feature", "Unknown"),
                patient_value=s_item.get("value", "—"),
                shap_value=float(s_item.get("shap_value", 0.0)),
                direction=s_item.get("direction", "NEUTRAL"),
            ))

        al_shap_list = []
        for s_item in exp_al.get("shap_clinical", {}).get("summary", []):
            al_shap_list.append(ClinicalSHAPItem(
                feature_name=s_item.get("feature", "Unknown"),
                patient_value=s_item.get("value", "—"),
                shap_value=float(s_item.get("shap_value", 0.0)),
                direction=s_item.get("direction", "NEUTRAL"),
            ))

        explainability_bundle = ExplainabilitySummary(
            modality_attributions=p10.get("modality_attribution", {}),
            stroke_gradcam=st_gcam,
            alzheimer_gradcam=al_gcam,
            stroke_shap_clinical=st_shap_list,
            alzheimer_shap_clinical=al_shap_list,
            stroke_shap_plot_path=exp_st.get("shap_clinical", {}).get("plot_path"),
            alzheimer_shap_plot_path=exp_al.get("shap_clinical", {}).get("plot_path"),
        )

        # 7. Summaries & Narratives
        clinical_narrative = build_clinical_narrative_summary(
            patient_id=patient_id,
            stroke_item=st_assessment,
            alzheimer_item=al_assessment,
            modalities_available=modalities_avail,
            demographics=demographics,
        )

        multimodal_summary = build_multimodal_structural_summary(
            modalities_available=modalities_avail,
            modality_attributions=p10.get("modality_attribution", {}),
            stroke_item=st_assessment,
            alzheimer_item=al_assessment,
        )

        report_data = ClinicalReportData(
            report_id=rep_id,
            patient_id=patient_id,
            generated_at=now_str,
            system_version=self.config.document_version,
            modalities_available=modalities_avail,
            patient_demographics=demographics,
            image_quality=quality_map,
            stroke_assessment=st_assessment,
            alzheimer_assessment=al_assessment,
            explainability=explainability_bundle,
            clinical_summary_text=clinical_narrative,
            multimodal_summary_text=multimodal_summary,
            limitations_text=get_default_limitations_text(),
            disclaimer_text=self.config.disclaimer,
        )

        return report_data

    def generate_full_report(
        self,
        patient_id: str,
        phase8_predictions: Dict[str, Any],
        phase9_uncertainty: Optional[Dict[str, Any]] = None,
        phase10_explainability: Optional[Dict[str, Any]] = None,
        phase3_quality: Optional[Dict[str, Any]] = None,
        clinical_record: Optional[Dict[str, Any]] = None,
        modalities_present: Optional[List[str]] = None,
        report_id: Optional[str] = None,
        pdf_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        save_csv_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        Compile report, write PDF and JSON artifacts, and append summary CSV.

        Returns:
            Dict containing 'pdf_path', 'json_path', 'csv_path', and 'report_data'.
        """
        report_data = self.build_report_data(
            patient_id=patient_id,
            phase8_predictions=phase8_predictions,
            phase9_uncertainty=phase9_uncertainty,
            phase10_explainability=phase10_explainability,
            phase3_quality=phase3_quality,
            clinical_record=clinical_record,
            modalities_present=modalities_present,
            report_id=report_id,
        )

        out_pdf = self.pdf_generator.generate_pdf(report_data=report_data, output_filepath=pdf_path)
        out_json = self.json_exporter.export_json(report_data=report_data, output_filepath=json_path)

        out_csv = None
        if save_csv_summary:
            out_csv = self.json_exporter.append_summary_csv(report_data=report_data)

        return {
            "report_id": report_data.report_id,
            "patient_id": report_data.patient_id,
            "pdf_path": str(out_pdf),
            "json_path": str(out_json),
            "csv_path": str(out_csv) if out_csv else None,
            "report_data": report_data,
        }
