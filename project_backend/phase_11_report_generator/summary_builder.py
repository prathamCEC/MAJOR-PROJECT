"""
Clinical-Style Summary and Narrative Synthesis Generator.

Generates objective, non-diagnostic research summaries from multimodal predictions,
uncertainty metrics, and explainability attributions.
"""

from typing import Dict, List, Optional
from .report_data import ClinicalReportData, DiseaseAssessmentItem, PatientDemographics


def build_clinical_narrative_summary(
    patient_id: str,
    stroke_item: DiseaseAssessmentItem,
    alzheimer_item: DiseaseAssessmentItem,
    modalities_available: Dict[str, bool],
    demographics: Optional[PatientDemographics] = None,
) -> str:
    """
    Construct a clinical-style narrative summary adhering strictly to research safety rules.
    """
    avail_list = [k.upper() for k, v in modalities_available.items() if v]
    modalities_str = ", ".join(avail_list) if avail_list else "Clinical Data Only"

    st_class_str = "Positive (Risk Indicator Present)" if stroke_item.predicted_class == 1 else "Negative (No Risk Indicator)"
    al_class_str = "Positive (Risk Indicator Present)" if alzheimer_item.predicted_class == 1 else "Negative (No Risk Indicator)"

    summary = (
        f"Multimodal AI assessment was performed for patient '{patient_id}' utilizing available retinal "
        f"modalities ({modalities_str}) and patient tabular clinical health variables. "
        f"For the Stroke target, the model estimated a predicted probability of {stroke_item.probability:.4f} "
        f"({stroke_item.risk_category}), with an estimated model confidence of {stroke_item.confidence_percent:.1f}% "
        f"and {stroke_item.uncertainty_level.lower()} predictive uncertainty (variance={stroke_item.predictive_variance:.4f}). "
        f"For the Alzheimer's Disease target, the model estimated a predicted probability of {alzheimer_item.probability:.4f} "
        f"({alzheimer_item.risk_category}), with an estimated model confidence of {alzheimer_item.confidence_percent:.1f}% "
        f"and {alzheimer_item.uncertainty_level.lower()} predictive uncertainty (variance={alzheimer_item.predictive_variance:.4f}). "
        f"Model explainability heatmaps (Grad-CAM) and game-theoretic clinical feature contributions (SHAP) "
        f"are integrated to highlight computational saliency patterns."
    )
    return summary


def build_multimodal_structural_summary(
    modalities_available: Dict[str, bool],
    modality_attributions: Dict[str, float],
    stroke_item: DiseaseAssessmentItem,
    alzheimer_item: DiseaseAssessmentItem,
) -> str:
    """
    Construct a structured multimodal pathway breakdown summary.
    """
    lines = ["Multimodal Pathway Utilization Summary:"]
    for mod, is_avail in modalities_available.items():
        status = "Processed" if is_avail else "Not available"
        attr_key = f"{mod}_attribution_percent"
        attr_pct = modality_attributions.get(attr_key, 0.0) if is_avail else 0.0
        lines.append(f"  • {mod.upper()}: Status = {status}, Feature Attribution = {attr_pct:.2f}%")

    clin_pct = modality_attributions.get("clinical_attribution_percent", 0.0)
    lines.append(f"  • Tabular Clinical Pathway: Status = Processed, Feature Attribution = {clin_pct:.2f}%")
    lines.append(f"  • Multi-Task Target Decoupling: Stroke ({stroke_item.risk_category}) | Alzheimer's ({alzheimer_item.risk_category})")

    return "\n".join(lines)


def get_default_limitations_text() -> str:
    """Return standard scientific and methodological limitations of the system."""
    return (
        "1. Experimental Model Scope: Algorithms are trained on curated academic datasets and require prospective clinical validation.\n"
        "2. Biological Complexity: Retinal microvascular and neurodegenerative patterns reflect multi-factorial systemic processes.\n"
        "3. Explainability Attribution: Grad-CAM heatmaps and SHAP values indicate model parameter attributions and do not establish medical causality.\n"
        "4. Modality Missingness: Missing imaging modalities degrade fusion representational capacity relative to complete tripartite scans."
    )
