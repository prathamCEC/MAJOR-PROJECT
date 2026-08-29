"""
Inference and Full-Pipeline Disease Predictor Module.

Coordinates end-to-end evaluation from patient multimodal data (Phases 4-7)
into research model predictions for Stroke and Alzheimer's Disease.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from phase_7_retina_clinical_fusion.feature_loader import PatientMultimodalPipeline
from .config import MultiTaskConfig, get_default_multitask_config
from .model import MultiTaskDiseasePredictionNetwork


class EndToEndDiseasePredictor:
    """
    High-level predictor orchestrating Phases 4 to 8.
    """

    def __init__(
        self,
        multitask_model: Optional[MultiTaskDiseasePredictionNetwork] = None,
        config: Optional[MultiTaskConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_multitask_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)

        # 1. Phase 4 -> Phase 5 -> Phase 6 -> Phase 7 Pipeline
        self.phase7_pipeline = PatientMultimodalPipeline(device=str(self.device))

        # 2. Phase 8 Multi-Task Model
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, self.ckpt_meta = MultiTaskDiseasePredictionNetwork.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
            )
        else:
            self.model = multitask_model or MultiTaskDiseasePredictionNetwork(config=self.config)
            self.model.to(self.device)
        self.model.eval()

    def predict_from_upr(
        self,
        upr: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run multi-task inference directly from Unified Patient Representation tensor.
        """
        self.model.eval()
        with torch.no_grad():
            upr_tensor = upr.to(self.device)
            out = self.model(upr_tensor, return_probabilities=True, threshold=threshold)

        return {
            "stroke": {
                "logit": out["stroke_logits"].cpu(),
                "probability": out["stroke_probabilities"].cpu(),
                "prediction": out["stroke_predictions"].cpu(),
            },
            "alzheimer": {
                "logit": out["alzheimer_logits"].cpu(),
                "probability": out["alzheimer_probabilities"].cpu(),
                "prediction": out["alzheimer_predictions"].cpu(),
            },
            "shared_features": out["shared_features"].cpu(),
            "is_research_prototype": True,
            "disclaimer": "RESEARCH PREDICTION ONLY — NOT CLINICAL DIAGNOSIS",
        }

    def predict_patient(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full end-to-end patient evaluation from raw scans & clinical variables to disease predictions.
        """
        # Step A: Run Phases 4 to 7 to extract Unified Patient Representation (UPR)
        patient_out = self.phase7_pipeline.extract_patient_upr(
            patient_id=patient_id,
            retinal_scans=retinal_scans,
            clinical_record=clinical_record,
        )
        upr = patient_out["upr"]  # [1, 512]

        # Step B: Run Phase 8 Multi-Task Classification
        pred_res = self.predict_from_upr(upr=upr, threshold=threshold)

        return {
            "patient_id": patient_id,
            "stroke": {
                "logit": float(pred_res["stroke"]["logit"][0, 0].item()),
                "probability": float(pred_res["stroke"]["probability"][0, 0].item()),
                "predicted_class": int(pred_res["stroke"]["prediction"][0, 0].item()),
            },
            "alzheimer": {
                "logit": float(pred_res["alzheimer"]["logit"][0, 0].item()),
                "probability": float(pred_res["alzheimer"]["probability"][0, 0].item()),
                "predicted_class": int(pred_res["alzheimer"]["prediction"][0, 0].item()),
            },
            "upr": upr.cpu(),
            "retinal_weights": patient_out["retinal_weights"],
            "gate_weights": patient_out["gate_weights"].cpu(),
            "disclaimer": "RESEARCH PREDICTION ONLY — NOT CLINICAL DIAGNOSIS",
        }
