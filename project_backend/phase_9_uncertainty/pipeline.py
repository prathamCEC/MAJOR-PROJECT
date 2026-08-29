"""
End-to-End Multimodal Patient Uncertainty Pipeline.

Orchestrates full multimodal extraction (Phases 4-7), Phase 8 disease classification,
and Phase 9 Monte Carlo Dropout uncertainty estimation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch

from phase_7_retina_clinical_fusion.feature_loader import PatientMultimodalPipeline
from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from .config import UncertaintyConfig, get_default_uncertainty_config
from .engine import MCDropoutUncertaintyEngine


class EndToEndUncertaintyPredictor:
    """
    High-level orchestrator connecting raw multimodal inputs to Phase 9 uncertainty estimates.
    """

    def __init__(
        self,
        config: Optional[UncertaintyConfig] = None,
        multitask_model: Optional[MultiTaskDiseasePredictionNetwork] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_uncertainty_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)

        # 1. Pipeline for Phases 4 -> 5 -> 6 -> 7
        self.phase7_pipeline = PatientMultimodalPipeline(device=str(self.device))

        # 2. Phase 9 Uncertainty Engine
        self.uncertainty_engine = MCDropoutUncertaintyEngine(
            model=multitask_model,
            config=self.config,
            checkpoint_path=checkpoint_path,
            device=str(self.device),
        )

    def evaluate_patient(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
        mc_samples: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full patient multimodal data and generate uncertainty-aware disease estimates.
        """
        # Step A: Extract Unified Patient Representation (UPR)
        patient_out = self.phase7_pipeline.extract_patient_upr(
            patient_id=patient_id,
            retinal_scans=retinal_scans,
            clinical_record=clinical_record,
        )
        upr = patient_out["upr"]  # [1, 512]

        # Step B: Run Phase 9 MC-Dropout Uncertainty Estimation
        unc_res = self.uncertainty_engine.estimate_uncertainty(
            upr=upr,
            mc_samples=mc_samples,
            threshold=threshold,
        )

        st_data = unc_res["stroke"]
        al_data = unc_res["alzheimer"]

        return {
            "patient_id": patient_id,
            "stroke": {
                "mc_mean_probability": float(st_data["mc_mean_probability"][0].item()),
                "mc_variance": float(st_data["mc_variance"][0].item()),
                "mc_std": float(st_data["mc_std"][0].item()),
                "predictive_entropy": float(st_data["predictive_entropy"][0].item()),
                "confidence_percent": float(st_data["confidence_percent"][0].item()),
                "predicted_class": int(st_data["prediction"][0].item()),
            },
            "alzheimer": {
                "mc_mean_probability": float(al_data["mc_mean_probability"][0].item()),
                "mc_variance": float(al_data["mc_variance"][0].item()),
                "mc_std": float(al_data["mc_std"][0].item()),
                "predictive_entropy": float(al_data["predictive_entropy"][0].item()),
                "confidence_percent": float(al_data["confidence_percent"][0].item()),
                "predicted_class": int(al_data["prediction"][0].item()),
            },
            "mc_samples": unc_res["mc_samples"],
            "upr": upr.cpu(),
            "retinal_weights": patient_out["retinal_weights"],
            "gate_weights": patient_out["gate_weights"].cpu(),
            "is_trained_checkpoint": unc_res["is_trained_checkpoint"],
            "disclaimer": unc_res["disclaimer"],
        }
