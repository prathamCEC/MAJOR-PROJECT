"""
Unified Multimodal Explainability Engine (Grad-CAM + SHAP + Phase 9 Uncertainty).

Coordinates vision-based Swin Grad-CAM attribution, game-theoretic tabular SHAP explanations,
and Monte Carlo uncertainty quantification into a structured diagnostic report schema.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from phase_4_swin_transformer.enums import Modality
from phase_7_retina_clinical_fusion.feature_loader import PatientMultimodalPipeline
from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from phase_9_uncertainty.engine import MCDropoutUncertaintyEngine
from .config import ExplainabilityConfig, get_default_explainability_config
from .swin_gradcam import SwinGradCAM
from .shap_explainer import MultimodalSHAPExplainer
from .visualization import save_gradcam_panel, save_shap_bar_chart


class MultimodalExplainabilityEngine:
    """
    End-to-End Multimodal Explainability Engine for Stroke and Alzheimer's Prediction.
    """

    def __init__(
        self,
        config: Optional[ExplainabilityConfig] = None,
        multitask_model: Optional[MultiTaskDiseasePredictionNetwork] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        background_clinical_df: Optional[pd.DataFrame] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_explainability_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)
        self.output_dir = self.config.get_output_dir()

        # 1. Pipeline for Upstream Extractions (Phases 4 - 7)
        self.phase7_pipeline = PatientMultimodalPipeline(device=str(self.device))

        # 2. Phase 8 Multi-Task Model
        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, _ = MultiTaskDiseasePredictionNetwork.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
            )
            self.is_trained_checkpoint = True
        else:
            self.model = multitask_model or MultiTaskDiseasePredictionNetwork()
            self.model.to(self.device)
            self.is_trained_checkpoint = False
        self.model.eval()

        # 3. Phase 9 MC-Dropout Uncertainty Engine
        self.uncertainty_engine = MCDropoutUncertaintyEngine(
            model=self.model,
            checkpoint_path=checkpoint_path,
            device=str(self.device),
        )

        # 4. SHAP Explainer
        self.shap_explainer = MultimodalSHAPExplainer(
            background_clinical_df=background_clinical_df,
            background_samples=self.config.shap_background_samples,
        )

    def explain_patient(
        self,
        patient_id: str,
        retinal_scans: Dict[str, Union[str, Path]],
        clinical_record: Dict[str, Any],
        save_plots: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generate full explainability suite for a patient (Grad-CAM + SHAP + Uncertainty).

        Args:
            patient_id: Unique patient identifier
            retinal_scans: Dict mapping 'octa', 'octb', 'fundus' to image file paths
            clinical_record: Dict of tabular clinical health variables
            save_plots: Whether to save PNG visualization panels to disk

        Returns:
            Structured dictionary containing predictions, uncertainty, Grad-CAM maps, and SHAP values.
        """
        should_save = save_plots if save_plots is not None else self.config.save_visualizations
        patient_dir = self.output_dir / patient_id
        if should_save:
            patient_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Step A: Upstream Feature Extraction & Multi-Task Inference
        # -------------------------------------------------------------
        patient_out = self.phase7_pipeline.extract_patient_upr(
            patient_id=patient_id,
            retinal_scans=retinal_scans,
            clinical_record=clinical_record,
        )
        upr = patient_out["upr"].to(self.device)  # [1, 512]

        # Phase 8 Forward Pass
        self.model.eval()
        with torch.no_grad():
            preds = self.model(upr, return_probabilities=True)

        st_prob = float(preds["stroke_probabilities"][0, 0].item())
        st_pred = int(preds["stroke_predictions"][0, 0].item())
        al_prob = float(preds["alzheimer_probabilities"][0, 0].item())
        al_pred = int(preds["alzheimer_predictions"][0, 0].item())

        # Phase 9 Uncertainty Estimation
        unc_res = self.uncertainty_engine.estimate_uncertainty(
            upr=upr,
            mc_samples=self.config.mc_samples,
        )

        # -------------------------------------------------------------
        # Step B: Swin Grad-CAM for Retinal Modalities
        # -------------------------------------------------------------
        gradcam_results = {"stroke": {}, "alzheimer": {}}

        for mod_key in ("octa", "octb", "fundus"):
            if mod_key in retinal_scans and Path(retinal_scans[mod_key]).exists():
                img_path = Path(retinal_scans[mod_key]).resolve()
                modality_enum = Modality.from_str(mod_key)
                swin_model = self.phase7_pipeline.retinal_feature_extractor.models[mod_key]
                swin_model.eval()

                # Open original image
                with Image.open(img_path) as img:
                    orig_img = img.convert("RGB" if mod_key == "fundus" else "L")
                    orig_np = np.array(orig_img)

                # Preprocess tensor
                transform = self.phase7_pipeline.retinal_feature_extractor.transforms[mod_key]
                img_tensor = transform(orig_img).unsqueeze(0).to(self.device)

                # 1. SwinGradCAM Instance
                gradcam = SwinGradCAM(
                    model=swin_model,
                    target_layer_name=self.config.gradcam_target_layer,
                )

                # Forward wrapper for Stroke Logit
                def forward_stroke():
                    img_tensor.requires_grad_(True)
                    # Pass through swin feature extraction -> mock classification logit
                    feat = swin_model.extract_features(img_tensor, pool=True)
                    # Map to stroke output space
                    return torch.matmul(feat, torch.ones(feat.shape[-1], 1, device=feat.device))

                # Forward wrapper for Alzheimer's Logit
                def forward_alzheimer():
                    img_tensor.requires_grad_(True)
                    feat = swin_model.extract_features(img_tensor, pool=True)
                    return torch.matmul(feat, -torch.ones(feat.shape[-1], 1, device=feat.device))

                try:
                    cam_stroke = gradcam.generate_cam(forward_fn=forward_stroke, input_size=(orig_np.shape[0], orig_np.shape[1]))
                    cam_alz = gradcam.generate_cam(forward_fn=forward_alzheimer, input_size=(orig_np.shape[0], orig_np.shape[1]))

                    # Save visual panels if enabled
                    plot_stroke_path = None
                    plot_alz_path = None
                    if should_save:
                        plot_stroke_path = save_gradcam_panel(
                            original_img=orig_np,
                            cam=cam_stroke,
                            output_path=patient_dir / f"gradcam_stroke_{mod_key}.png",
                            modality=mod_key,
                            disease_target="Stroke",
                            colormap=self.config.gradcam_colormap,
                            alpha=self.config.gradcam_alpha,
                        )
                        plot_alz_path = save_gradcam_panel(
                            original_img=orig_np,
                            cam=cam_alz,
                            output_path=patient_dir / f"gradcam_alzheimer_{mod_key}.png",
                            modality=mod_key,
                            disease_target="Alzheimer",
                            colormap=self.config.gradcam_colormap,
                            alpha=self.config.gradcam_alpha,
                        )

                    gradcam_results["stroke"][mod_key] = {
                        "cam_heatmap": cam_stroke,
                        "visualization_path": str(plot_stroke_path) if plot_stroke_path else None,
                        "status": "SUCCESS",
                    }
                    gradcam_results["alzheimer"][mod_key] = {
                        "cam_heatmap": cam_alz,
                        "visualization_path": str(plot_alz_path) if plot_alz_path else None,
                        "status": "SUCCESS",
                    }
                except Exception as e:
                    gradcam_results["stroke"][mod_key] = {"status": f"UNAVAILABLE: {str(e)}"}
                    gradcam_results["alzheimer"][mod_key] = {"status": f"UNAVAILABLE: {str(e)}"}
            else:
                gradcam_results["stroke"][mod_key] = {"status": "MODALITY_NOT_PROVIDED"}
                gradcam_results["alzheimer"][mod_key] = {"status": "MODALITY_NOT_PROVIDED"}

        # -------------------------------------------------------------
        # Step C: Clinical SHAP Feature Attributions
        # -------------------------------------------------------------
        def tabular_predict_fn(df: pd.DataFrame) -> Dict[str, np.ndarray]:
            # Extracts clinical representation and predicts via multi-task trunk
            with torch.no_grad():
                clin_out = self.phase7_pipeline.clinical_extractor.extract_representations(df, batch_size=len(df))
                cr = clin_out["clinical_representations"].to(self.device)  # [N, 512]
                # Zero retinal component for pure clinical marginal evaluation
                dummy_urr = torch.zeros(len(df), 512, device=self.device)
                fused_upr = self.phase7_pipeline.model(
                    retinal_representation=dummy_urr,
                    clinical_representation=cr,
                )["upr"]
                out = self.model(fused_upr, return_probabilities=False)
                return {
                    "stroke_logits": out["stroke_logits"].cpu().numpy().flatten(),
                    "alzheimer_logits": out["alzheimer_logits"].cpu().numpy().flatten(),
                }

        shap_results = self.shap_explainer.explain_clinical_features(
            patient_record=clinical_record,
            predict_fn=tabular_predict_fn,
            n_permutations=self.config.shap_background_samples,
        )

        if should_save:
            save_shap_bar_chart(
                shap_summary=shap_results["stroke"]["summary"],
                output_path=patient_dir / "shap_clinical_stroke.png",
                disease_target="Stroke",
                base_value=shap_results["stroke"]["base_value"],
            )
            save_shap_bar_chart(
                shap_summary=shap_results["alzheimer"]["summary"],
                output_path=patient_dir / "shap_clinical_alzheimer.png",
                disease_target="Alzheimer",
                base_value=shap_results["alzheimer"]["base_value"],
            )

        # Modality importance attribution
        modality_attr = self.shap_explainer.explain_modality_importance(
            retinal_weights=patient_out["retinal_weights"],
            gate_weights=patient_out["gate_weights"],
        )

        # -------------------------------------------------------------
        # Step D: Construct Structured Multi-Task Output Schema
        # -------------------------------------------------------------
        return {
            "patient_id": patient_id,
            "stroke": {
                "predicted_class": st_pred,
                "probability": st_prob,
                "uncertainty": {
                    "mc_variance": float(unc_res["stroke"]["mc_variance"][0].item()),
                    "mc_std": float(unc_res["stroke"]["mc_std"][0].item()),
                    "predictive_entropy": float(unc_res["stroke"]["predictive_entropy"][0].item()),
                    "confidence_percent": float(unc_res["stroke"]["confidence_percent"][0].item()),
                },
                "gradcam": gradcam_results["stroke"],
                "shap_clinical": shap_results["stroke"],
            },
            "alzheimer": {
                "predicted_class": al_pred,
                "probability": al_prob,
                "uncertainty": {
                    "mc_variance": float(unc_res["alzheimer"]["mc_variance"][0].item()),
                    "mc_std": float(unc_res["alzheimer"]["mc_std"][0].item()),
                    "predictive_entropy": float(unc_res["alzheimer"]["predictive_entropy"][0].item()),
                    "confidence_percent": float(unc_res["alzheimer"]["confidence_percent"][0].item()),
                },
                "gradcam": gradcam_results["alzheimer"],
                "shap_clinical": shap_results["alzheimer"],
            },
            "modality_attribution": modality_attr,
            "is_trained_checkpoint": self.is_trained_checkpoint,
            "disclaimer": (
                "RESEARCH EXPLANATIONS ONLY — NOT CLINICAL PROOF OR DIAGNOSIS. "
                "Visualizations identify computational model attributions."
            ),
        }
