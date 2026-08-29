"""
Monte Carlo Dropout Uncertainty Engine for Stroke and Alzheimer's Prediction.

Coordinates stochastic model passes, statistical aggregation, confidence scoring,
and Phase 10 explainability integration.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from phase_8_multitask_prediction.model import MultiTaskDiseasePredictionNetwork
from .config import UncertaintyConfig, get_default_uncertainty_config
from .mc_dropout import run_mc_forward_passes, enable_mc_dropout, disable_mc_dropout
from .uncertainty import calculate_predictive_statistics
from .confidence import calculate_confidence
from .validation import validate_uncertainty_inputs, validate_uncertainty_outputs


class MCDropoutUncertaintyEngine:
    """
    High-level engine for estimating predictive uncertainty and model confidence
    using Monte Carlo Dropout across Stroke and Alzheimer's disease prediction tasks.
    """

    def __init__(
        self,
        model: Optional[MultiTaskDiseasePredictionNetwork] = None,
        config: Optional[UncertaintyConfig] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
    ):
        self.config = config or get_default_uncertainty_config()
        self.device = self.config.get_device() if device == "auto" else torch.device(device)
        self.is_trained_checkpoint = False

        if checkpoint_path and Path(checkpoint_path).exists():
            self.model, ckpt_meta = MultiTaskDiseasePredictionNetwork.load_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
            )
            self.is_trained_checkpoint = True
        else:
            self.model = model or MultiTaskDiseasePredictionNetwork()
            self.model.to(self.device)

        self.model.eval()

    def estimate_uncertainty(
        self,
        upr: torch.Tensor,
        mc_samples: Optional[int] = None,
        threshold: Optional[float] = None,
        store_mc_predictions: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute Monte Carlo Dropout sampling and compute uncertainty metrics.

        Args:
            upr: Unified Patient Representation tensor [B, 512] or [B, 1, 512]
            mc_samples: Number of stochastic passes (default: config.mc_samples)
            threshold: Decision threshold for binary classification (default: config.classification_threshold)
            store_mc_predictions: Whether to include raw [B, T] prediction matrices

        Returns:
            Structured dictionary with Stroke and Alzheimer's uncertainty metrics.
        """
        n_samples = mc_samples or self.config.mc_samples
        thresh = threshold if threshold is not None else self.config.classification_threshold
        store_raw = (
            store_mc_predictions
            if store_mc_predictions is not None
            else self.config.store_mc_predictions
        )

        validate_uncertainty_inputs(upr=upr, config=self.config, mc_samples=n_samples)

        if upr.ndim == 3 and upr.shape[1] == 1:
            upr = upr.squeeze(1)

        batch_size = upr.shape[0]
        upr_device = upr.to(self.device)

        # 1. Run Stochastic Forward Passes
        mc_res = run_mc_forward_passes(
            model=self.model,
            upr=upr_device,
            mc_samples=n_samples,
        )

        stroke_mc_probs = mc_res["stroke_probabilities"]     # [B, T]
        alz_mc_probs = mc_res["alzheimer_probabilities"]      # [B, T]

        # 2. Compute Predictive Statistics (Mean, Variance, Std, Entropy)
        stroke_stats = calculate_predictive_statistics(
            stroke_mc_probs, epsilon=self.config.epsilon
        )
        alz_stats = calculate_predictive_statistics(
            alz_mc_probs, epsilon=self.config.epsilon
        )

        # 3. Compute Research Confidence Scores
        stroke_conf = calculate_confidence(
            stroke_stats["variance"], uncertainty_scale=self.config.uncertainty_scale
        )
        alz_conf = calculate_confidence(
            alz_stats["variance"], uncertainty_scale=self.config.uncertainty_scale
        )

        # 4. Generate Thresholded Class Predictions
        stroke_preds = (stroke_stats["mean_probability"] >= thresh).long()
        alz_preds = (alz_stats["mean_probability"] >= thresh).long()

        # Build output structure
        results = {
            "stroke": {
                "mc_mean_probability": stroke_stats["mean_probability"].cpu(),
                "mc_variance": stroke_stats["variance"].cpu(),
                "mc_std": stroke_stats["std_deviation"].cpu(),
                "predictive_entropy": stroke_stats["entropy"].cpu(),
                "confidence": stroke_conf["confidence"].cpu(),
                "confidence_percent": stroke_conf["confidence_percent"].cpu(),
                "prediction": stroke_preds.cpu(),
            },
            "alzheimer": {
                "mc_mean_probability": alz_stats["mean_probability"].cpu(),
                "mc_variance": alz_stats["variance"].cpu(),
                "mc_std": alz_stats["std_deviation"].cpu(),
                "predictive_entropy": alz_stats["entropy"].cpu(),
                "confidence": alz_conf["confidence"].cpu(),
                "confidence_percent": alz_conf["confidence_percent"].cpu(),
                "prediction": alz_preds.cpu(),
            },
            "mc_samples": n_samples,
            "threshold": thresh,
            "is_trained_checkpoint": self.is_trained_checkpoint,
            "disclaimer": (
                "RESEARCH PREDICTION AND UNCERTAINTY ESTIMATE ONLY — "
                "NOT A CLINICALLY CALIBRATED DIAGNOSIS OR PROBABILITY OF CORRECTNESS."
            ),
        }

        if store_raw:
            results["stroke"]["mc_predictions"] = stroke_mc_probs.cpu()
            results["alzheimer"]["mc_predictions"] = alz_mc_probs.cpu()

        validate_uncertainty_outputs(results, expected_batch_size=batch_size)
        return results
