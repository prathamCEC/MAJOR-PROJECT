"""
Monte Carlo Dropout Activation and Stochastic Forward Pass Module.

Manages fine-grained layer-level execution states: enables stochastic dropout mask generation
while preserving evaluation mode across normalization layers (LayerNorm, BatchNorm).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn

# Supported PyTorch Dropout module types
DROPOUT_MODULE_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def enable_mc_dropout(model: nn.Module) -> int:
    """
    Configure model for Monte Carlo Dropout inference.

    Sets model into eval mode, then specifically sets only dropout modules to train(True)
    so that stochastic sub-networks are sampled during inference without corrupting
    LayerNorm or BatchNorm running statistics.

    Args:
        model: PyTorch model instance

    Returns:
        Number of active dropout layers detected and enabled.
    """
    model.eval()
    dropout_count = 0

    for module in model.modules():
        if isinstance(module, DROPOUT_MODULE_TYPES):
            module.train(True)
            dropout_count += 1

    return dropout_count


def disable_mc_dropout(model: nn.Module) -> None:
    """
    Restore standard deterministic evaluation mode across all model layers.
    """
    model.eval()


def run_mc_forward_passes(
    model: nn.Module,
    upr: torch.Tensor,
    mc_samples: int = 30,
) -> Dict[str, torch.Tensor]:
    """
    Execute multiple stochastic forward passes through the model with active dropout.

    Args:
        model: Phase 8 MultiTaskDiseasePredictionNetwork
        upr: Input Unified Patient Representation tensor [B, upr_dim]
        mc_samples: Number of stochastic passes (T >= 2)

    Returns:
        Dict containing:
        - 'stroke_probabilities': Tensor [B, T]
        - 'alzheimer_probabilities': Tensor [B, T]
        - 'stroke_logits': Tensor [B, T]
        - 'alzheimer_logits': Tensor [B, T]
    """
    if mc_samples < 2:
        raise ValueError(f"Monte Carlo Dropout requires mc_samples >= 2, received {mc_samples}.")

    dropout_layers = enable_mc_dropout(model)
    if dropout_layers == 0:
        raise RuntimeError(
            "Monte Carlo Dropout requires at least one active dropout layer in the model architecture. "
            "Zero dropout layers were detected."
        )

    batch_size = upr.shape[0]
    stroke_probs_list = []
    alz_probs_list = []
    stroke_logits_list = []
    alz_logits_list = []

    try:
        with torch.no_grad():
            for _ in range(mc_samples):
                out = model(upr, return_probabilities=True)

                st_logit = out["stroke_logits"]          # [B, 1]
                al_logit = out["alzheimer_logits"]       # [B, 1]
                st_prob = out["stroke_probabilities"]    # [B, 1]
                al_prob = out["alzheimer_probabilities"] # [B, 1]

                stroke_logits_list.append(st_logit.squeeze(-1))
                alz_logits_list.append(al_logit.squeeze(-1))
                stroke_probs_list.append(st_prob.squeeze(-1))
                alz_probs_list.append(al_prob.squeeze(-1))

        # Stack over sample dimension: [B, T]
        stroke_probs = torch.stack(stroke_probs_list, dim=1)
        alz_probs = torch.stack(alz_probs_list, dim=1)
        stroke_logits = torch.stack(stroke_logits_list, dim=1)
        alz_logits = torch.stack(alz_logits_list, dim=1)

        return {
            "stroke_probabilities": stroke_probs,
            "alzheimer_probabilities": alz_probs,
            "stroke_logits": stroke_logits,
            "alzheimer_logits": alz_logits,
        }
    finally:
        disable_mc_dropout(model)
