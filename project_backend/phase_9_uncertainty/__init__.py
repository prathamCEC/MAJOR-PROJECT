"""
Phase 9: Monte Carlo Dropout & Confidence/Uncertainty Estimation Engine.

Performs stochastic forward passes on top of Phase 8 Multi-Task Disease Prediction Network
to compute predictive mean probability, variance, standard deviation, predictive entropy,
and bounded research confidence scores.
"""

from .config import (
    UncertaintyConfig,
    get_default_uncertainty_config,
    get_phase9_outputs_dir,
)
from .mc_dropout import (
    enable_mc_dropout,
    disable_mc_dropout,
    run_mc_forward_passes,
    DROPOUT_MODULE_TYPES,
)
from .uncertainty import (
    calculate_predictive_statistics,
    calculate_predictive_entropy,
)
from .confidence import calculate_confidence
from .validation import (
    validate_uncertainty_inputs,
    validate_uncertainty_outputs,
)
from .engine import MCDropoutUncertaintyEngine
from .pipeline import EndToEndUncertaintyPredictor

__version__ = "1.0.0"

__all__ = [
    "UncertaintyConfig",
    "get_default_uncertainty_config",
    "get_phase9_outputs_dir",
    "enable_mc_dropout",
    "disable_mc_dropout",
    "run_mc_forward_passes",
    "DROPOUT_MODULE_TYPES",
    "calculate_predictive_statistics",
    "calculate_predictive_entropy",
    "calculate_confidence",
    "validate_uncertainty_inputs",
    "validate_uncertainty_outputs",
    "MCDropoutUncertaintyEngine",
    "EndToEndUncertaintyPredictor",
]
