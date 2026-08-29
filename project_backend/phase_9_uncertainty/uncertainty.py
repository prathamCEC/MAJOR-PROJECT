"""
Uncertainty Calculation Module for Monte Carlo Predictions.

Computes sample-wise predictive mean, variance, standard deviation, and predictive entropy
across stochastic Monte Carlo draws.
"""

from typing import Dict
import torch


def calculate_predictive_entropy(
    probabilities: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Shannon binary predictive entropy for probabilities in [0, 1].

    H(p) = -p * ln(p) - (1-p) * ln(1-p)

    Args:
        probabilities: Tensor of predictive probabilities [B]
        epsilon: Small numerical clamp value to prevent log(0)

    Returns:
        Entropy tensor [B] (values in [0, ln(2)] ~= [0, 0.6931])
    """
    p = torch.clamp(probabilities, min=epsilon, max=1.0 - epsilon)
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return entropy


def calculate_predictive_statistics(
    mc_probabilities: torch.Tensor,
    epsilon: float = 1e-7,
) -> Dict[str, torch.Tensor]:
    """
    Calculate full suite of uncertainty and dispersion statistics from MC draws.

    Args:
        mc_probabilities: Tensor of stochastic probabilities with shape [B, T]
        epsilon: Numerical stability parameter for entropy

    Returns:
        Dict containing:
        - 'mean_probability': Tensor [B] (expected predictive probability)
        - 'variance': Tensor [B] (sample variance across MC passes)
        - 'std_deviation': Tensor [B] (standard deviation)
        - 'entropy': Tensor [B] (predictive Shannon entropy)
    """
    if mc_probabilities.ndim != 2:
        raise ValueError(
            f"Expected MC probabilities tensor of rank 2 ([B, T]), got shape {tuple(mc_probabilities.shape)}."
        )

    B, T = mc_probabilities.shape
    if T < 2:
        raise ValueError(f"Uncertainty calculation requires at least T >= 2 passes, got T={T}.")

    # 1. Predictive Mean: E[p] = (1/T) * sum(p_t)
    mean_prob = torch.mean(mc_probabilities, dim=1)  # [B]

    # 2. Predictive Variance: Var(p) = (1 / (T - 1)) * sum((p_t - E[p])^2)
    # Using unbiased sample variance
    var_prob = torch.var(mc_probabilities, dim=1, unbiased=True)  # [B]
    # Guard against minor floating point negatives
    var_prob = torch.clamp(var_prob, min=0.0)

    # 3. Standard Deviation: Std(p) = sqrt(Var(p))
    std_prob = torch.sqrt(var_prob)  # [B]

    # 4. Predictive Entropy: H(E[p])
    entropy_prob = calculate_predictive_entropy(mean_prob, epsilon=epsilon)  # [B]

    return {
        "mean_probability": mean_prob,
        "variance": var_prob,
        "std_deviation": std_prob,
        "entropy": entropy_prob,
    }
