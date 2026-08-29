"""
Dynamic Modality Reliability Attention (DMRA) Module.

Dynamically evaluates the informational quality and reliability of each retinal
modality (OCT-A, OCT-B, Fundus) per sample, producing learned, normalized modality weights.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleModalityReliabilityScorer(nn.Module):
    """
    Learnable reliability estimation network for a single modality.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute reliability logit from token sequence.

        Args:
            tokens: Tensor of shape [B, N, embed_dim]

        Returns:
            Logit tensor [B, 1]
        """
        # Global average pool over token dimension
        pooled = torch.mean(tokens, dim=1)  # [B, embed_dim]
        return self.net(pooled)  # [B, 1]


class DynamicModalityReliabilityAttention(nn.Module):
    """
    Dynamic Modality Reliability Attention (DMRA) Module.

    Calculates sample-wise modality reliability weights across available modalities
    with support for dynamic masking of missing modalities.
    """

    def __init__(
        self,
        modalities: Optional[List[str]] = None,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.modalities = modalities or ["octa", "octb", "fundus"]
        self.embed_dim = embed_dim
        self.temperature = temperature

        self.scorers = nn.ModuleDict({
            mod: SingleModalityReliabilityScorer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for mod in self.modalities
        })

    def forward(
        self,
        projected_features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward DMRA pass.

        Args:
            projected_features: Dict mapping modality name to projected tokens [B, N_m, embed_dim].
            modality_mask: Optional Dict mapping modality name to binary availability mask [B, 1]
                           (1.0 if present, 0.0 if missing).

        Returns:
            Tuple of:
            - modulated_features: Dict mapping modality to reliability-weighted tokens [B, N_m, embed_dim]
            - modality_weights: Dict mapping modality to normalized reliability weight [B, 1]
            - reliability_logits: Dict mapping modality to raw reliability logit [B, 1]
        """
        # Determine batch size from first available tensor
        first_feat = next(iter(projected_features.values()))
        batch_size = first_feat.shape[0]
        device = first_feat.device

        # Collect raw logits and masks in fixed order
        logits_list = []
        mask_list = []
        ordered_mods = []

        for mod in self.modalities:
            if mod in projected_features:
                ordered_mods.append(mod)
                logit = self.scorers[mod](projected_features[mod])  # [B, 1]
                logits_list.append(logit)

                if modality_mask and mod in modality_mask:
                    m_val = modality_mask[mod].to(device).float()
                    if m_val.ndim == 1:
                        m_val = m_val.unsqueeze(1)
                    mask_list.append(m_val)
                else:
                    mask_list.append(torch.ones(batch_size, 1, device=device))

        if not logits_list:
            raise ValueError("No valid modality features provided to DMRA.")

        # Stack into [B, M]
        logits_tensor = torch.cat(logits_list, dim=1) / self.temperature  # [B, M]
        mask_tensor = torch.cat(mask_list, dim=1)  # [B, M]

        # Numerically stable masked softmax
        # For masked-out modalities (mask == 0), set logits to large negative value
        very_negative = -1e9
        masked_logits = torch.where(mask_tensor > 0.5, logits_tensor, torch.full_like(logits_tensor, very_negative))
        normalized_weights = F.softmax(masked_logits, dim=1)  # [B, M]

        # Re-zero masked out entries in case all modalities in batch were masked
        normalized_weights = normalized_weights * mask_tensor
        sum_weights = torch.sum(normalized_weights, dim=1, keepdim=True)
        # Avoid division by zero if all modalities were 0
        normalized_weights = torch.where(sum_weights > 0, normalized_weights / (sum_weights + 1e-8), normalized_weights)

        # Unpack weights and apply modulation
        modality_weights = {}
        reliability_logits = {}
        modulated_features = {}

        for i, mod in enumerate(ordered_mods):
            w = normalized_weights[:, i : i + 1]  # [B, 1]
            raw_s = logits_list[i]
            modality_weights[mod] = w
            reliability_logits[mod] = raw_s

            # Modulate tokens: [B, N_m, D] * [B, 1, 1]
            modulated_features[mod] = projected_features[mod] * w.unsqueeze(1)

        return modulated_features, modality_weights, reliability_logits
