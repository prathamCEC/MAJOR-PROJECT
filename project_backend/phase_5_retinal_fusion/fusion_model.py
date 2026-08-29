"""
End-to-End Retinal Multimodal Fusion Model.

Integrates Modality Projections, Dynamic Modality Reliability Attention (DMRA),
Cross-Attention Transformer Fusion, and Unified Retinal Representation (URR).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .config import FusionConfig, get_default_fusion_config
from .modality_projection import MultiModalityProjection
from .reliability_attention import DynamicModalityReliabilityAttention
from .cross_attention import RetinalCrossAttentionFusion
from .urr import UnifiedRetinalRepresentationHead


class RetinalMultimodalFusionModel(nn.Module):
    """
    Complete Phase 5 Multimodal Retinal Fusion Architecture.

    Pipeline:
    1. Modality Feature Projections (OCT-A, OCT-B, Fundus -> common embed_dim)
    2. Dynamic Modality Reliability Attention (DMRA -> learned sample-wise weights)
    3. Multi-Head Cross-Attention Transformer Fusion (modality token interaction)
    4. Unified Retinal Representation Head (URR vector [B, urr_dim])
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or get_default_fusion_config()

        # 1. Modality Projections
        self.projections = MultiModalityProjection(
            input_dims=self.config.input_dims,
            embed_dim=self.config.embed_dim,
            dropout=self.config.dropout,
        )

        # 2. Dynamic Modality Reliability Attention (DMRA)
        self.dmra = DynamicModalityReliabilityAttention(
            modalities=self.config.modalities,
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.reliability_hidden_dim,
            dropout=self.config.dropout,
            temperature=self.config.reliability_temperature,
        )

        # 3. Cross-Attention Fusion
        self.cross_attention = RetinalCrossAttentionFusion(
            modalities=self.config.modalities,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_fusion_layers,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
        )

        # 4. URR Head
        self.urr_head = UnifiedRetinalRepresentationHead(
            embed_dim=self.config.embed_dim,
            urr_dim=self.config.urr_dim,
            pooling_type=self.config.urr_pooling,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full multimodal retinal fusion pipeline.

        Args:
            modality_features: Dict of input tensors, e.g.
                               {"octa": Tensor, "octb": Tensor, "fundus": Tensor}
            modality_mask: Optional Dict of availability masks [B, 1] per modality.

        Returns:
            Dict containing:
            - 'urr': Unified Retinal Representation vector [B, urr_dim]
            - 'urr_tokens': Fused token sequence [B, N_total, urr_dim]
            - 'modality_weights': Dict of learned reliability weights [B, 1]
            - 'projected_features': Dict of projected tokens [B, N_m, embed_dim]
            - 'reliability_scores': Dict of raw reliability logits [B, 1]
            - 'fused_tokens': Fused cross-attention tokens [B, N_total, embed_dim]
        """
        # 1. Feature Projections
        projected_feats = self.projections(modality_features)

        # 2. Dynamic Modality Reliability Attention
        modulated_feats, modality_weights, reliability_scores = self.dmra(
            projected_features=projected_feats,
            modality_mask=modality_mask,
        )

        # 3. Cross-Attention Fusion
        fused_tokens = self.cross_attention(modulated_features=modulated_feats)

        # 4. Unified Retinal Representation Head
        urr, urr_tokens = self.urr_head(fused_tokens=fused_tokens)

        return {
            "urr": urr,
            "urr_tokens": urr_tokens,
            "modality_weights": modality_weights,
            "projected_features": projected_feats,
            "reliability_scores": reliability_scores,
            "fused_tokens": fused_tokens,
        }

    def save_checkpoint(
        self,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save fusion model state dict and config."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config.__dict__,
            "epoch": epoch,
            "extra_metadata": extra_metadata or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(payload, str(path))
        return path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Tuple["RetinalMultimodalFusionModel", Dict[str, Any]]:
        """Load fusion model from checkpoint."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        ckpt = torch.load(str(path), map_location=device)
        cfg_dict = ckpt.get("config", {})
        config = FusionConfig(**cfg_dict)

        model = cls(config=config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(torch.device(device))

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return model, ckpt
