"""
End-to-End Retina-Clinical Cross-Attention Fusion Model.

Integrates Multimodal Projections, Bidirectional Cross-Attention Transformer,
Attentive Sequence Pooling, and Gated Multimodal Fusion into the Unified Patient Representation (UPR).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .config import RetinaClinicalConfig, get_default_retina_clinical_config
from .projection import RetinaClinicalProjection
from .cross_attention import BidirectionalRetinaClinicalTransformer
from .pooling import MultimodalTokenPooler
from .fusion import GatedMultimodalFusion
from .validation import validate_fusion_inputs, validate_upr_output


class RetinaClinicalFusionModel(nn.Module):
    """
    Complete Phase 7 Retina-Clinical Cross-Attention & UPR Fusion Architecture.
    """

    def __init__(self, config: Optional[RetinaClinicalConfig] = None):
        super().__init__()
        self.config = config or get_default_retina_clinical_config()

        # 1. Multimodal Feature Projections
        self.projection = RetinaClinicalProjection(
            retinal_input_dim=self.config.retinal_input_dim,
            clinical_input_dim=self.config.clinical_input_dim,
            common_embed_dim=self.config.common_embed_dim,
            dropout=self.config.dropout,
        )

        # 2. Bidirectional Cross-Attention Transformer
        self.cross_attention = BidirectionalRetinaClinicalTransformer(
            embed_dim=self.config.common_embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
        )

        # 3. Multimodal Token Pooling
        self.pooler = MultimodalTokenPooler(
            embed_dim=self.config.common_embed_dim,
            strategy=self.config.pooling_strategy,
        )

        # 4. Gated Multimodal Fusion & UPR Head
        self.fusion = GatedMultimodalFusion(
            embed_dim=self.config.common_embed_dim,
            upr_dim=self.config.upr_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        retinal_representation: torch.Tensor,
        clinical_representation: torch.Tensor,
        retinal_mask: Optional[torch.Tensor] = None,
        clinical_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute full Phase 7 multimodal fusion pipeline.

        Args:
            retinal_representation: Tensor [B, D_ret] or [B, N, D_ret] (Phase 5 URR / tokens)
            clinical_representation: Tensor [B, D_clin] or [B, M, D_clin] (Phase 6 CR / tokens)
            retinal_mask: Optional boolean padding mask [B, N] (True = padding)
            clinical_mask: Optional boolean padding mask [B, M] (True = padding)

        Returns:
            Dict containing:
            - 'upr': Unified Patient Representation vector [B, upr_dim] (e.g. [B, 512])
            - 'enhanced_retina': Contextualized retinal tokens [B, N, common_embed_dim]
            - 'enhanced_clinical': Contextualized clinical tokens [B, M, common_embed_dim]
            - 'gate_weights': Dynamic multimodal gate weights [B, common_embed_dim]
            - 'retinal_vector': Global pooled retinal vector [B, common_embed_dim]
            - 'clinical_vector': Global pooled clinical vector [B, common_embed_dim]
        """
        # Step 1: Input Validation
        validate_fusion_inputs(
            retinal_representation=retinal_representation,
            clinical_representation=clinical_representation,
            config=self.config,
            retinal_mask=retinal_mask,
            clinical_mask=clinical_mask,
        )
        batch_size = retinal_representation.shape[0]

        # Step 2: Modality Space Projections
        proj_ret, proj_clin = self.projection(
            retinal_features=retinal_representation,
            clinical_features=clinical_representation,
        )

        # Step 3: Bidirectional Cross-Attention Contextualization
        enh_ret, enh_clin = self.cross_attention(
            retinal_tokens=proj_ret,
            clinical_tokens=proj_clin,
            retinal_mask=retinal_mask,
            clinical_mask=clinical_mask,
        )

        # Step 4: Token Sequence Pooling
        v_ret, v_clin = self.pooler(
            retinal_tokens=enh_ret,
            clinical_tokens=enh_clin,
            retinal_mask=retinal_mask,
            clinical_mask=clinical_mask,
        )

        # Step 5: Gated Multimodal Fusion & UPR Projection
        upr, fused_vec, gate_weights = self.fusion(
            retinal_vector=v_ret,
            clinical_vector=v_clin,
        )

        # Step 6: Output Integrity Validation
        validate_upr_output(upr=upr, expected_batch_size=batch_size, expected_dim=self.config.upr_dim)

        return {
            "upr": upr,
            "enhanced_retina": enh_ret,
            "enhanced_clinical": enh_clin,
            "gate_weights": gate_weights,
            "retinal_vector": v_ret,
            "clinical_vector": v_clin,
        }

    def save_checkpoint(
        self,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model state, config, and training metadata."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "epoch": epoch,
            "metadata": metadata or {},
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
    ) -> Tuple["RetinaClinicalFusionModel", Dict[str, Any]]:
        """Load model from checkpoint."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location=device)

        cfg_dict = ckpt.get("config", {})
        config = RetinaClinicalConfig.from_dict(cfg_dict)

        model = cls(config=config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(torch.device(device))

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return model, ckpt
