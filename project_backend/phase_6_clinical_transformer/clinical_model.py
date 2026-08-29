"""
End-to-End Clinical FT-Transformer Model.

Integrates Clinical Feature Tokenization, FT-Transformer Multi-Head Self-Attention,
and Clinical Representation Projection into a unified PyTorch module.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .config import ClinicalTransformerConfig, get_default_clinical_config
from .schema import ClinicalSchema
from .feature_tokenizer import ClinicalFeatureTokenizer
from .ft_transformer import FTTransformerBackbone
from .clinical_representation import ClinicalRepresentationHead


class ClinicalFTTransformerModel(nn.Module):
    """
    Complete Phase 6 Clinical FT-Transformer Architecture.
    """

    def __init__(
        self,
        config: Optional[ClinicalTransformerConfig] = None,
        categorical_cardinalities: Optional[List[int]] = None,
    ):
        super().__init__()
        self.config = config or get_default_clinical_config()
        self.schema = self.config.schema

        num_numerical = self.schema.num_numerical
        # If explicit cardinalities provided (from fitted preprocessor), use them;
        # otherwise, default to binary / 5-category placeholders for initialization
        cards = categorical_cardinalities or [5] * self.schema.num_categorical

        # 1. Feature Tokenizer
        self.tokenizer = ClinicalFeatureTokenizer(
            num_numerical=num_numerical,
            categorical_cardinalities=cards,
            embed_dim=self.config.embed_dim,
            dropout=self.config.dropout,
        )

        # 2. FT-Transformer Backbone
        self.transformer = FTTransformerBackbone(
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            ffn_dropout=self.config.ffn_dropout,
        )

        # 3. Clinical Representation Head
        self.representation_head = ClinicalRepresentationHead(
            embed_dim=self.config.embed_dim,
            clinical_representation_dim=self.config.clinical_representation_dim,
            pooling_strategy=self.config.pooling_strategy,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_num: Numerical features tensor [B, num_numerical]
            x_cat: Categorical index tensor [B, num_categorical]
            key_padding_mask: Optional boolean mask [B, 1 + num_features]

        Returns:
            Dict containing:
            - 'clinical_representation': Output vector [B, clinical_representation_dim] (e.g. 512)
            - 'feature_tokens': Transformer output sequence [B, 1 + num_features, embed_dim]
            - 'cls_token': Unprojected CLS token [B, embed_dim]
        """
        # 1. Tokenize clinical features into sequence with [CLS]
        tokens = self.tokenizer(x_num, x_cat)  # [B, 1 + N_feat, embed_dim]

        # 2. Pass through FT-Transformer blocks
        transformed_tokens = self.transformer(tokens, key_padding_mask=key_padding_mask)

        # 3. Extract Clinical Representation (CR)
        clinical_rep, cls_token = self.representation_head(transformed_tokens)

        return {
            "clinical_representation": clinical_rep,
            "feature_tokens": transformed_tokens,
            "cls_token": cls_token,
        }

    def save_checkpoint(
        self,
        output_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        preprocessor_dict: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model state, config, schema, and preprocessing configuration."""
        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        cfg_dict = {k: v for k, v in self.config.__dict__.items() if k != "schema"}
        payload = {
            "model_state_dict": self.state_dict(),
            "config": cfg_dict,
            "schema": self.schema.to_dict(),
            "categorical_cardinalities": self.tokenizer.cat_tokenizer.cardinalities,
            "preprocessor_state": preprocessor_dict or {},
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
    ) -> Tuple["ClinicalFTTransformerModel", Dict[str, Any]]:
        """Load model from checkpoint."""
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        try:
            ckpt = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location=device)

        schema_dict = ckpt.get("schema", {})
        schema = ClinicalSchema.from_dict(schema_dict)

        cfg_dict = ckpt.get("config", {})
        cfg_dict["schema"] = schema
        config = ClinicalTransformerConfig(**cfg_dict)

        cards = ckpt.get("categorical_cardinalities")
        if cards is None:
            prep_state = ckpt.get("preprocessor_state", {})
            if prep_state and "state" in prep_state and "category_cardinalities" in prep_state["state"]:
                card_map = prep_state["state"]["category_cardinalities"]
                cards = [card_map[col] for col in schema.all_categorical_like if col in card_map]

        model = cls(config=config, categorical_cardinalities=cards)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(torch.device(device))

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        return model, ckpt
