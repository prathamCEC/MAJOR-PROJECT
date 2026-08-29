"""
Swin Transformer Factory for Retinal Disease Analysis.

Instantiates pretrained Swin Transformer backbones with dynamically replaced
classification heads, partial/full backbone freezing, and transfer learning capabilities.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

from ..enums import Modality


class SwinRetinalClassifier(nn.Module):
    """
    Standardized Swin Transformer Classifier for Retinal Imaging.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "swin_tiny_patch4_window7_224",
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ):
        """
        Initialize SwinRetinalClassifier.

        Args:
            num_classes: Number of target prediction classes.
            pretrained: Whether to load ImageNet pretrained weights.
            model_name: Name of Swin architecture in timm / torchvision.
            dropout: Dropout rate before classification head.
            freeze_backbone: If True, freezes backbone parameters for head-only tuning.
        """
        super().__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained
        self.dropout = dropout

        # Attempt instantiation via timm first, then torchvision fallback
        self.use_timm = False
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
            )
            self.use_timm = True
        except Exception:
            # Fallback to torchvision Swin-T
            weights = tv_models.Swin_T_Weights.DEFAULT if pretrained else None
            self.backbone = tv_models.swin_t(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )

        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Logits tensor [B, num_classes]
        """
        return self.backbone(x)

    def extract_features(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        """
        Extract feature representations prior to the classification head.

        Args:
            x: Input tensor [B, 3, H, W]
            pool: If True, returns pooled global feature vector [B, D].
                  If False, returns spatial token sequence [B, N, D] (e.g. [B, 49, 768]).

        Returns:
            Extracted feature tensor.
        """
        if self.use_timm:
            feat = self.backbone.forward_features(x)
            if pool:
                return self.backbone.forward_head(feat, pre_logits=True)
            return feat.reshape(feat.shape[0], -1, feat.shape[-1])
        else:
            feat = self.backbone.features(x)
            norm_feat = self.backbone.norm(feat)
            if pool:
                perm = self.backbone.permute(norm_feat)
                pooled = self.backbone.avgpool(perm)
                return torch.flatten(pooled, 1)
            return norm_feat.reshape(norm_feat.shape[0], -1, norm_feat.shape[-1])

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters, keeping only classifier head trainable."""
        for name, param in self.backbone.named_parameters():
            if "head" not in name and "fc" not in name and "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_swin_model(
    modality: Union[str, Modality],
    num_classes: int = 2,
    pretrained: bool = True,
    model_name: str = "swin_tiny_patch4_window7_224",
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> SwinRetinalClassifier:
    """
    Factory function instantiating a modality-dedicated Swin Transformer.

    Args:
        modality: Target modality (OCTA, OCTB, FUNDUS).
        num_classes: Number of output disease classes.
        pretrained: Load ImageNet pretrained weights.
        model_name: Swin model architecture.
        dropout: Classification head dropout probability.
        freeze_backbone: Initial backbone freeze setting.

    Returns:
        SwinRetinalClassifier instance.
    """
    mod_enum = Modality.from_str(modality) if isinstance(modality, str) else modality
    return SwinRetinalClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
