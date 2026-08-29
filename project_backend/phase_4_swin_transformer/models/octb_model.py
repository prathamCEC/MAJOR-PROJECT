"""
OCT-B Dedicated Swin Transformer Model Wrapper.
"""

from typing import Union
import torch
import torch.nn as nn

from ..enums import Modality
from .swin_factory import SwinRetinalClassifier, create_swin_model


class OCTBSwinModel(nn.Module):
    """
    Dedicated Swin Transformer model for Structural Cross-Sectional OCT (OCT-B).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        model_name: str = "swin_tiny_patch4_window7_224",
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.modality = Modality.OCTB
        self.model = create_swin_model(
            modality=self.modality,
            num_classes=num_classes,
            pretrained=pretrained,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor, pool: bool = False) -> torch.Tensor:
        return self.model.extract_features(x, pool=pool)

    def freeze_backbone(self) -> None:
        self.model.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        self.model.unfreeze_backbone()
