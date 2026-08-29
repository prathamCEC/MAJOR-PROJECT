"""
Explainability Module for Phase 4 Swin Transformer.

Generates feature activation and attention heatmaps overlaid on input retinal scans
for qualitative model inspection.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from .enums import Modality
from .transforms import get_transforms
from .utils import get_device


class SwinExplainabilityEngine:
    """
    Generates activation heatmaps for Swin Transformer backbones.
    """

    def __init__(self, model: nn.Module, modality: Union[str, Modality]):
        self.model = model
        self.modality = Modality.from_str(modality) if isinstance(modality, str) else modality
        self.model.eval()
        self.device = next(model.parameters()).device

    def generate_heatmap(
        self,
        image_path: Union[str, Path],
        target_layer: Optional[nn.Module] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate normalized activation heatmap for an image.
        """
        path = Path(image_path).resolve()
        with Image.open(path) as img:
            orig_rgb = np.array(img.convert("RGB"))
            if self.modality == Modality.FUNDUS:
                image_pil = img.convert("RGB")
            else:
                image_pil = img.convert("L")

        transform = get_transforms(self.modality, is_training=False, image_size=224)
        input_tensor = transform(image_pil).unsqueeze(0).to(self.device)

        # Forward pass hook for feature activations
        features = []
        def hook_fn(module, input, output):
            features.append(output)

        # Hook last normalization / stage of Swin
        hook_handle = None
        for name, module in self.model.named_modules():
            if "norm" in name or "layers" in name or "stages" in name:
                hook_handle = module.register_forward_hook(hook_fn)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        if hook_handle:
            hook_handle.remove()

        if features:
            feat = features[-1]
            if isinstance(feat, torch.Tensor):
                if feat.ndim == 3:  # [B, L, C]
                    B, L, C = feat.shape
                    H = W = int(np.sqrt(L))
                    if H * W == L:
                        feat_map = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
                    else:
                        feat_map = feat.permute(0, 2, 1).unsqueeze(-1)
                elif feat.ndim == 4:
                    feat_map = feat
                else:
                    feat_map = None

                if feat_map is not None:
                    cam = torch.mean(feat_map, dim=1).squeeze(0).cpu().numpy()
                    cam = np.maximum(cam, 0)
                    cam_min, cam_max = cam.min(), cam.max()
                    if cam_max > cam_min:
                        cam = (cam - cam_min) / (cam_max - cam_min)
                    else:
                        cam = np.zeros_like(cam)
                else:
                    cam = np.ones((7, 7)) * 0.5
            else:
                cam = np.ones((7, 7)) * 0.5
        else:
            cam = np.ones((7, 7)) * 0.5

        # Resize cam to original image shape
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_rgb.shape[1], orig_rgb.shape[0]), Image.BILINEAR)) / 255.0

        return orig_rgb, cam_resized

    def save_explanation(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate and save superimposed explanation overlay.
        """
        orig_img, cam = self.generate_heatmap(image_path)
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_img)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(cam, cmap="jet")
        axes[1].set_title("Activation Heatmap")
        axes[1].axis("off")

        axes[2].imshow(orig_img)
        axes[2].imshow(cam, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay (Research Only)")
        axes[2].axis("off")

        plt.suptitle("Swin Transformer Attention & Feature Activation Map", fontsize=12)
        fig.tight_layout()
        fig.savefig(str(out_p), dpi=150)
        plt.close(fig)

        return out_p
