"""
Swin Transformer Grad-CAM Implementation for Retinal Imaging.

Adapts gradient-weighted class activation mapping (Grad-CAM) to the shifted-window
hierarchical Transformer architecture, mapping 1D spatial token representations
back into 2D retinal coordinate spaces with guaranteed hook lifecycle cleanup.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinGradCAM:
    """
    Grad-CAM engine customized for Swin Transformer backbones across OCT-A, OCT-B, and Fundus scans.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None,
    ):
        """
        Initialize SwinGradCAM.

        Args:
            model: Full multimodal or modality-specific model
            target_layer: Specific PyTorch module to hook (if None, auto-discovered)
            target_layer_name: Optional string identifier of target layer
        """
        self.model = model
        self.target_layer = target_layer or self._auto_discover_target_layer(model, target_layer_name)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def _auto_discover_target_layer(self, model: nn.Module, preferred_name: Optional[str] = None) -> nn.Module:
        """
        Automatically discover the most suitable deep spatial feature layer in the model.
        """
        if preferred_name:
            for name, module in model.named_modules():
                if preferred_name in name:
                    return module

        # Strategy 1: Look for last Swin stage block / norm layer
        candidate = None
        for name, module in model.named_modules():
            # Check timm swin structure: layers.3, norm, blocks
            if "layers.3" in name or "stages.3" in name or "norm" in name:
                candidate = module
            elif isinstance(module, (nn.LayerNorm, nn.Conv2d)):
                candidate = module

        if candidate is not None:
            return candidate

        # Strategy 2: Fallback to last non-linear module
        modules = [m for m in model.modules() if len(list(m.children())) == 0]
        if len(modules) > 2:
            return modules[-2]
        return modules[-1]

    def _register_hooks(self) -> None:
        """Register forward and backward hooks with hook storage."""
        self._clear_hooks()

        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            # Handle tuple output from certain Transformer layers
            if isinstance(out, tuple):
                self.activations = out[0].detach()
            else:
                self.activations = out.detach()

        def backward_hook(module: nn.Module, grad_inp: Any, grad_out: Any) -> None:
            if isinstance(grad_out, tuple):
                self.gradients = grad_out[0].detach()
            else:
                self.gradients = grad_out.detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        # Use full backward hook or tensor hook
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h_fwd, h_bwd])

    def _clear_hooks(self) -> None:
        """Remove all active hooks."""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.hook_handles.clear()

    @staticmethod
    def _reshape_tokens_to_spatial(tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert token sequences [B, L, C] to 2D feature maps [B, C, H, W].

        In Swin-T at 224x224 input:
            Stage 4 has L=49 tokens (7x7) and C=768 channels.
        """
        if tensor.ndim == 4:
            # Check if channel-last [B, H, W, C] -> convert to [B, C, H, W]
            if tensor.shape[1] < tensor.shape[-1] and tensor.shape[1] in (7, 14, 28):
                return tensor.permute(0, 3, 1, 2)
            return tensor

        if tensor.ndim == 3:
            B, L, C = tensor.shape
            H = int(np.sqrt(L))
            W = int(np.ceil(L / H))
            if H * W == L:
                # [B, L, C] -> [B, H, W, C] -> [B, C, H, W]
                return tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
            else:
                # Pad to square if required
                return tensor.permute(0, 2, 1).unsqueeze(-1)

        raise ValueError(f"Unsupported activation rank for Grad-CAM: {tensor.ndim} (shape: {tuple(tensor.shape)})")

    def generate_cam(
        self,
        forward_fn: Any,
        target_logit_idx: int = 0,
        input_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a target logit.

        Args:
            forward_fn: Callable executing forward pass and returning target logit scalar or 1D tensor
            target_logit_idx: Index of logit to differentiate (e.g. 0 for binary task head)
            input_size: (H, W) target image dimensions for bilinear interpolation

        Returns:
            Numpy array of shape (H, W) with normalized CAM activations in [0.0, 1.0]
        """
        self._register_hooks()
        try:
            # 1. Forward Pass
            output_logit = forward_fn()

            if not isinstance(output_logit, torch.Tensor):
                raise TypeError(f"Forward function must return a torch.Tensor, got {type(output_logit)}")

            if not output_logit.requires_grad:
                raise RuntimeError("Target output logit does not require grad. Ensure gradient tracking is enabled.")

            # Select target scalar
            if output_logit.ndim == 2:
                target_scalar = output_logit[0, target_logit_idx]
            elif output_logit.ndim == 1:
                target_scalar = output_logit[target_logit_idx]
            else:
                target_scalar = output_logit.sum()

            # 2. Backward Pass
            self.model.zero_grad()
            target_scalar.backward(retain_graph=True)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Grad-CAM failed to capture activations or gradients from target layer.")

            # 3. Reshape Swin tokens to 2D Spatial Maps [B, C, H, W]
            act_2d = self._reshape_tokens_to_spatial(self.activations)   # [B, C, H, W]
            grad_2d = self._reshape_tokens_to_spatial(self.gradients)   # [B, C, H, W]

            # 4. Global Average Pooling over Spatial Dimensions (H, W) -> Channel Weights alpha_k
            alpha = torch.mean(grad_2d, dim=(2, 3), keepdim=True)       # [B, C, 1, 1]

            # 5. Weighted Combination: sum_k alpha_k * A_k
            weighted_cam = torch.sum(alpha * act_2d, dim=1, keepdim=True) # [B, 1, H, W]

            # 6. Apply ReLU
            cam_relu = F.relu(weighted_cam)                             # [B, 1, H, W]

            # 7. Bilinear Interpolation to Input Image Size
            cam_resized = F.interpolate(
                cam_relu,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()                                    # [H, W]

            # 8. Safe Min-Max Normalization to [0.0, 1.0]
            cam_min, cam_max = cam_resized.min(), cam_resized.max()
            if cam_max > cam_min:
                cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min + 1e-8)
            else:
                cam_normalized = np.zeros_like(cam_resized)

            return np.clip(cam_normalized, 0.0, 1.0)

        finally:
            self._clear_hooks()
            self.model.zero_grad()
