"""
Tests for Swin Transformer Grad-CAM activation mapping.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from phase_4_swin_transformer.models.swin_factory import SwinRetinalClassifier
from phase_10_explainability.swin_gradcam import SwinGradCAM


def test_gradcam_execution_and_hooks():
    model = SwinRetinalClassifier(num_classes=2, pretrained=False)
    model.eval()

    gradcam = SwinGradCAM(model=model)
    assert gradcam.target_layer is not None, "Failed to locate target layer in Swin architecture."

    img_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

    def forward_fn():
        return model(img_tensor)

    # Compute Grad-CAM for class index 0
    cam = gradcam.generate_cam(forward_fn=forward_fn, target_logit_idx=0, input_size=(224, 224))

    # Verify Output Dimensions & Bounds
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert np.isfinite(cam).all()

    # Verify Hooks are completely cleaned up
    assert len(gradcam.hook_handles) == 0


def test_gradcam_constant_activation_safety():
    """Verify that degenerate constant activations do not produce NaNs or division by zero."""
    dummy_model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 1),
    )
    dummy_model.eval()

    gradcam = SwinGradCAM(model=dummy_model, target_layer=dummy_model[0])
    img = torch.zeros(1, 3, 32, 32, requires_grad=True)

    def fwd():
        return dummy_model(img)

    cam = gradcam.generate_cam(forward_fn=fwd, target_logit_idx=0, input_size=(32, 32))
    assert cam.shape == (32, 32)
    assert np.isfinite(cam).all()
    assert (cam >= 0.0).all() and (cam <= 1.0).all()
